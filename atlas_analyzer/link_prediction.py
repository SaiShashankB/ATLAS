import pandas as pd
import networkx as nx
import random
import pickle
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

def create_node_features(graph):
    """
    Creates one-hot encoded features for the first and last letter of each country name.
    """
    nodes = list(graph.nodes())
    node_map = {node: i for i, node in enumerate(nodes)}
    features = torch.zeros(len(nodes), 52, dtype=torch.float)
    for node, i in node_map.items():
        if not node: continue
        first_letter_idx = ord(node[0].lower()) - ord('a')
        if 0 <= first_letter_idx < 26: features[i, first_letter_idx] = 1.0
        last_letter_idx = ord(node[-1].lower()) - ord('a')
        if 0 <= last_letter_idx < 26: features[i, 26 + last_letter_idx] = 1.0
    return features, node_map

def get_pyg_data(graph, features, node_map):
    """Converts NetworkX graph to PyG Data object."""
    edge_index = torch.tensor([(node_map[u], node_map[v]) for u, v in graph.edges()]).t().contiguous()
    data = Data(x=features, edge_index=edge_index)
    return data

def run_node2vec_link_prediction(graph):
    print("\n" + "="*20 + " Part 1: Link Prediction with Node2Vec (Tuned) " + "="*20)

    test_edges = random.sample(list(graph.edges()), int(0.2 * graph.number_of_edges()))
    train_graph = graph.copy()
    train_graph.remove_edges_from(test_edges)

    test_neg_edges = []
    while len(test_neg_edges) < len(test_edges):
        u, v = random.sample(list(graph.nodes()), 2)
        if not graph.has_edge(u, v): test_neg_edges.append((u, v))

    print("Training Node2Vec model with tuned parameters...")
    n2v = Node2Vec(train_graph, dimensions=128, walk_length=5, num_walks=200, p=1.0, q=0.9, workers=4, quiet=True)
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    embeddings = {node: model.wv[node] for node in graph.nodes()}

    train_pos_edges = list(train_graph.edges())
    train_neg_edges = []
    while len(train_neg_edges) < len(train_pos_edges):
        u, v = random.sample(list(graph.nodes()), 2)
        if not graph.has_edge(u, v): train_neg_edges.append((u, v))

    X_train = [(embeddings[u] * embeddings[v]) for u, v in train_pos_edges] + \
              [(embeddings[u] * embeddings[v]) for u, v in train_neg_edges]
    y_train = [1] * len(train_pos_edges) + [0] * len(train_neg_edges)

    print("Training a classifier on node embeddings...")
    # HYPERPARAMETER TUNING: Increased max_iter for convergence.
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(X_train, y_train)

    X_test = [(embeddings[u] * embeddings[v]) for u, v in test_edges] + \
             [(embeddings[u] * embeddings[v]) for u, v in test_neg_edges]
    y_test = [1] * len(test_edges) + [0] * len(test_neg_edges)

    predictions = classifier.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, predictions)
    print(f"✅ Node2Vec Link Prediction AUC: {auc_score:.4f}")

class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)
    def forward(self, x, edge_index):
        return self.conv1(x, edge_index).relu()

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, x_i, x_j):
        x = x_i * x_j
        return self.lin2(self.lin1(x).relu()).squeeze()

def gnn_train(model, predictor, data, optimizer, criterion):
    # This function remains the same
    model.train(); predictor.train(); optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    pos_out = predictor(z[data.edge_label_index[0]], z[data.edge_label_index[1]])
    neg_edge_index = torch.randint(0, data.num_nodes, data.edge_label_index.size(), dtype=torch.long)
    neg_out = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])
    out = torch.cat([pos_out, neg_out])
    y = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))])
    loss = criterion(out, y)
    loss.backward(); optimizer.step()
    return loss.item()

@torch.no_grad()
def gnn_test(model, predictor, data):
    # This function remains the same
    model.eval(); predictor.eval()
    z = model(data.x, data.edge_index)
    pos_preds = predictor(z[data.edge_label_index[0]], z[data.edge_label_index[1]]).sigmoid()
    neg_edge_index = torch.randint(0, data.num_nodes, data.edge_label_index.size(), dtype=torch.long)
    neg_preds = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]]).sigmoid()
    preds = torch.cat([pos_preds, neg_preds])
    y = torch.cat([torch.ones(pos_preds.size(0)), torch.zeros(neg_preds.size(0))])
    return roc_auc_score(y.cpu().numpy(), preds.cpu().numpy())

def run_simple_gnn_link_prediction(pyg_data):
    print("\n" + "="*20 + " Part 2: Link Prediction with Simple GNN (Tuned) " + "="*20)
    transform = RandomLinkSplit(is_undirected=False, add_negative_train_samples=False)
    train_data, val_data, test_data = transform(pyg_data)

    gnn = SimpleGNN(in_channels=pyg_data.num_node_features, out_channels=128)
    predictor = LinkPredictor(in_channels=128, hidden_channels=256, out_channels=1)

    optimizer = torch.optim.Adam(list(gnn.parameters()) + list(predictor.parameters()), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("Training Simple GNN model...")
    for epoch in range(1, 201):
        loss = gnn_train(gnn, predictor, train_data, optimizer, criterion)
        if epoch % 20 == 0:
            val_auc = gnn_test(gnn, predictor, val_data)
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

    test_auc = gnn_test(gnn, predictor, test_data)
    print(f"✅ Simple GNN Link Prediction Test AUC: {test_auc:.4f}")
