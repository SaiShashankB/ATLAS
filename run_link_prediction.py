import pickle
import os
from atlas_analyzer.link_prediction import (
    create_node_features, 
    get_pyg_data,
    run_node2vec_link_prediction,
    run_simple_gnn_link_prediction
)

if __name__ == "__main__":
    COUNTRY_CACHE = os.path.join("outputs", "graph_cache", "graph_countries.pkl")

    if not os.path.exists(COUNTRY_CACHE):
        print(f"Error: Cache '{COUNTRY_CACHE}' not found. Please run '01_create_graphs.py' first.")
    else:
        with open(COUNTRY_CACHE, 'rb') as f:
            G_country = pickle.load(f)
        
        # --- Run Node2Vec ---
        run_node2vec_link_prediction(G_country)
        
        # --- Run GNN ---
        features, node_map = create_node_features(G_country)
        pyg_data = get_pyg_data(G_country, features, node_map)
        run_simple_gnn_link_prediction(pyg_data)