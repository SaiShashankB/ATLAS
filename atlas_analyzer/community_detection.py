import networkx as nx
import matplotlib.pyplot as plt
import random
from itertools import islice
from networkx.drawing.nx_pydot import graphviz_layout

from networkx.algorithms.community import louvain_communities, girvan_newman
from networkx.algorithms.community.quality import modularity

def visualize_communities(G, communities, title, figsize=(30, 30)):
    """
    Visualizes graph with nodes colored by community.
    """
    print(f"🎨 Visualizing '{title}'...")
    try:
        pos = graphviz_layout(G, prog="fdp")
    except ImportError:
        print("Warning: `pygraphviz` or `pydot` not found. Falling back to spring_layout.")
        pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    node_to_community_map = {node: i for i, comm in enumerate(communities) for node in comm}
    node_colors = [node_to_community_map.get(node, -1) for node in G.nodes()]
    
    node_size = 1100
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, arrows=False, edge_color="gray", width=0.6)
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_size, node_color=node_colors, 
        cmap=plt.cm.get_cmap('tab20'), alpha=0.85, edgecolors="black", linewidths=0.5
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="black")

    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def run_community_analysis(graph):
    """Performs community detection using Louvain and Girvan-Newman algorithms."""
    print(f"--- Analyzing Communities for Graph ({graph.number_of_nodes()} nodes) ---")
    
    UG = graph.to_undirected()

    print("\n--- Algorithm 1: Louvain Method ---")
    louvain_comms = [set(c) for c in louvain_communities(UG)]
    print(f"Found {len(louvain_comms)} communities.")

    print("\n--- Algorithm 2: Girvan-Newman ---")
    print("(Searching for the optimal partition...)")
    gn_iterator = girvan_newman(UG)
    
    best_gn_modularity = -1.0
    best_gn_comms = []
    for communities in gn_iterator:
        current_modularity = modularity(UG, communities)
        if current_modularity > best_gn_modularity:
            best_gn_modularity = current_modularity
            best_gn_comms = [set(c) for c in communities]
    
    print(f"Found {len(best_gn_comms)} communities at the point of highest modularity.")

    print("\n" + "="*20 + " Quality Assessment " + "="*20)
    louvain_modularity = modularity(UG, louvain_comms)
    
    print(f"Louvain Modularity Score (Q): {louvain_modularity:.4f}")
    print(f"Girvan-Newman Best Modularity Score (Q): {best_gn_modularity:.4f}")
    
    if louvain_modularity > best_gn_modularity:
        best_communities = louvain_comms
        best_algo_name = "Louvain Method"
        best_modularity = louvain_modularity
    else:
        best_communities = best_gn_comms
        best_algo_name = "Girvan-Newman"
        best_modularity = best_gn_modularity
        
    print(f"\nInsight: The {best_algo_name} produced the partition with the highest modularity score.")
    
    viz_title = (f'Community Detection on the Atlas Country Graph\n'
                 f'({best_algo_name} | Q = {best_modularity:.3f})')
    
    visualize_communities(UG, best_communities, title=viz_title)