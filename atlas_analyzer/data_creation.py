import pandas as pd
import networkx as nx
import os
import pickle

def create_atlas_graph_from_list(names):
    """Builds an Atlas graph from a list of names."""
    names = [str(n).strip() for n in names if pd.notna(n)]
    names = list(set(names))

    G = nx.DiGraph()
    G.add_nodes_from(names)

    start_dict = {}
    for n in names:
        if not n: continue
        first = n[0].lower()
        start_dict.setdefault(first, []).append(n)

    for a in names:
        if not a: continue
        last_letter = a[-1].lower()
        if last_letter in start_dict:
            for b in start_dict[last_letter]:
                if b != a:
                    G.add_edge(a, b)
    return G

def load_or_create_graph(cache_path, name_list):
    """Loads a graph from a pickle cache or creates and saves it."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        print(f"✅ Loading graph from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            graph = pickle.load(f)
    else:
        print(f"⏳ Cache not found. Creating graph for '{cache_path}'...")
        graph = create_atlas_graph_from_list(name_list)
        print(f"💾 Saving new graph to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(graph, f)
    return graph