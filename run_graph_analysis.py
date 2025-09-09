import pickle
import pandas as pd
from atlas_analyzer.graph_analysis import analyze_graph_properties
import os

if __name__ == "__main__":
    cache_files = {
        "Countries Only": os.path.join("outputs", "graph_cache", "graph_countries.pkl"),
        "Cities Only": os.path.join("outputs", "graph_cache", "graph_cities.pkl"),
        "Countries + Cities": os.path.join("outputs", "graph_cache", "graph_combined.pkl")
    }
    
    all_stats = {}
    for name, path in cache_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                graph = pickle.load(f)
            all_stats[name] = analyze_graph_properties(graph, name)
        else:
            print(f"Error: Cache '{path}' not found. Please run '01_create_graphs.py' first.")

    if all_stats:
        df = pd.DataFrame(all_stats).T.drop(columns=["TopTraps"])
        print("\n\n" + "="*25 + " Final Comparative Summary " + "="*25)
        print(df)