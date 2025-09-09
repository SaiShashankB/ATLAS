import pickle
import os
from atlas_analyzer.simulation import run_simulation_analysis

if __name__ == "__main__":
    cache_files = {
        "Countries Only": os.path.join("outputs", "graph_cache", "graph_countries.pkl"),
        "Cities Only": os.path.join("outputs", "graph_cache", "graph_cities.pkl"),
        "Countries + Cities": os.path.join("outputs", "graph_cache", "graph_combined.pkl")
    }

    for name, path in cache_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                graph = pickle.load(f)
            # Use a higher number of simulations for more stable results
            run_simulation_analysis(graph, name, num_simulations=25000)
        else:
            print(f"Error: Cache '{path}' not found. Please run '01_create_graphs.py' first.")