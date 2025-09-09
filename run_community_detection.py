import pickle
import os
from atlas_analyzer.community_detection import run_community_analysis

if __name__ == "__main__":
    COUNTRY_CACHE = os.path.join("outputs", "graph_cache", "graph_countries.pkl")

    if not os.path.exists(COUNTRY_CACHE):
        print(f"Error: Cache '{COUNTRY_CACHE}' not found. Please run '01_create_graphs.py' first.")
    else:
        with open(COUNTRY_CACHE, 'rb') as f:
            G_country = pickle.load(f)
        
        run_community_analysis(G_country)