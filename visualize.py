import pickle
import os
from atlas_analyzer.visualization import visualize_graph

def load_graph(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache file '{path}' not found.")
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    # Define cache file paths
    country_cache = "graph_countries.pkl"
    city_cache = "graph_cities.pkl"
    combined_cache = "graph_combined.pkl"

    # Load graphs
    graph_countries = load_graph(country_cache)
    graph_cities = load_graph(city_cache)
    graph_combined = load_graph(combined_cache)

    # Visualize each graph
    visualize_graph(graph_countries, title="Atlas Graph: Countries Only", figsize=(22, 22))
    visualize_graph(graph_cities, title="Atlas Graph: Cities Only", figsize=(40, 40))
    visualize_graph(graph_combined, title="Atlas Graph: Countries + Cities", figsize=(50, 50))

if __name__ == "__main__":
    main()