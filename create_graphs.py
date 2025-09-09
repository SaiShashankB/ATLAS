import pandas as pd
from atlas_analyzer.data_creation import load_or_create_graph
import os

if __name__ == "__main__":
    # Define paths
    COUNTRY_CSV = os.path.join("data", "world-data-2023.csv")
    CITY_CSV = os.path.join("data", "largest-cities-by-population-2025.csv")
    
    COUNTRY_CACHE = os.path.join("outputs", "graph_cache", "graph_countries.pkl")
    CITY_CACHE = os.path.join("outputs", "graph_cache", "graph_cities.pkl")
    COMBINED_CACHE = os.path.join("outputs", "graph_cache", "graph_combined.pkl")

    # --- 1. Country Only ---
    df_countries = pd.read_csv(COUNTRY_CSV)
    countries = df_countries["Country"].dropna().unique()
    graph_countries = load_or_create_graph(COUNTRY_CACHE, countries)
    print(f"Country graph created with {graph_countries.number_of_nodes()} nodes.")

    # --- 2. City Only ---
    df_cities = pd.read_csv(CITY_CSV)
    top_cities = df_cities.sort_values(by="population", ascending=False).head(500)["city"].dropna().unique()
    graph_cities = load_or_create_graph(CITY_CACHE, top_cities)
    print(f"City graph created with {graph_cities.number_of_nodes()} nodes.")

    # --- 3. Country + City ---
    combined_names = list(countries) + list(top_cities)
    graph_combined = load_or_create_graph(COMBINED_CACHE, combined_names)
    print(f"Combined graph created with {graph_combined.number_of_nodes()} nodes.")