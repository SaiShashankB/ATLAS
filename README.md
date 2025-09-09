# Atlas Game Graph Analysis

This project performs a deep graph-theoretic analysis of the word game "Atlas". It constructs graphs from country and city data, analyzes their structural properties, simulates live gameplay, detects communities, and trains neural networks for link prediction.

## Repository Structure

- **`data/`**: Contains the raw input CSV files for countries and cities.
- **`outputs/graph_cache/`**: Stores the processed NetworkX graph objects (`.pkl` files) to speed up subsequent runs.
- **`outputs/plots/`**: Contains all generated visualizations, organized into subfolders for each graph type.
- **`atlas_analyzer/`**: The core Python package containing all logic for the project.
  - `data_creation.py`: Functions to read CSVs and create/cache the graphs.
  - `graph_analysis.py`: Functions for static graph property analysis (degree, PageRank, k-core, etc.) and plotting.
  - `simulation.py`: Functions for running Monte Carlo simulations of live games.
  - `community_detection.py`: Functions to perform and visualize community detection.
  - `link_prediction.py`: Code for Node2Vec and GNN link prediction models.
- **`create_graphs.py`**: Runner script to generate and cache the graph objects.
- **`run_graph_analysis.py`**: Runner script to perform the static analysis from Task 1.
- **`run_simulation.py`**: Runner script to simulate live games.
- **`run_community_detection.py`**: Runner script for the community detection task.
- **`run_link_prediction.py`**: Runner script for the link prediction bonus task.
- **`requirements.txt`**: A list of all required Python packages.

## How to Run

1.  **Setup:**

    - Place your data files (`world-data-2023.csv`, etc.) in the `data/` directory.
    - Install all required packages:
      ```bash
      pip install -r requirements.txt
      ```
    - For graph visualization, you may need to install Graphviz:
      ```bash
      # On Ubuntu/Debian
      sudo apt-get install -y graphviz graphviz-dev
      # On macOS (using Homebrew)
      brew install graphviz
      ```

2.  **Execution:**
    Run the scripts in numerical order. Each script performs a distinct part of the analysis.

    ```bash
    # Step 1: Create and cache the graphs
    python create_graphs.py

    # Step 2: Run the static graph analysis and generate plots
    python run_graph_analysis.py

    # Step 3: Run the live game simulations
    python run_simulation.py

    # Step 4: Run the community detection analysis
    python run_community_detection.py

    # Step 5: Run the link prediction models
    python run_link_prediction.py
    ```
