import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set_theme(style="whitegrid", context="talk")

def strategic_trap_analysis(G):
    """Identifies 'probabilistic' traps using a dynamic, percentile-based threshold."""
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    trap_scores = {}
    all_out_degrees = list(out_degrees.values())
    if not all_out_degrees: return []
    
    low_out_degree_threshold = np.percentile(all_out_degrees, 5.0)
    print(f"(Dynamic trap threshold: out-degree <= {low_out_degree_threshold:.2f})")

    for node, out_degree in out_degrees.items():
        if 1 <= out_degree <= low_out_degree_threshold:
            successors = list(G.successors(node))
            if not successors: continue
            avg_successor_in_degree = np.mean([in_degrees.get(succ, 0) for succ in successors])
            score = avg_successor_in_degree / out_degree
            trap_scores[node] = score
    return sorted(trap_scores.items(), key=lambda item: item[1], reverse=True)

def kcore_stats(G):
    """Calculates k-core decomposition stats."""
    UG = G.to_undirected()
    if UG.number_of_nodes() == 0: return 0, {}
    core_nums = nx.core_number(UG)
    kmax = max(core_nums.values()) if core_nums else 0
    core_sizes = {k: sum(1 for x in core_nums.values() if x == k) for k in sorted(set(core_nums.values()))}
    return kmax, core_sizes

def alphabet_transition_matrix(G):
    """Creates a 26x26 matrix of letter-to-letter transition frequencies."""
    letters = [chr(i) for i in range(ord('a'), ord('z')+1)]
    idx = {ch: i for i, ch in enumerate(letters)}
    M = np.zeros((26, 26), dtype=int)
    for u, v in G.edges():
        a = u[-1].lower() if u else ""
        b = v[0].lower() if v else ""
        if a in idx and b in idx:
            M[idx[a], idx[b]] += 1
    return letters, M

def generate_and_save_plots(name, in_deg, out_deg, core_sizes, M_data, output_dir):
    """Creates and saves a suite of beautiful, uncluttered analysis plots."""
    print(f"📊 Generating and saving plots to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    # ... (plotting code from previous steps)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Degree Distributions for {name}', fontsize=24)
    sns.histplot(list(out_deg.values()), bins=25, kde=True, ax=axes[0]).set_title("Out-Degree ('Offensive Power')")
    sns.histplot(list(in_deg.values()), bins=25, kde=True, ax=axes[1]).set_title("In-Degree ('Vulnerability')")
    plt.savefig(os.path.join(output_dir, "degree_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(12, 7))
    sns.barplot(x=list(core_sizes.keys()), y=list(core_sizes.values()), palette="viridis")
    plt.title(f'k-core Decomposition for {name}', fontsize=20)
    plt.xlabel("Core Number (k)")
    plt.ylabel("Number of Nodes in Shell")
    plt.savefig(os.path.join(output_dir, "k_core_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    letters, M = M_data
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(M, xticklabels=letters, yticklabels=letters, cmap="rocket", annot=True, fmt="d", ax=ax)
    ax.set_title(f'Alphabet Transition Heatmap for {name}', fontsize=24)
    ax.set_xlabel("Starting Letter of Next Word")
    ax.set_ylabel("Ending Letter of Previous Word")
    plt.savefig(os.path.join(output_dir, "alphabet_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Plots saved successfully.")

def analyze_graph_properties(G, name):
    """Performs the full static graph analysis for Task 1."""
    print(f"\n{'='*20} Analyzing: {name} {'='*20}")
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    print(f"Basic: {num_nodes} nodes, {num_edges} edges, Density: {density:.4f}")

    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    
    top_traps = strategic_trap_analysis(G)
    pagerank = nx.pagerank(G)
    kmax, core_sizes = kcore_stats(G)
    sccs = list(nx.strongly_connected_components(G))
    scc_size = len(max(sccs, key=len)) if sccs else 0
    M_data = alphabet_transition_matrix(G)
    
    # Print insights
    print("\n--- Insights ---")
    print("Top 5 'Offensive' Moves:", sorted(out_deg.items(), key=lambda item: item[1], reverse=True)[:5])
    print("Top 5 'Strategic Traps':", top_traps[:5])
    print("Top 5 'Hubs' (PageRank):", sorted(pagerank.items(), key=lambda item: item[1], reverse=True)[:5])
    print(f"Most Resilient Core (k-core): k_max = {kmax}")
    print(f"Largest 'Safe Zone' (SCC): {scc_size} nodes")

    output_dir = os.path.join("outputs", "plots", f"{name.replace(' ', '_').lower()}_plots")
    generate_and_save_plots(name, in_deg, out_deg, core_sizes, M_data, output_dir)
    
    return {"Nodes": num_nodes, "Edges": num_edges, "Density": density, "kmax": kmax, "LargestSCC": scc_size, "TopTraps": top_traps}