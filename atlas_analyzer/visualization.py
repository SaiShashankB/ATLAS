import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G, title="Atlas Graph", figsize=(30, 30)):
    """
    Visualize a directed graph with a suitable layout.
    Uses Graphviz if available, otherwise falls back to spring layout.
    """
    print(f"🎨 Visualizing '{title}'...")
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(G, prog="fdp")
    except Exception:
        print("Warning: `pygraphviz` or `pydot` not found. Falling back to spring_layout.")
        pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    node_colors = [G.in_degree(n) for n in G.nodes()]
    node_size = 1100

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        G, pos, ax=ax, alpha=0.25, arrows=True, edge_color="gray", width=0.6
    )
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_size,
        node_color=node_colors,
        cmap=plt.cm.plasma,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.3
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=8, font_color="black"
    )

    if node_colors:
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.plasma,
            norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label("In-degree (receivability)", fontsize=12)

    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()