import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

def simulate_one_game(graph):
    """
    Simulates a single game of Atlas between two random players.
    Returns: (game_length, winner, start_node)
    """
    available_nodes = set(graph.nodes())
    if not available_nodes:
        return (0, None, None)

    start_node = random.choice(list(available_nodes))
    available_nodes.remove(start_node)
    
    current_node = start_node
    turn_count = 1
    current_player = 1

    while True:
        potential_moves = list(graph.successors(current_node))
        valid_moves = [move for move in potential_moves if move in available_nodes]
        
        if not valid_moves:
            winner = 2 if current_player == 1 else 1
            return (turn_count, winner, start_node)
        
        next_move = random.choice(valid_moves)
        
        available_nodes.remove(next_move)
        current_node = next_move
        current_player = 2 if current_player == 1 else 1
        turn_count += 1

def run_simulation_analysis(graph, name, num_simulations=10000):
    """
    Runs a Monte Carlo simulation and analyzes the results.
    """
    print(f"\n{'='*20} Simulating: {name} ({num_simulations} games) {'='*20}")

    game_results = [simulate_one_game(graph) for _ in range(num_simulations)]
    df_results = pd.DataFrame(game_results, columns=['game_length', 'winner', 'start_node'])

    avg_length = df_results['game_length'].mean()
    p1_win_rate = (df_results['winner'] == 1).mean()
    p2_win_rate = (df_results['winner'] == 2).mean()

    print(f"Average Game Length: {avg_length:.2f} turns")
    print(f"Player 1 Win Rate: {p1_win_rate:.2%}")
    print(f"Player 2 Win Rate: {p2_win_rate:.2%}")

    wins_by_start_node = df_results[df_results['winner'] == 1]['start_node'].value_counts()
    games_by_start_node = df_results['start_node'].value_counts()
    
    win_rate_by_start = (wins_by_start_node / games_by_start_node).dropna().sort_values(ascending=False)
    
    print("\nTop 5 'Luckiest' Starting Nodes (Highest P1 Win Rate):")
    print(win_rate_by_start.head(5))

    plt.figure(figsize=(12, 6))
    sns.histplot(df_results['game_length'], bins=50, kde=True)
    plt.title(f'Distribution of Game Lengths for {name}', fontsize=16)
    plt.xlabel("Number of Turns")
    plt.ylabel("Frequency")
    plt.axvline(avg_length, color='red', linestyle='--', label=f'Average Length: {avg_length:.2f}')
    plt.legend()
    plt.show()
