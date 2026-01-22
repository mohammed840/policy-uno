"""
Game Statistics from Simulations.

Runs 100,000 simulations of Uno games to analyze:
1. Game length distribution (turns per game)
2. First player advantage
3. Common game situations

Generates publication-quality plots matching the reference style.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from uno.rlcard_env import UnoRLCardEnv, RLCARD_AVAILABLE


def run_random_game(env: UnoRLCardEnv) -> dict:
    """
    Run a single game with random agents.
    
    Returns dict with game statistics.
    """
    state, legal_mask, current_player = env.reset()
    starting_player = current_player
    
    done = False
    turn_count = 0
    player_turn_counts = {0: 0, 1: 0}
    cards_played = {0: 0, 1: 0}
    draws = {0: 0, 1: 0}
    
    while not done:
        turn_count += 1
        player_turn_counts[current_player] += 1
        
        # Random action selection from legal actions
        legal_actions = np.where(legal_mask > 0)[0]
        action = np.random.choice(legal_actions)
        
        # Track draw vs play
        if action == 60:  # Draw action
            draws[current_player] += 1
        else:
            cards_played[current_player] += 1
        
        # Take step
        next_state, reward, done, next_legal_mask, current_player, info = env.step(action)
        
        state = next_state
        legal_mask = next_legal_mask
    
    # Get winner
    payoffs = env.get_payoffs()
    winner = 0 if payoffs[0] > 0 else 1
    
    return {
        'winner': winner,
        'starting_player': starting_player,
        'first_player_won': winner == starting_player,
        'turns': turn_count,
        'player_0_turns': player_turn_counts[0],
        'player_1_turns': player_turn_counts[1],
        'player_0_cards_played': cards_played[0],
        'player_1_cards_played': cards_played[1],
        'player_0_draws': draws[0],
        'player_1_draws': draws[1],
    }


def run_simulations(num_games: int = 100000, seed: int = 42) -> list:
    """Run multiple game simulations."""
    np.random.seed(seed)
    
    if not RLCARD_AVAILABLE:
        print("ERROR: RLCard not available!")
        return []
    
    env = UnoRLCardEnv(num_players=2)
    results = []
    
    print(f"\n📊 Running {num_games:,} game simulations...")
    for i in tqdm(range(num_games), desc="Simulating games"):
        result = run_random_game(env)
        results.append(result)
    
    return results


def analyze_and_plot(results: list, output_dir: Path):
    """Analyze results and generate plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract statistics
    turns = [r['turns'] for r in results]
    first_player_wins = sum(1 for r in results if r['first_player_won'])
    player_0_wins = sum(1 for r in results if r['winner'] == 0)
    
    num_games = len(results)
    
    # Calculate statistics
    mean_turns = np.mean(turns)
    median_turns = np.median(turns)
    mode_turns = int(max(set(turns), key=turns.count))
    min_turns = min(turns)
    max_turns = max(turns)
    std_turns = np.std(turns)
    
    first_player_advantage = first_player_wins / num_games * 100
    
    print("\n" + "="*60)
    print("📈 GAME STATISTICS FROM SIMULATIONS")
    print("="*60)
    print(f"\n🎮 Total Games Simulated: {num_games:,}")
    print(f"\n📊 Game Length Statistics:")
    print(f"   • Mean turns:   {mean_turns:.1f}")
    print(f"   • Median turns: {median_turns:.1f}")
    print(f"   • Mode turns:   {mode_turns}")
    print(f"   • Std Dev:      {std_turns:.1f}")
    print(f"   • Range:        {min_turns} - {max_turns}")
    print(f"\n🏆 First Player Advantage:")
    print(f"   • First player win rate: {first_player_advantage:.1f}%")
    print(f"   • Player 0 wins: {player_0_wins:,} ({player_0_wins/num_games*100:.1f}%)")
    print("="*60)
    
    # Set up plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    
    # -------------------------------------------------------------------------
    # PLOT 1: Turns Distribution (matching reference style)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram with same style as reference
    bins = np.arange(0, max(turns) + 5, 5)
    counts, edges, patches = ax.hist(turns, bins=bins, color='#4a7298', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Turns per Game', fontsize=12)
    ax.set_ylabel('Games', fontsize=12)
    ax.set_title('Turns Distribution', fontsize=14, fontweight='bold')
    ax.set_xlim(0, min(max(turns) + 10, 350))
    
    # Add text annotation
    fig.text(0.5, 0.02, 
             f'Distribution of turns per game from {num_games:,} simulated games',
             ha='center', fontsize=11, style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plot_path_1 = output_dir / 'turns_distribution.png'
    plt.savefig(plot_path_1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ Saved: {plot_path_1}")
    
    # -------------------------------------------------------------------------
    # PLOT 2: Game Length Box Plot with Statistics
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    ax1 = axes[0]
    bp = ax1.boxplot([turns], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#4a7298')
    bp['boxes'][0].set_alpha(0.7)
    ax1.set_ylabel('Turns per Game', fontsize=12)
    ax1.set_xticklabels(['All Games'])
    ax1.set_title('Game Length Distribution', fontsize=14, fontweight='bold')
    
    # Add statistics text
    stats_text = f'Mean: {mean_turns:.1f}\nMedian: {median_turns:.1f}\nMode: {mode_turns}\nStd: {std_turns:.1f}'
    ax1.text(1.3, np.percentile(turns, 75), stats_text, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # First player advantage bar chart
    ax2 = axes[1]
    bars = ax2.bar(['First Player\nWins', 'Second Player\nWins'], 
                   [first_player_wins, num_games - first_player_wins],
                   color=['#2ecc71', '#e74c3c'])
    ax2.set_ylabel('Number of Games', fontsize=12)
    ax2.set_title('First Player Advantage', fontsize=14, fontweight='bold')
    
    # Add percentage labels on bars
    for bar, count in zip(bars, [first_player_wins, num_games - first_player_wins]):
        pct = count / num_games * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + num_games*0.01,
                f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plot_path_2 = output_dir / 'game_statistics.png'
    plt.savefig(plot_path_2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {plot_path_2}")
    
    # -------------------------------------------------------------------------
    # PLOT 3: Cumulative Distribution Function
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_turns = np.sort(turns)
    cdf = np.arange(1, len(sorted_turns) + 1) / len(sorted_turns)
    
    ax.plot(sorted_turns, cdf, color='#4a7298', linewidth=2)
    ax.fill_between(sorted_turns, cdf, alpha=0.3, color='#4a7298')
    
    # Add percentile lines
    for pct in [25, 50, 75, 90]:
        val = np.percentile(turns, pct)
        ax.axhline(y=pct/100, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=val, color='gray', linestyle='--', alpha=0.5)
        ax.text(val + 2, pct/100 + 0.02, f'{pct}th: {val:.0f} turns', fontsize=9)
    
    ax.set_xlabel('Turns per Game', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution of Game Length', fontsize=14, fontweight='bold')
    ax.set_xlim(0, np.percentile(turns, 99))
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plot_path_3 = output_dir / 'turns_cdf.png'
    plt.savefig(plot_path_3, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {plot_path_3}")
    
    # -------------------------------------------------------------------------
    # Save statistics to JSON
    # -------------------------------------------------------------------------
    stats = {
        'num_games': num_games,
        'timestamp': datetime.now().isoformat(),
        'game_length': {
            'mean': float(mean_turns),
            'median': float(median_turns),
            'mode': int(mode_turns),
            'std': float(std_turns),
            'min': int(min_turns),
            'max': int(max_turns),
            'percentiles': {
                '25': float(np.percentile(turns, 25)),
                '50': float(np.percentile(turns, 50)),
                '75': float(np.percentile(turns, 75)),
                '90': float(np.percentile(turns, 90)),
                '95': float(np.percentile(turns, 95)),
                '99': float(np.percentile(turns, 99)),
            }
        },
        'first_player_advantage': {
            'first_player_wins': first_player_wins,
            'first_player_win_rate': first_player_advantage,
        },
        'player_0_wins': player_0_wins,
        'player_0_win_rate': player_0_wins / num_games * 100,
    }
    
    stats_path = output_dir / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved: {stats_path}")
    
    return stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Uno game simulations')
    parser.add_argument('--games', type=int, default=100000, help='Number of games to simulate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='plots', help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent.parent / args.output
    
    # Run simulations
    results = run_simulations(num_games=args.games, seed=args.seed)
    
    if results:
        # Analyze and plot
        stats = analyze_and_plot(results, output_dir)
        
        print(f"\n🎉 All done! Check the '{output_dir}' directory for plots.")


if __name__ == '__main__':
    main()
