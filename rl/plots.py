"""
Plot generation from training logs.

Generates:
- fig1_network_arch.{png,svg} - Network architecture diagram
- fig2_dqn_avg_reward_over_iters.{png,svg} - DQN training curve
- fig3_sarsa_avg_reward_over_iters.{png,svg} - SARSA training curve
- fig_training_comparison_winrate.{png,svg} - Win rate comparison bar chart
- fig_training_comparison_reward.{png,svg} - Reward comparison bar chart
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


def load_training_metrics(run_dir: Path) -> dict:
    """Load training metrics from CSV."""
    import csv
    
    metrics_path = run_dir / 'metrics_train.csv'
    if not metrics_path.exists():
        return {}
    
    data = {'dqn': [], 'sarsa': []}
    
    with open(metrics_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row.get('algorithm', 'dqn')
            entry = {
                'iteration': int(row['iteration']),
                'epsilon': float(row['epsilon']),
                'avg_reward': float(row['avg_reward_tournament']),
                'win_rate': float(row['win_rate_tournament']),
                'avg_turns': float(row['avg_turns']),
            }
            if algo in data:
                data[algo].append(entry)
    
    return data


def load_eval_metrics(run_dir: Path) -> dict:
    """Load evaluation metrics from CSV."""
    import csv
    
    eval_path = run_dir / 'metrics_eval.csv'
    if not eval_path.exists():
        return {}
    
    data = {}
    
    with open(eval_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            agent = row['agent']
            data[agent] = {
                'win_rate': float(row['win_rate']),
                'avg_reward': float(row['avg_reward']),
                'games': int(row['games']),
            }
    
    return data


def plot_network_architecture(save_dir: Path):
    """
    Generate network architecture diagram.
    
    Architecture: 420 → 512 → 512 → 61
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Layer definitions
    layers = [
        {'name': 'Input', 'size': 420, 'x': 1, 'color': '#3498db'},
        {'name': 'Dense + ReLU', 'size': 512, 'x': 4, 'color': '#2ecc71'},
        {'name': 'Dense + ReLU', 'size': 512, 'x': 6.5, 'color': '#2ecc71'},
        {'name': 'Output', 'size': 61, 'x': 9, 'color': '#e74c3c'},
    ]
    
    y_center = 2
    box_height = 2
    box_width = 1.5
    
    # Draw layers
    for i, layer in enumerate(layers):
        x = layer['x']
        
        # Draw box
        rect = mpatches.FancyBboxPatch(
            (x - box_width/2, y_center - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=layer['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Draw text
        ax.text(x, y_center + 0.3, layer['name'],
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax.text(x, y_center - 0.3, f"({layer['size']})",
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # Draw arrow to next layer
        if i < len(layers) - 1:
            next_x = layers[i + 1]['x']
            arrow_start = x + box_width/2 + 0.1
            arrow_end = next_x - box_width/2 - 0.1
            ax.annotate('', xy=(arrow_end, y_center), xytext=(arrow_start, y_center),
                       arrowprops=dict(arrowstyle='->', lw=2, color='#34495e'))
    
    # Title
    ax.text(5, 3.7, 'Q-Network Architecture', ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax.text(5, 0.3, 'State (7×4×15 flattened) → Q-values for 61 actions',
            ha='center', va='center', fontsize=9, style='italic', color='#7f8c8d')
    
    plt.tight_layout()
    
    # Save
    fig.savefig(save_dir / 'fig1_network_arch.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(save_dir / 'fig1_network_arch.svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"  Saved: fig1_network_arch.png/svg")


def plot_training_curve(
    data: list,
    algorithm: str,
    save_dir: Path,
    fig_num: int
):
    """
    Plot training curve (average reward over iterations).
    
    Style matches Stanford DeepSARSA paper (Figure 3):
    - Single clean line with no raw data overlay
    - Proper axis labels
    - Shows progression from negative to positive rewards
    """
    if not data:
        print(f"  Skipping {algorithm} training curve (no data)")
        return
    
    iterations = [d['iteration'] for d in data]
    rewards = [d['avg_reward'] for d in data]
    
    # Apply moving average smoothing for cleaner visualization
    # Use adaptive window size based on data length
    window = max(1, min(50, len(rewards) // 20))
    if window > 1 and len(rewards) > window:
        rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
        # Pad the beginning to maintain array length
        pad_size = len(rewards) - len(rewards_smooth)
        rewards_smooth = np.concatenate([rewards[:pad_size], rewards_smooth])
    else:
        rewards_smooth = np.array(rewards)
    
    # Create figure matching Stanford paper style
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Single clean line plot (no raw data overlay to match Stanford style)
    ax.plot(iterations, rewards_smooth, color='#1f77b4', linewidth=1.5)
    
    # Axis labels matching Stanford paper
    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title(algorithm, fontsize=14, fontweight='bold')
    
    # Set x-axis limits
    ax.set_xlim(0, max(iterations))
    
    # Add subtle grid for readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Adjust tick formatting for large iteration counts
    if max(iterations) >= 10000:
        ax.ticklabel_format(style='plain', axis='x')
    
    plt.tight_layout()
    
    algo_lower = algorithm.lower().replace('deep', '').replace(' ', '_')
    filename = f'fig{fig_num}_{algo_lower}_avg_reward_over_iters'
    fig.savefig(save_dir / f'{filename}.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(save_dir / f'{filename}.svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"  Saved: {filename}.png/svg")


def plot_comparison_winrate(eval_data: dict, save_dir: Path):
    """Plot win rate comparison bar chart."""
    if not eval_data:
        print("  Skipping win rate comparison (no eval data)")
        return
    
    agents = list(eval_data.keys())
    win_rates = [eval_data[a]['win_rate'] for a in agents]
    
    colors = {'DQN': '#3498db', 'DeepSARSA': '#2ecc71', 'Random': '#95a5a6'}
    bar_colors = [colors.get(a, '#9b59b6') for a in agents]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(agents, win_rates, color=bar_colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Win Rate')
    ax.set_title('Agent Comparison: Win Rate')
    ax.set_ylim(0, max(win_rates) * 1.2)
    
    plt.tight_layout()
    
    fig.savefig(save_dir / 'fig_training_comparison_winrate.png', dpi=150, bbox_inches='tight')
    fig.savefig(save_dir / 'fig_training_comparison_winrate.svg', bbox_inches='tight')
    plt.close(fig)
    
    print("  Saved: fig_training_comparison_winrate.png/svg")


def plot_comparison_reward(eval_data: dict, save_dir: Path):
    """Plot average reward comparison bar chart."""
    if not eval_data:
        print("  Skipping reward comparison (no eval data)")
        return
    
    agents = list(eval_data.keys())
    rewards = [eval_data[a]['avg_reward'] for a in agents]
    
    colors = {'DQN': '#3498db', 'DeepSARSA': '#2ecc71', 'Random': '#95a5a6'}
    bar_colors = [colors.get(a, '#9b59b6') for a in agents]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(agents, rewards, color=bar_colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        y_offset = 0.02 if height >= 0 else -0.08
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                f'{reward:+.3f}', ha='center', va=va, fontweight='bold')
    
    ax.set_ylabel('Average Reward')
    ax.set_title('Agent Comparison: Average Reward')
    ax.axhline(y=0, color='#7f8c8d', linestyle='--', alpha=0.5)
    
    # Set y limits with some padding
    y_min, y_max = min(rewards), max(rewards)
    y_range = y_max - y_min if y_max != y_min else 1
    ax.set_ylim(y_min - 0.2 * y_range, y_max + 0.2 * y_range)
    
    plt.tight_layout()
    
    fig.savefig(save_dir / 'fig_training_comparison_reward.png', dpi=150, bbox_inches='tight')
    fig.savefig(save_dir / 'fig_training_comparison_reward.svg', bbox_inches='tight')
    plt.close(fig)
    
    print("  Saved: fig_training_comparison_reward.png/svg")


def generate_all_plots(run_id: str):
    """Generate all plots for a training run."""
    project_root = Path(__file__).parent.parent
    run_dir = project_root / 'runs' / run_id
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating plots for run: {run_id}")
    print(f"Output directory: {plots_dir}")
    print("-" * 40)
    
    # Network architecture (always generate)
    plot_network_architecture(plots_dir)
    
    # Load training metrics
    train_data = load_training_metrics(run_dir)
    
    # Training curves
    if train_data.get('dqn'):
        plot_training_curve(train_data['dqn'], 'DQN', plots_dir, 2)
    
    if train_data.get('sarsa'):
        plot_training_curve(train_data['sarsa'], 'DeepSARSA', plots_dir, 3)
    
    # Load eval metrics
    eval_data = load_eval_metrics(run_dir)
    
    # Comparison charts
    plot_comparison_winrate(eval_data, plots_dir)
    plot_comparison_reward(eval_data, plots_dir)
    
    print("-" * 40)
    print("Plot generation complete!")


def main():
    parser = argparse.ArgumentParser(description='Generate plots from training logs')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID to generate plots for')
    
    args = parser.parse_args()
    
    generate_all_plots(args.run_id)


if __name__ == '__main__':
    main()
