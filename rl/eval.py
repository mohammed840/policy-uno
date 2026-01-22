"""
Evaluation module for comparing trained agents.

Runs evaluation tournaments and generates comparison metrics.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from uno.rlcard_env import UnoRLCardEnv, RLCARD_AVAILABLE
from rl.qnet_torch import QNetwork, load_model
from rl.random_agent import RandomAgent


def evaluate_agent(
    env: UnoRLCardEnv,
    agent_fn,
    num_games: int,
    device: torch.device
) -> dict:
    """
    Evaluate an agent over multiple games.
    
    Args:
        env: Uno environment
        agent_fn: Function that takes (state, legal_mask) and returns action
        num_games: Number of games to play
        device: Torch device
        
    Returns:
        Dictionary with evaluation metrics
    """
    wins = 0
    total_reward = 0.0
    total_turns = 0
    
    random_opponent = RandomAgent()
    
    for _ in range(num_games):
        state, legal_mask, current_player = env.reset()
        done = False
        turns = 0
        
        while not done:
            turns += 1
            
            if current_player == 0:
                action = agent_fn(state, legal_mask)
            else:
                action = random_opponent.select_action(state, legal_mask)
            
            state, reward, done, legal_mask, current_player, info = env.step(action)
        
        payoffs = env.get_payoffs()
        if info.get('winner') == 0:
            wins += 1
        total_reward += payoffs[0]
        total_turns += turns
    
    return {
        'games': num_games,
        'wins': wins,
        'win_rate': wins / num_games,
        'avg_reward': total_reward / num_games,
        'avg_turns': total_turns / num_games
    }


def compare_agents(
    env: UnoRLCardEnv,
    agents: dict,
    num_games: int,
    device: torch.device
) -> dict:
    """
    Compare multiple agents in evaluation tournaments.
    
    Args:
        env: Uno environment
        agents: Dictionary of {name: agent_fn}
        num_games: Number of games per agent
        device: Torch device
        
    Returns:
        Dictionary with results for each agent
    """
    results = {}
    
    for name, agent_fn in agents.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_agent(env, agent_fn, num_games, device)
        print(f"  Win rate: {results[name]['win_rate']:.3f}, "
              f"Avg reward: {results[name]['avg_reward']:+.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Uno RL Agents')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID to evaluate')
    parser.add_argument('--games', type=int, default=1000, help='Number of evaluation games')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_players', type=int, default=2, help='Number of players')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if not RLCARD_AVAILABLE:
        print("ERROR: RLCard is not installed.")
        return
    
    # Create environment
    env = UnoRLCardEnv(num_players=args.num_players, seed=args.seed)
    
    # Load models
    project_root = Path(__file__).parent.parent
    run_dir = project_root / 'runs' / args.run_id
    models_dir = project_root / 'models'
    
    agents = {}
    
    # Random baseline
    random_agent = RandomAgent(seed=args.seed)
    agents['Random'] = lambda s, m, ra=random_agent: ra.select_action(s, m)
    
    # Try to load DQN model
    dqn_path = models_dir / 'best_qnet.pt'
    if dqn_path.exists():
        dqn_model = load_model(str(dqn_path), device)
        dqn_model.eval()
        agents['DQN'] = lambda s, m, model=dqn_model, d=device: model.select_greedy_action(s, m, d)
        print(f"Loaded DQN model from {dqn_path}")
    
    # Try to load SARSA model
    sarsa_path = models_dir / 'best_sarsa_qnet.pt'
    if sarsa_path.exists():
        sarsa_model = load_model(str(sarsa_path), device)
        sarsa_model.eval()
        agents['DeepSARSA'] = lambda s, m, model=sarsa_model, d=device: model.select_greedy_action(s, m, d)
        print(f"Loaded DeepSARSA model from {sarsa_path}")
    
    if len(agents) == 1:
        print("No trained models found. Only Random baseline available.")
    
    # Run evaluation
    print(f"\nRunning evaluation tournament ({args.games} games per agent)...")
    print("-" * 60)
    
    results = compare_agents(env, agents, args.games, device)
    
    # Save results
    run_dir.mkdir(parents=True, exist_ok=True)
    
    eval_path = run_dir / 'metrics_eval.csv'
    with open(eval_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['agent', 'games', 'wins', 'win_rate', 'avg_reward', 'avg_turns', 'timestamp'])
        
        timestamp = datetime.now().isoformat()
        for name, metrics in results.items():
            writer.writerow([
                name, metrics['games'], metrics['wins'],
                metrics['win_rate'], metrics['avg_reward'],
                metrics['avg_turns'], timestamp
            ])
    
    print("-" * 60)
    print(f"Evaluation results saved to: {eval_path}")
    
    # Summary
    print("\nSummary:")
    for name, metrics in sorted(results.items(), key=lambda x: -x[1]['win_rate']):
        print(f"  {name:12s}: Win Rate = {metrics['win_rate']:.3f}, "
              f"Avg Reward = {metrics['avg_reward']:+.3f}")


if __name__ == '__main__':
    main()
