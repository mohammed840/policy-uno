"""
DQN Tournament-based Training for Uno.

Training loop:
1. Each iteration simulates a tournament of n games
2. Collect experiences during games
3. Update Q-network using TD learning: y = r + γ * max_a' Q(s', a')
4. Decay epsilon: ε ← κ * ε
5. Log metrics and save checkpoints
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import csv

import numpy as np
import torch
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from uno.rlcard_env import UnoRLCardEnv, RLCARD_AVAILABLE
from rl.qnet_torch import (
    QNetwork, ReplayBuffer, create_target_network,
    update_target_network, save_model
)
from rl.random_agent import RandomAgent


def run_game(
    env: UnoRLCardEnv,
    q_network: QNetwork,
    epsilon: float,
    device: torch.device,
    replay_buffer: Optional[ReplayBuffer] = None,
    training: bool = True
) -> dict:
    """
    Run a single game and collect experiences.
    
    Returns dict with game statistics.
    """
    state, legal_mask, current_player = env.reset()
    
    done = False
    turn_count = 0
    player_experiences = {i: [] for i in range(env.num_players)}
    
    # Create random agent for opponents
    random_agent = RandomAgent()
    
    while not done:
        turn_count += 1
        
        if current_player == 0:
            # Our agent (being trained)
            action = q_network.select_action(state, legal_mask, epsilon, device)
        else:
            # Opponent (random)
            action = random_agent.select_action(state, legal_mask)
        
        # Store experience
        player_experiences[current_player].append({
            'state': state.copy(),
            'action': action,
            'legal_mask': legal_mask.copy()
        })
        
        # Take step
        next_state, reward, done, next_legal_mask, current_player, info = env.step(action)
        
        # Update last experience with reward and next state
        if player_experiences[0]:
            player_experiences[0][-1]['next_state'] = next_state.copy()
            player_experiences[0][-1]['reward'] = reward if done else 0.0
            player_experiences[0][-1]['done'] = done
            player_experiences[0][-1]['next_legal_mask'] = next_legal_mask.copy()
        
        state = next_state
        legal_mask = next_legal_mask
    
    # Get final payoffs
    payoffs = env.get_payoffs()
    winner = info.get('winner', 0)
    
    # Assign terminal rewards to all experiences
    for exp in player_experiences[0]:
        if 'reward' not in exp:
            exp['reward'] = 0.0
        exp['done'] = exp.get('done', False)
        if 'next_state' not in exp:
            exp['next_state'] = state.copy()
        if 'next_legal_mask' not in exp:
            exp['next_legal_mask'] = legal_mask.copy()
    
    # Terminal reward assignment
    if player_experiences[0]:
        player_experiences[0][-1]['reward'] = payoffs[0]
        player_experiences[0][-1]['done'] = True
    
    # Add to replay buffer
    if replay_buffer and training:
        for exp in player_experiences[0]:
            replay_buffer.push(
                exp['state'],
                exp['action'],
                exp['reward'],
                exp['next_state'],
                exp['done'],
                exp['next_legal_mask']
            )
    
    return {
        'winner': winner,
        'player_0_win': winner == 0,
        'reward': payoffs[0],
        'turns': turn_count,
        'experiences': len(player_experiences[0])
    }


def train_step(
    q_network: QNetwork,
    target_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device
) -> float:
    """
    Perform one training step on a batch from replay buffer.
    
    DQN TD target: y = r + γ * max_a' Q_target(s', a')
    
    Returns loss value.
    """
    if len(replay_buffer) < batch_size:
        return 0.0
    
    # Sample batch
    states, actions, rewards, next_states, dones, next_legal_masks = replay_buffer.sample(batch_size)
    
    # Convert to tensors
    states_t = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_t = torch.FloatTensor(rewards).to(device)
    next_states_t = torch.FloatTensor(next_states).to(device)
    dones_t = torch.FloatTensor(dones).to(device)
    next_legal_masks_t = torch.FloatTensor(next_legal_masks).to(device)
    
    # Current Q values
    q_values = q_network(states_t)
    q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
    
    # Target Q values with legal masking
    with torch.no_grad():
        next_q_values = target_network(next_states_t)
        # Mask illegal actions with large negative value
        next_q_values = next_q_values - (1 - next_legal_masks_t) * 1e9
        next_q_max = next_q_values.max(dim=1)[0]
        targets = rewards_t + gamma * next_q_max * (1 - dones_t)
    
    # Compute loss and update
    loss = F.mse_loss(q_values, targets)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
    optimizer.step()
    
    return loss.item()


def run_tournament(
    env: UnoRLCardEnv,
    q_network: QNetwork,
    epsilon: float,
    device: torch.device,
    num_games: int,
    replay_buffer: Optional[ReplayBuffer] = None,
    training: bool = True
) -> dict:
    """
    Run a tournament of multiple games.
    
    Returns tournament statistics.
    """
    wins = 0
    total_reward = 0.0
    total_turns = 0
    game_results = []
    
    for _ in range(num_games):
        result = run_game(env, q_network, epsilon, device, replay_buffer, training)
        
        if result['player_0_win']:
            wins += 1
        total_reward += result['reward']
        total_turns += result['turns']
        game_results.append(result)
    
    return {
        'games': num_games,
        'wins': wins,
        'win_rate': wins / num_games,
        'avg_reward': total_reward / num_games,
        'avg_turns': total_turns / num_games,
        'game_results': game_results
    }


def save_tournament_log(
    run_dir: Path,
    iteration: int,
    tournament_results: dict,
    algorithm: str = 'dqn'
):
    """Save tournament game logs to JSONL file."""
    tournaments_dir = run_dir / 'tournaments'
    tournaments_dir.mkdir(exist_ok=True)
    
    log_path = tournaments_dir / f'{algorithm}_iter_{iteration}.jsonl'
    
    def convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    with open(log_path, 'w') as f:
        for result in tournament_results['game_results']:
            # Remove experiences from log to save space
            log_entry = {k: v for k, v in result.items() if k != 'experiences'}
            log_entry = convert_to_serializable(log_entry)
            f.write(json.dumps(log_entry) + '\n')


def main():
    parser = argparse.ArgumentParser(description='DQN Training for Uno')
    parser.add_argument('--iters', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--games_per_iter', type=int, default=100, help='Games per tournament')
    parser.add_argument('--alpha', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=0.95, help='Initial epsilon')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum epsilon')
    parser.add_argument('--kappa', type=float, default=0.995, help='Epsilon decay factor')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--buffer_size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--target_update', type=int, default=10, help='Target network update frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--num_players', type=int, default=2, help='Number of players')
    parser.add_argument('--save_freq', type=int, default=100, help='Checkpoint save frequency')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID (auto-generated if not provided)')
    
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
    
    # Create run directory
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S_dqn')
    project_root = Path(__file__).parent.parent
    run_dir = project_root / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'plots').mkdir(exist_ok=True)
    (run_dir / 'tournaments').mkdir(exist_ok=True)
    
    # Save config
    config = vars(args)
    config['device'] = str(device)
    config['run_id'] = run_id
    config['algorithm'] = 'dqn'
    config['timestamp'] = datetime.now().isoformat()
    
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Run directory: {run_dir}")
    
    # Check RLCard availability
    if not RLCARD_AVAILABLE:
        print("ERROR: RLCard is not installed. Please install with: pip install rlcard")
        return
    
    # Create environment
    env = UnoRLCardEnv(num_players=args.num_players, seed=args.seed)
    
    # Create networks
    q_network = QNetwork().to(device)
    target_network = create_target_network(q_network).to(device)
    
    # Create optimizer and replay buffer
    optimizer = torch.optim.Adam(q_network.parameters(), lr=args.alpha)
    replay_buffer = ReplayBuffer(capacity=args.buffer_size)
    
    # Initialize metrics CSV
    metrics_path = run_dir / 'metrics_train.csv'
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iteration', 'algorithm', 'epsilon', 'avg_reward_tournament',
            'win_rate_tournament', 'avg_turns', 'loss', 'buffer_size', 'timestamp'
        ])
    
    # Training loop
    epsilon = args.epsilon_start
    best_win_rate = 0.0
    
    print(f"\nStarting DQN training: {args.iters} iterations, {args.games_per_iter} games/iter")
    print("-" * 60)
    
    for iteration in range(1, args.iters + 1):
        # Run tournament
        tournament = run_tournament(
            env, q_network, epsilon, device,
            num_games=args.games_per_iter,
            replay_buffer=replay_buffer,
            training=True
        )
        
        # Training steps
        total_loss = 0.0
        num_updates = min(args.games_per_iter, len(replay_buffer) // args.batch_size)
        
        for _ in range(num_updates):
            loss = train_step(
                q_network, target_network, optimizer,
                replay_buffer, args.batch_size, args.gamma, device
            )
            total_loss += loss
        
        avg_loss = total_loss / max(num_updates, 1)
        
        # Update target network
        if iteration % args.target_update == 0:
            update_target_network(q_network, target_network)
        
        # Decay epsilon
        epsilon = max(args.epsilon_min, epsilon * args.kappa)
        
        # Log metrics
        timestamp = datetime.now().isoformat()
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration, 'dqn', epsilon, tournament['avg_reward'],
                tournament['win_rate'], tournament['avg_turns'],
                avg_loss, len(replay_buffer), timestamp
            ])
        
        # Save tournament log
        save_tournament_log(run_dir, iteration, tournament, 'dqn')
        
        # Progress logging
        if iteration % 10 == 0 or iteration == 1:
            print(f"Iter {iteration:4d} | ε={epsilon:.3f} | "
                  f"WinRate={tournament['win_rate']:.3f} | "
                  f"AvgReward={tournament['avg_reward']:+.3f} | "
                  f"Loss={avg_loss:.4f}")
        
        # Save checkpoint
        if iteration % args.save_freq == 0 or tournament['win_rate'] > best_win_rate:
            if tournament['win_rate'] > best_win_rate:
                best_win_rate = tournament['win_rate']
                save_model(q_network, str(project_root / 'models' / 'best_qnet.pt'), optimizer)
                print(f"  → New best model saved (win rate: {best_win_rate:.3f})")
            
            save_model(q_network, str(run_dir / f'checkpoint_iter_{iteration}.pt'), optimizer)
    
    # Final save
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    save_model(q_network, str(models_dir / 'best_qnet.pt'), optimizer)
    save_model(q_network, str(run_dir / 'final_model.pt'), optimizer)
    
    print("-" * 60)
    print(f"Training complete! Best win rate: {best_win_rate:.3f}")
    print(f"Model saved to: {models_dir / 'best_qnet.pt'}")
    print(f"Logs saved to: {run_dir}")
    
    # Generate plots
    print("\nGenerating plots...")
    try:
        from rl.plots import generate_all_plots
        generate_all_plots(str(run_id))
        print("Plots generated successfully!")
    except Exception as e:
        print(f"Plot generation failed: {e}")
        print("Run 'python -m rl.plots --run_id {run_id}' manually to generate plots.")


if __name__ == '__main__':
    main()
