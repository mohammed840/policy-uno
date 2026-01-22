"""
DeepSARSA Tournament-based Training for Uno.

Training loop (same as DQN but with on-policy TD target):
1. Each iteration simulates a tournament of n games
2. Collect experiences during games
3. Update Q-network using TD learning: y = r + γ * Q(s', a_next)
   where a_next is the actual next action chosen by the policy
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
from rl.qnet_torch import QNetwork, save_model
from rl.random_agent import RandomAgent


class SARSABuffer:
    """On-policy experience buffer for SARSA (stores s, a, r, s', a')."""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
        done: bool,
        legal_mask: np.ndarray
    ):
        """Add SARSA experience to buffer."""
        experience = (state, action, reward, next_state, next_action, done, legal_mask)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> tuple:
        """Sample a batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, next_actions, dones, legal_masks = (
            [], [], [], [], [], [], []
        )
        
        for idx in indices:
            s, a, r, ns, na, d, lm = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            next_actions.append(na)
            dones.append(d)
            legal_masks.append(lm)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(next_actions, dtype=np.int64),
            np.array(dones, dtype=np.float32),
            np.array(legal_masks, dtype=np.float32)
        )
    
    def clear(self):
        """Clear buffer for on-policy learning."""
        self.buffer = []
        self.position = 0
    
    def __len__(self) -> int:
        return len(self.buffer)


def run_game_sarsa(
    env: UnoRLCardEnv,
    q_network: QNetwork,
    epsilon: float,
    device: torch.device,
    sarsa_buffer: Optional[SARSABuffer] = None,
    training: bool = True
) -> dict:
    """
    Run a single game and collect SARSA experiences.
    
    SARSA requires (s, a, r, s', a') tuples where a' is the actual next action.
    """
    state, legal_mask, current_player = env.reset()
    
    done = False
    turn_count = 0
    
    # Track experiences for player 0
    player_0_trajectory = []
    
    # Create random agent for opponents
    random_agent = RandomAgent()
    
    # Select first action for player 0
    if current_player == 0:
        action = q_network.select_action(state, legal_mask, epsilon, device)
    else:
        action = random_agent.select_action(state, legal_mask)
    
    while not done:
        turn_count += 1
        
        # Store current state/action for player 0
        if current_player == 0:
            player_0_trajectory.append({
                'state': state.copy(),
                'action': action,
                'legal_mask': legal_mask.copy()
            })
        
        # Take step
        next_state, reward, done, next_legal_mask, next_player, info = env.step(action)
        
        # Select next action
        if not done:
            if next_player == 0:
                next_action = q_network.select_action(
                    next_state, next_legal_mask, epsilon, device
                )
            else:
                next_action = random_agent.select_action(next_state, next_legal_mask)
        else:
            next_action = 0  # Dummy action for terminal state
        
        # Update player 0's last experience with SARSA tuple
        if current_player == 0 and player_0_trajectory:
            player_0_trajectory[-1]['next_state'] = next_state.copy()
            player_0_trajectory[-1]['next_action'] = next_action
            player_0_trajectory[-1]['reward'] = reward if done else 0.0
            player_0_trajectory[-1]['done'] = done
        
        state = next_state
        legal_mask = next_legal_mask
        action = next_action
        current_player = next_player
    
    # Get final payoffs
    payoffs = env.get_payoffs()
    winner = info.get('winner', 0)
    
    # Assign terminal reward
    if player_0_trajectory:
        player_0_trajectory[-1]['reward'] = payoffs[0]
        player_0_trajectory[-1]['done'] = True
    
    # Add to SARSA buffer
    if sarsa_buffer and training:
        for exp in player_0_trajectory:
            sarsa_buffer.push(
                exp['state'],
                exp['action'],
                exp['reward'],
                exp['next_state'],
                exp['next_action'],
                exp['done'],
                exp['legal_mask']
            )
    
    return {
        'winner': winner,
        'player_0_win': winner == 0,
        'reward': payoffs[0],
        'turns': turn_count,
        'experiences': len(player_0_trajectory)
    }


def sarsa_train_step(
    q_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    sarsa_buffer: SARSABuffer,
    batch_size: int,
    gamma: float,
    device: torch.device
) -> float:
    """
    Perform one SARSA training step.
    
    DeepSARSA TD target: y = r + γ * Q(s', a')
    where a' is the actual next action taken.
    
    Returns loss value.
    """
    if len(sarsa_buffer) < batch_size:
        return 0.0
    
    # Sample batch
    states, actions, rewards, next_states, next_actions, dones, _ = sarsa_buffer.sample(batch_size)
    
    # Convert to tensors
    states_t = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    rewards_t = torch.FloatTensor(rewards).to(device)
    next_states_t = torch.FloatTensor(next_states).to(device)
    next_actions_t = torch.LongTensor(next_actions).to(device)
    dones_t = torch.FloatTensor(dones).to(device)
    
    # Current Q values: Q(s, a)
    q_values = q_network(states_t)
    q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
    
    # Target Q values: Q(s', a') - the actual next action
    with torch.no_grad():
        next_q_values = q_network(next_states_t)
        # Use the actual next action (SARSA is on-policy)
        next_q_selected = next_q_values.gather(1, next_actions_t.unsqueeze(1)).squeeze(1)
        targets = rewards_t + gamma * next_q_selected * (1 - dones_t)
    
    # Compute loss and update
    loss = F.mse_loss(q_values, targets)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
    optimizer.step()
    
    return loss.item()


def run_tournament_sarsa(
    env: UnoRLCardEnv,
    q_network: QNetwork,
    epsilon: float,
    device: torch.device,
    num_games: int,
    sarsa_buffer: Optional[SARSABuffer] = None,
    training: bool = True
) -> dict:
    """Run a tournament of multiple games for SARSA."""
    wins = 0
    total_reward = 0.0
    total_turns = 0
    game_results = []
    
    for _ in range(num_games):
        result = run_game_sarsa(env, q_network, epsilon, device, sarsa_buffer, training)
        
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
    algorithm: str = 'sarsa'
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
            log_entry = {k: v for k, v in result.items() if k != 'experiences'}
            log_entry = convert_to_serializable(log_entry)
            f.write(json.dumps(log_entry) + '\n')


def main():
    parser = argparse.ArgumentParser(description='DeepSARSA Training for Uno')
    parser.add_argument('--iters', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--games_per_iter', type=int, default=100, help='Games per tournament')
    parser.add_argument('--alpha', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=0.95, help='Initial epsilon')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum epsilon')
    parser.add_argument('--kappa', type=float, default=0.995, help='Epsilon decay factor')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--buffer_size', type=int, default=100000, help='SARSA buffer size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--num_players', type=int, default=2, help='Number of players')
    parser.add_argument('--save_freq', type=int, default=100, help='Checkpoint save frequency')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID')
    
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
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S_sarsa')
    project_root = Path(__file__).parent.parent
    run_dir = project_root / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'plots').mkdir(exist_ok=True)
    (run_dir / 'tournaments').mkdir(exist_ok=True)
    
    # Save config
    config = vars(args)
    config['device'] = str(device)
    config['run_id'] = run_id
    config['algorithm'] = 'sarsa'
    config['timestamp'] = datetime.now().isoformat()
    
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Run directory: {run_dir}")
    
    # Check RLCard
    if not RLCARD_AVAILABLE:
        print("ERROR: RLCard is not installed. Please install with: pip install rlcard")
        return
    
    # Create environment
    env = UnoRLCardEnv(num_players=args.num_players, seed=args.seed)
    
    # Create network
    q_network = QNetwork().to(device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=args.alpha)
    sarsa_buffer = SARSABuffer(capacity=args.buffer_size)
    
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
    
    print(f"\nStarting DeepSARSA training: {args.iters} iterations, {args.games_per_iter} games/iter")
    print("-" * 60)
    
    for iteration in range(1, args.iters + 1):
        # Run tournament
        tournament = run_tournament_sarsa(
            env, q_network, epsilon, device,
            num_games=args.games_per_iter,
            sarsa_buffer=sarsa_buffer,
            training=True
        )
        
        # Training steps
        total_loss = 0.0
        num_updates = min(args.games_per_iter, len(sarsa_buffer) // args.batch_size)
        
        for _ in range(num_updates):
            loss = sarsa_train_step(
                q_network, optimizer,
                sarsa_buffer, args.batch_size, args.gamma, device
            )
            total_loss += loss
        
        avg_loss = total_loss / max(num_updates, 1)
        
        # Decay epsilon
        epsilon = max(args.epsilon_min, epsilon * args.kappa)
        
        # Log metrics
        timestamp = datetime.now().isoformat()
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration, 'sarsa', epsilon, tournament['avg_reward'],
                tournament['win_rate'], tournament['avg_turns'],
                avg_loss, len(sarsa_buffer), timestamp
            ])
        
        # Save tournament log
        save_tournament_log(run_dir, iteration, tournament, 'sarsa')
        
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
                save_model(q_network, str(project_root / 'models' / 'best_sarsa_qnet.pt'), optimizer)
                print(f"  → New best model saved (win rate: {best_win_rate:.3f})")
            
            save_model(q_network, str(run_dir / f'checkpoint_iter_{iteration}.pt'), optimizer)
    
    # Final save
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    save_model(q_network, str(models_dir / 'best_sarsa_qnet.pt'), optimizer)
    save_model(q_network, str(run_dir / 'final_model.pt'), optimizer)
    
    print("-" * 60)
    print(f"Training complete! Best win rate: {best_win_rate:.3f}")
    print(f"Model saved to: {models_dir / 'best_sarsa_qnet.pt'}")
    print(f"Logs saved to: {run_dir}")
    
    # Generate plots
    print("\nGenerating plots...")
    try:
        from rl.plots import generate_all_plots
        generate_all_plots(str(run_id))
        print("Plots generated successfully!")
    except Exception as e:
        print(f"Plot generation failed: {e}")


if __name__ == '__main__':
    main()
