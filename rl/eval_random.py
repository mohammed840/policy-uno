"""
Evaluate trained DQN agent against random opponents.

Matches the paper's evaluation setup:
- Agent plays against random opponent(s)
- Reports win rate and average reward
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from uno.rlcard_env import UnoRLCardEnv
from .qnet_torch import load_model


def evaluate_against_random(
    model_path: str,
    num_games: int = 100,
    device: str = 'auto'
) -> dict:
    """
    Evaluate a trained model against random opponents.
    
    Returns:
        dict with win_rate, avg_reward, and game details
    """
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    
    # Load model
    q_network = load_model(model_path, device)
    q_network.eval()
    
    # Create environment
    env = UnoRLCardEnv(num_players=2)
    
    wins = 0
    total_reward = 0.0
    game_lengths = []
    
    print(f"\nEvaluating over {num_games} games...")
    
    for game_idx in range(num_games):
        state, legal_mask, current_player = env.reset()
        done = False
        turn_count = 0
        
        while not done:
            turn_count += 1
            
            if current_player == 0:
                # DQN agent (greedy, no exploration)
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = q_network(state_t).cpu().numpy()[0]
                    # Mask illegal actions
                    q_values[legal_mask == 0] = float('-inf')
                    action = int(np.argmax(q_values))
            else:
                # Random opponent
                legal_actions = np.where(legal_mask > 0)[0]
                action = int(np.random.choice(legal_actions))
            
            next_state, reward, done, next_legal_mask, current_player, info = env.step(action)
            state = next_state
            legal_mask = next_legal_mask
        
        # Get final result
        payoffs = env.get_payoffs()
        winner = info.get('winner', 0)
        
        if winner == 0:
            wins += 1
        total_reward += payoffs[0]
        game_lengths.append(turn_count)
        
        # Progress update
        if (game_idx + 1) % 20 == 0:
            print(f"  Games: {game_idx + 1}/{num_games}, Win Rate: {wins/(game_idx+1):.3f}")
    
    win_rate = wins / num_games
    avg_reward = total_reward / num_games
    avg_length = np.mean(game_lengths)
    
    return {
        'games': num_games,
        'wins': wins,
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'avg_game_length': avg_length,
        'std_game_length': np.std(game_lengths)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate DQN against random opponents')
    parser.add_argument('--model', type=str, default='models/best_qnet.pt',
                        help='Path to trained model')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games to play')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Find model path
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent
        model_path = project_root / args.model
    
    if not model_path.exists():
        print(f"Error: Model not found at {args.model}")
        return
    
    # Run evaluation
    results = evaluate_against_random(str(model_path), args.games, args.device)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS (vs Random Opponent)")
    print("="*60)
    print(f"Games Played:     {results['games']}")
    print(f"Wins:             {results['wins']}")
    print(f"Win Rate:         {results['win_rate']:.1%}")
    print(f"Average Reward:   {results['avg_reward']:+.3f}")
    print(f"Avg Game Length:  {results['avg_game_length']:.1f} ± {results['std_game_length']:.1f} turns")
    print("="*60)
    
    # Compare to paper
    print("\nComparison to Reference Paper:")
    print(f"  Paper Win Rate:  62.2%")
    print(f"  Our Win Rate:    {results['win_rate']:.1%}")
    print(f"  Paper Avg Reward: +0.244")
    print(f"  Our Avg Reward:   {results['avg_reward']:+.3f}")


if __name__ == '__main__':
    main()
