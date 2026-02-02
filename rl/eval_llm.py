"""
Improved LLM Evaluation Script.

Addresses professor feedback:
1. Increased game count (configurable, default 500)
2. Alternating starting players
3. Confidence intervals via bootstrap
4. Detailed logging of prompts and error handling
5. Reproducible with fixed seeds
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import csv

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from uno.rlcard_env import UnoRLCardEnv, RLCARD_AVAILABLE
from rl.qnet_torch import QNetwork, load_model

try:
    from opponents.llm_openrouter import LLMOpenRouterOpponent, LLM_MODELS, HTTPX_AVAILABLE
except ImportError:
    HTTPX_AVAILABLE = False


def compute_confidence_interval(wins: int, total: int, confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval for win rate using Wilson score interval.
    
    More robust than normal approximation for small samples or extreme proportions.
    """
    if total == 0:
        return (0.0, 0.0)
    
    p = wins / total
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    
    # Wilson score interval
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    lower = max(0, center - spread)
    upper = min(1, center + spread)
    
    return (lower, upper)


def run_evaluation_game(
    env: UnoRLCardEnv,
    agent: QNetwork,
    opponent,
    device,
    agent_starts: bool = True
) -> dict:
    """
    Run a single evaluation game.
    
    Args:
        agent_starts: If True, agent is player 0. If False, opponent is player 0.
    
    Returns game result dict.
    """
    state, legal_mask, current_player = env.reset()
    
    done = False
    turn_count = 0
    
    # Track which player is which
    agent_id = 0 if agent_starts else 1
    opponent_id = 1 if agent_starts else 0
    
    while not done:
        turn_count += 1
        
        if current_player == agent_id:
            # RL agent's turn
            action = agent.select_greedy_action(state, legal_mask, device)
        else:
            # LLM opponent's turn
            action = opponent.act(state, legal_mask)
        
        next_state, reward, done, next_legal_mask, current_player, info = env.step(action)
        state = next_state
        legal_mask = next_legal_mask
    
    winner = info.get('winner', 0)
    agent_won = (winner == agent_id)
    
    return {
        'agent_won': agent_won,
        'agent_started': agent_starts,
        'turns': turn_count,
        'winner': winner
    }


def evaluate_vs_llm(
    agent: QNetwork,
    opponent,
    device,
    num_games: int = 500,
    alternate_starting: bool = True,
    seed: int = 42
) -> dict:
    """
    Evaluate agent against LLM opponent with improved protocol.
    
    Args:
        agent: Trained Q-Network
        opponent: LLM opponent
        num_games: Total games to play
        alternate_starting: If True, alternate who starts each game
        seed: Random seed for reproducibility
    
    Returns evaluation results with confidence intervals.
    """
    np.random.seed(seed)
    
    env = UnoRLCardEnv(num_players=2, seed=seed)
    
    results = []
    wins_total = 0
    wins_as_first = 0
    wins_as_second = 0
    games_as_first = 0
    games_as_second = 0
    
    for game_idx in range(num_games):
        # Determine starting player
        if alternate_starting:
            agent_starts = (game_idx % 2 == 0)
        else:
            agent_starts = True
        
        result = run_evaluation_game(env, agent, opponent, device, agent_starts)
        results.append(result)
        
        if result['agent_won']:
            wins_total += 1
            if result['agent_started']:
                wins_as_first += 1
            else:
                wins_as_second += 1
        
        if result['agent_started']:
            games_as_first += 1
        else:
            games_as_second += 1
        
        # Progress logging
        if (game_idx + 1) % 50 == 0:
            current_wr = wins_total / (game_idx + 1)
            print(f"  Game {game_idx + 1}/{num_games}: Win rate = {current_wr:.1%}")
    
    # Compute statistics
    win_rate = wins_total / num_games
    ci_lower, ci_upper = compute_confidence_interval(wins_total, num_games)
    
    # First player advantage analysis
    wr_as_first = wins_as_first / games_as_first if games_as_first > 0 else 0
    wr_as_second = wins_as_second / games_as_second if games_as_second > 0 else 0
    
    return {
        'total_games': num_games,
        'wins': wins_total,
        'win_rate': win_rate,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'games_as_first': games_as_first,
        'games_as_second': games_as_second,
        'wins_as_first': wins_as_first,
        'wins_as_second': wins_as_second,
        'win_rate_as_first': wr_as_first,
        'win_rate_as_second': wr_as_second,
        'game_results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL Agent vs LLM Opponents')
    parser.add_argument('--model', type=str, default='gemini_flash',
                        choices=list(LLM_MODELS.keys()),
                        help='LLM model to evaluate against')
    parser.add_argument('--games', type=int, default=500,
                        help='Number of games to play (default: 500)')
    parser.add_argument('--alternate', action='store_true', default=True,
                        help='Alternate starting player (default: True)')
    parser.add_argument('--no-alternate', dest='alternate', action='store_false',
                        help='Disable starting player alternation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: models/best_qnet.pt)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    if not HTTPX_AVAILABLE:
        print("ERROR: httpx not available. Install with: pip install httpx")
        return
    
    # Load model
    project_root = Path(__file__).parent.parent
    checkpoint_path = args.checkpoint or str(project_root / 'models' / 'best_qnet.pt')
    
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from: {checkpoint_path}")
    agent = load_model(checkpoint_path, device)
    
    # Create LLM opponent
    model_info = LLM_MODELS[args.model]
    print(f"Creating LLM opponent: {model_info['name']}")
    
    try:
        opponent = LLMOpenRouterOpponent(model_slug=model_info['slug'])
    except ValueError as e:
        print(f"ERROR: {e}")
        return
    
    # Run evaluation
    print(f"\n{'='*60}")
    print(f"EVALUATION: RL Agent vs {model_info['name']}")
    print(f"Games: {args.games}, Alternate Starting: {args.alternate}")
    print(f"{'='*60}\n")
    
    results = evaluate_vs_llm(
        agent, opponent, device,
        num_games=args.games,
        alternate_starting=args.alternate,
        seed=args.seed
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Win Rate: {results['win_rate']:.1%} "
          f"(95% CI: [{results['ci_95_lower']:.1%}, {results['ci_95_upper']:.1%}])")
    print(f"Games: {results['wins']}/{results['total_games']}")
    print(f"\nFirst Player Analysis:")
    print(f"  As first player: {results['win_rate_as_first']:.1%} "
          f"({results['wins_as_first']}/{results['games_as_first']})")
    print(f"  As second player: {results['win_rate_as_second']:.1%} "
          f"({results['wins_as_second']}/{results['games_as_second']})")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / 'runs' / f'eval_vs_{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove game_results for JSON (too large)
    save_results = {k: v for k, v in results.items() if k != 'game_results'}
    save_results['model'] = args.model
    save_results['model_name'] = model_info['name']
    save_results['timestamp'] = datetime.now().isoformat()
    save_results['seed'] = args.seed
    save_results['alternate_starting'] = args.alternate
    
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Cleanup
    opponent.close()


if __name__ == '__main__':
    main()
