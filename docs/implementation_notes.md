# Implementation Notes

This document provides technical details about the Uno RL implementation.

## State Encoding (7 × 4 × 15 = 420)

The state is represented as a 3D binary tensor that is flattened to 420 dimensions:

| Plane | Description | Encoding |
|-------|-------------|----------|
| 0-2 | Own hand (count buckets) | 0 cards, 1 card, 2+ cards per card type |
| 3-5 | Opponents' cards (count buckets) | 0 cards, 1 card, 2+ cards per card type |
| 6 | Top discard card | One-hot encoding |

Each plane has shape 4 × 15:
- **4 colors**: Red, Yellow, Green, Blue
- **15 card types**: 0-9, Skip, Reverse, +2, Wild, Wild+4

### Flatten Order
The state tensor is flattened in row-major order: `state.flatten()` produces a 420-element vector.

## Action Space (61 actions)

| Action Range | Description |
|--------------|-------------|
| 0-14 | Play Red card (types 0-9, Skip, Reverse, +2, Wild, Wild+4) |
| 15-29 | Play Yellow card |
| 30-44 | Play Green card |
| 45-59 | Play Blue card |
| 60 | Draw a card |

For Wild cards, the action encodes both playing the card AND choosing the color.

### Legal Action Mask
A 61-dimensional binary mask where 1 indicates a legal action. Computed from:
1. Cards in hand that match the top card's color or type
2. Wild cards (always playable)
3. Draw action (always legal)

## Reward Structure

**Terminal rewards only:**
- Win: +1
- Loss: -1
- During game: 0

## Q-Network Architecture

```
Input (420) → Dense (512, ReLU) → Dense (512, ReLU) → Output (61)
```

Total parameters: ~530,000

## Training Algorithms

### DQN
- TD target: `y = r + γ * max_a' Q_target(s', a')`
- Experience replay buffer
- Separate target network (updated every N iterations)
- Legal action masking: illegal actions set to -∞ before argmax

### DeepSARSA
- TD target: `y = r + γ * Q(s', a')` where `a'` is the actual next action
- On-policy: uses the action selected by the current policy
- No target network (can optionally add for stability)

### Hyperparameters (defaults)
- Learning rate (α): 1e-4
- Discount factor (γ): 0.95
- Initial epsilon (ε): 0.95
- Epsilon decay (κ): 0.995
- Batch size: 64
- Buffer size: 100,000

## Core ML Export

1. Load trained PyTorch model
2. Trace with TorchScript using example input `[1, 420]`
3. Convert to Core ML ML Program format
4. Input: `state` tensor [1, 420]
5. Output: `q_values` tensor [1, 61]

## Environment: RLCard

We use [RLCard](https://github.com/datamllab/rlcard) for the Uno environment because:
1. Correct Uno rule implementation
2. Well-tested game logic
3. Easy integration with RL training

The `UnoRLCardEnv` wrapper converts RLCard's native format to our encoding.

## File Organization

```
uno-coreml-tui/
├── uno/              # Game engine
├── rl/               # Training code
├── opponents/        # Opponent adapters
├── tui/              # Terminal UI
├── runs/             # Training artifacts
├── models/           # Saved models
└── docs/             # Documentation
```

## Artifact Outputs

Training creates:
- `runs/<run_id>/config.json` - Training configuration
- `runs/<run_id>/metrics_train.csv` - Per-iteration metrics
- `runs/<run_id>/metrics_eval.csv` - Evaluation results
- `runs/<run_id>/tournaments/*.jsonl` - Game-by-game logs
- `runs/<run_id>/plots/*.png` - Visualization figures
