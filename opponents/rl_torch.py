"""PyTorch inference opponent with Q-value explanation."""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Opponent

# RLCard color order
COLOR_NAMES = ['Red', 'Green', 'Blue', 'Yellow']
TYPE_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'Skip', 'Reverse', '+2', 'Wild', 'Wild+4']


def action_to_card_name(action_id: int) -> str:
    """Convert action ID to human-readable card name."""
    if action_id == 60:
        return "Draw"
    color_idx = action_id // 15
    type_idx = action_id % 15
    if type_idx >= 13:
        return TYPE_NAMES[type_idx]
    return f"{COLOR_NAMES[color_idx]} {TYPE_NAMES[type_idx]}"


class RLTorchOpponent(Opponent):
    """Opponent using PyTorch model for inference with explainable decisions."""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize PyTorch opponent.
        
        Args:
            model_path: Path to saved model (.pt file)
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        from rl.qnet_torch import load_model
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = load_model(model_path, self.device)
        self.model.eval()
        self._model_path = model_path
        
        # Store last decision info for explanation
        self.last_decision: Optional[Dict[str, Any]] = None
    
    def act(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        """Select action using greedy policy."""
        action, decision_info = self.act_with_explanation(obs, legal_mask)
        return action
    
    def act_with_explanation(self, obs: np.ndarray, legal_mask: np.ndarray) -> tuple:
        """
        Select action and return full explanation of the decision.
        
        Returns:
            Tuple of (action_id, decision_info_dict)
        """
        # Get Q-values from network
        q_values = self.model.get_q_values(obs, self.device)
        
        # Get legal action indices
        legal_actions = np.where(legal_mask > 0)[0]
        
        if len(legal_actions) == 0:
            legal_actions = [60]  # Fallback to draw
        
        # Mask illegal actions
        masked_q = q_values.copy()
        masked_q[legal_mask == 0] = float('-inf')
        
        # Select best legal action (greedy)
        best_action = int(np.argmax(masked_q))
        best_q = float(q_values[best_action])
        
        # Build top actions list for explanation
        top_actions = []
        sorted_indices = np.argsort(-masked_q)  # Sort descending
        
        for idx in sorted_indices[:5]:  # Top 5 actions
            if legal_mask[idx] > 0:
                top_actions.append({
                    'action_id': int(idx),
                    'card_name': action_to_card_name(int(idx)),
                    'q_value': float(q_values[idx]),
                    'is_selected': int(idx) == best_action
                })
        
        # Calculate softmax probabilities for legal actions
        legal_q = q_values[legal_actions]
        exp_q = np.exp(legal_q - np.max(legal_q))  # Subtract max for numerical stability
        softmax_probs = exp_q / np.sum(exp_q)
        
        # Store decision info
        self.last_decision = {
            'selected_action': best_action,
            'selected_card': action_to_card_name(best_action),
            'selected_q_value': best_q,
            'num_legal_actions': len(legal_actions),
            'top_actions': top_actions,
            'q_value_range': {
                'min': float(np.min(q_values)),
                'max': float(np.max(q_values)),
                'mean': float(np.mean(q_values))
            },
            'confidence': float(softmax_probs[np.where(legal_actions == best_action)[0][0]]) if best_action in legal_actions else 0.0,
            'legal_actions': [
                {'action_id': int(a), 'card_name': action_to_card_name(int(a)), 'q_value': float(q_values[a])}
                for a in legal_actions
            ]
        }
        
        return best_action, self.last_decision
    
    def get_explanation(self) -> Optional[Dict[str, Any]]:
        """Get the explanation for the last decision."""
        return self.last_decision
    
    @property
    def name(self) -> str:
        return "RL Agent (PyTorch)"


def get_default_torch_opponent(device: str = 'auto') -> Optional[RLTorchOpponent]:
    """Get default PyTorch opponent from models directory."""
    project_root = Path(__file__).parent.parent
    model_path = project_root / 'models' / 'best_qnet.pt'
    
    if model_path.exists():
        return RLTorchOpponent(str(model_path), device)
    
    return None


__all__ = ['RLTorchOpponent', 'get_default_torch_opponent']
