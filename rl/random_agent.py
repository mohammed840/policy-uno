"""Random baseline agent for Uno."""

import numpy as np
from typing import Optional


class RandomAgent:
    """Agent that selects random legal actions."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random agent.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
    
    def select_action(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray
    ) -> int:
        """
        Select a random legal action.
        
        Args:
            state: State array (ignored)
            legal_mask: Binary legal action mask of shape (61,)
            
        Returns:
            Selected action index
        """
        legal_actions = np.where(legal_mask > 0)[0]
        
        if len(legal_actions) == 0:
            # Fallback to draw
            return 60
        
        return int(self.rng.choice(legal_actions))
    
    def act(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        """Alias for select_action (matches Opponent interface)."""
        return self.select_action(obs, legal_mask)


__all__ = ['RandomAgent']
