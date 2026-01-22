"""Random bot opponent."""

import numpy as np
from .base import Opponent


class RandomBot(Opponent):
    """Opponent that selects random legal actions."""
    
    def __init__(self, seed: int = None):
        """
        Initialize random bot.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
    
    def act(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        """Select a random legal action."""
        legal_actions = np.where(legal_mask > 0)[0]
        
        if len(legal_actions) == 0:
            return 60  # Draw
        
        return int(self.rng.choice(legal_actions))
    
    @property
    def name(self) -> str:
        return "Random Bot"


__all__ = ['RandomBot']
