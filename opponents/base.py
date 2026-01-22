"""Base opponent interface."""

from abc import ABC, abstractmethod
import numpy as np


class Opponent(ABC):
    """Abstract base class for Uno opponents."""
    
    @abstractmethod
    def act(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        """
        Select an action given the current observation and legal action mask.
        
        Args:
            obs: State observation array of shape (420,)
            legal_mask: Binary legal action mask of shape (61,)
            
        Returns:
            Selected action index (0-60)
        """
        pass
    
    def reset(self):
        """Reset opponent state for a new game. Override if needed."""
        pass
    
    @property
    def name(self) -> str:
        """Return opponent name for display."""
        return self.__class__.__name__


class HumanOpponent(Opponent):
    """
    Placeholder for human opponent.
    
    In the TUI, this is handled by the UI directly,
    but this class exists for interface consistency.
    """
    
    def act(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        """Human action is handled by TUI - this should not be called."""
        raise NotImplementedError(
            "Human opponent actions should be handled by the TUI, not this method."
        )
    
    @property
    def name(self) -> str:
        return "Human"


__all__ = ['Opponent', 'HumanOpponent']
