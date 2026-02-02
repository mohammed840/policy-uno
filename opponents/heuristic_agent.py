"""
Heuristic (Rule-Based) Uno Agent.

Provides a baseline opponent using hand-crafted rules:
1. Play matching number cards (lowest value first)
2. Use action cards (Skip/Reverse/+2) strategically
3. Save Wild cards for when no other plays available
4. Draw only when absolutely necessary

This serves as a reproducible baseline for comparison with learned agents,
addressing the professor's concern about missing rule-based baselines.
"""

import numpy as np
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Opponent
from uno.cards import CardType, Color, COLOR_COUNT, CARD_TYPE_COUNT, DRAW_ACTION


class HeuristicAgent(Opponent):
    """
    Rule-based Uno agent using hand-crafted heuristics.
    
    Strategy priority:
    1. Play number cards matching color/number (prefer low numbers)
    2. Play action cards when strategically advantageous
    3. Play Wild cards only when no colored options
    4. Draw as last resort
    """
    
    def __init__(self, aggression: float = 0.5):
        """
        Initialize heuristic agent.
        
        Args:
            aggression: 0.0 = conservative (save action cards),
                       1.0 = aggressive (play action cards immediately)
        """
        self.aggression = aggression
    
    def act(
        self,
        obs: np.ndarray,
        legal_mask: np.ndarray,
        hand_cards: Optional[list] = None,
        top_card: Optional[str] = None,
        active_color: Optional[str] = None
    ) -> int:
        """
        Select action using rule-based heuristics.
        
        Args:
            obs: State observation (420-dim, ignored by heuristic)
            legal_mask: Binary mask of legal actions (61-dim)
            hand_cards: Optional list of card strings in hand
            top_card: Optional top card string
            active_color: Optional active color string
            
        Returns:
            Selected action index
        """
        legal_actions = np.where(legal_mask > 0)[0]
        
        if len(legal_actions) == 0:
            return DRAW_ACTION
        
        if len(legal_actions) == 1:
            return int(legal_actions[0])
        
        # Categorize legal actions by type
        number_cards = []
        action_cards = []  # Skip, Reverse, Draw Two
        wild_cards = []    # Wild, Wild+4
        draw_action = None
        
        for action_idx in legal_actions:
            if action_idx == DRAW_ACTION:
                draw_action = action_idx
                continue
            
            # Decode action: action_idx = color_idx * 15 + type_idx
            color_idx = action_idx // CARD_TYPE_COUNT
            type_idx = action_idx % CARD_TYPE_COUNT
            
            if type_idx <= CardType.NINE.value:
                # Number card (0-9)
                number_cards.append((action_idx, type_idx))
            elif type_idx in [CardType.SKIP.value, CardType.REVERSE.value, CardType.DRAW_TWO.value]:
                # Action card
                action_cards.append((action_idx, type_idx))
            elif type_idx in [CardType.WILD.value, CardType.WILD_DRAW_FOUR.value]:
                # Wild card
                wild_cards.append((action_idx, type_idx))
        
        # Priority 1: Play number cards (lowest value first for hand management)
        if number_cards:
            # Sort by card value (lower is better to play first)
            number_cards.sort(key=lambda x: x[1])
            return int(number_cards[0][0])
        
        # Priority 2: Play action cards based on aggression
        if action_cards:
            if np.random.random() < self.aggression or not wild_cards:
                # Prefer Draw Two > Skip > Reverse
                priority = {CardType.DRAW_TWO.value: 0, CardType.SKIP.value: 1, CardType.REVERSE.value: 2}
                action_cards.sort(key=lambda x: priority.get(x[1], 3))
                return int(action_cards[0][0])
        
        # Priority 3: Play Wild cards (prefer regular Wild over Wild+4)
        if wild_cards:
            # Wild+4 is more powerful, save it if possible
            wild_cards.sort(key=lambda x: x[1])  # Wild before Wild+4
            return int(wild_cards[0][0])
        
        # Priority 4: Draw (last resort)
        if draw_action is not None:
            return int(draw_action)
        
        # Fallback: random legal action
        return int(np.random.choice(legal_actions))
    
    @property
    def name(self) -> str:
        return f"Heuristic (aggr={self.aggression:.1f})"


class DefensiveHeuristic(HeuristicAgent):
    """Conservative heuristic that saves action and wild cards."""
    def __init__(self):
        super().__init__(aggression=0.2)


class AggressiveHeuristic(HeuristicAgent):
    """Aggressive heuristic that plays action cards immediately."""
    def __init__(self):
        super().__init__(aggression=0.9)


__all__ = ['HeuristicAgent', 'DefensiveHeuristic', 'AggressiveHeuristic']
