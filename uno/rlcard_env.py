"""RLCard environment wrapper for Uno."""

import numpy as np
from typing import Optional, Any

try:
    import rlcard
    from rlcard.envs.uno import UnoEnv
    RLCARD_AVAILABLE = True
except ImportError:
    RLCARD_AVAILABLE = False

from .encoding import (
    STATE_SIZE, ACTION_SPACE_SIZE, DRAW_ACTION,
    encode_state_from_rlcard, compute_legal_mask_from_rlcard,
    action_to_rlcard, parse_rlcard_cards
)
from .cards import Card


class UnoRLCardEnv:
    """
    Wrapper around RLCard's Uno environment.
    
    Provides:
    - State encoding as 420-dim vector
    - Action space as 61 discrete actions
    - Legal action masks
    """
    
    def __init__(self, num_players: int = 2, seed: Optional[int] = None):
        """
        Initialize the Uno environment.
        
        Args:
            num_players: Number of players (2-4)
            seed: Random seed for reproducibility
        """
        if not RLCARD_AVAILABLE:
            raise ImportError(
                "RLCard is not installed. Please install with: pip install rlcard"
            )
        
        self.num_players = num_players
        self.seed = seed
        
        # Create RLCard environment
        config = {
            'seed': seed,
            'allow_step_back': False,
        }
        self.env = rlcard.make('uno', config)
        
        # Track game state
        self._current_player = 0
        self._done = False
        self._obs = None
        self._legal_actions = None
        
    @property
    def state_size(self) -> int:
        return STATE_SIZE
    
    @property
    def action_size(self) -> int:
        return ACTION_SPACE_SIZE
    
    def reset(self) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Reset the environment for a new game.
        
        Returns:
            Tuple of (state, legal_mask, current_player)
        """
        state, player_id = self.env.reset()
        self._current_player = player_id
        self._done = False
        self._obs = state['obs']
        self._legal_actions = list(state['legal_actions'].keys())
        
        # Encode state and legal mask
        encoded_state = self._encode_current_state()
        legal_mask = self._compute_legal_mask()
        
        return encoded_state, legal_mask, self._current_player
    
    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, np.ndarray, int, dict]:
        """
        Take a step in the environment.
        
        Args:
            action_idx: Action index (0-60)
            
        Returns:
            Tuple of (next_state, reward, done, legal_mask, current_player, info)
        """
        # Convert our action to RLCard action
        hand = parse_rlcard_cards(self._obs if isinstance(self._obs, list) else 
                                   self._obs.get('hand', []) if isinstance(self._obs, dict) else [])
        rlcard_action = action_to_rlcard(action_idx, hand)
        
        # Find matching RLCard action
        legal_actions_dict = self.env.get_state(self._current_player)['legal_actions']
        
        # Map our action to RLCard's action format
        action_id = self._find_rlcard_action(action_idx, legal_actions_dict)
        
        if action_id is None:
            # Fallback: pick first legal action if mapping fails
            action_id = list(legal_actions_dict.keys())[0] if legal_actions_dict else 0
        
        # Step the environment
        next_state, next_player = self.env.step(action_id)
        
        self._current_player = next_player
        self._obs = next_state['obs']
        self._legal_actions = list(next_state['legal_actions'].keys())
        
        # Check if game is done
        self._done = self.env.is_over()
        
        # Compute reward (terminal only: +1 win, -1 loss)
        reward = 0.0
        info = {'winner': None}
        
        if self._done:
            payoffs = self.env.get_payoffs()
            reward = payoffs[0]  # Reward for player 0
            winner = np.argmax(payoffs)
            info['winner'] = winner
            info['payoffs'] = payoffs
        
        # Encode next state and legal mask
        encoded_state = self._encode_current_state()
        legal_mask = self._compute_legal_mask()
        
        return encoded_state, reward, self._done, legal_mask, self._current_player, info
    
    def _encode_current_state(self) -> np.ndarray:
        """Encode current observation to 420-dim state."""
        try:
            state = self.env.get_state(self._current_player)
            obs = state['obs']
            
            # RLCard obs can be array or dict
            if isinstance(obs, np.ndarray):
                # Use RLCard's encoding directly and pad/truncate to 420
                flat = obs.flatten()
                if len(flat) >= STATE_SIZE:
                    return flat[:STATE_SIZE].astype(np.float32)
                else:
                    padded = np.zeros(STATE_SIZE, dtype=np.float32)
                    padded[:len(flat)] = flat
                    return padded
            elif isinstance(obs, dict):
                return encode_state_from_rlcard(obs, self._current_player)
            else:
                return encode_state_from_rlcard({'hand': [], 'target': 'r-0'}, 0)
        except Exception:
            return np.zeros(STATE_SIZE, dtype=np.float32)
    
    def _compute_legal_mask(self, player_id: Optional[int] = None) -> np.ndarray:
        """Compute 61-dim legal action mask.
        
        Args:
            player_id: Player to compute mask for. Defaults to current player.
        """
        if player_id is None:
            player_id = self._current_player
            
        try:
            state = self.env.get_state(player_id)
            legal_actions = state['legal_actions']
            
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            
            # RLCard's legal_actions dict has integer keys that are action IDs (0-60)
            # The values are None, so we just use the keys directly
            for action_id in legal_actions.keys():
                if isinstance(action_id, int) and 0 <= action_id < ACTION_SPACE_SIZE:
                    mask[action_id] = 1.0
            
            # Always allow draw if no other actions
            if mask.sum() == 0:
                mask[DRAW_ACTION] = 1.0
                
            return mask
        except Exception as e:
            print(f"[DEBUG] Error computing legal mask: {e}")
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            mask[DRAW_ACTION] = 1.0
            return mask
    
    def _find_rlcard_action(self, action_idx: int, legal_actions: dict) -> Optional[int]:
        """Find RLCard action ID matching our action index.
        
        RLCard uses integer action IDs (0-60) directly, so we just need to check
        if our action index is in the legal_actions dict.
        """
        # RLCard uses the same action space as us (0-60)
        # Just check if the action is legal
        if action_idx in legal_actions:
            return action_idx
        
        # If exact match not found, return None
        return None
    
    def _rlcard_action_to_idx(self, action_id: int, action_info: Any) -> Optional[int]:
        """Convert RLCard action to our action index."""
        from .cards import Color, CardType, get_card_index
        
        # Handle draw action
        if isinstance(action_info, str):
            if 'draw' in action_info.lower():
                return DRAW_ACTION
            
            # Parse card action like 'r-5', 'wild', etc.
            card = parse_rlcard_cards([action_info])
            if card:
                c = card[0]
                if c.is_wild:
                    # Extract color choice if present
                    if '-' in action_info:
                        parts = action_info.split('-')
                        color_map = {'r': Color.RED, 'y': Color.YELLOW, 
                                   'g': Color.GREEN, 'b': Color.BLUE}
                        color = color_map.get(parts[-1].lower(), Color.RED)
                        return get_card_index(color, c.card_type)
                    return get_card_index(Color.RED, c.card_type)
                else:
                    return get_card_index(c.color, c.card_type)
        
        # If action_id is just an integer index
        if isinstance(action_id, int) and 0 <= action_id < ACTION_SPACE_SIZE:
            return action_id
            
        return None
    
    def get_current_player(self) -> int:
        """Get current player index."""
        return self._current_player
    
    def is_over(self) -> bool:
        """Check if game is over."""
        return self._done
    
    def get_payoffs(self) -> list[float]:
        """Get final payoffs for each player."""
        if self._done:
            return list(self.env.get_payoffs())
        return [0.0] * self.num_players
    
    def get_hand_sizes(self) -> list[int]:
        """Get hand sizes for all players."""
        try:
            state = self.env.get_state(self._current_player)
            if 'num_cards' in state:
                return list(state['num_cards'])
        except Exception:
            pass
        return [7] * self.num_players
    
    def render(self) -> str:
        """Render current game state as string."""
        try:
            state = self.env.get_state(self._current_player)
            obs = state.get('obs', {})
            
            lines = [
                f"Current Player: {self._current_player}",
                f"Hand Sizes: {self.get_hand_sizes()}",
            ]
            
            if isinstance(obs, dict):
                if 'hand' in obs:
                    lines.append(f"Hand: {obs['hand']}")
                if 'target' in obs:
                    lines.append(f"Target: {obs['target']}")
            
            return "\n".join(lines)
        except Exception as e:
            return f"Error rendering: {e}"


def make_env(num_players: int = 2, seed: Optional[int] = None) -> UnoRLCardEnv:
    """Factory function to create Uno environment."""
    return UnoRLCardEnv(num_players=num_players, seed=seed)


__all__ = ['UnoRLCardEnv', 'make_env', 'RLCARD_AVAILABLE']
