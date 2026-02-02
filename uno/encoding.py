"""
State and action encoding for Uno RL.

State Encoding (7 × 4 × 15 = 420):
- Planes 0-2: Own hand count buckets (0, 1, 2+) per card type
- Planes 3-5: Opponents' card counts (0, 1, 2+)
- Plane 6: Top discard card (one-hot)
- 4 colors × 15 card types = 60 features per plane

Action Encoding (61):
- Actions 0-59: Play card (color_idx * 15 + type_idx)
- Action 60: Draw a card
"""

import numpy as np
from typing import Optional

from .cards import (
    Card, CardType, Color, COLOR_COUNT, CARD_TYPE_COUNT,
    ACTION_SPACE_SIZE, DRAW_ACTION, get_card_index, card_from_action_index
)


# State encoding dimensions
NUM_PLANES = 7
STATE_SHAPE = (NUM_PLANES, COLOR_COUNT, CARD_TYPE_COUNT)  # 7 × 4 × 15
STATE_SIZE = NUM_PLANES * COLOR_COUNT * CARD_TYPE_COUNT  # 420


# =============================================================================
# PUBLIC INFORMATION TRACKING FOR OPPONENT MODELING
# =============================================================================
# This class implements transparent opponent hand estimation using ONLY
# publicly observable information, addressing potential information leakage
# concerns. All estimates are derived from:
#   1. Cards played (visible to all players)
#   2. Number of cards drawn (count only, not card identities)
#   3. Card conservation (total deck composition is known)
# =============================================================================

class PublicInfoTracker:
    """
    Track public information for opponent hand estimation.
    
    Uses only publicly observable information:
    - Cards played by all players (visible)
    - Number of cards drawn per player (count only)
    - Card conservation (deck composition is known)
    
    No privileged information is used - opponent hand contents are never accessed.
    """
    
    # Standard Uno deck composition (108 cards total for 2-player)
    DECK_COMPOSITION = {
        # Number cards: one 0 per color, two of each 1-9 per color
        # Action cards: two each of Skip, Reverse, Draw Two per color
        # Wild cards: 4 Wild, 4 Wild Draw Four
    }
    TOTAL_CARDS = 108
    
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.reset()
    
    def reset(self):
        """Reset tracker for new game."""
        # Cards that have been played (publicly visible)
        self.played_cards: list[Card] = []
        # Number of cards each player has drawn (count only, not values)
        self.draw_counts: dict[int, int] = {i: 0 for i in range(self.num_players)}
        # Initial hand sizes (typically 7 cards each)
        self.initial_hand_size = 7
        # Track cards played by each player
        self.cards_played_by: dict[int, list[Card]] = {i: [] for i in range(self.num_players)}
    
    def record_play(self, player_id: int, card: Card):
        """Record a card being played (public information)."""
        self.played_cards.append(card)
        self.cards_played_by[player_id].append(card)
    
    def record_draw(self, player_id: int, count: int = 1):
        """Record player drawing cards (count only, not values)."""
        self.draw_counts[player_id] += count
    
    def estimate_opponent_hand_size(self, player_id: int) -> int:
        """Estimate opponent's current hand size from public info."""
        cards_played = len(self.cards_played_by[player_id])
        cards_drawn = self.draw_counts[player_id]
        return self.initial_hand_size - cards_played + cards_drawn
    
    def estimate_remaining_cards(self, own_hand: list[Card]) -> np.ndarray:
        """
        Estimate remaining cards not in own hand using card conservation.
        
        Returns array of shape (4, 15) with estimated counts for each card type.
        These are the cards that could be in opponents' hands or the deck.
        """
        # Start with full deck composition
        remaining = get_full_deck_counts()
        
        # Subtract own hand
        own_counts = count_cards_by_type(own_hand)
        remaining -= own_counts
        
        # Subtract played cards
        played_counts = count_cards_by_type(self.played_cards)
        remaining -= played_counts
        
        # Clamp to non-negative
        remaining = np.maximum(remaining, 0)
        
        return remaining
    
    def estimate_opponent_distribution(
        self, 
        own_hand: list[Card],
        opponent_hand_sizes: list[int]
    ) -> np.ndarray:
        """
        Estimate opponent card distribution using public information only.
        
        Uses maximum entropy assumption: distribute remaining cards uniformly
        across unknown positions (deck + opponent hands).
        
        Args:
            own_hand: Player's own cards (known exactly)
            opponent_hand_sizes: Number of cards each opponent has
            
        Returns:
            Array of shape (4, 15) with expected opponent card counts
        """
        remaining = self.estimate_remaining_cards(own_hand)
        total_remaining = remaining.sum()
        
        if total_remaining == 0:
            return np.zeros((COLOR_COUNT, CARD_TYPE_COUNT), dtype=np.float32)
        
        # Total opponent cards
        total_opponent_cards = sum(opponent_hand_sizes)
        
        # Distribute proportionally (maximum entropy assumption)
        # P(opponent has card X) = (remaining X) / total_remaining * opponent_cards
        opponent_estimate = remaining * (total_opponent_cards / max(total_remaining, 1))
        
        return opponent_estimate.astype(np.float32)


def get_full_deck_counts() -> np.ndarray:
    """
    Get counts of each card type in a full Uno deck.
    
    Standard deck:
    - 1x Zero per color (4 total)
    - 2x each of 1-9 per color (72 total)
    - 2x Skip per color (8 total)
    - 2x Reverse per color (8 total)
    - 2x Draw Two per color (8 total)
    - 4x Wild (spread across colors for encoding)
    - 4x Wild Draw Four (spread across colors for encoding)
    """
    counts = np.zeros((COLOR_COUNT, CARD_TYPE_COUNT), dtype=np.float32)
    
    for color_idx in range(COLOR_COUNT):
        # Zeros: 1 per color
        counts[color_idx, CardType.ZERO.value] = 1
        # Numbers 1-9: 2 per color
        for num in [CardType.ONE, CardType.TWO, CardType.THREE, CardType.FOUR,
                    CardType.FIVE, CardType.SIX, CardType.SEVEN, CardType.EIGHT, CardType.NINE]:
            counts[color_idx, num.value] = 2
        # Action cards: 2 per color
        counts[color_idx, CardType.SKIP.value] = 2
        counts[color_idx, CardType.REVERSE.value] = 2
        counts[color_idx, CardType.DRAW_TWO.value] = 2
        # Wild cards: 4 total, spread as 1 per color for encoding
        counts[color_idx, CardType.WILD.value] = 1
        counts[color_idx, CardType.WILD_DRAW_FOUR.value] = 1
    
    return counts


def count_cards_by_type(cards: list[Card]) -> np.ndarray:
    """
    Count cards by (color, type) combination.
    
    Returns array of shape (4, 15) with counts.
    """
    counts = np.zeros((COLOR_COUNT, CARD_TYPE_COUNT), dtype=np.float32)
    
    for card in cards:
        if card.is_wild:
            # Wild cards: count across all colors for their type
            # We'll put them in a designated spot
            for color_idx in range(COLOR_COUNT):
                counts[color_idx, card.card_type.value] += 0.25  # Spread across colors
        else:
            counts[card.color.value, card.card_type.value] += 1
    
    return counts


def bucket_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert counts to 3 bucket planes: 0 cards, 1 card, 2+ cards.
    
    Args:
        counts: Array of shape (4, 15) with card counts
        
    Returns:
        Tuple of 3 arrays, each (4, 15), representing buckets
    """
    bucket_0 = (counts == 0).astype(np.float32)  # 0 cards
    bucket_1 = (counts == 1).astype(np.float32)  # 1 card
    bucket_2 = (counts >= 2).astype(np.float32)  # 2+ cards
    
    return bucket_0, bucket_1, bucket_2


def encode_top_card(top_card: Card, active_color: Color) -> np.ndarray:
    """
    Encode top discard card as one-hot plane.
    
    Args:
        top_card: The top card on discard pile
        active_color: Current active color (for Wild cards)
        
    Returns:
        Array of shape (4, 15) with one-hot encoding
    """
    plane = np.zeros((COLOR_COUNT, CARD_TYPE_COUNT), dtype=np.float32)
    
    if top_card.is_wild:
        # For wild cards, use the active color
        plane[active_color.value, top_card.card_type.value] = 1.0
    else:
        plane[top_card.color.value, top_card.card_type.value] = 1.0
    
    return plane


def encode_state(
    own_hand: list[Card],
    opponent_hand_sizes: list[int],
    top_card: Card,
    active_color: Color,
    opponent_known_cards: Optional[list[list[Card]]] = None
) -> np.ndarray:
    """
    Encode game state as 7 × 4 × 15 tensor.
    
    Args:
        own_hand: Player's cards
        opponent_hand_sizes: Number of cards each opponent has
        top_card: Top discard card
        active_color: Current active color
        opponent_known_cards: Optional list of known opponent cards (for full observability)
        
    Returns:
        Flattened state array of shape (420,)
    """
    state = np.zeros(STATE_SHAPE, dtype=np.float32)
    
    # Planes 0-2: Own hand bucket encoding
    own_counts = count_cards_by_type(own_hand)
    bucket_0, bucket_1, bucket_2 = bucket_counts(own_counts)
    state[0] = bucket_0
    state[1] = bucket_1
    state[2] = bucket_2
    
    # Planes 3-5: Opponent hand encoding
    # If we know opponent cards, use them; otherwise use hand sizes to estimate
    if opponent_known_cards:
        all_opponent_cards = [card for hand in opponent_known_cards for card in hand]
        opp_counts = count_cards_by_type(all_opponent_cards)
    else:
        # Estimate from hand sizes (uniform distribution assumption)
        total_opp_cards = sum(opponent_hand_sizes)
        # Create uniform distribution based on total cards
        opp_counts = np.ones((COLOR_COUNT, CARD_TYPE_COUNT), dtype=np.float32)
        opp_counts *= total_opp_cards / (COLOR_COUNT * CARD_TYPE_COUNT)
    
    opp_bucket_0, opp_bucket_1, opp_bucket_2 = bucket_counts(opp_counts)
    state[3] = opp_bucket_0
    state[4] = opp_bucket_1
    state[5] = opp_bucket_2
    
    # Plane 6: Top card one-hot
    state[6] = encode_top_card(top_card, active_color)
    
    return state.flatten()


def encode_state_from_rlcard(obs: dict, player_id: int = 0) -> np.ndarray:
    """
    Convert RLCard observation to our encoding format.
    
    RLCard's Uno observation includes:
    - 'hand': list of card strings like 'r-1', 'b-skip', 'wild'
    - 'target': the target card to match
    - 'played_cards': cards already played
    - etc.
    
    Returns flattened state of shape (420,)
    """
    # Parse hand from RLCard format
    hand = parse_rlcard_cards(obs.get('hand', []))
    
    # Parse target card
    target_str = obs.get('target', 'r-0')
    top_card, active_color = parse_rlcard_target(target_str)
    
    # Get opponent hand sizes (RLCard provides this)
    num_cards = obs.get('num_cards', [7, 7, 7, 7])
    if isinstance(num_cards, list):
        opponent_hand_sizes = [n for i, n in enumerate(num_cards) if i != player_id]
    else:
        opponent_hand_sizes = [7] * 3  # Default
    
    return encode_state(hand, opponent_hand_sizes, top_card, active_color)


def parse_rlcard_cards(card_strings: list[str]) -> list[Card]:
    """Parse RLCard card strings to Card objects."""
    cards = []
    for s in card_strings:
        card = parse_rlcard_card(s)
        if card:
            cards.append(card)
    return cards


def parse_rlcard_card(s: str) -> Optional[Card]:
    """
    Parse a single RLCard card string.
    
    Formats:
    - 'r-5' -> Red 5
    - 'b-skip' -> Blue Skip
    - 'wild' -> Wild
    - 'wild_draw_four' -> Wild+4
    """
    s = s.lower().strip()
    
    if s == 'wild':
        return Card(Color.WILD, CardType.WILD)
    if s in ('wild_draw_four', 'wild-draw-four', 'w4'):
        return Card(Color.WILD, CardType.WILD_DRAW_FOUR)
    
    if '-' in s:
        parts = s.split('-', 1)
        color_char = parts[0]
        type_str = parts[1]
        
        # Parse color
        color_map = {'r': Color.RED, 'y': Color.YELLOW, 'g': Color.GREEN, 'b': Color.BLUE}
        if color_char not in color_map:
            return None
        color = color_map[color_char]
        
        # Parse type
        type_map = {
            '0': CardType.ZERO, '1': CardType.ONE, '2': CardType.TWO,
            '3': CardType.THREE, '4': CardType.FOUR, '5': CardType.FIVE,
            '6': CardType.SIX, '7': CardType.SEVEN, '8': CardType.EIGHT,
            '9': CardType.NINE, 'skip': CardType.SKIP, 'reverse': CardType.REVERSE,
            'draw_two': CardType.DRAW_TWO, 'draw-two': CardType.DRAW_TWO,
            '+2': CardType.DRAW_TWO
        }
        if type_str in type_map:
            return Card(color, type_map[type_str])
    
    return None


def parse_rlcard_target(target_str: str) -> tuple[Card, Color]:
    """
    Parse RLCard target card string.
    
    Returns (top_card, active_color) tuple.
    """
    card = parse_rlcard_card(target_str)
    if card is None:
        # Default fallback
        card = Card(Color.RED, CardType.ZERO)
    
    # Determine active color
    if card.is_wild:
        # For wild cards, try to extract color from context
        # RLCard format might be 'wild-r' meaning wild with red chosen
        if '-' in target_str and target_str.startswith('wild'):
            parts = target_str.split('-')
            if len(parts) > 1:
                color_map = {'r': Color.RED, 'y': Color.YELLOW, 'g': Color.GREEN, 'b': Color.BLUE}
                active_color = color_map.get(parts[-1], Color.RED)
            else:
                active_color = Color.RED
        else:
            active_color = Color.RED
    else:
        active_color = card.color
    
    return card, active_color


def action_to_rlcard(action_idx: int, hand: list[Card]) -> str:
    """
    Convert our action index to RLCard action string.
    
    Args:
        action_idx: Our action index (0-60)
        hand: Current hand (to find the actual card)
        
    Returns:
        RLCard action string
    """
    if action_idx == DRAW_ACTION:
        return 'draw'
    
    color, card_type = card_from_action_index(action_idx)
    
    # Build RLCard string
    if card_type == CardType.WILD:
        return f'wild-{color.name[0].lower()}'  # e.g., 'wild-r'
    elif card_type == CardType.WILD_DRAW_FOUR:
        return f'wild_draw_four-{color.name[0].lower()}'
    else:
        color_char = color.name[0].lower()
        type_strs = {
            CardType.ZERO: '0', CardType.ONE: '1', CardType.TWO: '2',
            CardType.THREE: '3', CardType.FOUR: '4', CardType.FIVE: '5',
            CardType.SIX: '6', CardType.SEVEN: '7', CardType.EIGHT: '8',
            CardType.NINE: '9', CardType.SKIP: 'skip', CardType.REVERSE: 'reverse',
            CardType.DRAW_TWO: 'draw_two'
        }
        return f'{color_char}-{type_strs[card_type]}'


def compute_legal_mask_from_rlcard(legal_actions: list[str], hand: list[Card]) -> np.ndarray:
    """
    Convert RLCard legal actions to our 61-dimensional binary mask.
    
    Args:
        legal_actions: List of RLCard action strings
        hand: Current hand
        
    Returns:
        Binary mask array of shape (61,)
    """
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    
    for action_str in legal_actions:
        action_str = action_str.lower()
        
        if action_str == 'draw':
            mask[DRAW_ACTION] = 1.0
            continue
        
        # Parse the action
        if action_str.startswith('wild_draw_four') or action_str.startswith('wild-draw-four'):
            # Wild+4 - check if it includes color choice
            if '-' in action_str:
                parts = action_str.replace('wild_draw_four', 'wd4').replace('wild-draw-four', 'wd4').split('-')
                if len(parts) > 1:
                    color_map = {'r': Color.RED, 'y': Color.YELLOW, 'g': Color.GREEN, 'b': Color.BLUE}
                    color = color_map.get(parts[-1], None)
                    if color:
                        idx = get_card_index(color, CardType.WILD_DRAW_FOUR)
                        mask[idx] = 1.0
                        continue
            # Allow all color choices for wild+4
            for c in [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]:
                mask[get_card_index(c, CardType.WILD_DRAW_FOUR)] = 1.0
                
        elif action_str.startswith('wild'):
            if '-' in action_str:
                parts = action_str.split('-')
                if len(parts) > 1:
                    color_map = {'r': Color.RED, 'y': Color.YELLOW, 'g': Color.GREEN, 'b': Color.BLUE}
                    color = color_map.get(parts[-1], None)
                    if color:
                        idx = get_card_index(color, CardType.WILD)
                        mask[idx] = 1.0
                        continue
            # Allow all color choices for wild
            for c in [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]:
                mask[get_card_index(c, CardType.WILD)] = 1.0
        else:
            # Regular card
            card = parse_rlcard_card(action_str)
            if card and not card.is_wild:
                idx = get_card_index(card.color, card.card_type)
                mask[idx] = 1.0
    
    return mask


# Export key constants
__all__ = [
    'STATE_SIZE', 'STATE_SHAPE', 'ACTION_SPACE_SIZE', 'DRAW_ACTION',
    'encode_state', 'encode_state_from_rlcard',
    'compute_legal_mask_from_rlcard', 'action_to_rlcard',
    'parse_rlcard_card', 'parse_rlcard_cards'
]
