"""Uno game rules and legality checks."""

from typing import Optional
from .cards import Card, CardType, Color, DRAW_ACTION, get_card_index


def is_playable(card: Card, top_card: Card, active_color: Color) -> bool:
    """
    Check if a card can be played on the current top card.
    
    A card is playable if:
    1. It's a Wild or Wild+4 (always playable)
    2. It matches the active color
    3. It matches the top card's type (number or action)
    """
    # Wild cards are always playable
    if card.is_wild:
        return True
    
    # Match active color
    if card.color == active_color:
        return True
    
    # Match card type (number matches number, action matches action)
    if not top_card.is_wild and card.card_type == top_card.card_type:
        return True
    
    return False


def get_legal_actions(
    hand: list[Card],
    top_card: Card,
    active_color: Color,
    include_draw: bool = True
) -> list[int]:
    """
    Get list of legal action indices given current game state.
    
    Returns list of action indices (0-60).
    """
    legal_actions = []
    
    for card in hand:
        if is_playable(card, top_card, active_color):
            if card.is_wild:
                # For Wild cards, add actions for all 4 color choices
                for color in [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]:
                    action_idx = get_card_index(color, card.card_type)
                    if action_idx not in legal_actions:
                        legal_actions.append(action_idx)
            else:
                action_idx = get_card_index(card.color, card.card_type)
                if action_idx not in legal_actions:
                    legal_actions.append(action_idx)
    
    # Always can draw if enabled
    if include_draw:
        legal_actions.append(DRAW_ACTION)
    
    return sorted(legal_actions)


def compute_legal_mask(
    hand: list[Card],
    top_card: Card,
    active_color: Color,
    include_draw: bool = True
) -> list[int]:
    """
    Compute a binary legal action mask.
    
    Returns list of 61 binary values (0 or 1).
    """
    legal_actions = get_legal_actions(hand, top_card, active_color, include_draw)
    mask = [0] * 61
    for action_idx in legal_actions:
        mask[action_idx] = 1
    return mask


def format_hand(hand: list[Card], playable_mask: Optional[list[int]] = None) -> str:
    """Format hand for display, optionally highlighting playable cards."""
    parts = []
    for i, card in enumerate(hand):
        card_str = f"[{i}] {card.to_string()}"
        if playable_mask is not None:
            # Check if any action for this card is legal
            if card.is_wild:
                # Check any color choice for wild
                is_legal = any(playable_mask[get_card_index(c, card.card_type)] 
                              for c in [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE])
            else:
                is_legal = playable_mask[get_card_index(card.color, card.card_type)]
            if is_legal:
                card_str = f"*{card_str}"
        parts.append(card_str)
    return " | ".join(parts)


def format_top_card(top_card: Card, active_color: Color) -> str:
    """Format the top discard card and active color."""
    color_names = {
        Color.RED: "Red", Color.YELLOW: "Yellow",
        Color.GREEN: "Green", Color.BLUE: "Blue", Color.WILD: "Wild"
    }
    
    if top_card.is_wild:
        return f"{top_card.to_string()} (Active: {color_names[active_color]})"
    return top_card.to_string()


def apply_card_effect(card: Card) -> dict:
    """
    Get the effect of playing a card.
    
    Returns dict with effect information.
    """
    effects = {
        'skip_next': False,
        'reverse': False,
        'draw_cards': 0,
        'is_wild': card.is_wild
    }
    
    if card.card_type == CardType.SKIP:
        effects['skip_next'] = True
    elif card.card_type == CardType.REVERSE:
        effects['reverse'] = True
    elif card.card_type == CardType.DRAW_TWO:
        effects['draw_cards'] = 2
    elif card.card_type == CardType.WILD_DRAW_FOUR:
        effects['draw_cards'] = 4
    
    return effects


def get_color_emoji(color: Color) -> str:
    """Get emoji representation of color for TUI."""
    emojis = {
        Color.RED: "🔴",
        Color.YELLOW: "🟡",
        Color.GREEN: "🟢",
        Color.BLUE: "🔵",
        Color.WILD: "🌈"
    }
    return emojis.get(color, "⚪")


def get_card_display_color(card: Card) -> str:
    """Get rich color string for card display."""
    if card.is_wild:
        return "magenta"
    
    color_map = {
        Color.RED: "red",
        Color.YELLOW: "yellow",
        Color.GREEN: "green",
        Color.BLUE: "blue"
    }
    return color_map.get(card.color, "white")
