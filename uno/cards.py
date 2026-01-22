"""Uno card definitions and utilities."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Color(Enum):
    """Card colors in Uno."""
    RED = 0
    YELLOW = 1
    GREEN = 2
    BLUE = 3
    WILD = 4  # For Wild cards before color is chosen


class CardType(Enum):
    """Card types in Uno - 15 total types matching the encoding spec."""
    # Number cards (0-9)
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    # Action cards
    SKIP = 10
    REVERSE = 11
    DRAW_TWO = 12
    WILD = 13
    WILD_DRAW_FOUR = 14


# Card type indices for encoding
CARD_TYPE_COUNT = 15
COLOR_COUNT = 4  # Red, Yellow, Green, Blue (excluding Wild)


@dataclass(frozen=True)
class Card:
    """Represents an Uno card."""
    color: Color
    card_type: CardType
    
    def __post_init__(self):
        # Validate Wild cards have WILD color
        if self.card_type in (CardType.WILD, CardType.WILD_DRAW_FOUR):
            if self.color != Color.WILD:
                object.__setattr__(self, 'color', Color.WILD)
    
    @property
    def is_wild(self) -> bool:
        """Check if card is a Wild or Wild+4."""
        return self.card_type in (CardType.WILD, CardType.WILD_DRAW_FOUR)
    
    @property
    def is_action(self) -> bool:
        """Check if card is an action card (Skip, Reverse, Draw Two)."""
        return self.card_type in (CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO)
    
    @property
    def is_number(self) -> bool:
        """Check if card is a number card (0-9)."""
        return self.card_type.value <= 9
    
    def to_string(self) -> str:
        """Format card as human-readable string."""
        type_names = {
            CardType.ZERO: "0", CardType.ONE: "1", CardType.TWO: "2",
            CardType.THREE: "3", CardType.FOUR: "4", CardType.FIVE: "5",
            CardType.SIX: "6", CardType.SEVEN: "7", CardType.EIGHT: "8",
            CardType.NINE: "9", CardType.SKIP: "Skip", CardType.REVERSE: "Reverse",
            CardType.DRAW_TWO: "+2", CardType.WILD: "Wild", CardType.WILD_DRAW_FOUR: "Wild+4"
        }
        
        if self.is_wild:
            return type_names[self.card_type]
        
        color_names = {
            Color.RED: "Red", Color.YELLOW: "Yellow",
            Color.GREEN: "Green", Color.BLUE: "Blue"
        }
        return f"{color_names[self.color]} {type_names[self.card_type]}"
    
    def to_short_string(self) -> str:
        """Format card as short string for TUI."""
        type_chars = {
            CardType.ZERO: "0", CardType.ONE: "1", CardType.TWO: "2",
            CardType.THREE: "3", CardType.FOUR: "4", CardType.FIVE: "5",
            CardType.SIX: "6", CardType.SEVEN: "7", CardType.EIGHT: "8",
            CardType.NINE: "9", CardType.SKIP: "S", CardType.REVERSE: "R",
            CardType.DRAW_TWO: "D", CardType.WILD: "W", CardType.WILD_DRAW_FOUR: "W4"
        }
        
        if self.is_wild:
            return type_chars[self.card_type]
        
        color_chars = {
            Color.RED: "R", Color.YELLOW: "Y",
            Color.GREEN: "G", Color.BLUE: "B"
        }
        return f"{color_chars[self.color]}{type_chars[self.card_type]}"
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __repr__(self) -> str:
        return f"Card({self.color.name}, {self.card_type.name})"


def color_from_string(s: str) -> Color:
    """Parse color from string."""
    s = s.lower().strip()
    mapping = {
        'r': Color.RED, 'red': Color.RED,
        'y': Color.YELLOW, 'yellow': Color.YELLOW,
        'g': Color.GREEN, 'green': Color.GREEN,
        'b': Color.BLUE, 'blue': Color.BLUE,
    }
    if s in mapping:
        return mapping[s]
    raise ValueError(f"Unknown color: {s}")


def card_type_from_value(value: int) -> CardType:
    """Get CardType from integer value."""
    return CardType(value)


def get_card_index(color: Color, card_type: CardType) -> int:
    """
    Get the action index for playing a specific card.
    
    Action space mapping:
    - Actions 0-59: Play card (color_idx * 15 + type_idx)
    - Action 60: Draw
    
    For Wild cards, we use color index based on the chosen color.
    """
    if card_type in (CardType.WILD, CardType.WILD_DRAW_FOUR):
        # Wild cards can be played with any color choice
        # The action index depends on the chosen color
        if color == Color.WILD:
            raise ValueError("Must specify a color choice for Wild cards")
        color_idx = color.value
    else:
        color_idx = color.value
    
    type_idx = card_type.value
    return color_idx * CARD_TYPE_COUNT + type_idx


def card_from_action_index(action_idx: int) -> tuple[Color, CardType]:
    """
    Convert action index back to color and card type.
    
    Returns (color, card_type) tuple.
    """
    if action_idx == 60:
        raise ValueError("Action 60 is draw, not a card")
    
    color_idx = action_idx // CARD_TYPE_COUNT
    type_idx = action_idx % CARD_TYPE_COUNT
    
    return Color(color_idx), CardType(type_idx)


# Action index for drawing a card
DRAW_ACTION = 60

# Total action space size
ACTION_SPACE_SIZE = 61
