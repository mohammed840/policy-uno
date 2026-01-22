"""TUI widgets for Uno game."""

from rich.console import RenderableType
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from textual.widget import Widget
from textual.widgets import Static, Button
from textual.reactive import reactive
from textual.message import Message
from textual.containers import Container, Vertical, Horizontal


# Color mappings
COLOR_NAMES = ['Red', 'Yellow', 'Green', 'Blue']
TYPE_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
              'Skip', 'Reverse', '+2', 'Wild', 'Wild+4']


def action_to_card_string(action_id: int) -> str:
    """Convert action ID to card string."""
    if action_id == 60:
        return "Draw"
    
    color_idx = action_id // 15
    type_idx = action_id % 15
    
    if type_idx >= 13:  # Wild cards
        return f"{TYPE_NAMES[type_idx]} ({COLOR_NAMES[color_idx]})"
    else:
        return f"{COLOR_NAMES[color_idx]} {TYPE_NAMES[type_idx]}"


def get_card_color(action_id: int) -> str:
    """Get color name for a card."""
    if action_id == 60:
        return "white"
    color_idx = action_id // 15
    type_idx = action_id % 15
    if type_idx >= 13:
        return "magenta"
    return ['red', 'yellow', 'green', 'blue'][color_idx]


def get_card_style(action_id: int) -> str:
    """Get Rich style for a card."""
    return f"{get_card_color(action_id)} bold"


class CardButton(Button):
    """A button representing a card."""
    
    class CardPlayed(Message):
        """Message sent when a card is played."""
        def __init__(self, index: int, action_id: int):
            self.index = index
            self.action_id = action_id
            super().__init__()
    
    def __init__(self, action_id: int, index: int, playable: bool = True, **kwargs):
        self.action_id = action_id
        self.index = index
        self.is_playable = playable
        
        card_str = action_to_card_string(action_id)
        label = f"[{index}] {card_str}"
        
        variant = "primary" if playable else "default"
        super().__init__(label, variant=variant, disabled=not playable, **kwargs)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if self.is_playable:
            self.post_message(self.CardPlayed(self.index, self.action_id))


class DrawButton(Button):
    """Button to draw a card."""
    
    class DrawPressed(Message):
        """Message sent when draw is pressed."""
        pass
    
    def __init__(self, **kwargs):
        super().__init__("[D] Draw Card", variant="warning", **kwargs)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        self.post_message(self.DrawPressed())


class HandContainer(Vertical):
    """Container for displaying hand with buttons."""
    
    DEFAULT_CSS = """
    HandContainer {
        height: auto;
        padding: 1;
        border: solid blue;
    }
    
    HandContainer Button {
        width: 100%;
        margin: 0 0 1 0;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hand = []
        self._legal_mask = [0] * 61
        self._rebuild_counter = 0  # Counter for unique IDs
    
    def update_hand(self, hand: list, legal_mask: list):
        """Update the displayed hand."""
        self._hand = hand
        self._legal_mask = legal_mask
        self._rebuild_buttons()
    
    def _rebuild_buttons(self):
        """Rebuild card buttons."""
        # Increment counter to ensure unique IDs for this rebuild cycle
        self._rebuild_counter += 1
        prefix = f"r{self._rebuild_counter}"
        
        # Remove old children (async, but new IDs won't conflict)
        for child in list(self.children):
            child.remove()
        
        # Add title
        self.mount(Static(Text("Your Hand", style="bold cyan")))
        
        # Add card buttons with unique IDs
        for i, action_id in enumerate(self._hand):
            is_playable = self._legal_mask[action_id] > 0 if action_id < len(self._legal_mask) else False
            btn = CardButton(action_id, i, is_playable, id=f"{prefix}-card-{i}")
            self.mount(btn)
        
        # Add draw button with unique ID
        self.mount(DrawButton(id=f"{prefix}-draw-btn"))


class HandWidget(Static):
    """Simple static hand display (fallback)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hand = []
        self._legal_mask = [0] * 61
    
    def update_hand(self, hand: list, legal_mask: list):
        """Update the displayed hand."""
        self._hand = hand
        self._legal_mask = legal_mask
        self.refresh()
    
    def render(self) -> RenderableType:
        if not self._hand:
            return Panel("No cards", title="Your Hand", border_style="blue")
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        
        for i, action_id in enumerate(self._hand):
            card_str = action_to_card_string(action_id)
            is_playable = self._legal_mask[action_id] > 0 if action_id < len(self._legal_mask) else False
            
            style = get_card_style(action_id)
            if not is_playable:
                style = "dim"
            
            prefix = "▶ " if is_playable else "  "
            table.add_row(f"{prefix}[{i}]", Text(card_str, style=style))
        
        table.add_row("▶ [D]", Text("Draw Card", style="cyan bold"))
        
        return Panel(table, title="Your Hand (0-9 to play, D to draw)", border_style="blue")


class DiscardWidget(Static):
    """Widget displaying the discard pile top card."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._top_card = None
        self._active_color = None
    
    def update_discard(self, top_card_action: int, active_color: str = None):
        """Update the displayed discard."""
        self._top_card = top_card_action
        self._active_color = active_color
        self.refresh()
    
    def render(self) -> RenderableType:
        if self._top_card is None:
            return Panel("No card", title="Discard Pile", border_style="yellow")
        
        card_str = action_to_card_string(self._top_card)
        style = get_card_style(self._top_card)
        
        content = Text(card_str, style=style)
        
        if self._active_color:
            content.append(f"\nActive: {self._active_color}", style="bold")
        
        return Panel(content, title="Discard Pile", border_style="yellow")


class OpponentWidget(Static):
    """Widget displaying opponent info."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = "Opponent"
        self._card_count = 0
        self._is_turn = False
    
    def update_opponent(self, name: str, card_count: int, is_turn: bool = False):
        """Update opponent display."""
        self._name = name
        self._card_count = card_count
        self._is_turn = is_turn
        self.refresh()
    
    def render(self) -> RenderableType:
        style = "bold green" if self._is_turn else "white"
        turn_indicator = " ◀" if self._is_turn else ""
        
        content = Text()
        content.append(f"{self._name}{turn_indicator}\n", style=style)
        content.append(f"Cards: {self._card_count}", style="dim")
        
        return Panel(content, title="Opponent", border_style="red")


class LogWidget(Static):
    """Widget displaying game log."""
    
    def __init__(self, max_lines: int = 10, **kwargs):
        super().__init__(**kwargs)
        self._logs = []
        self._max_lines = max_lines
    
    def add_log(self, message: str, style: str = "white"):
        """Add a log message."""
        self._logs.append((message, style))
        if len(self._logs) > self._max_lines:
            self._logs = self._logs[-self._max_lines:]
        self.refresh()
    
    def clear(self):
        """Clear all logs."""
        self._logs = []
        self.refresh()
    
    def render(self) -> RenderableType:
        content = Text()
        
        for i, (msg, style) in enumerate(self._logs):
            if i > 0:
                content.append("\n")
            content.append(msg, style=style)
        
        if not self._logs:
            content.append("Game started...", style="dim")
        
        return Panel(content, title="Game Log", border_style="cyan")


class GameStatusWidget(Static):
    """Widget showing game status."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._status = "Waiting..."
        self._is_player_turn = False
    
    def update_status(self, status: str, is_player_turn: bool = False):
        """Update status display."""
        self._status = status
        self._is_player_turn = is_player_turn
        self.refresh()
    
    def render(self) -> RenderableType:
        style = "bold green" if self._is_player_turn else "yellow"
        return Panel(
            Text(self._status, style=style),
            title="Status",
            border_style="green" if self._is_player_turn else "yellow"
        )


__all__ = [
    'CardButton', 'DrawButton', 'HandContainer', 'HandWidget',
    'DiscardWidget', 'OpponentWidget', 'LogWidget', 'GameStatusWidget',
    'action_to_card_string', 'get_card_style'
]
