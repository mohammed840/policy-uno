"""TUI screens for Uno game."""

import sys
from pathlib import Path
from typing import Optional, Callable

from textual import on
from textual.app import ComposeResult
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Header, Footer, Button, Static, Label,
    ListView, ListItem, OptionList, Rule
)
from textual.widgets.option_list import Option
from textual.containers import Container, Horizontal, Vertical, Center, Middle
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from .widgets import (
    HandWidget, HandContainer, CardButton, DrawButton,
    DiscardWidget, OpponentWidget, LogWidget, GameStatusWidget,
    action_to_card_string
)


class MenuScreen(Screen):
    """Main menu screen."""
    
    BINDINGS = [
        Binding("1", "select_rl", "Play vs RL Agent"),
        Binding("2", "select_llm", "Play vs LLM"),
        Binding("3", "select_human", "Human vs Human"),
        Binding("q", "quit", "Quit"),
    ]
    
    CSS = """
    MenuScreen {
        align: center middle;
    }
    
    #menu-container {
        width: 60;
        height: auto;
        padding: 2;
        border: solid green;
        background: $surface;
    }
    
    #menu-title {
        text-align: center;
        text-style: bold;
        color: $text;
        padding-bottom: 1;
    }
    
    .menu-button {
        width: 100%;
        margin: 1 0;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Middle():
                with Vertical(id="menu-container"):
                    yield Static(
                        Text("🃏 UNO Terminal Game 🃏", style="bold magenta"),
                        id="menu-title"
                    )
                    yield Rule()
                    yield Button(
                        "1. Play vs RL Agent (Core ML)",
                        id="btn-rl",
                        classes="menu-button",
                        variant="primary"
                    )
                    yield Button(
                        "2. Play vs LLM Opponent (OpenRouter)",
                        id="btn-llm", 
                        classes="menu-button",
                        variant="default"
                    )
                    yield Button(
                        "3. Human vs Human (Local)",
                        id="btn-human",
                        classes="menu-button",
                        variant="default"
                    )
                    yield Rule()
                    yield Button(
                        "Quit",
                        id="btn-quit",
                        classes="menu-button",
                        variant="error"
                    )
        yield Footer()
    
    @on(Button.Pressed, "#btn-rl")
    def on_rl_pressed(self) -> None:
        self.app.start_game("rl")
    
    @on(Button.Pressed, "#btn-llm")
    def on_llm_pressed(self) -> None:
        self.app.push_screen(ModelPickerScreen())
    
    @on(Button.Pressed, "#btn-human")
    def on_human_pressed(self) -> None:
        self.app.start_game("human")
    
    @on(Button.Pressed, "#btn-quit")
    def on_quit_pressed(self) -> None:
        self.app.exit()
    
    def action_select_rl(self) -> None:
        self.app.start_game("rl")
    
    def action_select_llm(self) -> None:
        self.app.push_screen(ModelPickerScreen())
    
    def action_select_human(self) -> None:
        self.app.start_game("human")
    
    def action_quit(self) -> None:
        self.app.exit()


class ModelPickerScreen(Screen):
    """LLM model selection screen."""
    
    BINDINGS = [
        Binding("1", "select_gemini", "Gemini 3 Flash"),
        Binding("2", "select_gpt", "GPT 5.2"),
        Binding("3", "select_opus", "Opus 4.5"),
        Binding("escape", "go_back", "Back"),
    ]
    
    CSS = """
    ModelPickerScreen {
        align: center middle;
    }
    
    #model-container {
        width: 60;
        height: auto;
        padding: 2;
        border: solid blue;
        background: $surface;
    }
    
    #model-title {
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }
    
    .model-button {
        width: 100%;
        margin: 1 0;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Middle():
                with Vertical(id="model-container"):
                    yield Static(
                        Text("Select LLM Model", style="bold cyan"),
                        id="model-title"
                    )
                    yield Rule()
                    yield Button(
                        "1. Gemini 3 Flash (google/gemini-3-flash-preview)",
                        id="btn-gemini",
                        classes="model-button",
                        variant="primary"
                    )
                    yield Button(
                        "2. GPT 5.2 (openai/gpt-5.2)",
                        id="btn-gpt",
                        classes="model-button",
                        variant="default"
                    )
                    yield Button(
                        "3. Opus 4.5 (anthropic/claude-opus-4.5)",
                        id="btn-opus",
                        classes="model-button",
                        variant="default"
                    )
                    yield Rule()
                    yield Button(
                        "← Back",
                        id="btn-back",
                        classes="model-button",
                        variant="warning"
                    )
        yield Footer()
    
    @on(Button.Pressed, "#btn-gemini")
    def on_gemini_pressed(self) -> None:
        self.app.start_game("llm", model_key="gemini_flash")
    
    @on(Button.Pressed, "#btn-gpt")
    def on_gpt_pressed(self) -> None:
        self.app.start_game("llm", model_key="gpt_5")
    
    @on(Button.Pressed, "#btn-opus")
    def on_opus_pressed(self) -> None:
        self.app.start_game("llm", model_key="opus")
    
    @on(Button.Pressed, "#btn-back")
    def on_back_pressed(self) -> None:
        self.app.pop_screen()
    
    def action_select_gemini(self) -> None:
        self.app.start_game("llm", model_key="gemini_flash")
    
    def action_select_gpt(self) -> None:
        self.app.start_game("llm", model_key="gpt_5")
    
    def action_select_opus(self) -> None:
        self.app.start_game("llm", model_key="opus")
    
    def action_go_back(self) -> None:
        self.app.pop_screen()


class ColorPickerModal(ModalScreen):
    """Modal for selecting Wild card color."""
    
    BINDINGS = [
        Binding("r", "select_red", "Red"),
        Binding("y", "select_yellow", "Yellow"),
        Binding("g", "select_green", "Green"),
        Binding("b", "select_blue", "Blue"),
        Binding("escape", "cancel", "Cancel"),
    ]
    
    CSS = """
    ColorPickerModal {
        align: center middle;
    }
    
    #color-container {
        width: 40;
        height: auto;
        padding: 2;
        border: solid magenta;
        background: $surface;
    }
    
    .color-button {
        width: 100%;
        margin: 1 0;
    }
    
    #btn-red { background: red; }
    #btn-yellow { background: yellow; color: black; }
    #btn-green { background: green; }
    #btn-blue { background: blue; }
    """
    
    def __init__(self, callback: Callable[[str], None], **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
    
    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Vertical(id="color-container"):
                    yield Static(
                        Text("Choose Wild Color", style="bold magenta"),
                        id="color-title"
                    )
                    yield Button("[R] Red", id="btn-red", classes="color-button")
                    yield Button("[Y] Yellow", id="btn-yellow", classes="color-button")
                    yield Button("[G] Green", id="btn-green", classes="color-button")
                    yield Button("[B] Blue", id="btn-blue", classes="color-button")
    
    def _select_color(self, color: str) -> None:
        self.dismiss(color)
        self.callback(color)
    
    @on(Button.Pressed, "#btn-red")
    def on_red(self) -> None:
        self._select_color("red")
    
    @on(Button.Pressed, "#btn-yellow")
    def on_yellow(self) -> None:
        self._select_color("yellow")
    
    @on(Button.Pressed, "#btn-green")
    def on_green(self) -> None:
        self._select_color("green")
    
    @on(Button.Pressed, "#btn-blue")
    def on_blue(self) -> None:
        self._select_color("blue")
    
    def action_select_red(self) -> None:
        self._select_color("red")
    
    def action_select_yellow(self) -> None:
        self._select_color("yellow")
    
    def action_select_green(self) -> None:
        self._select_color("green")
    
    def action_select_blue(self) -> None:
        self._select_color("blue")
    
    def action_cancel(self) -> None:
        self.dismiss(None)


class GameScreen(Screen):
    """Main gameplay screen."""
    
    BINDINGS = [
        Binding("0", "select_card_0", "Card 0", show=False),
        Binding("1", "select_card_1", "Card 1", show=False),
        Binding("2", "select_card_2", "Card 2", show=False),
        Binding("3", "select_card_3", "Card 3", show=False),
        Binding("4", "select_card_4", "Card 4", show=False),
        Binding("5", "select_card_5", "Card 5", show=False),
        Binding("6", "select_card_6", "Card 6", show=False),
        Binding("7", "select_card_7", "Card 7", show=False),
        Binding("8", "select_card_8", "Card 8", show=False),
        Binding("9", "select_card_9", "Card 9", show=False),
        Binding("d", "draw_card", "Draw"),
        Binding("q", "quit_game", "Quit"),
    ]
    
    CSS = """
    GameScreen {
        layout: grid;
        grid-size: 3 3;
        grid-columns: 1fr 2fr 1fr;
        grid-rows: 1fr 2fr 1fr;
    }
    
    #opponent-area {
        column-span: 3;
        height: 100%;
        align: center middle;
    }
    
    #left-panel {
        height: 100%;
    }
    
    #center-panel {
        height: 100%;
        align: center middle;
    }
    
    #right-panel {
        height: 100%;
    }
    
    #player-area {
        column-span: 3;
        height: 100%;
    }
    
    HandWidget {
        height: 100%;
    }
    
    DiscardWidget {
        height: 100%;
        width: 100%;
    }
    
    OpponentWidget {
        height: auto;
    }
    
    LogWidget {
        height: 100%;
    }
    
    GameStatusWidget {
        height: auto;
    }
    """
    
    def __init__(
        self,
        game_mode: str,
        model_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.game_mode = game_mode
        self.model_key = model_key
        self.game_engine = None
        self.player_hand = []
        self.legal_mask = [0] * 61
        self.is_player_turn = True
        self.pending_wild_action = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="opponent-area"):
            yield OpponentWidget(id="opponent")
        
        with Container(id="left-panel"):
            yield LogWidget(id="log")
        
        with Container(id="center-panel"):
            yield DiscardWidget(id="discard")
            yield GameStatusWidget(id="status")
        
        with Container(id="right-panel"):
            yield Static(
                Panel(
                    Text("Controls:\n0-9: Play card\nD: Draw\nQ: Quit", style="dim"),
                    title="Help",
                    border_style="dim"
                ),
                id="help"
            )
        
        with Container(id="player-area"):
            yield HandContainer(id="hand")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize game when screen is mounted."""
        self.log_widget = self.query_one("#log", LogWidget)
        self.hand_widget = self.query_one("#hand", HandContainer)
        self.discard_widget = self.query_one("#discard", DiscardWidget)
        self.opponent_widget = self.query_one("#opponent", OpponentWidget)
        self.status_widget = self.query_one("#status", GameStatusWidget)
        
        self.log_widget.add_log(f"Game started: {self.game_mode} mode", "green")
        
        # Initialize game
        self.call_later(self._initialize_game)
    
    def _initialize_game(self) -> None:
        """Initialize the game engine."""
        from tui.game_logic import GameEngine
        
        try:
            self.game_engine = GameEngine(
                mode=self.game_mode,
                model_key=self.model_key,
                on_update=self._on_game_update,
                on_log=self._on_log
            )
            self.game_engine.start_game()
            self._update_display()
        except Exception as e:
            self.log_widget.add_log(f"Error: {e}", "red")
            self.status_widget.update_status(f"Error: {e}")
    
    def _on_game_update(self) -> None:
        """Called when game state changes."""
        # Use call_later for safe UI updates from callbacks
        self.call_later(self._update_display)
    
    def _on_log(self, message: str, style: str = "white") -> None:
        """Called when game logs a message."""
        # Use call_later for safe UI updates from callbacks
        self.call_later(lambda: self.log_widget.add_log(message, style))
    
    def _update_display(self) -> None:
        """Update all display widgets."""
        if not self.game_engine:
            return
        
        state = self.game_engine.get_state()
        
        self.player_hand = state.get('player_hand', [])
        self.legal_mask = state.get('legal_mask', [0] * 61)
        self.is_player_turn = state.get('is_player_turn', True)
        
        # Update widgets
        self.hand_widget.update_hand(self.player_hand, self.legal_mask)
        self.discard_widget.update_discard(
            state.get('top_card_action'),
            state.get('active_color')
        )
        self.opponent_widget.update_opponent(
            state.get('opponent_name', 'Opponent'),
            state.get('opponent_cards', 0),
            not self.is_player_turn
        )
        
        # Update status
        if state.get('game_over'):
            winner = state.get('winner', 'Unknown')
            if winner == 'player':
                self.status_widget.update_status("🎉 You Won! 🎉", True)
                self.log_widget.add_log("Congratulations! You won!", "green bold")
            else:
                self.status_widget.update_status("Game Over - Opponent Won", False)
                self.log_widget.add_log("Opponent won. Better luck next time!", "red")
        elif self.is_player_turn:
            self.status_widget.update_status("Your Turn - Select a card", True)
        else:
            self.status_widget.update_status("Opponent's Turn...", False)
    
    @on(CardButton.CardPlayed)
    def on_card_played(self, event: CardButton.CardPlayed) -> None:
        """Handle card button click."""
        self._play_card(event.index)
    
    @on(DrawButton.DrawPressed)
    def on_draw_pressed(self, event: DrawButton.DrawPressed) -> None:
        """Handle draw button click."""
        self.action_draw_card()
    
    def _play_card(self, card_index: int) -> None:
        """Play a card from hand."""
        if not self.game_engine or not self.is_player_turn:
            return
        
        if card_index >= len(self.player_hand):
            self.log_widget.add_log(f"Invalid card index: {card_index}", "red")
            return
        
        action_id = self.player_hand[card_index]
        
        # Check if legal (use float comparison for numpy types)
        if float(self.legal_mask[action_id]) <= 0:
            self.log_widget.add_log("That card can't be played right now", "yellow")
            return
        
        # Check if Wild card (needs color selection)
        type_idx = action_id % 15
        if type_idx >= 13:  # Wild or Wild+4
            self.pending_wild_action = action_id
            self.app.push_screen(ColorPickerModal(self._on_color_selected))
            return
        
        self.game_engine.play_action(action_id)
    
    def _on_color_selected(self, color: Optional[str]) -> None:
        """Handle Wild card color selection."""
        if color is None or self.pending_wild_action is None:
            self.pending_wild_action = None
            return
        
        # Compute new action with color
        color_map = {'red': 0, 'yellow': 1, 'green': 2, 'blue': 3}
        color_idx = color_map.get(color, 0)
        type_idx = self.pending_wild_action % 15
        
        new_action = color_idx * 15 + type_idx
        self.game_engine.play_action(new_action)
        self.pending_wild_action = None
    
    def action_draw_card(self) -> None:
        """Draw a card."""
        if not self.game_engine or not self.is_player_turn:
            return
        
        self.game_engine.play_action(60)  # Draw action
    
    def action_quit_game(self) -> None:
        """Quit the game."""
        self.app.pop_screen()
    
    # Card selection actions
    def action_select_card_0(self) -> None:
        self._play_card(0)
    
    def action_select_card_1(self) -> None:
        self._play_card(1)
    
    def action_select_card_2(self) -> None:
        self._play_card(2)
    
    def action_select_card_3(self) -> None:
        self._play_card(3)
    
    def action_select_card_4(self) -> None:
        self._play_card(4)
    
    def action_select_card_5(self) -> None:
        self._play_card(5)
    
    def action_select_card_6(self) -> None:
        self._play_card(6)
    
    def action_select_card_7(self) -> None:
        self._play_card(7)
    
    def action_select_card_8(self) -> None:
        self._play_card(8)
    
    def action_select_card_9(self) -> None:
        self._play_card(9)
    
    def on_key(self, event) -> None:
        """Handle key presses directly."""
        key = event.key
        
        # Number keys 0-9
        if key in "0123456789":
            index = int(key)
            self._play_card(index)
            event.prevent_default()
            event.stop()
        elif key.lower() == "d":
            self.action_draw_card()
            event.prevent_default()
            event.stop()
        elif key.lower() == "q":
            self.action_quit_game()
            event.prevent_default()
            event.stop()


class ResultScreen(Screen):
    """Game result screen."""
    
    BINDINGS = [
        Binding("enter", "continue", "Continue"),
        Binding("q", "quit", "Quit"),
    ]
    
    CSS = """
    ResultScreen {
        align: center middle;
    }
    
    #result-container {
        width: 50;
        height: auto;
        padding: 2;
        border: solid green;
        background: $surface;
    }
    
    #result-title {
        text-align: center;
        text-style: bold;
        padding: 1;
    }
    
    .result-button {
        width: 100%;
        margin: 1 0;
    }
    """
    
    def __init__(self, winner: str, **kwargs):
        super().__init__(**kwargs)
        self.winner = winner
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Middle():
                with Vertical(id="result-container"):
                    if self.winner == "player":
                        yield Static(
                            Text("🎉 You Won! 🎉", style="bold green"),
                            id="result-title"
                        )
                    else:
                        yield Static(
                            Text("Game Over", style="bold red"),
                            id="result-title"
                        )
                    yield Rule()
                    yield Button(
                        "Play Again",
                        id="btn-again",
                        classes="result-button",
                        variant="primary"
                    )
                    yield Button(
                        "Main Menu",
                        id="btn-menu",
                        classes="result-button",
                        variant="default"
                    )
                    yield Button(
                        "Quit",
                        id="btn-quit",
                        classes="result-button",
                        variant="error"
                    )
        yield Footer()
    
    @on(Button.Pressed, "#btn-again")
    def on_again_pressed(self) -> None:
        self.app.pop_screen()  # Go back to game
    
    @on(Button.Pressed, "#btn-menu")
    def on_menu_pressed(self) -> None:
        self.app.switch_screen(MenuScreen())
    
    @on(Button.Pressed, "#btn-quit")
    def on_quit_pressed(self) -> None:
        self.app.exit()
    
    def action_continue(self) -> None:
        self.app.switch_screen(MenuScreen())
    
    def action_quit(self) -> None:
        self.app.exit()


__all__ = [
    'MenuScreen', 'ModelPickerScreen', 'ColorPickerModal',
    'GameScreen', 'ResultScreen'
]
