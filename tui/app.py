"""
Uno TUI Application.

Terminal interface for playing Uno against RL agents, LLM opponents, or other humans.
"""

import sys
from pathlib import Path
from typing import Optional

from textual.app import App
from textual.binding import Binding

sys.path.insert(0, str(Path(__file__).parent.parent))

from .screens import MenuScreen, GameScreen


class UnoApp(App):
    """Main Uno TUI application."""
    
    TITLE = "Uno Terminal Game"
    CSS = """
    Screen {
        background: $surface;
    }
    
    Header {
        background: $primary;
    }
    
    Footer {
        background: $primary-darken-1;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+q", "quit", "Quit", show=False),
    ]
    
    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.push_screen(MenuScreen())
    
    def start_game(self, mode: str, model_key: Optional[str] = None) -> None:
        """
        Start a new game.
        
        Args:
            mode: Game mode ('rl', 'llm', 'human')
            model_key: LLM model key if mode is 'llm'
        """
        self.push_screen(GameScreen(game_mode=mode, model_key=model_key))
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Uno Terminal Game')
    parser.add_argument(
        '--mode',
        choices=['menu', 'rl', 'llm', 'human'],
        default='menu',
        help='Start mode (default: menu)'
    )
    parser.add_argument(
        '--model',
        choices=['gemini_flash', 'gpt_5', 'opus'],
        default='gemini_flash',
        help='LLM model for llm mode'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    app = UnoApp()
    
    if args.debug:
        app.run(inline=False)
    else:
        app.run()


if __name__ == '__main__':
    main()
