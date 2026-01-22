"""Game logic for TUI."""

import sys
import time
import threading
from pathlib import Path
from typing import Optional, Callable
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class GameEngine:
    """
    Game engine that manages Uno game state and opponent interactions.
    
    Wraps the RLCard environment and opponent adapters for the TUI.
    """
    
    def __init__(
        self,
        mode: str,
        model_key: Optional[str] = None,
        on_update: Optional[Callable] = None,
        on_log: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize game engine.
        
        Args:
            mode: Game mode ('rl', 'llm', 'human')
            model_key: LLM model key if mode is 'llm'
            on_update: Callback when game state changes
            on_log: Callback for logging (message, style)
        """
        self.mode = mode
        self.model_key = model_key
        self.on_update = on_update
        self.on_log = on_log
        
        self.env = None
        self.opponent = None
        self.state = None
        self.legal_mask = None
        self.current_player = 0
        self.game_over = False
        self.winner = None
        
        # Player hands (action IDs)
        self.player_hand = []
        self.opponent_cards = 0
        
        # Current top card
        self.top_card_action = None
        self.active_color = None
        
        self._setup_opponent()
        
        # Store last RL decision for explanation panel
        self.last_rl_decision = None
    
    def _setup_opponent(self):
        """Set up the opponent based on mode."""
        self.opponent_name = "Opponent"
        self.player0_agent = None  # For spectator mode
        self.is_spectator = False
        
        if self.mode == 'rl':
            try:
                from opponents import get_rl_opponent
                self.opponent = get_rl_opponent()
                if self.opponent:
                    self.opponent_name = self.opponent.name
                    self._log(f"Loaded: {self.opponent_name}", "green")
                else:
                    self._log("No trained model found, using random", "yellow")
                    from opponents import RandomBot
                    self.opponent = RandomBot()
                    self.opponent_name = "Random Bot"
            except Exception as e:
                self._log(f"Error loading RL model: {e}", "red")
                from opponents import RandomBot
                self.opponent = RandomBot()
                self.opponent_name = "Random Bot"
                
        elif self.mode == 'llm':
            try:
                from opponents import get_llm_opponent
                self.opponent = get_llm_opponent(self.model_key or 'gemini_flash')
                if self.opponent:
                    self.opponent_name = self.opponent.name
                    self._log(f"Connected: {self.opponent_name}", "green")
                else:
                    self._log("OpenRouter API key not set", "red")
                    from opponents import RandomBot
                    self.opponent = RandomBot()
                    self.opponent_name = "Random Bot (LLM unavailable)"
            except Exception as e:
                self._log(f"LLM error: {e}", "red")
                from opponents import RandomBot
                self.opponent = RandomBot()
                self.opponent_name = "Random Bot"
                
        elif self.mode == 'spectator':
            # Spectator mode: RL (player 0) vs LLM (player 1)
            self.is_spectator = True
            self._log("👁️ Spectator Mode: RL vs LLM", "cyan")
            
            # Set up RL as player 0
            try:
                from opponents import get_rl_opponent
                self.player0_agent = get_rl_opponent()
                if self.player0_agent:
                    self._log(f"Player 0: {self.player0_agent.name}", "green")
                else:
                    from opponents import RandomBot
                    self.player0_agent = RandomBot()
                    self._log("Player 0: Random Bot (no RL model)", "yellow")
            except Exception as e:
                from opponents import RandomBot
                self.player0_agent = RandomBot()
                self._log(f"Player 0: Random Bot (error: {e})", "red")
            
            # Set up LLM as player 1 (opponent) - use selected model or default to gemini_flash
            try:
                from opponents import get_llm_opponent
                llm_model = self.model_key or 'gemini_flash'
                self.opponent = get_llm_opponent(llm_model)
                if self.opponent:
                    self.opponent_name = self.opponent.name
                    self._log(f"Player 1: {self.opponent_name}", "green")
                else:
                    from opponents import RandomBot
                    self.opponent = RandomBot()
                    self.opponent_name = "Random Bot (no API key)"
                    self._log("Player 1: Random Bot (no API key)", "yellow")
            except Exception as e:
                from opponents import RandomBot
                self.opponent = RandomBot()
                self.opponent_name = "Random Bot"
                self._log(f"Player 1: Random Bot (error: {e})", "red")
                
        elif self.mode == 'human':
            self.opponent = None
            self.opponent_name = "Human Player 2"
            self._log("Human vs Human mode", "cyan")
    
    def _log(self, message: str, style: str = "white"):
        """Log a message."""
        if self.on_log:
            self.on_log(message, style)
    
    def start_game(self):
        """Start a new game."""
        try:
            from uno.rlcard_env import UnoRLCardEnv, RLCARD_AVAILABLE
            
            if not RLCARD_AVAILABLE:
                self._log("RLCard not available, using mock game", "yellow")
                self._start_mock_game()
                return
            
            self.env = UnoRLCardEnv(num_players=2)
            self.state, self.legal_mask, self.current_player = self.env.reset()
            
            self._update_game_state()
            self._log("Game started!", "green")
            
            # In spectator mode, trigger AI vs AI play in a background thread
            # so it doesn't block the Flask-SocketIO event loop
            if self.is_spectator:
                import threading
                spectator_thread = threading.Thread(target=self._spectator_play, daemon=True)
                spectator_thread.start()
            # If opponent goes first, make their move
            elif self.current_player != 0:
                self._opponent_turn()
                
        except Exception as e:
            self._log(f"Error starting game: {e}", "red")
            self._start_mock_game()
    
    def _start_mock_game(self):
        """Start a mock game for testing without RLCard."""
        self._log("Starting mock game (no RLCard)", "yellow")
        
        # Initialize mock state
        self.player_hand = [0, 16, 31, 47, 58]  # Some sample cards
        self.opponent_cards = 7
        self.top_card_action = 15  # Yellow 0
        self.active_color = "Yellow"
        self.legal_mask = [0] * 61
        
        # Make some cards legal
        for action_id in self.player_hand:
            self.legal_mask[action_id] = 1
        self.legal_mask[60] = 1  # Draw always legal
        
        self.current_player = 0
        self.game_over = False
        
        if self.on_update:
            self.on_update()
    
    def _update_game_state(self):
        """Update internal game state from environment."""
        if not self.env:
            return
        
        try:
            # Get player state from environment
            state_dict = self.env.env.get_state(0)
            
            # RLCard uses raw_obs for actual game data
            raw_obs = state_dict.get('raw_obs', {})
            hand_cards = raw_obs.get('hand', [])
            
            # Convert hand to action IDs
            self.player_hand = self._cards_to_actions(hand_cards)
            
            # Get opponent card count
            num_cards = raw_obs.get('num_cards', [7, 7])
            self.opponent_cards = num_cards[1] if len(num_cards) > 1 else 7
            
            # Get top card (from target)
            target = raw_obs.get('target', 'r-0')
            self.top_card_action, self.active_color = self._parse_target(target)
            
            # Update legal mask - always compute for player 0 since we display player 0's hand
            self.legal_mask = list(self.env._compute_legal_mask(player_id=0))
            
        except Exception as e:
            self._log(f"State update error: {e}", "dim")
        
        if self.on_update:
            self.on_update()
    
    def _cards_to_actions(self, card_strings: list) -> list:
        """Convert card strings to action IDs."""
        actions = []
        
        # RLCard color order: Red=0, Green=1, Blue=2, Yellow=3
        color_map = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
        type_map = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'skip': 10, 'reverse': 11, 'draw_two': 12, 'draw_2': 12,
            'wild': 13, 'wild_draw_four': 14, 'wild_draw_4': 14
        }
        
        for card_str in card_strings:
            card_str = card_str.lower()
            
            if card_str == 'wild':
                actions.append(13)  # Wild with default color (red)
            elif card_str in ('wild_draw_four', 'wild-draw-four', 'wild_draw_4'):
                actions.append(14)  # Wild+4 with default color (red)
            elif '-' in card_str:
                parts = card_str.split('-', 1)
                color_idx = color_map.get(parts[0], 0)
                type_idx = type_map.get(parts[1], 0)
                actions.append(color_idx * 15 + type_idx)
        
        return actions
    
    def _parse_target(self, target_str: str) -> tuple:
        """Parse target card string to action and color."""
        target_str = target_str.lower()
        
        # RLCard color order: Red=0, Green=1, Blue=2, Yellow=3
        color_names = {0: 'Red', 1: 'Green', 2: 'Blue', 3: 'Yellow'}
        color_map = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
        type_map = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'skip': 10, 'reverse': 11, 'draw_two': 12, 'draw_2': 12
        }
        
        if target_str.startswith('wild'):
            # Wild card - extract color if present
            if '-' in target_str:
                parts = target_str.split('-')
                color_char = parts[-1]
                color_idx = color_map.get(color_char, 0)
                type_idx = 14 if 'draw' in target_str else 13
            else:
                color_idx = 0
                type_idx = 13
            
            return (color_idx * 15 + type_idx, color_names[color_idx])
        
        elif '-' in target_str:
            parts = target_str.split('-', 1)
            color_idx = color_map.get(parts[0], 0)
            type_idx = type_map.get(parts[1], 0)
            return (color_idx * 15 + type_idx, color_names[color_idx])
        
        return (0, 'Red')
    
    def play_action(self, action_id: int):
        """Play an action (card or draw)."""
        if self.game_over:
            return
        
        if not self.env:
            # Mock game
            self._mock_play(action_id)
            return
        
        try:
            from tui.widgets import action_to_card_string
            
            card_str = action_to_card_string(action_id)
            self._log(f"You played: {card_str}", "green")
            
            # Take action
            state, reward, done, legal_mask, player, info = self.env.step(action_id)
            
            self.state = state
            self.legal_mask = list(legal_mask)
            self.current_player = player
            
            if done:
                self.game_over = True
                self.winner = 'player' if reward > 0 else 'opponent'
                self._update_game_state()
                return
            
            self._update_game_state()
            
            # Opponent's turn
            if self.current_player != 0:
                self._opponent_turn()
                
        except Exception as e:
            self._log(f"Action error: {e}", "red")
    
    def _mock_play(self, action_id: int):
        """Handle play in mock mode."""
        from tui.widgets import action_to_card_string
        
        card_str = action_to_card_string(action_id)
        self._log(f"You played: {card_str}", "green")
        
        if action_id != 60 and action_id in self.player_hand:
            self.player_hand.remove(action_id)
            self.top_card_action = action_id
        
        if len(self.player_hand) == 0:
            self.game_over = True
            self.winner = 'player'
        else:
            # Mock opponent play
            self._log("Opponent thinking...", "cyan")
            self.opponent_cards = max(0, self.opponent_cards - 1)
            
            if self.opponent_cards == 0:
                self.game_over = True
                self.winner = 'opponent'
        
        if self.on_update:
            self.on_update()
    
    def _opponent_turn(self):
        """Handle opponent's turn."""
        if self.mode == 'human':
            # In human vs human, just update display
            self._log("Player 2's turn", "cyan")
            if self.on_update:
                self.on_update()
            return
        
        if not self.opponent:
            return
        
        try:
            self._log(f"{self.opponent_name} thinking...", "cyan")
            
            # Brief delay to show thinking state in UI
            if self.on_update:
                self.on_update()  # Update UI immediately to show thinking message
            time.sleep(1.5)  # Let user see the thinking state
            
            # Get opponent's legal mask (for the current player, not player 0)
            legal_mask = np.array(self.env._compute_legal_mask(player_id=self.current_player), dtype=np.float32)
            
            # Get state encoded from the opponent's perspective (current_player, not player 0)
            # This is critical - the RL agent must see the game from its own viewpoint
            opponent_state = self.env._encode_current_state()
            
            # Get action from opponent using their properly encoded state
            # Check if opponent supports explanation (RLTorchOpponent does)
            if hasattr(self.opponent, 'act_with_explanation'):
                action, decision_info = self.opponent.act_with_explanation(opponent_state, legal_mask)
                self.last_rl_decision = decision_info
                
                # Log detailed decision info
                self._log(f"Q({decision_info['selected_card']}) = {decision_info['selected_q_value']:.3f}", "dim")
                self._log(f"Confidence: {decision_info['confidence']*100:.1f}%", "dim")
            else:
                action = self.opponent.act(opponent_state, legal_mask)
                self.last_rl_decision = None
            
            from tui.widgets import action_to_card_string
            card_str = action_to_card_string(action)
            self._log(f"{self.opponent_name} played: {card_str}", "yellow")
            
            # Take action
            state, reward, done, legal_mask, player, info = self.env.step(action)
            
            self.state = state
            self.legal_mask = list(legal_mask)
            self.current_player = player
            
            if done:
                self.game_over = True
                self.winner = 'opponent' if reward < 0 else 'player'
            
            self._update_game_state()
            
            # If still opponent's turn, continue
            if not self.game_over and self.current_player != 0:
                self._opponent_turn()
                
        except Exception as e:
            self._log(f"Opponent error: {e}", "red")
    
    def _spectator_play(self):
        """Run AI vs AI game loop for spectator mode."""
        from tui.widgets import action_to_card_string
        
        while not self.game_over:
            try:
                # Determine which AI is playing
                if self.current_player == 0:
                    agent = self.player0_agent
                    agent_name = "RL Agent"
                else:
                    agent = self.opponent
                    agent_name = self.opponent_name
                
                if not agent:
                    self._log(f"No agent for player {self.current_player}", "red")
                    break
                
                self._log(f"{agent_name} thinking...", "cyan")
                
                # Update UI to show thinking state
                if self.on_update:
                    self.on_update()
                time.sleep(1.5)  # Delay for watchability
                
                # Get legal mask for current player
                legal_mask = np.array(self.env._compute_legal_mask(player_id=self.current_player), dtype=np.float32)
                
                # Get state from current player's perspective
                current_state = self.env._encode_current_state()
                
                # Get action from the appropriate agent
                if hasattr(agent, 'act_with_explanation'):
                    action, decision_info = agent.act_with_explanation(current_state, legal_mask)
                    self.last_rl_decision = decision_info
                    self._log(f"Q({decision_info['selected_card']}) = {decision_info['selected_q_value']:.3f}", "dim")
                    self._log(f"Confidence: {decision_info['confidence']*100:.1f}%", "dim")
                else:
                    action = agent.act(current_state, legal_mask)
                    self.last_rl_decision = None
                
                card_str = action_to_card_string(action)
                self._log(f"{agent_name} played: {card_str}", "yellow")
                
                # Take action
                state, reward, done, legal_mask_new, player, info = self.env.step(action)
                
                self.state = state
                self.legal_mask = list(legal_mask_new)
                self.current_player = player
                
                if done:
                    self.game_over = True
                    # In spectator mode, determine winner based on who just played
                    if self.current_player == 0 or (done and reward > 0):
                        self.winner = "RL Agent" 
                    else:
                        self.winner = self.opponent_name
                    self._log(f"🎉 {self.winner} wins!", "green")
                
                self._update_game_state()
                
            except Exception as e:
                self._log(f"Spectator mode error: {e}", "red")
                import traceback
                traceback.print_exc()
                break
    
    def get_state(self) -> dict:
        """Get current game state for display."""
        # Convert legal_mask to native Python types for JSON serialization
        legal_mask_native = [int(x) for x in self.legal_mask] if self.legal_mask else [0] * 61
        
        # Get player 0 agent name for spectator mode
        player0_name = "RL Agent"
        if self.player0_agent and hasattr(self.player0_agent, 'name'):
            player0_name = self.player0_agent.name
        
        # Determine current player name for spectator mode
        if self.current_player == 0:
            current_player_name = player0_name
        else:
            current_player_name = self.opponent_name
        
        return {
            'player_hand': [int(x) for x in self.player_hand] if self.player_hand else [],
            'legal_mask': legal_mask_native,
            'top_card_action': int(self.top_card_action) if self.top_card_action is not None else None,
            'active_color': self.active_color,
            'opponent_name': self.opponent_name,
            'opponent_cards': int(self.opponent_cards) if self.opponent_cards else 0,
            'is_player_turn': self.current_player == 0,
            'game_over': self.game_over,
            'winner': self.winner,
            'rl_decision': self.last_rl_decision,
            'is_spectator': self.is_spectator,
            'player0_name': player0_name,
            'current_player_name': current_player_name
        }


__all__ = ['GameEngine']
