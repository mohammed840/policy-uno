"""OpenRouter LLM opponent adapter."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Opponent

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# Available LLM models
LLM_MODELS = {
    "gemini_flash": {
        "name": "Gemini 3 Flash",
        "slug": "google/gemini-3-flash-preview"
    },
    "gpt_5": {
        "name": "GPT 5.2", 
        "slug": "openai/gpt-5.2"
    },
    "opus": {
        "name": "Opus 4.5",
        "slug": "anthropic/claude-opus-4.5"
    }
}


def get_available_models() -> dict:
    """Get available LLM models."""
    return LLM_MODELS


class LLMOpenRouterOpponent(Opponent):
    """
    Opponent using OpenRouter LLM API.
    
    Sends game state as prompt and parses JSON action response.
    """
    
    def __init__(
        self,
        model_slug: str = "google/gemini-3-flash-preview",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 1
    ):
        """
        Initialize OpenRouter opponent.
        
        Args:
            model_slug: OpenRouter model identifier
            api_key: API key (or from OPENROUTER_API_KEY env var)
            base_url: API base URL (or from OPENROUTER_BASE_URL env var)
            max_retries: Number of retries on invalid response
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is not installed. Install with: pip install httpx")
        
        self.model_slug = model_slug
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.max_retries = max_retries
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
        
        self.client = httpx.Client(timeout=30.0)
        self._model_name = None
        
        # Find display name
        for key, info in LLM_MODELS.items():
            if info["slug"] == model_slug:
                self._model_name = info["name"]
                break
        
        if self._model_name is None:
            self._model_name = model_slug.split("/")[-1]
    
    def _format_state_prompt(
        self,
        obs: np.ndarray,
        legal_mask: np.ndarray,
        hand_cards: Optional[list] = None,
        top_card: Optional[str] = None,
        active_color: Optional[str] = None
    ) -> str:
        """Format game state as prompt for LLM."""
        # Get legal action indices
        legal_actions = np.where(legal_mask > 0)[0].tolist()
        
        # Build prompt
        prompt = """You are playing Uno. Choose the best action.

GAME STATE:
"""
        if top_card:
            prompt += f"- Top card: {top_card}\n"
        if active_color:
            prompt += f"- Active color: {active_color}\n"
        if hand_cards:
            prompt += f"- Your hand: {', '.join(hand_cards)}\n"
        
        prompt += f"""
LEGAL ACTIONS (choose one action_id):
"""
        for action_id in legal_actions:
            if action_id == 60:
                prompt += f"  {action_id}: Draw a card\n"
            else:
                # Decode action to card
                color_idx = action_id // 15
                type_idx = action_id % 15
                colors = ["Red", "Yellow", "Green", "Blue"]
                types = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
                        "Skip", "Reverse", "+2", "Wild", "Wild+4"]
                
                if type_idx >= 13:  # Wild cards
                    card_name = f"{types[type_idx]} (choose {colors[color_idx]})"
                else:
                    card_name = f"{colors[color_idx]} {types[type_idx]}"
                prompt += f"  {action_id}: Play {card_name}\n"
        
        prompt += """
RESPOND WITH ONLY JSON: {"action_id": <number>}
"""
        return prompt
    
    def _call_llm(self, prompt: str) -> dict:
        """Call OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/uno-coreml-tui",
        }
        
        payload = {
            "model": self.model_slug,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an Uno card game player. Respond only with valid JSON."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        return response.json()
    
    def _parse_action(self, response: dict, legal_mask: np.ndarray) -> Optional[int]:
        """Parse action from LLM response."""
        try:
            content = response["choices"][0]["message"]["content"]
            
            # Try to extract JSON from response
            content = content.strip()
            
            # Handle markdown code blocks
            if "```" in content:
                # Extract content between code blocks
                start = content.find("```")
                end = content.rfind("```")
                if start != end:
                    content = content[start:end]
                    # Remove language identifier if present
                    if content.startswith("```json"):
                        content = content[7:]
                    elif content.startswith("```"):
                        content = content[3:]
            
            # Find JSON object
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                data = json.loads(json_str)
                
                action_id = data.get("action_id")
                
                if isinstance(action_id, int) and 0 <= action_id < 61:
                    # Verify action is legal
                    if legal_mask[action_id] > 0:
                        return action_id
            
            return None
            
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return None
    
    def act(
        self,
        obs: np.ndarray,
        legal_mask: np.ndarray,
        hand_cards: Optional[list] = None,
        top_card: Optional[str] = None,
        active_color: Optional[str] = None
    ) -> int:
        """
        Select action using LLM.
        
        Falls back to random legal action on failure.
        """
        prompt = self._format_state_prompt(
            obs, legal_mask, hand_cards, top_card, active_color
        )
        
        # Try to get valid action from LLM
        for attempt in range(self.max_retries + 1):
            try:
                response = self._call_llm(prompt)
                action = self._parse_action(response, legal_mask)
                
                if action is not None:
                    return action
                
                # Add correction message for retry
                if attempt < self.max_retries:
                    prompt = prompt + "\n\nYour previous response was invalid. Respond with ONLY: {\"action_id\": <number>}"
                    
            except Exception as e:
                print(f"LLM API error (attempt {attempt + 1}): {e}")
        
        # Fallback to random legal action
        legal_actions = np.where(legal_mask > 0)[0]
        if len(legal_actions) == 0:
            return 60  # Draw
        
        return int(np.random.choice(legal_actions))
    
    @property
    def name(self) -> str:
        return f"LLM ({self._model_name})"
    
    def close(self):
        """Close HTTP client."""
        self.client.close()


def get_llm_opponent(
    model_key: str = "gemini_flash",
    api_key: Optional[str] = None
) -> Optional[LLMOpenRouterOpponent]:
    """
    Get LLM opponent by model key.
    
    Args:
        model_key: Key from LLM_MODELS ('gemini_flash', 'gpt_5', 'opus')
        api_key: Optional API key override
        
    Returns:
        LLMOpenRouterOpponent or None if API key not available
    """
    if model_key not in LLM_MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(LLM_MODELS.keys())}")
    
    model_slug = LLM_MODELS[model_key]["slug"]
    
    try:
        return LLMOpenRouterOpponent(model_slug=model_slug, api_key=api_key)
    except ValueError:
        return None


__all__ = [
    'LLMOpenRouterOpponent', 'get_llm_opponent', 
    'LLM_MODELS', 'get_available_models', 'HTTPX_AVAILABLE'
]
