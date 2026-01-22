"""Opponent adapters package."""

from .base import Opponent, HumanOpponent
from .random_bot import RandomBot
from .rl_torch import RLTorchOpponent, get_default_torch_opponent
from .rl_coreml import RLCoreMLOpponent, RLOpponent, get_rl_opponent, COREML_AVAILABLE
from .llm_openrouter import (
    LLMOpenRouterOpponent, get_llm_opponent,
    LLM_MODELS, get_available_models, HTTPX_AVAILABLE
)

__all__ = [
    'Opponent', 'HumanOpponent',
    'RandomBot',
    'RLTorchOpponent', 'get_default_torch_opponent',
    'RLCoreMLOpponent', 'RLOpponent', 'get_rl_opponent', 'COREML_AVAILABLE',
    'LLMOpenRouterOpponent', 'get_llm_opponent',
    'LLM_MODELS', 'get_available_models', 'HTTPX_AVAILABLE'
]
