"""Core ML inference opponent."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Opponent


# Check Core ML availability
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


class RLCoreMLOpponent(Opponent):
    """Opponent using Core ML model for inference."""
    
    def __init__(self, model_path: str):
        """
        Initialize Core ML opponent.
        
        Args:
            model_path: Path to Core ML model (.mlpackage)
        """
        if not COREML_AVAILABLE:
            raise ImportError(
                "coremltools is not installed. "
                "Install with: pip install coremltools"
            )
        
        self.model = ct.models.MLModel(model_path)
        self._model_path = model_path
    
    def act(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        """Select action using greedy policy with Core ML inference."""
        # Prepare input
        state = obs.astype(np.float32).reshape(1, -1)
        
        # Run inference
        predictions = self.model.predict({"state": state})
        q_values = predictions["q_values"].flatten()
        
        # Mask illegal actions
        masked_q = q_values.copy()
        masked_q[legal_mask == 0] = float('-inf')
        
        return int(np.argmax(masked_q))
    
    @property
    def name(self) -> str:
        return "RL Agent (Core ML)"


class RLOpponent(Opponent):
    """
    RL Opponent with automatic fallback.
    
    Uses Core ML if available, otherwise falls back to PyTorch.
    """
    
    def __init__(self, prefer_coreml: bool = True, device: str = 'auto'):
        """
        Initialize RL opponent with fallback support.
        
        Args:
            prefer_coreml: Try Core ML first if available
            device: PyTorch device for fallback
        """
        self._opponent = None
        self._backend = None
        
        project_root = Path(__file__).parent.parent
        coreml_path = project_root / 'models' / 'best_qnet.mlpackage'
        torch_path = project_root / 'models' / 'best_qnet.pt'
        
        # Try Core ML first
        if prefer_coreml and COREML_AVAILABLE and coreml_path.exists():
            try:
                self._opponent = RLCoreMLOpponent(str(coreml_path))
                self._backend = 'coreml'
                return
            except Exception as e:
                print(f"Core ML loading failed: {e}")
        
        # Fall back to PyTorch
        if torch_path.exists():
            from .rl_torch import RLTorchOpponent
            self._opponent = RLTorchOpponent(str(torch_path), device)
            self._backend = 'pytorch'
            return
        
        raise FileNotFoundError(
            "No trained model found. "
            f"Expected at {coreml_path} or {torch_path}. "
            "Train a model first with: python -m rl.dqn_train"
        )
    
    def act(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        """Select action using the loaded model."""
        return self._opponent.act(obs, legal_mask)
    
    def act_with_explanation(self, obs: np.ndarray, legal_mask: np.ndarray) -> tuple:
        """Select action with explanation if supported."""
        if hasattr(self._opponent, 'act_with_explanation'):
            return self._opponent.act_with_explanation(obs, legal_mask)
        else:
            return self._opponent.act(obs, legal_mask), None
    
    @property
    def name(self) -> str:
        backend_name = "Core ML" if self._backend == 'coreml' else "PyTorch"
        return f"RL Agent ({backend_name})"
    
    @property
    def backend(self) -> str:
        return self._backend


def get_rl_opponent(prefer_coreml: bool = True, device: str = 'auto') -> Optional[RLOpponent]:
    """Get RL opponent with automatic fallback."""
    try:
        return RLOpponent(prefer_coreml=prefer_coreml, device=device)
    except FileNotFoundError:
        return None


__all__ = ['RLCoreMLOpponent', 'RLOpponent', 'get_rl_opponent', 'COREML_AVAILABLE']
