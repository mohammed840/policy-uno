"""
PyTorch Q-Network for Uno RL.

Network Architecture (matching reference PDF):
    Input (420) → Dense (512, ReLU) → Dense (512, ReLU) → Output (61)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class QNetwork(nn.Module):
    """
    Q-Network MLP for Uno.
    
    Architecture: 420 → 512 → 512 → 61
    """
    
    def __init__(
        self,
        state_size: int = 420,
        action_size: int = 61,
        hidden_size: int = 512
    ):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: State tensor of shape (batch, 420)
            
        Returns:
            Q-values tensor of shape (batch, 61)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_q_values(
        self,
        state: np.ndarray,
        device: torch.device
    ) -> np.ndarray:
        """
        Get Q-values for a state.
        
        Args:
            state: State array of shape (420,) or (batch, 420)
            device: Torch device
            
        Returns:
            Q-values array of shape (61,) or (batch, 61)
        """
        was_1d = state.ndim == 1
        if was_1d:
            state = state[np.newaxis, :]
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device)
            q_values = self.forward(state_t).cpu().numpy()
        
        if was_1d:
            return q_values[0]
        return q_values
    
    def select_action(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray,
        epsilon: float,
        device: torch.device
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: State array of shape (420,)
            legal_mask: Binary legal action mask of shape (61,)
            epsilon: Exploration probability
            device: Torch device
            
        Returns:
            Selected action index
        """
        # Get legal action indices
        legal_actions = np.where(legal_mask > 0)[0]
        
        if len(legal_actions) == 0:
            # Fallback to draw if no legal actions
            return 60
        
        # Epsilon-greedy
        if np.random.random() < epsilon:
            # Random legal action
            return int(np.random.choice(legal_actions))
        
        # Greedy: select best legal action
        q_values = self.get_q_values(state, device)
        
        # Mask illegal actions with -inf
        masked_q = q_values.copy()
        masked_q[legal_mask == 0] = float('-inf')
        
        return int(np.argmax(masked_q))
    
    def select_greedy_action(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray,
        device: torch.device
    ) -> int:
        """
        Select greedy action (no exploration).
        
        Args:
            state: State array of shape (420,)
            legal_mask: Binary legal action mask of shape (61,)
            device: Torch device
            
        Returns:
            Selected action index
        """
        return self.select_action(state, legal_mask, epsilon=0.0, device=device)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        legal_mask: np.ndarray
    ):
        """Add experience to buffer."""
        experience = (state, action, reward, next_state, done, legal_mask)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> tuple:
        """Sample a batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones, legal_masks = [], [], [], [], [], []
        
        for idx in indices:
            s, a, r, ns, d, lm = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            legal_masks.append(lm)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(legal_masks, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


def create_target_network(q_network: QNetwork) -> QNetwork:
    """Create a target network as a copy of the Q-network."""
    target = QNetwork(
        state_size=q_network.state_size,
        action_size=q_network.action_size,
        hidden_size=q_network.hidden_size
    )
    target.load_state_dict(q_network.state_dict())
    return target


def update_target_network(q_network: QNetwork, target_network: QNetwork, tau: float = 1.0):
    """
    Soft update target network parameters.
    
    θ_target = τ * θ_local + (1 - τ) * θ_target
    """
    for target_param, local_param in zip(target_network.parameters(), q_network.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def save_model(model: QNetwork, path: str, optimizer: Optional[torch.optim.Optimizer] = None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'state_size': model.state_size,
        'action_size': model.action_size,
        'hidden_size': model.hidden_size,
    }
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, path)


def load_model(path: str, device: torch.device) -> QNetwork:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    model = QNetwork(
        state_size=checkpoint.get('state_size', 420),
        action_size=checkpoint.get('action_size', 61),
        hidden_size=checkpoint.get('hidden_size', 512)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


__all__ = [
    'QNetwork', 'ReplayBuffer',
    'create_target_network', 'update_target_network',
    'save_model', 'load_model'
]
