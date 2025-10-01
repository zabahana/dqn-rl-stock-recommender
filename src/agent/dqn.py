from dataclasses import dataclass
from typing import Tuple, List, Dict
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class AdvancedDQN(nn.Module):
    """Enhanced DQN with deeper architecture, hyperparameter optimization, and advanced techniques."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, num_heads: int = 8, 
                 num_blocks: int = 6, dropout_rate: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # Enhanced input projection with multiple layers
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Multi-head self-attention for temporal dependencies
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout_rate)
        
        # Enhanced residual blocks with deeper architecture
        self.residual_blocks = nn.ModuleList([
            EnhancedResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        
        # Cross-attention layer for feature interaction
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads // 2, batch_first=True, dropout=dropout_rate)
        
        # Enhanced dueling architecture with deeper networks
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Enhanced output layer with optimization
        self.output_norm = nn.LayerNorm(action_dim)
        self.output_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Reshape for attention (batch, seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Enhanced input projection
        x = self.input_proj(x)
        
        # Self-attention for temporal dependencies
        attn_out, attn_weights = self.attention(x, x, x)
        x = x + attn_out  # Residual connection
        
        # Enhanced residual blocks with gating
        for block in self.residual_blocks:
            x = block(x)
        
        # Cross-attention for feature interaction
        cross_attn_out, _ = self.cross_attention(x, x, x)
        x = x + cross_attn_out
        
        # Global average pooling with attention weights
        if x.dim() == 3:
            # Weighted pooling based on attention
            pooled_x = torch.sum(x * attn_weights.mean(dim=1, keepdim=True), dim=1)
        else:
            pooled_x = x
        
        # Enhanced dueling architecture
        value = self.value_stream(pooled_x)
        advantage = self.advantage_stream(pooled_x)
        
        # Combine value and advantage with improved normalization
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Enhanced output processing
        q_values = self.output_norm(q_values)
        q_values = self.output_dropout(q_values)
        
        return q_values


class EnhancedResidualBlock(nn.Module):
    """Enhanced residual block with deeper architecture and optimization layers."""
    
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Gating mechanism for adaptive feature selection
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.layers(x)
        gate_weights = self.gate(x)
        return F.relu(residual + out * gate_weights)


class ResidualBlock(nn.Module):
    """Original residual block for backward compatibility."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.layers(x))


@dataclass
class AdvancedDQNConfig:
    state_dim: int
    action_dim: int
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 100000
    target_update_freq: int = 100
    learning_starts: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    gradient_clip: float = 1.0
    # Enhanced hyperparameters
    hidden_dim: int = 256
    num_heads: int = 8
    num_blocks: int = 6
    dropout_rate: float = 0.1
    weight_decay: float = 1e-5
    tau: float = 0.005  # Soft target update
    prioritized_replay: bool = True
    double_dqn: bool = True


class AdvancedDQNAgent:
    """Advanced DQN agent with experience replay, target network, and optimization techniques."""
    
    def __init__(self, cfg: AdvancedDQNConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_net = AdvancedDQN(cfg.state_dim, cfg.action_dim).to(self.device)
        self.target_net = AdvancedDQN(cfg.state_dim, cfg.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=cfg.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000)
        
        # Experience replay buffer
        self.buffer = deque(maxlen=cfg.buffer_size)
        
        # Training statistics
        self.step_count = 0
        self.episode_rewards = []
        self.losses = []
        self.q_values_history = []
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection with annealing."""
        if training:
            epsilon = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
                     np.exp(-self.step_count / self.cfg.epsilon_decay)
        else:
            epsilon = 0.0
            
        if random.random() < epsilon:
            return random.randint(0, self.cfg.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            return int(q_values.argmax().item())
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def learn(self) -> Dict[str, float]:
        """Train the network on a batch of experiences."""
        if len(self.buffer) < self.cfg.learning_starts:
            return {}
        
        # Sample batch
        batch = random.sample(self.buffer, self.cfg.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values with double DQN
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.cfg.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update target network
        if self.step_count % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.step_count += 1
        
        # Store statistics
        self.losses.append(loss.item())
        self.q_values_history.append(current_q_values.mean().item())
        
        return {
            'loss': loss.item(),
            'q_value': current_q_values.mean().item(),
            'epsilon': self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * 
                      np.exp(-self.step_count / self.cfg.epsilon_decay)
        }
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics for visualization."""
        return {
            'losses': self.losses,
            'q_values': self.q_values_history,
            'episode_rewards': self.episode_rewards
        }


