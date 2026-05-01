"""Deep Q-Network model and agent implementation."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim


class DQN(nn.Module):
    """Feed-forward Q-network with two ReLU hidden layers."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class AgentConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 64
    replay_buffer_size: int = 50_000
    target_update_frequency: int = 10


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards.astype(np.float32), next_states, dones.astype(np.float32)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN traffic signal control agent."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: AgentConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or AgentConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.update_target_network()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(self.config.replay_buffer_size)
        self.epsilon = self.config.epsilon_start

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose an action using epsilon-greedy exploration."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """Run one optimization step from replay memory."""
        if len(self.memory) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.config.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.epsilon > self.config.epsilon_min:
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

        return float(loss.item())

    def update_target_network(self) -> None:
        """Synchronize target network weights with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.policy_net.state_dict(),
                "epsilon": self.epsilon,
                "state_size": self.state_size,
                "action_size": self.action_size,
            },
            path,
        )

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"Invalid model checkpoint: {path}")

        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.update_target_network()
        self.epsilon = float(checkpoint.get("epsilon", self.config.epsilon_min))
