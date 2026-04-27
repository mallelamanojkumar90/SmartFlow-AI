"""Custom traffic intersection environment for SmartFlow AI.

The environment intentionally follows a small Gym-like API without depending on
Gymnasium. It models a single three-lane intersection where one lane receives a
green signal at a time and a DQN agent learns to minimize queues and waiting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrafficConfig:
    """Configuration values controlling the traffic simulation."""

    num_lanes: int = 3
    max_steps: int = 200
    arrival_rate: float = 2.0
    departure_rate: int = 3
    max_queue: int = 50
    seed: int | None = None


class TrafficEnv:
    """Three-lane traffic signal optimization environment.

    State:
        [lane1_count, lane2_count, lane3_count, current_signal]

    Actions:
        0: Lane 1 green
        1: Lane 2 green
        2: Lane 3 green
        3: Extend current signal
    """

    def __init__(self, config: TrafficConfig | None = None) -> None:
        self.config = config or TrafficConfig()
        self.num_lanes = self.config.num_lanes
        self.action_size = self.num_lanes + 1
        self.state_size = self.num_lanes + 1
        self.rng = np.random.default_rng(self.config.seed)

        self.queues = np.zeros(self.num_lanes, dtype=np.float32)
        self.waiting_times = np.zeros(self.num_lanes, dtype=np.float32)
        self.current_signal = 0
        self.current_step = 0

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state."""
        self.queues = self.rng.integers(0, 6, size=self.num_lanes).astype(np.float32)
        self.waiting_times = np.zeros(self.num_lanes, dtype=np.float32)
        self.current_signal = int(self.rng.integers(0, self.num_lanes))
        self.current_step = 0
        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float | int]]:
        """Advance the simulation by one time step."""
        if action < 0 or action >= self.action_size:
            raise ValueError(f"Invalid action {action}; expected 0-{self.action_size - 1}")

        if action != self.num_lanes:
            self.current_signal = int(action)

        arrivals = self.rng.poisson(self.config.arrival_rate, size=self.num_lanes)
        self.queues = np.minimum(self.queues + arrivals, self.config.max_queue)

        departed = min(self.config.departure_rate, int(self.queues[self.current_signal]))
        self.queues[self.current_signal] -= departed

        # Cars remaining in each queue accumulate one unit of waiting per step.
        self.waiting_times += self.queues
        self.current_step += 1

        total_queue = float(np.sum(self.queues))
        total_waiting_time = float(np.sum(self.waiting_times))
        reward = -(total_waiting_time + total_queue)
        done = self.current_step >= self.config.max_steps

        info = {
            "total_queue": total_queue,
            "total_waiting_time": total_waiting_time,
            "average_waiting_time": total_waiting_time / max(1, self.current_step),
            "current_signal": self.current_signal,
            "departed": departed,
        }
        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        state = np.append(self.queues, self.current_signal).astype(np.float32)
        state[:-1] = state[:-1] / float(self.config.max_queue)
        state[-1] = state[-1] / float(max(1, self.num_lanes - 1))
        return state

    def render_data(self) -> dict[str, list[float] | int]:
        """Return display-friendly values for dashboards and debugging."""
        return {
            "queues": self.queues.astype(int).tolist(),
            "waiting_times": self.waiting_times.round(2).tolist(),
            "current_signal": self.current_signal,
            "step": self.current_step,
        }
