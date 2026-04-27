"""Train the SmartFlow AI DQN agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from env.traffic_env import TrafficConfig, TrafficEnv
from model.dqn_model import AgentConfig, DQNAgent
from utils import ensure_dir, save_training_log, set_seed


def train(
    episodes: int = 500,
    max_steps: int = 200,
    seed: int = 42,
    model_path: str = "model/dqn_traffic.pth",
) -> tuple[list[float], list[float]]:
    set_seed(seed)
    env = TrafficEnv(TrafficConfig(max_steps=max_steps, seed=seed))
    agent = DQNAgent(env.state_size, env.action_size, AgentConfig())

    rewards_history: list[float] = []
    avg_wait_history: list[float] = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        final_avg_wait = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            episode_reward += reward
            final_avg_wait = float(info["average_waiting_time"])

            if done:
                break

        if episode % agent.config.target_update_frequency == 0:
            agent.update_target_network()

        rewards_history.append(episode_reward)
        avg_wait_history.append(final_avg_wait)

        if episode == 1 or episode % 10 == 0:
            print(
                f"Episode {episode:4d}/{episodes} | "
                f"Reward: {episode_reward:10.2f} | "
                f"Avg wait: {final_avg_wait:8.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    output_path = Path(model_path)
    ensure_dir(output_path.parent)
    agent.save(str(output_path))
    save_training_log(output_path.parent / "training_log.csv", rewards_history, avg_wait_history)
    print(f"Saved trained model to {output_path}")
    print(f"Saved training log to {output_path.parent / 'training_log.csv'}")
    return rewards_history, avg_wait_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmartFlow AI DQN traffic controller.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Steps per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model-path", type=str, default="model/dqn_traffic.pth", help="Model output path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.episodes, args.max_steps, args.seed, args.model_path)
