"""Streamlit dashboard for SmartFlow AI."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from env.traffic_env import TrafficConfig, TrafficEnv
from model.dqn_model import AgentConfig, DQNAgent
from utils import moving_average


MODEL_PATH = Path("model/dqn_traffic.pth")


@st.cache_resource
def load_agent() -> DQNAgent:
    env = TrafficEnv(TrafficConfig())
    agent = DQNAgent(env.state_size, env.action_size, AgentConfig(epsilon_start=0.0, epsilon_min=0.0))
    if MODEL_PATH.exists():
        agent.load(str(MODEL_PATH))
        agent.epsilon = 0.0
    return agent


def initialize_session() -> None:
    if "env" not in st.session_state:
        st.session_state.env = TrafficEnv(TrafficConfig(seed=7))
        st.session_state.state = st.session_state.env.reset()
        st.session_state.running = False
        st.session_state.waiting_history = []
        st.session_state.reward_history = []
        st.session_state.queue_history = []


def reset_simulation() -> None:
    st.session_state.env = TrafficEnv(TrafficConfig(seed=7))
    st.session_state.state = st.session_state.env.reset()
    st.session_state.running = False
    st.session_state.waiting_history = []
    st.session_state.reward_history = []
    st.session_state.queue_history = []


def plot_line(values: list[float], title: str, ylabel: str, smooth: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3))
    series = moving_average(values, 10) if smooth else values
    ax.plot(series, color="#2563eb", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="SmartFlow AI", layout="wide")
    initialize_session()
    agent = load_agent()

    st.title("SmartFlow AI: Real-Time Traffic Signal Optimization")
    if not MODEL_PATH.exists():
        st.warning("No trained model found at model/dqn_traffic.pth. Run `python train.py` first for learned behavior.")

    controls = st.columns([1, 1, 1, 4])
    if controls[0].button("Start", use_container_width=True):
        st.session_state.running = True
    if controls[1].button("Stop", use_container_width=True):
        st.session_state.running = False
    if controls[2].button("Reset", use_container_width=True):
        reset_simulation()

    speed = st.sidebar.slider("Simulation delay (seconds)", 0.05, 1.0, 0.25, 0.05)
    max_live_steps = st.sidebar.slider("Live steps per refresh", 1, 25, 5)

    status_placeholder = st.empty()
    metric_placeholder = st.container()
    chart_placeholder = st.container()

    if st.session_state.running:
        for _ in range(max_live_steps):
            action = agent.select_action(st.session_state.state, training=False)
            next_state, reward, done, info = st.session_state.env.step(action)
            st.session_state.state = next_state
            st.session_state.waiting_history.append(float(info["average_waiting_time"]))
            st.session_state.reward_history.append(float(reward))
            st.session_state.queue_history.append(float(info["total_queue"]))
            if done:
                st.session_state.state = st.session_state.env.reset()
                break
            time.sleep(speed)
        st.rerun()

    data = st.session_state.env.render_data()
    active_lane = int(data["current_signal"]) + 1
    status_placeholder.subheader(f"Active Green Signal: Lane {active_lane}")

    with metric_placeholder:
        cols = st.columns(3)
        for index, queue in enumerate(data["queues"], start=1):
            cols[index - 1].metric(f"Lane {index} Queue", int(queue))

        wait_cols = st.columns(3)
        for index, wait in enumerate(data["waiting_times"], start=1):
            wait_cols[index - 1].metric(f"Lane {index} Waiting Time", f"{wait:.0f}")

    with chart_placeholder:
        left, right = st.columns(2)
        with left:
            st.pyplot(plot_line(st.session_state.waiting_history, "Average Waiting Time Over Time", "Avg waiting"))
        with right:
            st.pyplot(plot_line(st.session_state.reward_history, "Reward Trend", "Reward", smooth=True))

        st.bar_chart(
            {"Queue Length": {f"Lane {idx + 1}": value for idx, value in enumerate(data["queues"])}},
            height=250,
        )


if __name__ == "__main__":
    main()
