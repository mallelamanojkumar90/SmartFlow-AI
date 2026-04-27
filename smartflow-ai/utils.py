"""Utility helpers for training and visualization."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def moving_average(values: list[float], window: int = 20) -> list[float]:
    if not values:
        return []
    window = max(1, window)
    averages: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averages.append(float(np.mean(values[start : index + 1])))
    return averages


def save_training_log(path: str | Path, rewards: list[float], avg_waits: list[float]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "average_waiting_time"])
        for episode, (reward, wait) in enumerate(zip(rewards, avg_waits), start=1):
            writer.writerow([episode, reward, wait])
