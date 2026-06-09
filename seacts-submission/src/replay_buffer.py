from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_valid_actions: np.ndarray
    done: bool
    n_steps: int = 1
    context_index: int = 0


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = Random(seed)

    def append(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self._items):
            raise ValueError("Cannot sample more transitions than are available.")
        return self._rng.sample(list(self._items), batch_size)

    def __len__(self) -> int:
        return len(self._items)
