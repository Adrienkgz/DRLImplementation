from dataclasses import dataclass
import numpy as np
from collections import deque
@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    logprobs: float
    done: bool
    td_error: float = 0.0
    advantage: float = 0.0
    