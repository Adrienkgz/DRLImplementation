from dataclasses import dataclass
import numpy as np

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    advantage: float