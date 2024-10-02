import numpy as np
from abc import ABC, abstractmethod
from policy.abstract_policy import AbstractPolicy


class AbstractEvaluator(ABC):

    @abstractmethod
    def evaluate(self, policy: AbstractPolicy, num_episodes: int) -> np.ndarray:
        pass
