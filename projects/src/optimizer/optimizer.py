"""Optimizer base class."""
import numpy

from src.utils.config import Config
from src.environment import Environment


class Optimizer:
    """Optimizer base class.

    Attributes:
        ...
    """

    num_engines = 3
    num_states = 6

    def __init__(self, environment: Environment = None, config: Config = None) -> None:
        """Initializes optimizer base class."""
        # self.boosters = environment.boosters
        # self.config = config

        # Parameters to be visualized with Tensorboard.
        self.stats = {
            "reward": None,
            "loss": None,
            "epsilon": None,
            "temperature": None,
        }

        # Scalars
        self.iteration = 0
        self.idx_best = 0

    def _gather_rewards(self, reduction: str = "sum") -> numpy.ndarray:
        """Gathers rewards of all agents.

        Args:
            reduction: Specifies the reduction applied to the rewards:
            `sum` or `last`. `last`: only last reward will be considered.

        Returns:
            Numpy array of reduced rewards.
        """
        if reduction == "sum":
            rewards = [sum(booster.rewards) for booster in self.boosters]
        elif reduction == "last":
            rewards = [booster.rewards[-1] for booster in self.boosters]
        else:
            raise NotImplementedError(f"Reduction '{reduction}' not implemented.")
        return numpy.array(rewards)

    def step(self) -> None:
        """Runs single optimization step."""
        raise NotImplementedError
