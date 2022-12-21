"""Optimizer base class."""
from src.utils.config import Config
from src.environment import Environment


class Optimizer:
    """Optimizer base class.

    Attributes:
        ...
    """
    num_engines = 3
    num_states = 6

    def __init__(self) -> None:
        """Initializes optimizer base class."""

        # Parameters to be visualized with Tensorboard. 
        self.stats = {
            "reward": None, 
            "loss": None, 
            "epsilon": None,
            "temperature": None,
        }

        # Scalars
        self.iteration = 0

    def step(self) -> None:
        """Runs single optimization step."""
        raise NotImplementedError
