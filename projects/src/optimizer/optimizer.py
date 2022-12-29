"""Optimizer base class."""
import numpy

from src.utils.config import Config
from src.environment import Environment


class Optimizer:
    """Optimizer base class.

    Attributes:
        stats: Dictionary holding stats.
    """

    num_engines = 3
    num_states = 6

    def __init__(self, environment: Environment = None, config: Config = None) -> None:
        """Initializes optimizer base class."""
        self.boosters = None
        self.config = config

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
            `mean`, `sum` or `last`. `last`: only last reward will be considered.

        Returns:
            Numpy array of reduced rewards.
        """
        if reduction == "sum":
            rewards = [numpy.sum(booster.rewards) for booster in self.boosters]
        elif reduction == "mean":
            rewards = [numpy.mean(booster.rewards) for booster in self.boosters]
        elif reduction == "last":
            rewards = [booster.rewards[-1] for booster in self.boosters]
        else:
            raise NotImplementedError(f"Reduction '{reduction}' not implemented.")
        return numpy.array(rewards)

    @staticmethod
    def _perturb_weights(model: object, prob: float, rate: float) -> None:
        """Perturbs the network's weights.

        Args:
            model: Neural network model.
            prob: Probability for perturbation.
            rate: Rate of perturbation.
        """
        for weight, bias in model.parameters:

            mask = numpy.random.random(size=weight.shape) < prob
            mutation = rate * numpy.random.normal(size=weight.shape)
            weight += mask * mutation

            mask = numpy.random.random(size=bias.shape) < prob
            mutation = rate * numpy.random.normal(size=bias.shape)
            bias += mask * mutation

    def step(self) -> None:
        """Runs single optimization step."""
        raise NotImplementedError
