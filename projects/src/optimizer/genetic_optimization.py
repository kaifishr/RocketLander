"""Genetic optimization class.

Implements genetic optimization inspired by biological evolution. 

"""
import copy
import numpy

from projects.src.optimizer import Optimizer

from src.utils.config import Config
from src.environment import Environment


class GeneticOptimizer(Optimizer):
    """Genetic optimizer class.

    Attr:
        boosters:
        mutation_prob
        mutation_rate
    """
    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""
        super().__init__()

        self.boosters = environment.boosters
        self.mutation_prob = config.optimizer.mutation_probability
        self.mutation_rate = config.optimizer.mutation_rate

        # Index of currently fittest agent.
        self.idx_best = 0

    def _select(self) -> None:
        """Selects best agent for reproduction."""

        # Fetch rewards of each booster.
        # Cumulative rewards:
        # rewards = numpy.array([sum(booster.rewards) for booster in self.boosters])  
        # Use final reward
        rewards = numpy.array([booster.rewards[-1] for booster in self.boosters]) 
        self.idx_best = rewards.argmax()
        self.stats["reward"] = rewards[self.idx_best]

    def _mutate(self) -> None:
        """Mutates network parameters of each agent."""

        # Get neural network of fittest booster to reproduce.
        model = self.boosters[self.idx_best].model

        # Pass best model to other boosters and mutate their weights.
        for booster in self.boosters:
            # Assign best model to each booster and mutate weights.
            booster.model = copy.deepcopy(model)  # TODO: Why not just weights?
            self._mutate_weights(booster.model)

    def _mutate_weights(self, model: object) -> None:
        """Mutates the network's weights.

        Args:
            model: Neural network model.
        """
        for weight, bias in model.parameters:

            mask = numpy.random.random(size=weight.shape) < self.mutation_prob
            mutation = self.mutation_rate * numpy.random.normal(size=weight.shape)
            weight += mask * mutation

            mask = numpy.random.random(size=bias.shape) < self.mutation_prob
            mutation = self.mutation_rate * numpy.random.normal(size=bias.shape)
            bias += mask * mutation

    def step(self) -> None:
        """Runs single genetic optimization step."""

        # Select fittest agent (booster) based on reward.
        self._select()

        # Reproduce and mutate weights of best agent.
        self._mutate()