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

    def _select(self) -> None:
        """Selects best agent for reproduction."""
        rewards = self._gather_rewards(reduction="sum")
        self.idx_best = rewards.argmax()
        self.stats["reward"] = rewards[self.idx_best]

    def _mutate(self) -> None:
        """Mutates network parameters of each agent."""

        # Get neural network of fittest booster to reproduce.
        parameters = self.boosters[self.idx_best].model.parameters

        # Pass best model to other boosters and mutate their weights.
        for booster in self.boosters:
            # Assign parameters of best model to each booster and mutate weights.
            booster.model.parameters = copy.deepcopy(parameters)
            self._perturb_weights(booster.model, self.mutation_prob, self.mutation_rate)

    def step(self) -> None:
        """Runs single genetic optimization step."""

        # Select fittest agent based on reward.
        self._select()

        # Reproduce and mutate weights of best agent.
        self._mutate()
