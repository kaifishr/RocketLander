"""Genetic optimization class."""
import copy

import numpy

from src.utils.config import Config
from src.environment import Environment


class GeneticOptimizer:
    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""

        self.boosters = environment.boosters
        self.mutation_prob = config.optimizer.mutation_probability
        self.mutation_rate = config.optimizer.mutation_rate

        # Index of currently fittest agent (booster)
        self.idx_best = 0

        # Maximum reward for current epoch.
        self.reward = 0

    def step(self) -> None:
        """Runs single genetic optimization step."""

        # Select fittest agent (booster) based on reward.
        self._select()

        # Reproduce and mutate weights of best agent.
        self._mutate()

    def _select(self) -> None:
        """Selects best agent for reproduction."""

        # Fetch rewards of each booster.
        rewards = [booster.reward for booster in self.boosters]

        # Select and store reward of best booster.
        self.idx_best = numpy.argmax(rewards)
        self.reward = rewards[self.idx_best]

    def _mutate(self) -> None:
        """Mutates network parameters of each booster."""

        # Get neural network of fittest booster to reproduce.
        model = self.boosters[self.idx_best].model

        # Pass best model to other boosters and mutate their weights.
        for booster in self.boosters:
            # Assign best model to each booster and mutate weights.
            booster.model = copy.deepcopy(model)
            self._mutate_weights(booster.model)
            # booster.model.mutate_weights()

    def _mutate_weights(self, model: object) -> None:
        """Mutates the network's weights."""

        for weight, bias in model.parameters:

            mask = numpy.random.random(size=weight.shape) < self.mutation_prob
            mutation = self.mutation_rate * numpy.random.normal(size=weight.shape)
            weight += mask * mutation

            mask = numpy.random.random(size=bias.shape) < self.mutation_prob
            mutation = self.mutation_rate * numpy.random.normal(size=bias.shape)
            bias += mask * mutation
