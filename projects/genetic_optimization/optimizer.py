"""Genetic optimization class."""
import copy

import numpy as np

from src.utils.config import Config
from src.environment import Environment


class GeneticOptimizer:

    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""

        self.boosters = environment.boosters
        self.config = config

        # Index of currently fittest agent (booster)
        self.idx_best = 0

        # Maximum reward for current epoch.
        self.reward = 0

    def step(self) -> None:
        """Runs single genetic optimization step."""

        # Select fittest agent based on distance traveled.
        self._select()

        # Reproduce and mutate weights of best agent.
        self._mutate()

    def _select(self) -> None:
        """Selects best agent for reproduction."""

        # Fetch rewards of each booster.
        rewards = [booster.reward for booster in self.boosters]

        # Select and store reward of best booster.
        self.idx_best = np.argmax(rewards)
        self.reward = rewards[self.idx_best]

    def _mutate(self) -> None:
        """Mutates network parameters of each booster."""

        # Get neural network of fittest booster to reproduce.
        model = self.boosters[self.idx_best].model

        # Pass best model to other boosters and mutate their weights.
        for booster in self.boosters:

            # Assign best model to each booster and mutate weights.
            booster.model = copy.deepcopy(model)
            booster.model.mutate_weights()
