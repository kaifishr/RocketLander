"""Genetic optimization class."""
import copy
import math
import random

import numpy as np

from src.utils.config import Config
from src.environment import Environment


class SimulatedAnnealing:
    """Optimizer class for simulated annealing."""

    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""

        self.boosters = environment.boosters
        self.config = config

        self.model_old = copy.deepcopy(self.boosters[0].model)

        # Scheduler
        self.gamma = 0.00001
        self.temp_initial = 1.0
        # self.temp_final = 0.01
        # self.prob_perturbation = 1.0

        self.iteration = 0
        self.temp = self.temp_initial

        # Maximum reward for current epoch.
        self.reward = 0.0
        self.reward_old = 0.0
        # self.reward_best = 0.0

        # Backup network
        # self.model_tmp = None

    def step(self) -> None:
        """Runs single simulated annealing step."""

        # Get reward of booster.
        self.reward = self.boosters[0].reward

        # delta_reward = self.reward - self.reward_old

        if self.reward > self.reward_old:
            # Save network if current reward is higher
            # Use current weights as they yield a higher reward
            pass
        elif math.exp((self.reward - self.reward_old) / self.temp) > random.random():
            # Keep current weights even though the reward is lower
            pass
        else:
            self.boosters[0].model = copy.deepcopy(self.model_old)

        # Save current model and reward
        self.model_old = copy.deepcopy(self.boosters[0].model)  # TODO: Just copy the parameters
        self.reward_old = self.reward

        # Reduce temperature according to scheduler
        self._scheduler()

        # Perturb weights for next iteration.
        self._perturb()

        self.iteration += 1

        # print(f"{self.temp = }")

    def _scheduler(self) -> None:
        self.temp = self.temp_initial * math.exp(-self.gamma * self.iteration)

    def _perturb(self) -> None:
        """Perturbs network parameters of each booster.
        TODO: Move perturbation to optimizer
        """
        # Perturb parameters.
        self.boosters[0].model.mutate_weights()
