"""Optimization class for simulated annealing."""
import copy
import math
import numpy
import random

from src.utils.config import Config
from src.environment import Environment


class SimulatedAnnealing:
    """Optimizer class for simulated annealing."""

    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""

        self.booster = environment.boosters[0]

        self.config = config.optimizer

        self.perturbation_probability_initial = self.config.perturbation_probability_initial
        self.perturbation_probability_final = self.config.perturbation_probability_final
        self.perturbation_rate = self.config.perturbation_rate
        self.temp_initial = self.config.temp_initial
        self.temp_final = self.config.temp_final
        self.temp = self.temp_initial

        num_epochs = config.trainer.num_generations
        self.gamma = (1.0 / num_epochs) * math.log(self.temp_initial / self.temp_final)

        self.model_old = None
        self.iteration = 0

        # Maximum reward of current epoch.
        self.reward = 0.0
        self.reward_old = 0.0

    def step(self) -> None:
        """Runs single simulated annealing step."""

        # Get reward of booster.
        self.reward = self.booster.reward

        delta_reward = self.reward - self.reward_old
        if delta_reward > 0:
            # Save network if current reward is higher
            self.model_old = copy.deepcopy(self.booster.model)
            self.reward_old = self.reward
        elif math.exp(delta_reward / self.temp) > random.random():
            # Keep current weights even though the reward is lower
            self.model_old = copy.deepcopy(self.booster.model)
            self.reward_old = self.reward
        else:
            # Do not accept current state. Return to previous state.
            self.booster.model = copy.deepcopy(self.model_old)

        # Reduce temperature according to scheduler
        self._scheduler()

        # Perturb weights for next iteration.
        self._perturb()

        self.iteration += 1

    def _scheduler(self) -> None:
        """Decreases temperature according to exponential decay."""
        self.temp = self.temp_initial * math.exp(-self.gamma * self.iteration)
        if self.temp < self.temp_final:
            self.temp = self.temp_final

    def _perturb(self) -> None:
        """Perturbs network weights."""

        perturbation_probability = (self.temp / self.temp_initial) + self.temp_final

        for weight, bias in self.booster.model.parameters:

            mask = numpy.random.random(size=weight.shape) < perturbation_probability
            mutation = self.perturbation_rate * numpy.random.normal(size=weight.shape)
            weight += mask * mutation

            mask = numpy.random.random(size=bias.shape) < perturbation_probability
            mutation = self.perturbation_rate * numpy.random.normal(size=bias.shape)
            bias += mask * mutation
