"""Evolution Strategies Optimizer

Implements the black box optimization algorithm Evolution Strategies (ES). 

See also: https://arxiv.org/abs/1703.03864
"""
import copy
import numpy

from src.utils.config import Config
from src.environment import Environment


class EvolutionStrategies:
    """Class information.
    
    Longer class information.
    
    Attr:
        boosters:
        learning_rate:
        momentum:
        standard_deviation:
        reward:
    """
    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""

        self.boosters = environment.boosters
        self.learning_rate = config.optimizer.learning_rate
        self.momentum = config.optimizer.momentum
        self.standard_deviation = config.optimizer.standard_deviation
        self.noise_probability = config.optimizer.noise_probability

        # Maximum reward for current epoch.
        self.reward = 0.0

    def step(self) -> None:
        """Performs a single optimization step."""

        # ... 1) Get rewards for each agent in population.
        rewards = numpy.array([booster.reward for booster in self.boosters])

        # ... 2) Standardize rewards to be N(0, 1) gaussian.
        r_mean, r_std = rewards.mean(), rewards.std()
        rewards = (rewards - r_mean) / (r_std + 1e-6)

        rewards = numpy.array([1, 1])

        # OR:
        # ... 2.2) Normalize rewards to be [0, 1].
        # r_min, r_max = rewards.min(), rewards.max()
        # rewards = (rewards - r_min) / (r_max - r_min)
        # ... 2.3) Softmax?
        for reward in rewards:
            print(f"{reward =}")

        # ... 3) Compute estimated gradients

        # ... 3.1) Get all the parameters
        for i, booster in enumerate(self.boosters):
            print(f"booster {i+1}")
            for weight, bias in booster.model.parameters:
                print(f"{weight}")
                print(f"{bias}")
            print()

        # Weight parameters according to reward
        for booster, reward in zip(self.boosters, rewards):
            for weight, bias in booster.model.parameters:
                weight *= reward
                bias *= reward

        parameter_target = self.boosters[0].model.parameters
        for booster in self.boosters[1:]:
            for (weight_target, bias_target), (weight, bias) in zip(parameter_target, booster.model.parameters):
                weight_target += weight
                bias_target += bias

        # Weight parameters according to reward
        ## for weight, bias in self.boosters[0].model.parameters:
        ##     weight *= rewards[0]
        ##     bias *= rewards[0]

        ## for booster, reward in zip(self.boosters[1:], rewards[1:]):
        ##     for (weight_target, bias_target), (weight, bias) in zip(self.boosters[0].model.parameters, booster.model.parameters):
        ##         weight_target += reward * weight
        ##         bias_target += reward * bias

        for i, booster in enumerate(self.boosters):
            print(f"booster {i+1}")
            for weight, bias in booster.model.parameters:
                print(f"{weight}")
                print(f"{bias}")
            print()
        exit(0)

        # ... 3.2) Compute weighted sum

        # ... 4) Update parameters

        # ... 5) Distribute parameters across all agents.

        # Add gaussian noise.
        self._add_noise()

    def _get_reward(self) -> None:
        """Collects rewards of agents."""
        # Fetch rewards of each booster.
        rewards = [booster.reward for booster in self.boosters]
        # Get maximum reward of current population.
        idx_best = numpy.argmax(rewards)
        self.reward = rewards[idx_best]

    def _add_noise(self) -> None:
        """Adds noise to network parameters of each booster."""
        # Pass best model to other boosters and mutate their weights.
        for booster in self.boosters:
            # Assign best model to each booster and mutate weights.
            self._noise(booster.model)
            # booster.model.mutate_weights()

    def _noise(self, model: object) -> None:
        """Adds noise the network's weights."""

        for weight, bias in model.parameters:

            mask = numpy.random.random(size=weight.shape) < self.noise_probability
            noise = self.standard_deviation * numpy.random.normal(size=weight.shape)
            weight += mask * noise

            mask = numpy.random.random(size=bias.shape) < self.noise_probability
            noise = self.standard_deviation * numpy.random.normal(size=bias.shape)
            bias += mask * noise
