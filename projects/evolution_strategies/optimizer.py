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

        self.model_old = copy.deepcopy(self.boosters[0].model.parameters)
        self.gradients = copy.deepcopy(self.boosters[0].model.parameters)

        # Maximum reward for current epoch.
        self.reward = 0.0

    def step(self) -> None:
        """Performs a single optimization step."""

        # ... 1) Get rewards for each agent in population.
        rewards = numpy.array([booster.reward for booster in self.boosters])
        idx_best = numpy.argmax(rewards)
        self.reward = rewards[idx_best]

        # ... 2) Standardize rewards to be N(0, 1) gaussian.
        r_mean, r_std = rewards.mean(), rewards.std()
        rewards = (rewards - r_mean) / (r_std + 1e-6)

        # OR:
        # ... 2.2) Normalize rewards to be [0, 1].
        # r_min, r_max = rewards.min(), rewards.max()
        # rewards = (rewards - r_min) / (r_max - r_min)
        # rewards /= rewards.sum()  # So rewards sum up to 1
        # ... 2.3) Softmax?

        # ... 3) Compute estimated gradients
        # ... 3.1) Weight parameters according to reward
        # for booster, reward in zip(self.boosters, rewards):
        #     for weight, bias in booster.model.parameters:
        #         weight *= reward
        #         bias *= reward

        # ... 3.2) Compute weighted sum
        # Zero gradients
        for grad_w, grad_b in self.gradients:
            grad_w *= 0.0
            grad_b *= 0.0

        for booster, reward in zip(self.boosters, rewards):
            for (grad_w, grad_b), (weight, bias) in zip(self.gradients, booster.model.parameters):
                grad_w += reward * weight
                grad_b += reward * bias

        # ... 4) Update parameters and distribute new parameters across all agents
        for booster in self.boosters:
            for (grad_w, grad_b), (weight, bias) in zip(self.gradients, booster.model.parameters):
                weight += self.learning_rate * grad_w
                bias += self.learning_rate * grad_b 

        # ... 5) Save model.
        self.model_old = copy.deepcopy(self.boosters[0].model.parameters)

        # Add gaussian noise.
        self._add_noise()

    # def _get_reward(self) -> None:
    #     """Collects rewards of agents."""
    #     # Fetch rewards of each booster.
    #     rewards = [booster.reward for booster in self.boosters]
    #     # Get maximum reward of current population.
    #     idx_best = numpy.argmax(rewards)
    #     self.reward = rewards[idx_best]

    def _add_noise(self) -> None:
        """Adds noise to network parameters of each booster."""
        for booster in self.boosters:
            self._noise(booster.model)

    def _noise(self, model: object) -> None:
        """Adds noise the network's weights."""
        for weight, bias in model.parameters:
            mask = numpy.random.random(size=weight.shape) < self.noise_probability
            noise = self.standard_deviation * numpy.random.normal(size=weight.shape)
            weight += mask * noise

            mask = numpy.random.random(size=bias.shape) < self.noise_probability
            noise = self.standard_deviation * numpy.random.normal(size=bias.shape)
            bias += mask * noise
