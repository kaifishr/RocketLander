"""Evolution Strategies Optimizer

Implements the black box optimization algorithm Evolution Strategies (ES). 

See also: https://arxiv.org/abs/1703.03864

TODO: 
    - Test sparse noise:

        mask = numpy.random.random(size=weight.shape) < self.noise_probability
        noise = self.standard_deviation * numpy.random.normal(size=weight.shape)
        weight += mask * noise
"""
import copy
import numpy

from projects.src.optimizer import Optimizer

from src.utils.config import Config
from src.environment import Environment


class EvolutionStrategies(Optimizer):
    """Evolution strategies optimizer.

    Implementation according to paper:
    https://arxiv.org/abs/1703.03864

    Attr:
        boosters:
        learning_rate:
        momentum:
        standard_deviation:
        noise_probability:
    """

    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""
        super().__init__()

        self.boosters = environment.boosters
        self.learning_rate = config.optimizer.learning_rate
        self.momentum = config.optimizer.momentum
        self.standard_deviation = config.optimizer.standard_deviation
        self.noise_probability = config.optimizer.noise_probability

        self.parameters = copy.deepcopy(self.boosters[0].model.parameters)
        self.gradients = copy.deepcopy(self.boosters[0].model.parameters)

        # Initializes agents with same set of parameters plus noise.
        self._init_agents()

        self.num_agents = len(self.boosters)

    def _init_agents(self) -> None:
        """Initializes agents for Evolution Strategies Optimization.

        1) Assigns `noise` attribute for weights and biases to each agent.
        2) Assigns same initial parameters to all the agents' networks.
        3) Sets initial noise for all parameters.
        """
        # Noise has same structure as parameters.
        noise = copy.deepcopy(self.boosters[0].model.parameters)

        for booster in self.boosters:
            # Set new attribute `noise` to neural network of booster.
            booster.model.noise = copy.deepcopy(noise)
            booster.model.parameters = copy.deepcopy(self.parameters)
            self._noise(booster.model)

    def _noise(self, model: object) -> None:
        """Adds noise the network's weights.
        
        Args:
            model: The neural network.
        """
        scale = self.standard_deviation

        for (weight, bias), (noise_w, noise_b) in zip(model.parameters, model.noise):

            noise_w[...] = numpy.random.normal(scale=scale, size=weight.shape)
            numpy.add(weight, noise_w, out=weight)

            noise_b[...] = numpy.random.normal(scale=scale, size=bias.shape)
            numpy.add(bias, noise_b, out=bias)

    def _add_noise(self) -> None:
        """Adds noise to network parameters of each booster."""
        for booster in self.boosters:
            self._noise(booster.model)

    def step(self) -> None:
        """Performs a single optimization step."""

        # Get rewards for each agent in population.
        # rewards = numpy.array([sum(booster.rewards) for booster in self.boosters])  # cumulative reward
        rewards = numpy.array([booster.rewards[-1] for booster in self.boosters])  # final reward
        self.stats["reward"] = max(rewards)

        # Standardize rewards to be N(0, 1) gaussian.
        r_mean, r_std = rewards.mean(), rewards.std()
        rewards = (rewards - r_mean) / (r_std + 1e-6)

        # Compute estimated gradients

        # Zero gradients
        for grad_w, grad_b in self.gradients:
            numpy.multiply(grad_w, 0.0, out=grad_w)
            numpy.multiply(grad_b, 0.0, out=grad_b)

        # Compute weighted sum
        for booster, reward in zip(self.boosters, rewards):
            for (grad_w, grad_b), (noise_w, noise_b) in zip(
                self.gradients, booster.model.noise
            ):
                numpy.add(grad_w, reward * noise_w, out=grad_w)
                numpy.add(grad_b, reward * noise_b, out=grad_b)

        # Update parameters
        eta = self.learning_rate / (self.num_agents * self.standard_deviation)
        for (weight, bias), (grad_w, grad_b) in zip(self.parameters, self.gradients):
            numpy.add(weight, eta * grad_w, out=weight)
            numpy.add(bias, eta * grad_b, out=bias)

        # Broadcast weights to agents.
        for booster in self.boosters:
            booster.model.parameters = copy.deepcopy(self.parameters)

        # Add gaussian noise.
        self._add_noise()