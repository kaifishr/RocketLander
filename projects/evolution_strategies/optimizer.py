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

from src.utils.config import Config
from src.environment import Environment


class EvolutionStrategies_v2:
    """Evolution strategies optimizer. Version 2.
    
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

        self.parameters = copy.deepcopy(self.boosters[0].model.parameters)
        self.gradients = copy.deepcopy(self.boosters[0].model.parameters)

        # Maximum reward for current epoch.
        self.reward = 0.0

        self.num_agents = len(self.boosters)

    def step(self) -> None:
        """Performs a single optimization step."""

        # ... 1) Get rewards for each agent in population.
        rewards = numpy.array([booster.reward for booster in self.boosters])
        idx_best = numpy.argmax(rewards)
        self.reward = rewards[idx_best]

        # ... 2) Standardize rewards to be N(0, 1) gaussian.
        r_mean, r_std = rewards.mean(), rewards.std()
        rewards = (rewards - r_mean) / (r_std + 1e-6)

        # ... 3) Compute estimated gradients

        # ... 3.1) Zero gradients
        for grad_w, grad_b in self.gradients:
            grad_w *= 0.0
            grad_b *= 0.0

        # ... 3.2) Compute weighted sum (TODO: make faster by saving noise to model)
        for booster, reward in zip(self.boosters, rewards):
            for (grad_w, grad_b), (weight_p, bias_p), (weight, bias) in zip(self.gradients, booster.model.parameters, self.parameters):
                grad_w += reward * (weight_p - weight)  # (weight_p - weight) = noise
                grad_b += reward * (bias_p - bias)

        # ... 4) Update parameters 
        eta = self.learning_rate / (self.num_agents * self.standard_deviation)
        for (weight, bias), (grad_w, grad_b) in zip(self.parameters, self.gradients):
            weight += eta * grad_w
            bias += eta * grad_b

        # ... 5) Broadcast weights to agents. 
        for booster in self.boosters:
            booster.model.parameters = copy.deepcopy(self.parameters)

        # Add gaussian noise.
        self._add_noise()

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


class EvolutionStrategies:
    """Evolution strategies optimizer. Version 1.
    
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
        # self.noise_probability = config.optimizer.noise_probability

        self.parameters = copy.deepcopy(self.boosters[0].model.parameters)
        self.gradients = copy.deepcopy(self.boosters[0].model.parameters)

        # Maximum reward for current epoch.
        self.reward = 0.0
        self.num_agents = len(self.boosters)

    def step(self) -> None:
        """Performs a single optimization step."""

        # ... 1) Get rewards for each agent in population.
        rewards = numpy.array([booster.reward for booster in self.boosters])
        idx_best = numpy.argmax(rewards)
        self.reward = rewards[idx_best]

        # ... 2) Standardize rewards to be N(0, 1) gaussian.
        r_mean, r_std = rewards.mean(), rewards.std()
        rewards = (rewards - r_mean) / (r_std + 1e-6)

        # ... 3) Compute estimated gradients

        # ... 3.1) Zero gradients
        for grad_w, grad_b in self.gradients:
            grad_w *= 0.0
            grad_b *= 0.0

        # ... 3.2) Compute weighted sum (TODO: make faster by saving noise to model)
        for booster, reward in zip(self.boosters, rewards):
            for (grad_w, grad_b), (weight, bias) in zip(self.gradients, booster.model.parameters):
                grad_w += reward * weight 
                grad_b += reward * bias

        # ... 4) Update parameters 
        eta = self.learning_rate / (self.num_agents * self.standard_deviation)
        for (weight, bias), (grad_w, grad_b) in zip(self.parameters, self.gradients):
            weight += eta * grad_w
            bias += eta * grad_b

        # ... 5) Broadcast weights to agents. 
        for booster in self.boosters:
            booster.model.parameters = copy.deepcopy(self.parameters)

        # Add gaussian noise.
        self._add_noise()

    def _add_noise(self) -> None:
        """Adds noise to network parameters of each booster."""
        for booster in self.boosters:
            self._noise(booster.model)

    def _noise(self, model: object) -> None:
        """Adds noise the network's weights."""
        for weight, bias in model.parameters:
            weight += numpy.random.normal(scale=self.standard_deviation, size=weight.shape)
            bias += numpy.random.normal(scale=self.standard_deviation, size=bias.shape)

            ## mask = numpy.random.random(size=weight.shape) < self.noise_probability
            ## noise = self.standard_deviation * numpy.random.normal(size=weight.shape)
            ## weight += mask * noise

            ## mask = numpy.random.random(size=bias.shape) < self.noise_probability
            ## noise = self.standard_deviation * numpy.random.normal(size=bias.shape)
            ## bias += mask * noise


class EvolutionStrategies:
    """Evolution strategies optimizer. Version 3.

    Implementation according to paper:
    https://arxiv.org/abs/1703.03864
    
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

        self.parameters = copy.deepcopy(self.boosters[0].model.parameters)
        self.gradients = copy.deepcopy(self.boosters[0].model.parameters)

        # Maximum reward for current epoch.
        self.reward = 0.0

        # Initializes agents with same set of parameters plus noise.
        self._init_agents()

        self.num_agents = len(self.boosters)

    def step(self) -> None:
        """Performs a single optimization step."""

        # ... 1) Get rewards for each agent in population.
        rewards = numpy.array([booster.reward for booster in self.boosters])
        idx_best = numpy.argmax(rewards)
        self.reward = rewards[idx_best]

        # ... 2) Standardize rewards to be N(0, 1) gaussian.
        r_mean, r_std = rewards.mean(), rewards.std()
        rewards = (rewards - r_mean) / (r_std + 1e-6)

        # ... 3) Compute estimated gradients

        # ... 3.1) Zero gradients
        for grad_w, grad_b in self.gradients:
            numpy.multiply(grad_w, 0.0, out=grad_w)
            numpy.multiply(grad_b, 0.0, out=grad_b)

        # ... 3.2) Compute weighted sum 
        for booster, reward in zip(self.boosters, rewards):
            for (grad_w, grad_b), (noise_w, noise_b) in zip(self.gradients, booster.model.noise):
                numpy.add(grad_w, reward * noise_w, out=grad_w)
                numpy.add(grad_b, reward * noise_b, out=grad_b)

        # ... 4) Update parameters 
        eta = self.learning_rate / (self.num_agents * self.standard_deviation)
        for (weight, bias), (grad_w, grad_b) in zip(self.parameters, self.gradients):
            numpy.add(weight, eta * grad_w, out=weight)
            numpy.add(bias, eta * grad_b, out=bias)

        # ... 5) Broadcast weights to agents. 
        for booster in self.boosters:
            booster.model.parameters = copy.deepcopy(self.parameters)

        # Reset old noise. 
        # self._reset_noise()

        # Add gaussian noise.
        self._add_noise()

    def _init_agents(self) -> None:
        """Initializes agents for Evolution Strategies Optimization.

        1) Assigns `noise` attribute for weights and biases to each agent.
        2) Assigns same initial parameters to all the agents' networks.
        3) Sets initial noise for all parameters.
        """
        # Noise has same structure as parameters.
        noise = copy.deepcopy(self.boosters[0].model.parameters)

        # # Zero noise
        # for noise_w, noise_b in noise:
        #     numpy.multiply(noise_w, 0.0, out=noise_w)
        #     numpy.multiply(noise_b, 0.0, out=noise_b)

        for booster in self.boosters:
            # Set new attribute `noise` to neural network of booster.
            booster.model.noise = copy.deepcopy(noise)
            booster.model.parameters = copy.deepcopy(self.parameters)
            self._noise(booster.model)

    # def _reset_noise(self) -> None:
    #     """Resets noise of model."""
    #     for booster in self.boosters:
    #         for noise_w, noise_b in booster.model.noise:
    #             numpy.multiply(noise_w, 0.0, out=noise_w)
    #             numpy.multiply(noise_b, 0.0, out=noise_b)

    def _add_noise(self) -> None:
        """Adds noise to network parameters of each booster."""
        for booster in self.boosters:
            self._noise(booster.model)

    def _noise(self, model: object) -> None:
        """Adds noise the network's weights."""
        for (weight, bias), (noise_w, noise_b) in zip(model.parameters, model.noise):

            # mask = numpy.random.random(size=weight.shape) < self.noise_probability
            # noise = self.standard_deviation * numpy.random.normal(size=weight.shape)
            # # numpy.add(noise_w, mask * noise, out=noise_w)
            noise_w[:] = numpy.random.normal(scale=self.standard_deviation, size=weight.shape)
            numpy.add(weight, noise_w, out=weight) 

            # mask = numpy.random.random(size=bias.shape) < self.noise_probability
            # noise = self.standard_deviation * numpy.random.normal(size=bias.shape)
            # # numpy.add(noise_b, mask * noise, out=noise_w)
            # noise_b[...] = mask * noise
            noise_b[:] = numpy.random.normal(scale=self.standard_deviation, size=bias.shape)
            numpy.add(bias, noise_b, out=bias) 