"""Optimization class for Deep Q Reinforcement Learning."""
import copy
import math
import numpy

from src.utils.config import Config
from src.environment import Environment


class DeepQOptimizer:
    """Optimizer class for Deep Q Reinforcement Learning."""

    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""

        self.boosters = environment.boosters

        self.config = config.optimizer
        self.epsilon = self.config.epsilon
        self.gamma = self.config.gamma
        self.decay_rate = self.config.decay_rate
        self.eps_min = self.config.epsilon_min

        self.model = copy.deepcopy(self.boosters[0].model)

        # Scalars
        self.epsilon = 1.0
        self.reward = 0.0
        self.idx_best = 0
        self.iteration = 0

        self._init_agents()

    def step(self) -> None:
        """Runs single optimization step."""
        self.model.train()

        # Select booster with highest reward in current population.
        rewards = [booster.reward for booster in self.boosters]
        self.idx_best = numpy.argmax(rewards)
        self.reward = rewards[self.idx_best]
        print(f"optimizer.step() {rewards = }")

        ###
        for i, booster in enumerate(self.boosters):
            print()
            print("booster", i)
            print(booster.model.memory)
        exit()
        ###

        # Reduce temperature according to scheduler
        self._scheduler()

        # Broadcast model to all agents
        for booster in self.boosters:
            booster.model = copy.deepcopy(self.model)

        self.model.eval()
        self.iteration += 1

    def _epsilon_scheduler(self) -> None:
        """Decreases epsilon exponentially."""
        self.epsilon = self.eps_min + (1.0 - self.eps_min) * math.exp(-self.decay_rate * self.iteration)

    def _init_agents(self) -> None:
        """Initializes agents for reinforcement learning.
        
        Assigns same initial network parameters to all agents.
        """
        for booster in self.boosters:
            booster.model.parameters = copy.deepcopy(self.model.parameters)
            # booster.model.load_state_dict(self.model.state_dict())