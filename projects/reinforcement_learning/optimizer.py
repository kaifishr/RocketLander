"""Optimization class for Deep Q Reinforcement Learning."""
import copy
import math

from collections import deque

import torch

from src.utils.config import Config
from src.environment import Environment


class DeepQOptimizer:
    """Optimizer class for deep reinforcement learning."""

    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""

        self.boosters = environment.boosters

        self.config = config.optimizer
        self.epsilon = self.config.epsilon
        self.gamma = self.config.gamma
        self.decay_rate = self.config.decay_rate
        self.eps_min = self.config.epsilon_min
        self.num_states = 6
        self.num_actions = 15  # TODO
        self.num_engines = 3 
        self.num_thrust_levels = 3  # Thrust levels of engines. Minimum is 2 for on/off
        self.num_thrust_angles = 3  # Thrust angles of engines. Must be an odd number.
        self.num_actions = 1 + self.num_engines * self.num_thrust_levels * self.num_thrust_angles

        self.model = copy.deepcopy(self.boosters[0].model)

        # Scalars
        self.epsilon = 1.0
        self.reward = 0.0
        self.iteration = 0

        self._init_agents()

    def _epsilon_scheduler(self) -> None:
        """Decreases epsilon exponentially."""
        decay_rate = self.decay_rate
        iteration = self.iteration
        eps_min = self.eps_min
        self.epsilon = eps_min + (1.0 - eps_min) * math.exp(-decay_rate * iteration)

    def _init_agents(self) -> None:
        """Initializes agents for reinforcement learning.
        
        Assigns same initial network parameters to all agents.
        """
        self._copy_agents()

    def _copy_agents(self) -> None:
        """Broadcasts base network parameters to all agents."""
        for booster in self.boosters:
            booster.model.parameters = copy.deepcopy(self.model.parameters)
            # booster.model.load_state_dict(self.model.state_dict())  # TODO: Test this method.

    def _create_training_set(self, replay: deque):
        """
        Args:
            replay: List of tuples holding [state, action, reward, new_state]
        """
        # Select states and new states from replay
        states = torch.Tensor([memory[0] for memory in replay])
        new_states = torch.Tensor([memory[3] for memory in replay])

        # Predict expected utility (Q-value) of current state and new state
        # TODO: Instead of saving agent integer, save expected utility 
        #       predicted by network.
        with torch.no_grad():
            self.eval()
            expected_utility = self.forward(states)
            expected_utility_new = self.forward(new_states)
            self.train()

        replay_length = len(replay)
        x_data = torch.empty(size=(replay_length, self.num_states))
        y_data = torch.empty(size=(replay_length, self.num_actions))

        # Create training set
        for i in range(replay_length):

            # Unpack replay
            state, action, reward, new_state, done = replay[i]

            # Utility is the reward of performing an action a in state s.
            target = expected_utility[i]
            target[action] = reward

            # Add expected maximum future reward if not done.
            if not done:
                target[action] += self.gamma * torch.amax(expected_utility_new[i])

            x_data[i] = state
            y_data[i] = target

        return x_data, y_data

    def step(self) -> None:
        """Runs single optimization step."""
        self.model.train()

        # Select booster with highest reward in current population.
        rewards = [booster.reward for booster in self.boosters]
        self.reward = max(rewards)
        print(f"optimizer.step() {rewards = }")

        # Create data set from recorded state-action-reward pairs.

        ###
        # for i, booster in enumerate(self.boosters):
        #     print("booster", i)
        #     for m in booster.model.memory:
        #         print(m)
        #     print()
        # exit()
        ###

        # Train model on training set.
        ...

        # Reduce  according to scheduler
        self._epsilon_scheduler()

        # Broadcast model to all agents
        self._copy_agents()  # TODO: Use this also for other optimizers. Add to optimizer base class.

        self.model.eval()
        self.iteration += 1

        if self.iteration == 3:
            exit()
