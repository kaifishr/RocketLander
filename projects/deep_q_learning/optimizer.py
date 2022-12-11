"""Optimization class for Deep Q Reinforcement Learning."""
import copy
import math
import random

from collections import deque

import numpy
import torch
from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from src.utils.config import Config
from src.environment import Environment


class DeepQOptimizer:
    """Optimizer class for deep reinforcement learning.

    Attributes:
        epsilon: The epsilon-greedy value. This value defines the probability, 
            that the agent selects a random action instead of the action that 
            maximizes the expected utility (Q-value).
        epsilon_min: Minimal value of the epsilon-greedy value.
        decay_rate: Determines the decay of the epsilon-greedy value after each epoch.
        gamma: A discount factor determining how much the agent considers future rewards.
    """
    num_engines = 3    # FIX
    num_states = 6     # FIX

    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""

        self.boosters = environment.boosters

        config = config.optimizer
        self.epsilon_max = config.epsilon_max
        self.epsilon_min = config.epsilon_min
        self.gamma = config.gamma
        self.decay_rate = config.decay_rate
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.memory_size = config.memory_size
        self.num_thrust_levels = config.num_thrust_levels
        self.num_thrust_angles = config.num_thrust_angles
        self.num_actions = 1 + self.num_engines * self.num_thrust_levels * self.num_thrust_angles

        # Scalars
        self.stats = {"reward": -1, "loss": -1, "epsilon": -1}

        self.model = copy.deepcopy(self.boosters[0].model)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), 
            lr=self.learning_rate, 
        )

        self.replay_memory = deque()
        self._init_agents()
        self.iteration = 0

    def _init_agents(self) -> None:
        """Initializes agents for reinforcement learning.
        
        Assigns same initial network parameters to all agents.
        """
        self._broadcast_agents()
        for booster in self.boosters:
            booster.model.epsilon = self.epsilon_max

    @torch.no_grad()
    def _broadcast_agents(self) -> None:
        """Broadcasts base network parameters to all agents."""
        for booster in self.boosters:
            for params1, params2 in zip(booster.model.parameters(), self.model.parameters()):
                params1.data = params2.data
            #     # params1.data.copy_(params2.data)
            #     params1.data[...] = params2.data[...]
            # booster.model.load_state_dict(state_dict=self.model.state_dict())

    def _epsilon_scheduler(self) -> None:
        """Decreases epsilon-greedy value exponentially."""
        iteration = self.iteration
        decay_rate = self.decay_rate
        eps_min = self.epsilon_min
        eps_max = self.epsilon_max
        epsilon = eps_min + (eps_max - eps_min) * math.exp(-decay_rate * iteration)
        self.stats["epsilon"] = epsilon

        for booster in self.boosters:
            booster.model.epsilon = epsilon

    @torch.no_grad()
    def _create_training_set(self) -> None:
        """Create training set from entire memory.

        Creates dataset from memory that each of the booster's network holds.
        """
        # Build replay memory with [state, action, reward, next_state, done]
        # with memory from each booster.
        for booster in self.boosters:
            for (s0, a0, r0), (s1, _, _) in zip(list(booster.model.memory)[:-1], list(booster.model.memory)[1:]):
                self.replay_memory.append([s0, a0, r0, s1, False])
                # TODO: Set `done` to True if booster has landed.
                if len(self.replay_memory) > self.memory_size:
                    self.replay_memory.popleft()

        # Normalize the rewards.
        rewards = numpy.array([memory[2] for memory in self.replay_memory])
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        for memory, reward in zip(self.replay_memory, rewards):
            memory[2] = reward

        # Use subset of replay memory for training as transitions are strongly correlated.  
        replay_memory_batch = random.sample(
            self.replay_memory, 
            min(len(self.replay_memory), self.batch_size)
        )

        # Select states from replay memory.
        states = torch.stack([memory[0] for memory in replay_memory_batch])
        next_states = torch.stack([memory[3] for memory in replay_memory_batch])

        # Predict expected utilities (Q target values) for states from replay memory.
        self.model.eval()
        q_targets = self.model(states)
        q_targets_new = self.model(next_states)
        self.model.train()

        for i, (_, action, reward, next_state, done) in enumerate(replay_memory_batch):
            q_targets[i, action] = reward
            if not done:  # Discount the reward by gamma as landing was not successful.
                q_targets[i, action] = reward + self.gamma * torch.amax(q_targets_new[i]).item()
            else:  # Full reward if booster landed successfully.
                q_targets[i, action] = reward

        return states, q_targets

    def _train_network(self, states, q_targets) -> None:
        """Trains network on random batch of memory.
        
        """
        self.model.train()

        self.optimizer.zero_grad()
        # Predict expected utility from state with policy (network).
        q_values = self.model(states)
        # Compute loss
        loss = self.criterion(input=q_values, target=q_targets)
        # Backpropagation
        loss.backward()
        # Gradient descent
        self.optimizer.step()

        self.model.eval()
        self.stats["loss"] = loss.item() / len(states)


    def step(self) -> None:
        """Runs single optimization step."""
        self.reward = max([booster.reward for booster in self.boosters])
        self.stats["reward"] = self.reward

        # Create training set from memory
        states, q_targets = self._create_training_set()
        self._train_network(states, q_targets)

        # Broadcast model weights to all agents
        self._broadcast_agents()

        # Reduce epsilon according to scheduler
        self._epsilon_scheduler()

        self.iteration += 1