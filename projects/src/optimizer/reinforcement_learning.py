"""Optimization class for Deep Q Reinforcement Learning."""
import copy
import random

from collections import deque

import numpy
import torch

from projects.src.optimizer import Optimizer
from src.utils.config import Config
from src.environment import Environment


class DeepQOptimizer(Optimizer):
    """Optimizer class for deep reinforcement learning.

    Attributes:
        epsilon: The epsilon-greedy value. This value defines the probability,
            that the agent selects a random action instead of the action that
            maximizes the expected utility (Q-value).
        epsilon_min: Minimal value of the epsilon-greedy value.
        gamma: A discount factor determining how much the agent considers future rewards.
    """

    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""
        super().__init__()

        self.boosters = environment.boosters
        self.config = config

        config = config.optimizer
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.gamma = config.gamma
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.num_thrust_levels = config.num_thrust_levels
        self.num_thrust_angles = config.num_thrust_angles
        self.num_actions = (
            1 + self.num_engines * self.num_thrust_levels * self.num_thrust_angles
        )

        self.model = copy.deepcopy(self.boosters[0].model)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.learning_rate,
        )

        self.replay_memory = deque(maxlen=config.memory_size)
        self._init_agents()

    def _init_agents(self) -> None:
        """Initializes agents for reinforcement learning.

        Assigns same initial network parameters to all agents.
        """
        self._broadcast_agents()
        for booster in self.boosters:
            booster.model.epsilon = self.epsilon

    @torch.no_grad()
    def _broadcast_agents(self) -> None:
        """Broadcasts base network parameters to all agents."""
        for booster in self.boosters:
            for params1, params2 in zip(
                booster.model.parameters(), self.model.parameters()
            ):
                # Here it is sufficient to assign the parameters by reference.
                params1.data = params2.data
                # params1.data.copy_(params2.data)

    def _epsilon_scheduler(self) -> None:
        """Decays epsilon-greedy value."""

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.stats["epsilon"] = self.epsilon

        for booster in self.boosters:
            booster.model.epsilon = self.epsilon

    def _gather_data(self) -> None:
        """Gathers memory data from all agents.

        Build replay memory with [state, action, reward, next_state, done]
        with data from each booster. Terminal state in memory is set to `True`
        if one of the following conditions is met:

            - crash
            - leaving domain
            - exceeding maximum stress
            - exceeding maximum number of simulation steps
            - successful landing
        """
        for booster in self.boosters:
            memory = booster.model.memory
            rewards = booster.rewards
            for (s0, a0), r0, (s1, _) in zip(memory[:-1], rewards, memory[1:]):
                self.replay_memory.append(copy.deepcopy([s0, a0, r0, s1, False]))
            self.replay_memory[-1][-1] = True

    @torch.no_grad()
    def _create_training_set(self) -> None:
        """Create training set from entire memory.

        Creates dataset from memory that each of the booster's network holds.
        """
        # Use subset of replay memory for training as transitions are strongly correlated.
        replay_batch = random.sample(
            self.replay_memory, min(len(self.replay_memory), self.batch_size)
        )

        # Normalize the rewards of batch.
        rewards = numpy.array([memory[2] for memory in replay_batch])
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        for memory, reward in zip(replay_batch, rewards):
            memory[2] = reward

        # Select states from replay memory.
        states = torch.stack([memory[0] for memory in replay_batch])
        next_states = torch.stack([memory[3] for memory in replay_batch])

        # Predict expected utilities (Q target values) for states from replay memory.
        with torch.no_grad():
            self.model.eval()
            q_targets = self.model(states)
            q_targets_new = self.model(next_states)
            self.model.train()

        for i, (_, action, reward, next_state, done) in enumerate(replay_batch):
            q_targets[i, action] = reward
            if not done:  # Discount the reward by gamma as landing was not successful.
                q_targets[i, action] = (
                    reward + self.gamma * torch.amax(q_targets_new[i]).item()
                )
            else:  # Full reward if booster landed successfully.
                q_targets[i, action] = reward

        return states, q_targets

    def _train_network(self, states, q_targets) -> None:
        """Trains network on random batch of memory."""
        self.model.train()

        self.optimizer.zero_grad()
        # Predict expected utility from state with policy (network).
        q_values = self.model(states)
        # Compute loss.
        loss = self.criterion(input=q_values, target=q_targets)
        self.stats["loss"] = loss.item()
        # Backpropagation.
        loss.backward()
        # Clip gradients.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # Gradient descent.
        self.optimizer.step()

        self.model.eval()

    def step(self) -> None:
        """Runs single optimization step."""
        self.stats["reward"] = max(self._gather_rewards(reduction="sum"))

        # Gather data from all agents.
        self._gather_data()

        if len(self.replay_memory) > self.config.optimizer.num_warmup_steps:

            # Create training set from memory and train network.
            states, q_targets = self._create_training_set()
            self._train_network(states, q_targets)

            # Broadcast model weights to all agents.
            self._broadcast_agents()

        # Reduce epsilon according to scheduler.
        self._epsilon_scheduler()

        self.iteration += 1
