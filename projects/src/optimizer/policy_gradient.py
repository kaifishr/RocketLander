"""Optimization class for Deep Q Reinforcement Learning."""
import copy

import torch
import torch.nn.functional as F

from projects.src.optimizer import Optimizer
from src.utils.config import Config
from src.environment import Environment


class PolicyGradient(Optimizer):
    """Optimizer class for policy gradient reinforcement learning.

    Attributes:
        gamma: A discount factor determining how much the agent considers future rewards.
    """

    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""
        super().__init__()

        self.boosters = environment.boosters
        self.config = config

        config = config.optimizer
        self.gamma = config.gamma
        self.learning_rate = config.learning_rate
        self.num_thrust_levels = config.num_thrust_levels
        self.num_thrust_angles = config.num_thrust_angles
        self.num_actions = 1 + self.num_engines * self.num_thrust_levels * self.num_thrust_angles

        self.model = copy.deepcopy(self.boosters[0].model)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        self._init_agents()

    def _init_agents(self) -> None:
        """Initializes agents for reinforcement learning.

        Assigns same initial network parameters to all agents.
        """
        self._broadcast_agents()

    @torch.no_grad()
    def _broadcast_agents(self) -> None:
        """Broadcasts base network parameters to all agents."""
        for booster in self.boosters:
            for params1, params2 in zip(booster.model.parameters(), self.model.parameters()):
                # Here it is sufficient to assign the parameters by reference.
                params1.data = params2.data
                # params1.data.copy_(params2.data)

    def _gather_data(self) -> tuple[list, list, list]:
        """Gathers memory data from all agents.

        Builds state, action, reward lists.

        Returns:
            Tuple holding states, actions, and rewards lists.
        """
        states = []
        actions = []
        rewards = []

        for booster in self.boosters:
            memory = booster.model.memory
            rewards_ = booster.rewards
            for (state, action), reward in zip(memory[:-1], rewards_):
                states.append(state)
                actions.append(action)
                rewards.append(reward)

        return states, actions, rewards

    def _normalize_rewards(self, rewards: torch.Tensor, eps: float = 1e-05) -> torch.Tensor:
        """Normalizes rewards.

        Normalizes rewards if there is more than one reward 
        and if standard-deviation is non-zeros.
        
        Args:
            rewards: The agent's rewards.
            eps: Value added to the denominator for numerical stability.
            
        Returns:
            Normalized rewards.
        """
        if len(rewards) > 1:
            std = torch.std(rewards)
            if std != 0:
                mean = torch.mean(rewards)
                rewards = (rewards - mean) / (std + eps)
        return rewards

    def step(self) -> None:
        """Runs single optimization step."""
        self.stats["reward"] = max(self._gather_rewards(reduction="sum"))

        # Gather data from all agents.
        states, actions, rewards = self._gather_data()

        reward_sum = 0.0
        discounted_rewards = []

        for reward in rewards[::-1]:
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards = discounted_rewards[::-1]

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = self._normalize_rewards(rewards=discounted_rewards)

        states = torch.vstack(states)
        target_actions = F.one_hot(torch.tensor(actions), num_classes=self.num_actions).float()

        # Train policy network on single batch.
        self.optimizer.zero_grad()
        output_actions = self.model(states)
        loss = self.criterion(output_actions, target_actions)
        loss = discounted_rewards * loss
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        # Broadcast model weights to all agents.
        self._broadcast_agents()

        self.iteration += 1
