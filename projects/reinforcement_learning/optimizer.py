"""Optimization class for Deep Q Reinforcement Learning."""
import copy
import math

import torch
from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from src.utils.config import Config
from src.environment import Environment


class DeepQOptimizer:
    """Optimizer class for deep reinforcement learning."""

    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer"""
        self.num_engines = 3    # FIX
        self.num_states = 6     # FIX

        self.boosters = environment.boosters

        self.config = config.optimizer
        self.epsilon = self.config.epsilon
        self.gamma = self.config.gamma
        self.decay_rate = self.config.decay_rate
        self.eps_min = self.config.epsilon_min
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.batch_size
        self.num_thrust_levels = self.config.num_thrust_levels
        self.num_thrust_angles = self.config.num_thrust_angles
        self.num_actions = 1 + self.num_engines * self.num_thrust_levels * self.num_thrust_angles

        # Scalars
        self.reward = 0.0
        self.loss = 0.0
        self.iteration = 0
        self.stats = {"reward": 0.0, "loss": 0.0}

        self.model = copy.deepcopy(self.boosters[0].model)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), 
            lr=self.learning_rate, 
        )

        self._init_agents()

    def _init_agents(self) -> None:
        """Initializes agents for reinforcement learning.
        
        Assigns same initial network parameters to all agents.
        """
        self._broadcast_agents()
        for booster in self.boosters:
            booster.model.epsilon = self.epsilon

    def _broadcast_agents(self) -> None:
        """Broadcasts base network parameters to all agents."""
        for booster in self.boosters:
            booster.model.parameters = copy.deepcopy(self.model.parameters)
            # booster.model.load_state_dict(self.model.state_dict())  # TODO: Test this method.

    def _epsilon_scheduler(self) -> None:
        """Decreases epsilon-greedy value exponentially."""
        decay_rate = self.decay_rate
        iteration = self.iteration
        eps_min = self.eps_min
        eps_max = self.epsilon
        epsilon = eps_min + (eps_max - eps_min) * math.exp(-decay_rate * iteration)

        for booster in self.boosters:
            booster.model.epsilon = epsilon

    def _create_training_set(self):
        """Create training set from memory.

        Creates dataset from memory that each of the booster's network holds.

        Memory is a list of state-action-reward-tuples: 

        memory = [
            [state_0, action_0, reward_1],
            [state_1, action_1, reward_2],
            .
            .
            .
            [state_n, action_n, reward_n+1],
        ]
        """
        replay_memory = []

        # Gather the memory that each booster's network holds.
        for booster in self.boosters:
            for memory in booster.model.memory:
                # Create replay memory and add `done` field.
                # `done` indicates if simulation has come to an end.
                replay_memory.append(memory + [False, ])
            # Set `done` to true for last memory before simulation stopped.
            # Not happy with this. Consider time constraints and actual landing.
            replay_memory[-1][-1] = True

        # Select states and new states from replay
        # NOTE: replay = replay_memory
        # new_states = torch.Tensor([memory[3] for memory in replay])
        states = torch.stack([memory[0] for memory in replay_memory])

        # Predict expected utilities (Q-values) for all states 
        # TODO: Save expected utility during simulation.
        with torch.no_grad():
            self.model.eval()
            expected_utility = self.model(states)
            # expected_utility_new = self.forward(new_states)
            self.model.train()

        replay_memory_length = len(replay_memory)
        x_data = torch.empty(size=(replay_memory_length, self.num_states))
        y_data = torch.empty(size=(replay_memory_length, self.num_actions))

        # Create the actual training set
        x_data = states

        for i in range(replay_memory_length):

            # Unpack replay memory.
            state, action, reward, done = replay_memory[i]
            # Utility is the reward of performing an action a in state s.
            target = expected_utility[i]
            target[action] = reward

            # Add expected maximum future reward if not done.
            if not done:
                target[action] += self.gamma * torch.amax(expected_utility[i+1])

            # x_data[i] = state
            y_data[i] = target

        # Create dataset
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Train model for single epoch
        for x, y in dataloader:
            self.optimizer.zero_grad()
            # Forward: Predict action pred from state x
            pred = self.model(x)
            # Compute loss
            loss = self.criterion(input=pred, target=y)
            # Backpropagation
            loss.backward()
            # Gradient descent
            self.optimizer.step()

        # return x_data, y_data

    def step(self) -> None:
        """Runs single optimization step."""
        self.model.train()

        # Select booster with highest reward in current population.
        rewards = [booster.reward for booster in self.boosters]
        self.reward = max(rewards)
        # print(f"optimizer.step() {rewards = }")

        # Create data set from recorded state-action-reward pairs.
        self._create_training_set()

        # Broadcast model to all agents
        self.model.eval()

        # Broadcast model weights to all agents
        # TODO: Add _broadcast_params() to optimizer base class.
        self._broadcast_agents()

        # Reduce  according to scheduler
        self._epsilon_scheduler()

        self.iteration += 1