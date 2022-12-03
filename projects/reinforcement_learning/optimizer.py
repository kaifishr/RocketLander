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
    """Optimizer class for deep reinforcement learning.

    Attributes:
        epsilon: The epsilon-greedy value. This value defines the probability, 
            that the agent selects a random action instead of the action that 
            maximizes the expected utility (Q-value).
        epsilon_min: Minimal value of the epsilon-greedy value.
        decay_rate: Determines the decay of the epsilon-greedy value after each epoch.
        gamma: A discount factor determining how much the agent considers future rewards.
    
    """

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
        self.num_epochs = self.config.num_epochs
        self.num_thrust_levels = self.config.num_thrust_levels
        self.num_thrust_angles = self.config.num_thrust_angles
        self.num_actions = 1 + self.num_engines * self.num_thrust_levels * self.num_thrust_angles

        # Scalars
        self.reward = 0.0
        self.iteration = 0
        self.stats = {"reward": 0.0, "loss": 0.0}

        # TODO: Pass models to agents by reference as they all use the same model.
        self.model = copy.deepcopy(self.boosters[0].model)

        # Mean squared error loss function
        self.criterion = torch.nn.MSELoss()

        # Adam optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), 
            lr=self.learning_rate, 
        )

        # Dataloader
        self.dataloader = None

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

    @torch.no_grad()
    def _create_training_set(self) -> None:
        """Create training set from entire memory.

        Creates dataset from memory that each of the booster's network holds.

        Memory is a list of state-action-reward-tuples: 

        memory = [
            [state_0, action_0, reward_1],
            [state_1, action_1, reward_2],
            ...
            [state_n, action_n, reward_n+1],
        ]
        """
        replay_memory = []

        # Gather the memory from each booster's.
        for booster in self.boosters:
            for memory in booster.model.memory:
                # Create replay memory and add `done` field.
                # `done` indicates if simulation has come to an end.
                replay_memory.append(memory + [False, ])
            # Set `done` to true for last memory before simulation stopped.
            # Not happy with this. Consider time constraints and actual landing.
            replay_memory[-1][-1] = True

        # Select states from replay memory.
        states = torch.stack([memory[0] for memory in replay_memory])

        # Predict expected utilities (Q target values) for states from replay memory.
        # TODO: Save expected utility directly during simulation for higher efficiency?
        self.model.eval()
        q_targets = self.model(states)
        self.model.train()

        # Create the actual training set
        for i in range(len(replay_memory)): 
            # Unpack replay memory.
            state, action, reward, done = replay_memory[i]

            if done:
                q_targets[i, action] = reward
            else:
                q_targets[i, action] = reward + self.gamma * torch.amax(q_targets[i+1])

        # Create dataloader 
        dataset = TensorDataset(states, q_targets)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def _train_network(self) -> None:
        """Trains network on memory.
        
        Trans network for specified number of epochs and mini batch size
        on memory.
        """
        self.model.train()

        running_loss = 0.0
        running_counter = 0

        for epoch in range(self.num_epochs):
            for state, q_target in self.dataloader:

                self.optimizer.zero_grad()
                # Forward: Predict expected utility from state.
                q_value = self.model(state)
                # Compute loss
                loss = self.criterion(input=q_value, target=q_target)
                # Backpropagation
                loss.backward()
                # Gradient descent
                self.optimizer.step()

                running_loss += loss.item()
                running_counter += len(state)

        self.stats["loss"] = running_loss / running_counter

        self.model.eval()

    def step(self) -> None:
        """Runs single optimization step."""

        # Select booster with highest reward in current population.
        rewards = [booster.reward for booster in self.boosters]
        self.reward = max(rewards)
        self.stats["reward"] = self.reward

        # Create training set from memory.
        self._create_training_set()

        self._train_network()

        # Broadcast model weights to all agents
        self._broadcast_agents()

        # Reduce epsilon according to scheduler
        self._epsilon_scheduler()

        self.iteration += 1