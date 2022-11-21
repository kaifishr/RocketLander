"""Contains neural network definitions.

The booster's brain represented by a feedforward neural network.
"""
import math
import random
from collections import deque

import numpy
import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit

from src.utils.config import Config
from src.utils.utils import load_checkpoint


class ModelLoader:
    """Loads NumPy or PyTorch neural network."""

    def __init__(self, config: Config) -> None:
        """Initializes Network wrapper."""
        self.config = config

    def __call__(self):
        """Loads and returns model.

        Args:
            lib: Library "numpy" or "pytorch".
        """
        lib = self.config.optimizer.lib

        if lib == "numpy":
            model = NumpyNeuralNetwork(self.config)
            if self.config.checkpoints.load_model:
                raise NotImplementedError(
                    f"Loading checkpoints not implemented for NumPy neural networks."
                )

        elif lib == "torch":
            model = TorchNeuralNetwork(self.config)
            if self.config.checkpoints.load_model:
                load_checkpoint(model=self.model, config=self.config)

        else:
            raise NotImplementedError(f"Network for {lib} not implemented.")

        model.eval()  # No gradients for genetic optimization required.

        return model


class NumpyNeuralNetwork:
    """Neural network written with Numpy.

    Attributes:
        mutation_prob:
        mutation_rate:
        weights:
        biases:
    """

    def __init__(self, config: Config) -> None:
        """Initializes NeuralNetwork."""

        config = config.env.booster.neural_network

        in_features = config.num_dim_in
        out_features = config.num_dim_out
        hidden_features = config.num_dim_hidden
        num_hidden_layers = config.num_hidden_layers

        # nonlinearity = "leaky_relu"
        # nonlinearity = "sigmoid"
        nonlinearity = "tanh"

        # Install activation function
        self.act_fun = self._install_activation_function(nonlinearity)

        # Parameters
        self.parameters = []

        # Input layer weights
        size = (hidden_features, in_features)
        self.parameters.append(self._init_weights(size=size, nonlinearity=nonlinearity))

        # Hidden layer weights
        size = (hidden_features, hidden_features)
        for _ in range(num_hidden_layers):
            self.parameters.append(
                self._init_weights(size=size, nonlinearity=nonlinearity)
            )

        # Output layer weights
        size = (out_features, hidden_features)
        self.parameters.append(self._init_weights(size=size, nonlinearity="sigmoid"))

    def _install_activation_function(self, nonlinearity: str):
        """Installs activation function."""
        if nonlinearity == "leaky_relu":
            act_fun = lambda x: np.where(x > 0, x, 0.01 * x)
        elif nonlinearity == "sigmoid":
            act_fun = expit
        elif nonlinearity == "tanh":
            act_fun = np.tanh
        else:
            raise NotImplementedError(
                f"Initialization for '{nonlinearity}' not implemented."
            )
        return act_fun

    @staticmethod
    def _init_weights(size: tuple[int, int], nonlinearity: str) -> None:
        """Initializes model weights.

        Xavier normal initialization for feedforward neural networks described in
        'Understanding the difficulty of training deep feedforward neural networks'
        by Glorot and Bengio (2010).

            std = gain * (2 / (fan_in + fan_out)) ** 0.5

        """
        if nonlinearity == "leaky_relu":
            gain = math.sqrt(2.0 / (1.0 + 0.01**2))
        elif nonlinearity == "sigmoid":
            gain = 1.0
        elif nonlinearity == "tanh":
            gain = 5.0 / 3.0
        else:
            raise NotImplementedError(
                f"Initialization for '{nonlinearity}' not implemented."
            )
        std = gain * (2.0 / sum(size)) ** 0.5

        weights = np.random.normal(loc=0.0, scale=std, size=size)
        biases = np.zeros(shape=(size[0], 1))

        return weights, biases

    def __call__(self, x: numpy.ndarray):
        return self.forward(x)

    def eval(self):
        pass

    def forward(self, x: numpy.ndarray):
        """Feedforward state.

        Args:
            x: State of booster.
        """
        for weight, bias in self.parameters[:-1]:
            x = self.act_fun(np.matmul(x, weight.T) + bias.T)

        weight, bias = self.parameters[-1]
        x = expit(np.matmul(x, weight.T) + bias.T)[0, :]

        return x


class TorchNeuralNetwork(nn.Module):
    """Network class.

    Simple fully-connected neural network.

    Attributes:
        mutation_prob:
        mutation_rate:
        net:
    """

    def __init__(self, config: Config) -> None:
        """Initializes NeuralNetwork class."""
        super().__init__()

        config = config.env.booster.neural_network

        in_features = config.num_dim_in
        out_features = config.num_dim_out
        hidden_features = config.num_dim_hidden
        num_hidden_layers = config.num_hidden_layers

        layers = [
            nn.Flatten(start_dim=0),
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.Tanh(),
        ]

        for _ in range(num_hidden_layers):
            layers += [
                nn.Linear(in_features=hidden_features, out_features=hidden_features),
                nn.Tanh(),
            ]

        layers += [
            nn.Linear(in_features=hidden_features, out_features=out_features),
            nn.Sigmoid(),
        ]

        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

        # <Genetic optimization>
        self.mutation_prob = config.optimizer.mutation_probability
        self.mutation_rate = config.optimizer.mutation_rate
        # </Genetic optimization>

        # <Reinforcement learning (Deep Q-Learning)>
        self.epsilon = 1.0
        self.gamma = 0.99
        self.decay_rate = 0.005
        self.epsilon_min = 0.05
        self.memory_size = 1000
        self.memory = deque()
        self.num_states = (
            6  # State of booster (pos_x, pos_y, vel_x, vel_y, omega, angle)
        )
        self.num_actions = 9  # Three engines at maximum thrust at three angles
        # </Reinforcement learning (Deep Q-Learning)>

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.5)
            gain = 5.0 / 3.0  # Gain for tanh nonlinearity.
            torch.nn.init.xavier_normal_(module.weight, gain=gain)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    # <Genetic optimization>

    @torch.no_grad()
    def _mutate_weights(self, module: nn.Module) -> None:
        """Mutates weights."""
        if isinstance(module, nn.Linear):
            mask = torch.rand_like(module.weight) < self.mutation_prob
            mutation = self.mutation_rate * torch.randn_like(module.weight)
            module.weight.add_(mask * mutation)
            if module.bias is not None:
                mask = torch.rand_like(module.bias) < self.mutation_prob
                mutation = self.mutation_rate * torch.randn_like(module.bias)
                module.bias.add_(mask * mutation)

    def mutate_weights(self) -> None:
        """Mutates the network's weights."""
        self.apply(self._mutate_weights)

    # </Genetic optimization>

    # <Reinforcement learning (Deep Q-Learning)>

    def _select_action(self, state: torch.Tensor, epsilon: float = 0.9) -> int:
        """Selects action from a discrete action space.

        Args:
            epsilon: Probability of choosing a random action (epsilon-greedy value).

        Returns:
            Action to be performed by booster (Firing engine at certain angle).
        """
        if random.random() < epsilon:
            # Choose a random action
            action = random.randint(0, self.num_actions - 1)
        else:
            # Select action with highest predicted utility given state
            with torch.no_grad():
                self.eval()
                pred = self.forward(state)
                action = torch.argmax(pred)
                self.train()

        return action

    def _memorize(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        new_state: torch.Tensor,
        done,
    ) -> None:
        """Stores past events."""
        self.memory.append((state, action, reward, new_state, done))
        # Make sure that the memory is not exceeded.
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def _create_training_set(self, replay: deque):
        """"""
        # Select states and new states from replay
        states = torch.Tensor([memory[0] for memory in self.replay])
        new_states = torch.Tensor([memory[3] for memory in self.replay])

        # Predict expected utility (Q-value) of current state and new state
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

    # </Reinforcement learning (Deep Q-Learning)>

    @torch.no_grad()  # TODO: Test this
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)

        # Detach prediction from graph.
        x = x.detach().numpy().astype(np.float)

        return x
