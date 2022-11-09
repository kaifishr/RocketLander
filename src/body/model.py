"""Contains neural network definitions.

The drone's brain represented by a feedforward neural network.
"""
import numpy
import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit

from src.utils.config import Config
from src.utils.utils import load_checkpoint


class NetworkLoader:
    """Loads NumPy or PyTorch neural network."""

    def __init__(self, config: Config) -> None:
        """Initializes Network wrapper."""
        self.config = config

        # Compute normalization parameter for input data
        x_min = config.env.domain.limit.x_min
        x_max = config.env.domain.limit.x_max
        y_min = config.env.domain.limit.y_min
        y_max = config.env.domain.limit.y_max

        domain_diam_x = x_max - x_min
        domain_diam_y = y_max - y_min

        self.normalizer = 1.0 / (domain_diam_x**2 + domain_diam_y**2) ** 0.5

    def __call__(self):
        """Loads and returns model.

        Args:
            lib: Library "numpy" or "pytorch".
        """
        lib = self.config.optimizer.lib

        if lib == "numpy":
            model = NumpyNeuralNetwork(self.config, normalizer=self.normalizer)
            if self.config.checkpoints.load_model:
                raise NotImplementedError(
                    f"Loading checkpoints not implemented for NumPy neural networks."
                )

        elif lib == "torch":
            model = TorchNeuralNetwork(self.config, normalizer=self.normalizer)
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

    def __init__(self, config: Config, normalizer: float) -> None:
        """Initializes NeuralNetwork."""

        self.normalizer = normalizer

        self.mutation_prob = config.optimizer.mutation_probability
        self.mutation_rate = config.optimizer.mutation_rate

        config = config.env.drone.neural_network

        in_features = config.num_dim_in
        out_features = config.num_dim_out
        hidden_features = config.num_dim_hidden
        num_hidden_layers = config.num_hidden_layers

        # Input layer weights
        size = (hidden_features, in_features)
        self.weights = [self._init_weights(size=size, nonlinearity="tanh")]
        self.biases = [np.zeros(shape=(hidden_features, 1))]

        # Hidden layer weights
        size = (hidden_features, hidden_features)
        for _ in range(num_hidden_layers):
            self.weights += [self._init_weights(size=size, nonlinearity="tanh")]
            self.biases += [np.zeros(shape=(hidden_features, 1))]

        # Output layer weights
        size = (out_features, hidden_features)
        self.weights += [self._init_weights(size=size, nonlinearity="sigmoid")]
        self.biases += [np.zeros(shape=(out_features, 1))]

    @staticmethod
    def _init_weights(size: tuple[int, int], nonlinearity: str) -> None:
        """Initializes model weights.

        Xavier normal initialization for feedforward neural networks described in
        'Understanding the difficulty of training deep feedforward neural networks'
        by Glorot and Bengio (2010).

            std = gain * (2 / (fan_in + fan_out)) ** 0.5

        """
        if nonlinearity == "tanh":
            gain = 5.0 / 3.0
        elif nonlinearity == "sigmoid":
            gain = 1.0
        else:
            raise NotImplementedError(
                f"Initialization for '{nonlinearity}' not implemented."
            )
        std = gain * (2.0 / sum(size)) ** 0.5
        return np.random.normal(loc=0.0, scale=std, size=size)

    def __call__(self, x: numpy.ndarray):
        return self.forward(x)

    def mutate_weights(self) -> None:
        """Mutates the network's weights."""
        for weight, bias in zip(self.weights, self.biases):
            mask = numpy.random.random(size=weight.shape) < self.mutation_prob
            mutation = self.mutation_rate * numpy.random.normal(size=weight.shape)
            weight[:] = weight[:] + mask * mutation
            # weight += mask * mutation

            mask = numpy.random.random(size=bias.shape) < self.mutation_prob
            mutation = self.mutation_rate * numpy.random.normal(size=bias.shape)
            bias[:] = bias[:] + mask * mutation
            # bias += mask * mutation

    @staticmethod
    def _sigmoid(x: numpy.ndarray) -> numpy.ndarray:
        # return 1.0 / (1.0 + np.exp(-x))
        return expit(x)  # Numerically stable sigmoid

    def eval(self):
        pass

    def forward(self, data: list):

        # Normalize data
        x = self.normalizer * np.array(data)

        # Feedforward
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(np.matmul(x, weight.T) + bias.T)
        x = self._sigmoid(np.matmul(x, self.weights[-1].T) + self.biases[-1].T)[0, :]

        return x


class TorchNeuralNetwork(nn.Module):
    """Network class.

    Simple fully-connected neural network.

    Attributes:
        mutation_prob:
        mutation_rate:
        net:
    """

    def __init__(self, config: Config, normalizer: float) -> None:
        """Initializes NeuralNetwork class."""
        super().__init__()

        self.normalizer = normalizer

        self.mutation_prob = config.optimizer.mutation_probability
        self.mutation_rate = config.optimizer.mutation_rate

        config = config.env.drone.neural_network

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

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.5)
            gain = 5.0 / 3.0  # Gain for tanh nonlinearity.
            torch.nn.init.xavier_normal_(module.weight, gain=gain)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @torch.no_grad()  # TODO: add to NetLab as regularization tool if not used here.
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

    def forward(self, data: list) -> torch.Tensor:

        # Normalize data.
        x = self.normalizer * torch.tensor(data)

        # Feedforward.
        x = self.net(x)

        # Detach prediction from graph.
        x = x.detach().numpy().astype(np.float)

        return x
