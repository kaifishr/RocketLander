"""Optimizer base class."""
from src.utils.config import Config
from src.environment import Environment


class Optimizer:
    def __init__(self, environment: Environment, config: Config) -> None:
        """Initializes optimizer base class."""

        self.stats = {"reward": 0.0, "loss": 0.0}

    def step(self) -> None:
        """Runs single optimization step."""

    def _broadcast_params(self) -> None:
        """Broadcasts set of network parameters to all agents."""
        for booster in self.boosters:
            # booster.model.parameters = copy.deepcopy(self.model.parameters)
            booster.model.load_state_dict(
                self.model.state_dict()
            )  # TODO: Test this method.
