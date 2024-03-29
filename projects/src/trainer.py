"""Trainer class."""
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from src.utils.config import Config
from src.utils.utils import save_checkpoint
from src.environment import Environment

from projects.src.optimizer import Optimizer


class Trainer:
    """Trainer class.

    Trains neural network with specified optimization method.

    Attributes:
        config:
        env:
        writer:
    """

    def __init__(self, optimizer: Optimizer, config: Config) -> None:
        """Initializes Trainer."""
        self.config = config
        self.writer = SummaryWriter()

        # Get UUID for PyGame window.
        self.config.id = self.writer.log_dir.split("/")[-1]

        self.env = Environment(config=config)
        self.optimizer = optimizer(environment=self.env, config=config)

        # Save config file
        file_path = Path(self.writer.log_dir) / "config.txt"
        with open(file_path, "w") as file:
            file.write(self.config.__str__())

    def run(self) -> None:
        """Runs training."""

        config = self.config
        num_episodes = self.config.trainer.num_episodes
        num_simulation_steps = self.config.optimizer.num_simulation_steps
        save_model = self.config.checkpoints.save_model
        every_num_episodes = self.config.checkpoints.every_num_episodes

        step = 0
        episode = 0

        is_running = True
        t0 = time.time()

        while is_running:

            # Step the environment.
            self.env.step_()

            # Method that run at end of simulation epoch
            if ((step + 1) % num_simulation_steps == 0) or not self.env.is_active():

                self.optimizer.step()

                # Reset environment to start over again.
                self.env.reset()

                step = 0
                episode += 1

                # Write stats to Tensorboard.
                for name, scalar in self.optimizer.stats.items():
                    if scalar is not None:
                        self.writer.add_scalar(name, scalar, episode)
                self.writer.add_scalar("seconds", time.time() - t0, episode)
                print(f"{episode = }", end="\r")

                # Save model
                if save_model and (episode % every_num_episodes == 0):
                    model = self.optimizer.model
                    save_checkpoint(model=model, config=config)

                t0 = time.time()

                if episode == num_episodes:
                    is_running = False

            step += 1
