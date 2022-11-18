"""Trainer class."""
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from src.utils.config import Config
from src.utils.utils import save_checkpoint
from src.environment import Environment

from .optimizer import SimulatedAnnealing 


class Trainer:
    """Trainer class.

    Trains neural network with specified optimization method.

    Attributes:
        config:
        env:
        writer:
    """

    def __init__(self, config: Config) -> None:
        """Initializes Trainer."""
        self.config = config
        self.writer = SummaryWriter() 

        # Create UUID for PyGame window.
        self.config.id = self.writer.log_dir.split("/")[-1]

        self.env = Environment(config=config)
        self.optimizer = SimulatedAnnealing(environment=self.env, config=config)

        # Save config file
        file_path = Path(self.writer.log_dir) / "config.txt"
        with open(file_path, "w") as file:
            file.write(self.config.__str__())

    def run(self) -> None:
        """Runs training."""

        num_generations = self.config.trainer.num_generations
        num_simulation_steps = self.config.optimizer.num_simulation_steps

        step = 0
        generation = 0
        max_reward = 0.0

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
                generation += 1

                reward = self.optimizer.reward

                # Write stats to Tensorboard.
                self.writer.add_scalar("Reward", reward, generation)
                self.writer.add_scalar("Seconds", time.time() - t0, generation)
                print(f"{generation = }")

                # Save model
                if self.config.checkpoints.save_model:
                    if reward > max_reward:
                        model = self.env.drones[self.optimizer.idx_best].model
                        save_checkpoint(
                            model=model, config=self.config, generation=generation
                        )
                        max_reward = reward 

                t0 = time.time()

                if generation == num_generations:
                    is_running = False


            step += 1
