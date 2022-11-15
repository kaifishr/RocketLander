"""Optimizer class for genetic optimization."""
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from src.utils.config import Config
from src.utils.utils import save_checkpoint
from src.environment import Environment


class Optimizer:
    """Optimizer class.

    Optimizer uses genetic optimization.

    Attributes:
        config:
        env:
        writer:
    """

    def __init__(self, config: Config) -> None:
        """Initializes Optimizer"""
        self.config = config
        self.writer = SummaryWriter()  # move to trainer

        # Create UUID for PyGame window.
        self.config.id = self.writer.log_dir.split("/")[-1]

        self.env = Environment(config=config)

        # Save config file
        file_path = Path(self.writer.log_dir) / "config.txt"
        with open(file_path, "w") as file:
            file.write(self.config.__str__())

    def run(self) -> None:
        """Runs genetic optimization."""

        num_max_steps = self.config.optimizer.num_max_steps
        step = 0
        generation = 0
        best_score = 0.0

        is_running = True
        t0 = time.time()

        while is_running:

            # Physics and rendering
            self.env.step()

            # Detect landing 
            # Turns off engines
            # self.env.detect_landing()

            # Detect high stresses
            self.env.detect_stress()

            # Detect collision with ground
            self.env.detect_impact()

            # Detect leaving the domain
            self.env.detect_escape()

            # Compute current fitness / score of booster
            self.env.comp_score()

            # Fetch data of each booster used for neural network
            self.env.fetch_data()

            # Run neural network prediction
            self.env.comp_action()

            # Apply network predictions to drone
            self.env.apply_action()

            # Method that run at end of simulation.
            if ((step + 1) % num_max_steps == 0) or self.env.is_active():

                # Select fittest agent based on distance traveled.
                score = self.env.select()

                # Reproduce and mutate weights of best agent.
                self.env.mutate()

                # Reset drones to start over again.
                self.env.reset()

                step = 0
                generation += 1

                # Write stats to Tensorboard.
                self.writer.add_scalar("score", score, generation)
                self.writer.add_scalar("seconds", time.time() - t0, generation)
                print(f"{generation = }")

                # Save model
                if self.config.checkpoints.save_model:
                    if score > best_score:
                        model = self.env.drones[self.env.idx_best].model
                        save_checkpoint(
                            model=model, config=self.config, generation=generation
                        )
                        best_score = score

                t0 = time.time()

            step += 1
