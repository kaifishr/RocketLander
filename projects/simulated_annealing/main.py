"""Runs the simulated annealing."""
import pathlib

from src.utils.config import init_config
from src.utils.utils import set_random_seed

from projects.src.trainer import Trainer
from projects.src.optimizer import AsyncSimulatedAnnealing
from projects.src.optimizer import SimulatedAnnealing


if __name__ == "__main__":
    file_name = "config.yml"
    file_dir = pathlib.Path(__file__).parent.resolve()
    file_path = file_dir / file_name
    config = init_config(path=file_path)
    set_random_seed(seed=config.random_seed)
    trainer = Trainer(optimizer=AsyncSimulatedAnnealing, config=config)
    # trainer = Trainer(optimizer=SimulatedAnnealing, config=config)
    trainer.run()
