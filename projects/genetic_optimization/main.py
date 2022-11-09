"""Runs the genetic optimization."""
from src.utils.config import init_config
from src.utils.utils import set_random_seed
from src.optimizer import Optimizer


if __name__ == "__main__":
    config = init_config(path="config.yml")
    set_random_seed(seed=config.random_seed)
    optimizer = Optimizer(config=config)
    optimizer.run()
