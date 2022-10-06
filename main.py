from src.config import load_config
from src.optimizer import Optimizer


def main():
    config = load_config(path="config.yml")
    print(config)
    optimizer = Optimizer(config=config)
    optimizer.optimize()
    # optimizer.run()


if __name__ == "__main__":
    main()
