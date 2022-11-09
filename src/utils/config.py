"""Class to hold configuration."""
from pathlib import Path

import yaml


class Config:
    """Configuration class.
    Class creates nested configuration for parameters used
    in different modules during training.
    """

    def __init__(self, d: dict = None) -> None:
        """Initializes config class."""
        self.merge_dict(d)

    @staticmethod
    def _merge_dict(self: object, d: object) -> None:
        if d is not None:
            for key, value in d.items():
                if isinstance(value, dict):
                    if not hasattr(self, key):
                        self.__setattr__(key, Config())
                    self._merge_dict(self=self.__getattribute__(key), d=value)
                else:
                    self.__setattr__(key, value)

    def merge_dict(self, d: dict) -> None:
        self._merge_dict(self, d)

    def __str__(self) -> str:
        """Prints nested config."""
        cfg = []
        self._build_str(self, cfg)
        return "".join(cfg)

    @staticmethod
    def _build_str(self: object, cfg: list, indent: int = 0) -> None:
        """Recursively iterates through all configuration nodes."""
        for key, value in self.__dict__.items():
            indent_ = 4 * indent * " "
            if isinstance(value, Config):
                cfg.append(f"{indent_}{key}\n")
                self._build_str(
                    self=self.__getattribute__(key), cfg=cfg, indent=indent + 1
                )
            else:
                cfg.append(f"{indent_}{key}: {value}\n")


def load_config(path: str) -> Config:
    """Loads configuration file.

    Args:
        path: Path to yaml file.

    Returns:
        Dictionary holding content of yaml file.

    """
    with open(path, "r") as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def init_config(path: str) -> Config:
    """Initializes configuration class.

    Args:
        file_path: File to configuration file.

    Returns:
        Config class.
    """
    # Loads yaml file as dictionary.
    config = load_config(path=path)

    # Convert dictionary to configuration class.
    config = Config(d=config)
    print(config)

    # Create folder if not exist.
    Path(config.checkpoints.model_path).mkdir(parents=True, exist_ok=True)
    Path(config.dirs.frames).mkdir(parents=True, exist_ok=True)

    return config
