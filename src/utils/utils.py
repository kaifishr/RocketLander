"""Script with helper functions."""
import random
from pathlib import Path

import numpy
import pygame
import torch

from src.utils.config import Config


def set_random_seed(seed: int = 0) -> None:
    """Sets random seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def capture_screen(screen, iteration: int) -> None:
    """Captures screen every simulation step.

    Args:
        screen:
        iteration:

    Returns:
        None
    """
    filename = Path("frames") / f"screen_{iteration:04d}"
    pygame.image.save(screen, filename)


def save_checkpoint(model: torch.nn.Module, config: Config, generation: int) -> None:
    """Saves model checkpoint.

    Args:
        model:
        ckpt_dir:
        model_name:
    """
    model_name = f"weights_{generation}"
    model_path = Path(config.checkpoints.model_path) / f"{model_name}.pth"
    torch.save(obj=model.state_dict(), f=model_path)


def load_checkpoint(model: torch.nn.Module, config: Config) -> None:
    """Loads model from checkpoint.

    Args:
        model:
        ckpt_dir:
        model_name:
    """
    model_name = "weights"
    model_path = Path(config.checkpoints.model_path) / f"{model_name}.pth"
    if model_path.is_file():
        state_dict = torch.load(f=model_path)
        model.load_state_dict(state_dict=state_dict)
    else:
        print(f"Model '{model_name}' not found. Continuing with random weights.")
