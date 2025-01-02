import logging
import os
from typing import Optional

import numpy as np
import torch
from lightning.pytorch import seed_everything
from dotenv import load_dotenv

from omegaconf import DictConfig

MAX_SEED = 10000


class Seed:
    seed_history: dict[int, int] = {}
    next_history: int = 0

    @staticmethod
    def set_seed(seed: Optional[int] = None, workers: bool = True, history: Optional[int] = None):
        """Set seed for reproducibility.

        :param seed: seed value, defaults to None (random seed selected up to 10000)
        """
        if seed is None:
            if history is not None and history in Seed.seed_history:
                seed = Seed.seed_history[history]
            else:
                seed = np.random.randint(MAX_SEED)
        if history is None:
            history = Seed.next_history

        seed_everything(seed, workers=True)
        Seed.seed_history[history] = seed

        # Seed.next_history always tracks the highest history value + 1. When we log a new history, that new history
        # is either the new largest value (possibly tied), in which we take +1, or it is not, in which case next_history
        # is still the largest unused.
        Seed.next_history = max(Seed.next_history, history + 1)


set_seed = Seed.set_seed


def setup_environment(setup_config: DictConfig):
    load_dotenv()

    # set logging to prevent pytorch_lightning from printing seed logs on every change
    log = logging.getLogger("lightning.fabric.utilities.seed")
    log.propagate = False
    log.setLevel(logging.WARNING)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt=r"%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOG_LEVEL", "INFO"),
    )

    default_logger = logging.getLogger()
    default_logger.setLevel(os.getenv("LOG_LEVEL"))

    http_logger = logging.getLogger("httpx")
    http_logger.setLevel(os.getenv("LOG_LEVEL"))

    if not os.path.exists(setup_config.save_dir):
        os.makedirs(setup_config.save_dir, exist_ok=True)

    set_seed(setup_config.seed, history=0)

    if hasattr(setup_config, "precision"):
        torch.set_float32_matmul_precision(setup_config.precision)


def log_slurm(run):
    for var in os.environ:
        if var.startswith("SLURM"):
            run[f"slurm/{var.lower()}"] = os.environ[var]


def find_folder(prefix: str, options: list[str]) -> str:
    for opt in options:
        if os.path.exists(os.path.join(prefix, opt)):
            return os.path.join(prefix, opt)

    raise FileNotFoundError(
        f"Could not find a file or directory with any of the specified files or subfolders: {prefix=}, {options=}"
    )


def find_split_folder(prefix: str, split: str) -> str:
    match split:
        case "train":
            return find_folder(prefix, ["train", "Training", "Train", "training"])
        case "val":
            return find_folder(prefix, ["val", "Validation", "Val", "validation", "valid"])
        case "test":
            return find_folder(prefix, ["test", "Testing", "Test", "testing"])
        case "dev":
            return find_folder(prefix, ["dev", "Development", "Dev", "development", "develop"])
        case _:
            raise ValueError(f"Invalid split name. Must be one of 'train', 'val', 'test', or 'dev', but found: {split}")


def makedirs_for_file(file: str):
    os.makedirs(os.path.dirname(file), exist_ok=True)
