import datetime
import random

import numpy as np
import torch
import yaml


def generate_run_name(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.get("run_name", f"DCGAN_CelebA_lr{config['lr']}_bs{config['batch_size']}_{timestamp}")


def get_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_random_seed(seed):
    """Set random seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
