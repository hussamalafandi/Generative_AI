import datetime
import random

import numpy as np
import torch
import yaml


def generate_wandb_run_name(config):
    """Generate a run name based on the configuration and timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = config.get("wandb_project", "Experiment")
    return f"{base_name}_lr{config['lr']}_bs{config['batch_size']}_{timestamp}"


def get_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
