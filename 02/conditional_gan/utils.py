import datetime

import yaml


def generate_run_name(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.get("run_name", f"DCGAN_CelebA_lr{config['lr']}_bs{config['batch_size']}_{timestamp}")

def get_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)