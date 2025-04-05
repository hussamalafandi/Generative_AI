import datetime
import hashlib
import json
import logging
import os

import torch
import yaml

import wandb
from zmq import device

logger = logging.getLogger(__name__)


def get_dict_hash(dictionary):
    # Convert the dictionary to a JSON string (sorted keys ensure consistent hash)
    dict_str = json.dumps(dictionary, sort_keys=True)
    # Generate a SHA-256 hash of the string
    return hashlib.sha256(dict_str.encode()).hexdigest()[:8]

def generate_checkpoint_folder(config):
    config_hash = get_dict_hash(config)

    base_path = config.get("checkpoint_base_dir", "checkpoints")
    os.makedirs(base_path, exist_ok=True)
    
    folder_name = f"{config.get('model_name', 'model')}_lr{config['lr']}_bs{config['batch_size']}_{config_hash}"
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    logger.info(f"Checkpoint folder created at: {folder_path}")
    return folder_path


# ------------------------------
# Checkpoint Saving & Loading
# ------------------------------
def save_checkpoint(folder, models, optimizers, epoch, config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"ckpt_epoch_{epoch}_{timestamp}.pth"
    checkpoint_path = os.path.join(folder, checkpoint_filename)
    state = {
        "epoch": epoch,
        "models_state_dict": {name: model.state_dict() for name, model in models.items()},
        "optimizers_state_dict": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "wandb_run_id": wandb.run.id if wandb.run else None,
        "config": config
    }
    torch.save(state, checkpoint_path)
    logger.info(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

    config_path = os.path.join(folder, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Configuration saved at {config_path}")

    # Log checkpoint as a WandB artifact with unique naming.
    artifact = wandb.Artifact(
        f"model-checkpoint-epoch-{epoch}-{timestamp}", type="model", description=f"Checkpoint at epoch {epoch}")
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)


def load_checkpoint(folder, models, optimizers, current_config):
    # Find all checkpoint files in the folder
    ckpt_files = [f for f in os.listdir(folder) if f.startswith(
        "ckpt_epoch_") and f.endswith(".pth")]
    
    if not ckpt_files:
        logger.info("No checkpoint files found; starting fresh.")
        return 0, None
    
    # Determine the latest checkpoint (highest epoch number)
    latest_ckpt = max(ckpt_files, key=lambda f: int(
        f.split("_")[-1].split(".")[0]))
    
    device = torch.device(
        "cuda" if current_config["use_cuda"] and torch.cuda.is_available() else "cpu")
    
    checkpoint_path = os.path.join(folder, latest_ckpt)
    state = torch.load(checkpoint_path, map_location=device)

    checkpoint_config = state.get("config", {})

    if checkpoint_config != current_config:
        logger.warning(
            "Configuration differences between checkpoint and current run:")
        for key in set(checkpoint_config.keys()).union(current_config.keys()):
            if checkpoint_config.get(key) != current_config.get(key):
                logger.warning(
                    f"  {key}: checkpoint={checkpoint_config.get(key)}, current={current_config.get(key)}")
        logger.warning("Configuration mismatch. Skipping checkpoint loading.")
        return 0, None

    for name, model in models.items():
        model.load_state_dict(state["models_state_dict"][name])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(state["optimizers_state_dict"][name])

    start_epoch = state["epoch"] + 1
    wandb_run_id = state.get("wandb_run_id", None)

    logger.info(
        f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")

    return start_epoch, wandb_run_id
