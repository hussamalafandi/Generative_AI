import argparse
import os

import torch
import wandb
import yaml
from checkpoint import (generate_checkpoint_folder, load_checkpoint,
                        save_checkpoint)
from logger import setup_logger
from c_gan import Discriminator, Generator
from train import evaluate, train_epoch
from utils import generate_run_name

from dataProcessor import get_dataloader


def main(config):
    if config["use_cuda"] and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but 'use_cuda' is set to True.")

    device = torch.device(
        "cuda" if config["use_cuda"] and torch.cuda.is_available() else "cpu")

    # Create a unique checkpoint folder based on hyperparameters
    checkpoint_folder = generate_checkpoint_folder(config)

    # Set up a logger that writes to a file in the checkpoint folder.
    log_file = os.path.join(checkpoint_folder, "experiment.log")
    logger = setup_logger(log_file)

    logger.info("\n" + "=" * 80)
    logger.info("Logger is set up.")


    # Initialize models and optimizers
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config["lr"], betas=(
        config["beta1"], 0.999), weight_decay=config["weight_decay"])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr"], betas=(
        config["beta1"], 0.999), weight_decay=config["weight_decay"])

    dataloader = get_dataloader(config)

    start_epoch = 0
    run_id = None
    global_step = 0
    if os.path.exists(checkpoint_folder) and config.get("resume_training", True):
        start_epoch, global_step, run_id = load_checkpoint(
            checkpoint_folder, generator, discriminator, optimizer_g, optimizer_d, device, config)

    # Generate a meaningful run name if not provided in config
    run_name = generate_run_name(config)
    tags = config.get("tags", [])

    wandb.init(project=config["wandb_project"], config=config,
               resume="allow", id=run_id, name=run_name, tags=tags, mode=config["wandb_mode"])
    wandb.watch(generator, log="all")
    wandb.watch(discriminator, log="all")

    try:
        for epoch in range(start_epoch, config["epochs"]):
            global_step = train_epoch(generator, discriminator, optimizer_g,
                                      optimizer_d, dataloader, device, config, epoch, global_step)

            if epoch % config["checkpoint_interval"] == 0 or epoch == config["epochs"] - 1:
                evaluate(generator, device, config, step=global_step)
                save_checkpoint(checkpoint_folder, generator, discriminator,
                                optimizer_g, optimizer_d, epoch, global_step, config)
    finally:
        wandb.finish()
        logger.info("WandB run finished.")
        logger.info("\n" + "=" * 80 + "\n\n")


# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DCGAN on CelebA")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
