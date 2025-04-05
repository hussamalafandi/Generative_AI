import argparse
import os

import torch
from genai_core import models
from genai_core.models.gans.c_gan import Discriminator, Generator
from genai_core.utils.checkpoint import (generate_checkpoint_folder, load_checkpoint,
                        save_checkpoint)
from genai_core.data.image_loader import get_image_dataloader
from genai_core.utils.logger import setup_logger
from genai_core.trainers.c_gan_trainer import ConditionalGANTrainer
from genai_core.utils.utils import generate_wandb_run_name, get_config, set_random_seed

import wandb


def main(config):
    if config["use_cuda"] and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but 'use_cuda' is set to True.")

    # Set the random seed
    set_random_seed(config["seed"])

    device = torch.device(
        "cuda" if config["use_cuda"] and torch.cuda.is_available() else "cpu")

    # Create a unique checkpoint folder based on hyperparameters
    checkpoint_folder = generate_checkpoint_folder(config)
    config["checkpoint_dir"] = checkpoint_folder

    # Set up a logger that writes to a file in the checkpoint folder.
    log_file = os.path.join(checkpoint_folder, "experiment.log")
    logger = setup_logger(log_file)

    logger.info("\n" + "=" * 80)
    logger.info("Logger is set up.")

    # Initialize models and optimizers
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    models = {"generator": generator,
             "discriminator": discriminator}

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config["lr"], betas=(
        config["beta1"], 0.999), weight_decay=config["weight_decay"])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr"], betas=(
        config["beta1"], 0.999), weight_decay=config["weight_decay"])
    optimizers = {"optimizer_g": optimizer_g,
                 "optimizer_d": optimizer_d}

    dataloader = get_image_dataloader(config, split="train")

    start_epoch = 0
    run_id = None
    if os.path.exists(checkpoint_folder) and config.get("resume_training", True):
        start_epoch, run_id = load_checkpoint(checkpoint_folder, models, optimizers, config)

    # Generate a meaningful run name if not provided in config
    run_name = generate_wandb_run_name(config)
    tags = config.get("tags", [])

    wandb.init(project=config["wandb_project"], config=config,
               resume="allow", id=run_id, name=run_name, tags=tags, mode=config["wandb_mode"], save_code=True)
    # wandb.run.log_code('../')
    wandb.watch([generator, discriminator], log="all")  # Combine watch calls


    conditional_gan_trainer = ConditionalGANTrainer(config, generator, discriminator, optimizer_g, optimizer_d)

    try:
        conditional_gan_trainer.train(dataloader, start_epoch)

    except KeyboardInterrupt:
        logger.error(
            "Training interrupted by user (KeyboardInterrupt).", exc_info=True)
        wandb.log(
            {"error": "Training interrupted by user (KeyboardInterrupt)."})
        raise

    except Exception as e:
        logger.error(
            f"An error occurred during training: {str(e)}", exc_info=True)
        wandb.log({"error": str(e)})
        raise

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

    config = get_config(args.config)

    main(config)
