import argparse
import os

import torch
from genai_core.models.rnns.lstm import LSTMLanguageModel
from genai_core.utils.checkpoint import generate_checkpoint_folder, load_checkpoint
from genai_core.data.text_loader import create_dataloader
from genai_core.utils.logger import setup_logger
from genai_core.trainers.lstm_trainer import LSTMTrainer
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
    config["checkpoint_folder"] = checkpoint_folder

    # Set up a logger that writes to a file in the checkpoint folder.
    log_file = os.path.join(checkpoint_folder, "experiment.log")
    logger = setup_logger(log_file)

    logger.info("\n" + "=" * 80)
    logger.info("Logger is set up.")

    # Load pre-trained embeddings if specified
    pretrained_embeddings = None
    if config["pretrained_embeddings_path"]:
        logger.info("Loading pre-trained embeddings from Hugging Face: %s", config["pretrained_embeddings_path"])
        from transformers import AutoModel
        embedding_model = AutoModel.from_pretrained(config["pretrained_embeddings_path"])
        pretrained_embeddings = embedding_model.get_input_embeddings().weight.detach().clone()

    # Initialize model and optimizer
    model = LSTMLanguageModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        pretrained_embeddings=pretrained_embeddings,
        freeze=config["freeze_embeddings"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()

    dataloader, tokenizer = create_dataloader(
        dataset_name=config["dataset_name"],
        tokenizer_name=config["tokenizer_name"],
        block_size=config["block_size"],
        batch_size=config["batch_size"],
        shuffle=True,
        mlm=False,
        split=config["split"],
        num_workers=config["num_workers"],
    )

    start_epoch = 0
    run_id = None
    if os.path.exists(checkpoint_folder) and config.get("resume_training", True):
        start_epoch, run_id = load_checkpoint(checkpoint_folder, {"model": model}, {"optimizer": optimizer}, config)

    # Generate a meaningful run name if not provided in config
    run_name = generate_wandb_run_name(config)
    tags = config.get("tags", [])

    wandb.init(
        project=config["wandb_project"],
        config=config,
        resume="allow",
        id=run_id,
        name=run_name,
        tags=tags,
        mode=config["wandb_mode"],
        save_code=True,
    )
    wandb.watch(model, log="all")

    lstm_trainer = LSTMTrainer(config, model, optimizer, criterion)
    lstm_trainer.tokenizer = tokenizer

    try:
        lstm_trainer.train(dataloader, start_epoch)

    except KeyboardInterrupt:
        logger.error("Training interrupted by user (KeyboardInterrupt).", exc_info=True)
        wandb.log({"error": "Training interrupted by user (KeyboardInterrupt)."})
        raise

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}", exc_info=True)
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
    parser = argparse.ArgumentParser(description="Train an LSTM language model")
    parser.add_argument("--config", type=str, default="./lstm_config.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    config = get_config(args.config)

    main(config)
