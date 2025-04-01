import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm

# ------------------------------
# Define the Generator Network
# ------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.net(z)
        img = img.view(z.size(0), 1, 28, 28)
        return img

# ------------------------------
# Define the Discriminator Network
# ------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.net(img_flat)
        return validity

# ------------------------------
# Checkpoint Saving & Loading
# ------------------------------
def save_checkpoint(path, generator, discriminator, optimizer_G, optimizer_D, epoch, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "wandb_run_id": wandb.run.id if wandb.run else None,
        "config": config  # Save the config used for training
    }
    torch.save(state, path)
    print(f"Checkpoint saved at epoch {epoch} to {path}")

def load_checkpoint(path, generator, discriminator, optimizer_G, optimizer_D, device, current_config):
    state = torch.load(path, map_location=device)
    
    checkpoint_config = state.get("config", {})

    if checkpoint_config != current_config:
        raise ValueError("Configuration mismatch between checkpoint and current run.")
    
    generator.load_state_dict(state["generator_state_dict"])
    discriminator.load_state_dict(state["discriminator_state_dict"])
    optimizer_G.load_state_dict(state["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(state["optimizer_D_state_dict"])

    start_epoch = state["epoch"] + 1
    wandb_run_id = state.get("wandb_run_id", None)

    print(f"Loaded checkpoint from {path}, resuming from epoch {start_epoch}")

    return start_epoch, wandb_run_id

# ------------------------------
# Training Epoch Function
# ------------------------------
def train_epoch(generator, discriminator, optimizer_G, optimizer_D, dataloader, device, config, epoch, global_step):
    generator.train()
    discriminator.train()
    
    # Wrap dataloader with tqdm for progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{config['epochs']}", unit="batch", leave=False)
    for i, (imgs, _) in enumerate(progress_bar):
        batch_size = imgs.size(0)
        imgs = imgs.to(device)

        # Define adversarial ground truths
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, config["latent_dim"], device=device)
        gen_imgs = generator(z)
        # Loss: generator tries to fool the discriminator
        g_loss = F.binary_cross_entropy(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = F.binary_cross_entropy(discriminator(imgs), valid)
        fake_loss = F.binary_cross_entropy(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Update tqdm progress bar with current losses
        progress_bar.set_postfix(g_loss=g_loss.item(), d_loss=d_loss.item())

        # Log training losses
        if i % config["log_interval"] == 0:
            wandb.log({
                "g_loss": g_loss.item(),
                "d_loss": d_loss.item(),
                "epoch": epoch,
                "batch": i
            }, step=global_step)
        
        global_step += 1

    return global_step

# ------------------------------
# Evaluation Function
# ------------------------------
def evaluate(generator, device, config, step):
    """Generates images from fixed noise and logs them to WandB."""
    generator.eval()
    with torch.no_grad():
        fixed_noise = torch.randn(config["num_eval_samples"], config["latent_dim"], device=device)
        gen_imgs = generator(fixed_noise)
        # Create a grid of generated images
        grid = torchvision.utils.make_grid(gen_imgs, nrow=4, normalize=True)
        wandb.log({"generated_images": [wandb.Image(grid, caption="Generated Images")]}, step=step)
    generator.train()

# ------------------------------
# DataLoader Initialization
# ------------------------------
def get_dataloader(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root=config["data_root"], train=True, transform=transform, download=True)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# ------------------------------
# Main Training Function
# ------------------------------
def main(config):
    if config["use_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but 'use_cuda' is set to True in the configuration.")
    
    device = torch.device("cuda" if config["use_cuda"] and torch.cuda.is_available() else "cpu")

    # Initialize models and optimizers
    generator = Generator(config["latent_dim"]).to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config["lr"], betas=(config["beta1"], 0.999), weight_decay=config["weight_decay"])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config["lr"], betas=(config["beta1"], 0.999), weight_decay=config["weight_decay"])
    
    dataloader = get_dataloader(config)

    # If a checkpoint exists, load it and extract the wandb run id
    start_epoch = 0
    run_id = None
    if os.path.exists(config["checkpoint_path"]):
        start_epoch, run_id = load_checkpoint(config["checkpoint_path"], generator, discriminator, optimizer_G, optimizer_D, device, config)

    # Resume the WandB run if possible. Using resume="allow" tells WandB to try to reconnect to the previous run.
    wandb.init(project=config["wandb_project"], config=config, resume="allow", id=run_id)
    wandb.save(__file__)  # Save the current script to WandB
    wandb.watch(generator, log="all")
    wandb.watch(discriminator, log="all")

    # Initialize global step (set based on previous training if resuming)
    global_step = start_epoch * len(dataloader)

    # Training Loop: Pass dependencies explicitly to each function
    for epoch in range(start_epoch, config["epochs"]):
        global_step = train_epoch(generator, discriminator, optimizer_G, optimizer_D, dataloader, device, config, epoch, global_step)

        if epoch % config["checkpoint_interval"] == 0:
            evaluate(generator, device, config, step=global_step)
            save_checkpoint(config["checkpoint_path"], generator, discriminator, optimizer_G, optimizer_D, epoch, config)

    wandb.finish()

def get_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ------------------------------
# Main Execution: Load Config & Start Training
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    config = get_config(args.config)

    main(config)
