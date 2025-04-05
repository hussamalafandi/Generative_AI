from typing import override

import torch
import torch.nn.functional as F
import wandb
from numpy import indices
from torch import ge, nn
from tqdm.auto import tqdm

from genai_core.trainers.base_trainer import BaseTrainer
from genai_core.utils.checkpoint import save_checkpoint


class ConditionalGANTrainer(BaseTrainer):
    def __init__(self, config, generator, discriminator, optimizer_g, optimizer_d):
        super().__init__(config)
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d

        self.noise_dim = config["latent_dim"]
        self.num_classes = config["num_classes"]
        self.num_eval_samples = config.get("num_eval_samples", 10)


        self.criterion = torch.nn.BCELoss()

    @override
    def train_epoch(self, dataloader, epoch):
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0
        n_batches = 0

        # Precompute adversarial ground truths
        valid = torch.ones(self.batch_size, device=self.device)
        fake = torch.zeros(self.batch_size, device=self.device)

        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch}/{self.epochs}", unit="batch", leave=False)
        for i, (imgs, labels) in enumerate(progress_bar):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            batch_size = imgs.size(0)
            n_batches += 1

            # Adjust valid and fake tensors for the current batch size
            valid = valid[:batch_size]
            fake = fake[:batch_size]

            # -----------------
            #  Train Generator
            # -----------------
            self.optimizer_g.zero_grad()
            noise = torch.randn(batch_size, self.noise_dim,
                                1, 1, device=self.device)
            gen_imgs = self.generator(noise, labels)
            g_loss = F.binary_cross_entropy(
                self.discriminator(gen_imgs, labels), valid)
            g_loss.backward()
            self.optimizer_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.optimizer_d.zero_grad()
            real_loss = F.binary_cross_entropy(
                self.discriminator(imgs, labels), valid)
            fake_loss = F.binary_cross_entropy(
                self.discriminator(gen_imgs.detach(), labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            self.optimizer_d.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            progress_bar.set_postfix(
                g_loss=g_loss.item(), d_loss=d_loss.item())

            if self.log_interval != -1 and i % self.log_interval == 0:
                wandb.log({
                    "batch_g_loss": g_loss.item(),
                    "batch_d_loss": d_loss.item(),
                    "epoch": epoch,
                    "batch": i
                })

        # Compute average losses for the epoch
        avg_g_loss = total_g_loss / n_batches
        avg_d_loss = total_d_loss / n_batches
        wandb.log({
            "epoch_avg_g_loss": avg_g_loss,
            "epoch_avg_d_loss": avg_d_loss,
            "epoch": epoch
        })

        self.logger.info(
            f"Epoch {epoch} finished: avg_g_loss={avg_g_loss:.4f}, avg_d_loss={avg_d_loss:.4f}")

    @override
    def train(self, dataloader, start_epoch):
        for epoch in range(start_epoch, self.epochs):
            self.train_epoch(dataloader, epoch)

            if epoch % self.checkpoint_interval == 0 or epoch == self.epochs - 1:
                self.generate_samples()
                
                # Save checkpoint
                models = {
                    "generator": self.generator,
                    "discriminator": self.discriminator
                }
                optimizers = {
                    "optimizer_g": self.optimizer_g,
                    "optimizer_d": self.optimizer_d
                }
                save_checkpoint(self.checkpoint_folder, models, optimizers,
                                epoch, self.config)

    def generate_samples(self):
        self.generator.eval()
        noise = torch.randn(self.num_eval_samples,
                            self.noise_dim, 1, 1, device=self.device)
        labels = torch.randint(0, self.num_classes,
                               (self.num_eval_samples,), device=self.device)

        with torch.no_grad():
            gen_imgs = self.generator(noise, labels)

        gen_imgs = gen_imgs.cpu().numpy()
        labels = labels.cpu().numpy()

        if self.num_eval_samples > 5:
            indices = torch.randperm(self.num_eval_samples)[:5]
        else:
            indices = torch.arange(self.num_eval_samples)

        gen_imgs = gen_imgs[indices]
        labels = labels[indices]

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))

        for i, ax in enumerate(axes):
            img = gen_imgs[i].transpose(1, 2, 0)
            img = (img + 1) / 2

            label = labels[i].item()

            ax.imshow(img, cmap='gray')
            ax.set_title(f"Label: {label}")
            ax.axis('off')

        wandb.log({"generated_images": wandb.Image(
            fig, caption="Generated images")})
        plt.close(fig)
        self.generator.train()
