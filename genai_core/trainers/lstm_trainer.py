import torch
from torch import nn
from tqdm.auto import tqdm
import wandb

from genai_core.trainers.base_trainer import BaseTrainer
from genai_core.utils.checkpoint import save_checkpoint

from typing import override


class LSTMTrainer(BaseTrainer):
    def __init__(self, config, model, optimizer, criterion):
        super().__init__(config)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.vocab_size = model.vocab_size

    @override
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch}/{self.epochs}", unit="batch", leave=False
        )
        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = self.criterion(outputs.view(-1, self.vocab_size), labels.view(-1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            progress_bar.set_postfix(loss=loss.item())

            if self.log_interval != -1 and i % self.log_interval == 0:
                wandb.log({"training/batch_loss": loss.item(), "training/epoch": epoch, "training/batch": i})

        avg_loss = total_loss / n_batches
        wandb.log({"training/epoch_avg_loss": avg_loss, "training/epoch": epoch})
        self.logger.info(f"Epoch {epoch} finished: avg_loss={avg_loss:.4f}")

    @override
    def train(self, dataloader, start_epoch):
        for epoch in range(start_epoch, self.epochs):
            self.train_epoch(dataloader, epoch)

            if epoch % self.checkpoint_interval == 0 or epoch == self.epochs - 1:
                # Evaluate the model at the end of each epoch
                self.evaluate(dataloader)

                # Generate a sample text after each epoch
                self.generate_text(
                    tokenizer=self.tokenizer,
                    start_sequence="Once upon a time",
                    max_length=50,
                    temperature=1.0
                )

                # Save checkpoint
                models = {"model": self.model}
                optimizers = {"optimizer": self.optimizer}
                save_checkpoint(self.config['checkpoint_folder'], models, optimizers, epoch, self.config)

    def evaluate(self, dataloader):
        """
        Evaluate the model on a validation or test dataset.

        Args:
            dataloader (DataLoader): DataLoader for the evaluation dataset.

        Returns:
            float: Average loss over the evaluation dataset.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", unit="batch", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids)
                loss = self.criterion(outputs.view(-1, self.vocab_size), labels.view(-1))

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        self.logger.info(f"Evaluation finished: avg_loss={avg_loss:.4f}")
        wandb.log({"evaluation/avg_loss": avg_loss})
        return avg_loss

    def generate_text(self, tokenizer, start_sequence, max_length=50, temperature=1.0):
        """
        Generate text using the trained LSTM model.

        Args:
            tokenizer (AutoTokenizer): Tokenizer used for encoding and decoding text.
            start_sequence (str): Initial text to start the generation.
            max_length (int): Maximum length of the generated text.
            temperature (float): Sampling temperature for diversity in generation.

        Returns:
            str: Generated text.
        """
        self.model.eval()
        input_ids = tokenizer.encode(start_sequence, return_tensors="pt").to(self.device)
        generated_ids = input_ids

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(generated_ids)
                logits = outputs[:, -1, :] / temperature
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

                if next_token_id.item() == tokenizer.eos_token_id:
                    break

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.logger.info(f"\nGenerated text: {generated_text}\n")

        wandb_table = wandb.Table(columns=["Starting Sequence", "Generated Text"])
        wandb_table.add_data(start_sequence, generated_text)
        wandb.log({"evaluation/generated_text": wandb_table})
        
        return generated_text
