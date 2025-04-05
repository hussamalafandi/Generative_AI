import os
import torch
import wandb
from logging import getLogger
from abc import ABC, abstractmethod



class BaseTrainer(ABC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if config["use_cuda"] and torch.cuda.is_available() else "cpu")
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.checkpoint_interval = config.get("checkpoint_interval", 10)
        self.log_interval = config.get("log_interval", -1)
        self.checkpoint_folder = config.get("checkpoint_dir", "checkpoints")


        self.logger = getLogger(self.__class__.__name__)


    @abstractmethod
    def train_epoch(self, dataloader, epoch):
        pass

    @abstractmethod
    def train(self, dataloader, epochs):
        pass