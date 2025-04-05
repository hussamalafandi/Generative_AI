import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, **kwargs):
        super().__init__()
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
    def __init__(self, **kwargs):
        super().__init__()
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