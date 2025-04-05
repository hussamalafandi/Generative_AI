import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__()

        self.latent_dim = kwargs.get("latent_dim", 100)
        self.ngf = kwargs.get("ngf", 64)
        self.nc = kwargs.get("nc", 1)

        self.n_classes = kwargs.get("num_classes", 10)
        self.embed_dim = kwargs.get("embed_dim", 50)

        # Label embedding: maps labels to vectors of size embed_dim.
        self.label_embed = nn.Embedding(self.n_classes, self.embed_dim)

        self.main = nn.Sequential(
            # Combine noise and label embedding -> output shape: (latent_dim + embed_dim, 1, 1)
            # upscale to 7x7 with ngf*4 channels.
            nn.ConvTranspose2d(self.latent_dim + self.embed_dim, self.ngf * 4,
                               kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),

            # 7x7 -> 14x14
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            # 14x14 -> 28x28.
            nn.ConvTranspose2d(self.ngf * 2, self.ngf,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            # Final layer: convert to 1 channel, preserving 28x28.
            nn.ConvTranspose2d(self.ngf, self.nc, kernel_size=3,
                               stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed labels and reshape to (batch, embed_dim, 1, 1)
        label_embedding = self.label_embed(labels).unsqueeze(2).unsqueeze(3)
        # Concatenate noise and embedded labels along the channel dimension.
        gen_input = torch.cat([noise, label_embedding], dim=1)
        return self.main(gen_input)


class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__()

        self.ndf = kwargs.get("ndf", 64)
        self.nc = kwargs.get("nc", 1)

        self.n_classes = kwargs.get("num_classes", 10)
        self.embed_dim = kwargs.get("embed_dim", 50)

        # Label embedding: maps labels to vectors of size embed_dim.
        self.label_embed = nn.Embedding(self.n_classes, self.embed_dim)

        # DCGAN discriminator architecture
        self.main = nn.Sequential(

            # Input: (nc + embed_dim) x 28 x 28
            nn.Conv2d(self.nc + self.embed_dim, self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (ndf) x 14 x 14
            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (ndf*2) x 1 x 1
            nn.Conv2d(self.ndf * 2, 1, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Embed the labels and replicate them spatially
        label_embedding = self.label_embed(labels).unsqueeze(2).unsqueeze(3)
        # Assume img is of shape (batch, nc, H, W)
        label_embedding = label_embedding.expand(-1, -1, img.size(2), img.size(3))
        
        # Concatenate the image with the label embedding
        d_in = torch.cat((img, label_embedding), dim=1)

        return self.main(d_in).view(-1, 1).squeeze(1)
