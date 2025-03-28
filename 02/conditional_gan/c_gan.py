import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        
        self.latent_dim = config["latent_dim"]
        self.ngf = config["ngf"]
        self.nc = config["nc"]

        self.n_classes = config["num_classes"]
        self.embed_dim = config["embed_dim"]

        # Label embedding: maps labels to vectors of size embed_dim.
        self.label_embed = nn.Embedding(self.n_classes, self.embed_dim)
        
        # DCGAN generator architecture
        self.main = nn.Sequential(

            # Input: latent vector Z (batch_size, latent_dim + embed_dim, 1, 1)
            nn.ConvTranspose2d(self.latent_dim + self.embed_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),

            # State: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),

            # State: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            # State: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            # State: (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: (nc) x 64 x 64
        )
    
    def forward(self, noise, labels):
        # Embed the labels and reshape to (batch, embed_dim, 1, 1)
        label_embedding = self.label_embed(labels).unsqueeze(2).unsqueeze(3)

        # Concatenate noise and label embedding along the channel dimension.
        gen_input = torch.cat([noise, label_embedding], dim=1)

        return self.main(gen_input)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.ndf = config["ndf"]
        self.nc = config["nc"]

        self.n_classes = config["num_classes"]
        self.embed_dim = config["embed_dim"]

        # Label embedding: maps labels to vectors of size embed_dim.
        self.label_embed = nn.Embedding(self.n_classes, self.embed_dim)

        # DCGAN discriminator architecture
        self.main = nn.Sequential(

            # Input: (nc + embed_dim) x 64 x 64
            nn.Conv2d(self.nc + self.embed_dim, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # State: (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
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

