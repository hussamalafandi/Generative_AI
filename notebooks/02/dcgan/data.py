import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(config):
    transform = transforms.Compose([
        transforms.CenterCrop(178),            # Crop the faces to square
        # Resize to the desired image size (e.g., 64x64)
        transforms.Resize(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CelebA(
        root=config["data_root"], split="train", transform=transform, download=True)

    # Use a subset of the dataset
    subset_size = config.get("subset_size", len(dataset))
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = torch.utils.data.Subset(dataset, indices)

    return DataLoader(subset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True)
