import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


def get_image_transforms(config):
    size = config["image_size"]
    channels = config["nc"]
    normalize_mean = [0.5] * channels
    normalize_std = [0.5] * channels

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])


def get_image_dataset(config, split="train"):
    transform = get_image_transforms(config)
    dataset_name = config["dataset"].lower()
    data_root = config["data_root"]

    if dataset_name == "mnist":
        is_train = split == "train"
        return datasets.MNIST(data_root, train=is_train, download=True, transform=transform)

    elif dataset_name == "cifar10":
        is_train = split == "train"
        return datasets.CIFAR10(data_root, train=is_train, download=True, transform=transform)

    elif dataset_name == "celeba":
        return datasets.CelebA(data_root, split=split, download=True, transform=transform)

    elif dataset_name == "custom":
        # We'll split manually below
        full_dataset = ImageFolder(root=os.path.join(
            data_root, "all"), transform=transform)
        indices = list(range(len(full_dataset)))
        split_idx = int(len(full_dataset) * config["train_ratio"])

        if split == "train":
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]

        return Subset(full_dataset, selected)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_image_dataloader(config, split="train"):
    dataset = get_image_dataset(config, split)

    # Use a subset of the dataset
    subset_size = config.get("subset_size", len(dataset))
    indices = torch.randperm(len(dataset))[:subset_size]
    dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=(split == "train"),
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
