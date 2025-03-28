from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(config):
    transform = transforms.Compose([
        transforms.Resize(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root=config["data_root"], train=True, transform=transform, download=True)

    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
