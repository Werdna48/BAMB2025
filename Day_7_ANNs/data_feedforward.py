
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders(batch_size: int = 64):
    """Return train & test DataLoader objects for MNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),  # → [0,1]
        transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST mean/std
    ])

    train_ds = datasets.MNIST(
        root="./data",
        train=True,
        download=True,  # downloaded from Yann LeCun's site via torchvision mirror
        transform=transform,
    )
    test_ds = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader
