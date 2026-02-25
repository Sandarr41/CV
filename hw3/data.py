import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from hw3.SSL.config import *

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ])

def get_subset(dataset, percent, seed=SEED):
    np.random.seed(seed)
    n_total = len(dataset)
    n_subset = int(n_total * percent / 100)
    indices = np.random.permutation(n_total)[:n_subset]
    return Subset(dataset, indices)

def get_dataloaders():
    train_dataset = torchvision.datasets.CIFAR10(
        DATA_ROOT, train=True, download=True, transform=get_transforms(True)
    )
    test_dataset = torchvision.datasets.CIFAR10(
        DATA_ROOT, train=False, download=True, transform=get_transforms(False)
    )

    train_subset = get_subset(train_dataset, N_PERCENT)

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    return train_loader, test_loader