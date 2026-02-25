import random

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def get_transforms():
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    return transform_train, transform_test


def get_dataloaders(batch_size=64, num_workers=0, labeled_fraction=1.0, seed=42):
    transform_train, transform_test = get_transforms()

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    if labeled_fraction < 1.0:
        if not 0 < labeled_fraction <= 1.0:
            raise ValueError("labeled_fraction must be in the range (0, 1].")

        total_indices = list(range(len(train_dataset)))
        rng = random.Random(seed)
        rng.shuffle(total_indices)
        keep_count = int(len(total_indices) * labeled_fraction)
        keep_count = max(1, keep_count)
        selected_indices = total_indices[:keep_count]
        train_subset = Subset(train_dataset, selected_indices)
    else:
        train_subset = train_dataset

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, train_dataset