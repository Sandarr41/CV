import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import ImageFolderDataset, TestDataset
from model import AutoEncoder
from train import train_model
from evaluate import reconstruction_error, find_optimal_threshold, evaluate_test


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_ds = ImageFolderDataset("dataset/train", transform)
    proliv_ds = ImageFolderDataset("dataset/proliv", transform)

    model = AutoEncoder().to(device)

    model = train_model(model, train_ds, device)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
    proliv_loader = DataLoader(proliv_ds, batch_size=128, shuffle=False)

    train_errors = reconstruction_error(model, train_loader, device)
    proliv_errors = reconstruction_error(model, proliv_loader, device)

    threshold = find_optimal_threshold(train_errors, proliv_errors)

    print("Optimized threshold:", threshold)
    print("Mean normal error:", train_errors.mean())
    print("Mean proliv error:", proliv_errors.mean())

    test_dataset = TestDataset(
        images_dir="dataset/test/imgs",
        annotations_path="dataset/test/test_annotation.txt",
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    TPR, TNR = evaluate_test(model, test_loader, device, threshold)

    print(f"TPR (пролив): {TPR:.4f}")
    print(f"TNR (норма):  {TNR:.4f}")


if __name__ == "__main__":
    main()