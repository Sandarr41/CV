import torch
import torch.nn as nn
import torch.optim as optim

from datasets import get_dataloaders
from model import get_model
from train import train_one_epoch, evaluate
from metrics import get_all_predictions, compute_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, train_dataset = get_dataloaders(
        batch_size=64,
        num_workers=0
    )

    model = get_model(num_classes=10, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    y_true, y_pred = get_all_predictions(model, test_loader, device)
    compute_metrics(y_true, y_pred, train_dataset.classes)


if __name__ == "__main__":
    main()
