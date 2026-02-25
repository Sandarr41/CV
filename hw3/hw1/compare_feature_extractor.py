import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import get_dataloaders
from model import get_model
from train import train_one_epoch, evaluate_with_metrics


def find_plateau_epoch(values, patience=2, min_improvement=0.002):
    best = values[0]
    stale = 0

    for idx, value in enumerate(values[1:], start=2):
        if value > best + min_improvement:
            best = value
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                return idx

    return len(values)


def run_experiment(mode, train_loader, test_loader, device, epochs, lr, ssl_encoder_path):
    if mode == "baseline":
        model = get_model(num_classes=10, pretrained=False, freeze_features=False).to(device)
        description = "Без feature extractor (обучение всей сети с нуля)"
    elif mode == "feature_extractor":
        model = get_model(
            num_classes=10,
            pretrained=False,
            freeze_features=True,
            ssl_encoder_path=ssl_encoder_path,
        ).to(device)
        description = (
            "С предобученным и замороженным feature extractor "
            f"({ssl_encoder_path})"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    print(f"\n=== {description} ===")
    for epoch in range(1, epochs + 1):
        train_loss, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1 = evaluate_with_metrics(
            model, test_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

    plateau_epoch = find_plateau_epoch(history["val_f1"])

    result = {
        "mode": mode,
        "description": description,
        "plateau_epoch": plateau_epoch,
        "max_val_acc": max(history["val_acc"]),
        "max_val_f1": max(history["val_f1"]),
        "history": history,
    }
    return result


def print_comparison(baseline_result, extractor_result):
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ")
    print("=" * 70)

    print("\n1) Скорость изменения лоссов")
    print(
        f"- Baseline: start={baseline_result['history']['train_loss'][0]:.4f}, "
        f"end={baseline_result['history']['train_loss'][-1]:.4f}"
    )
    print(
        f"- Feature extractor: start={extractor_result['history']['train_loss'][0]:.4f}, "
        f"end={extractor_result['history']['train_loss'][-1]:.4f}"
    )

    print("\n2) Скорость изменения метрик (Val Acc / Val Macro-F1)")
    print(
        f"- Baseline: {baseline_result['history']['val_acc'][0]:.2f}% -> "
        f"{baseline_result['history']['val_acc'][-1]:.2f}% | "
        f"F1: {baseline_result['history']['val_f1'][0]:.4f} -> "
        f"{baseline_result['history']['val_f1'][-1]:.4f}"
    )
    print(
        f"- Feature extractor: {extractor_result['history']['val_acc'][0]:.2f}% -> "
        f"{extractor_result['history']['val_acc'][-1]:.2f}% | "
        f"F1: {extractor_result['history']['val_f1'][0]:.4f} -> "
        f"{extractor_result['history']['val_f1'][-1]:.4f}"
    )

    print("\n3) Какая сеть достигла плато быстрее")
    print(
        f"- Baseline plateau epoch: {baseline_result['plateau_epoch']}"
    )
    print(
        f"- Feature extractor plateau epoch: {extractor_result['plateau_epoch']}"
    )

    faster = "baseline" if baseline_result["plateau_epoch"] < extractor_result["plateau_epoch"] else "feature extractor"
    if baseline_result["plateau_epoch"] == extractor_result["plateau_epoch"]:
        faster = "одновременно"
    print(f"=> Быстрее до плато: {faster}")

    print("\n4) Максимальные метрики")
    print(
        f"- Baseline max Val Acc: {baseline_result['max_val_acc']:.2f}% | "
        f"max Val F1: {baseline_result['max_val_f1']:.4f}"
    )
    print(
        f"- Feature extractor max Val Acc: {extractor_result['max_val_acc']:.2f}% | "
        f"max Val F1: {extractor_result['max_val_f1']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-percent", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssl-encoder-path", type=str, default="hw3/SSL/ssl_encoder.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled_fraction = args.labeled_percent / 100.0
    train_loader, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        labeled_fraction=labeled_fraction,
        seed=args.seed,
    )

    print(
        f"Используется {args.labeled_percent:.1f}% размеченной train-выборки "
        f"(fraction={labeled_fraction:.3f})"
    )

    baseline = run_experiment(
        mode="baseline",
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        ssl_encoder_path=args.ssl_encoder_path,
    )

    extractor = run_experiment(
        mode="feature_extractor",
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        ssl_encoder_path=args.ssl_encoder_path,
    )

    print_comparison(baseline, extractor)


if __name__ == "__main__":
    main()