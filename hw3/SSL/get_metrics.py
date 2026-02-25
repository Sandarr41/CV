import argparse
import csv
from pathlib import Path


def load_losses(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    losses = []
    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "epoch" not in reader.fieldnames or "loss" not in reader.fieldnames:
            raise ValueError("CSV must contain 'epoch' and 'loss' columns")

        for row in reader:
            losses.append((int(row["epoch"]), float(row["loss"])))

    if not losses:
        raise ValueError("No metric rows were found in the log")

    return losses


def summarize(losses):
    epochs = [epoch for epoch, _ in losses]
    values = [loss for _, loss in losses]

    best_idx = min(range(len(values)), key=values.__getitem__)
    best_epoch, best_loss = losses[best_idx]

    return {
        "epochs_tracked": len(losses),
        "first_epoch": epochs[0],
        "last_epoch": epochs[-1],
        "first_loss": values[0],
        "final_loss": values[-1],
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "mean_loss": sum(values) / len(values),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract SSL metrics from CSV log")
    parser.add_argument(
        "--log_path",
        type=Path,
        default=Path("hw3/logs/ssl_loss.csv"),
        help="Path to SSL CSV log with columns: epoch,loss",
    )
    args = parser.parse_args()

    try:
        losses = load_losses(args.log_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Cannot read SSL metrics: {exc}")

    metrics = summarize(losses)

    print("=== SSL Metrics Summary ===")
    print(f"Log file: {args.log_path}")
    print(f"Epochs tracked: {metrics['epochs_tracked']}")
    print(f"First epoch/loss: {metrics['first_epoch']} / {metrics['first_loss']:.6f}")
    print(f"Final epoch/loss: {metrics['last_epoch']} / {metrics['final_loss']:.6f}")
    print(f"Best epoch/loss: {metrics['best_epoch']} / {metrics['best_loss']:.6f}")
    print(f"Mean loss: {metrics['mean_loss']:.6f}")


if __name__ == "__main__":
    main()