import sys
import os
sys.path.append(os.path.abspath("SSL"))

import torch
import argparse

from config import SSL_ENCODER_PATH
from pretrain import pretrain_ssl


def main():
    parser = argparse.ArgumentParser(description="SimCLR SSL pretraining")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_path", type=str, default=SSL_ENCODER_PATH)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Self-Supervised Pretraining (SimCLR) ===")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    encoder = pretrain_ssl(
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    torch.save(encoder.state_dict(), args.save_path)
    print(f"\nEncoder saved to: {args.save_path}")


if __name__ == "__main__":
    main()