import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 distillation experiments.")
    parser.add_argument("--teacher", type=str, default="efficientnet_b0")
    parser.add_argument("--student", type=str, default="mobilenetv3_small_100")
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--student-checkpoint", type=str, default=None)
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["1", "2", "3", "all"],
        required=True,
        help="1|2|3 for single run or all for sequential run with summary.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for logit distillation.")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--labeled-fraction", type=float, default=1.0)
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Optional path to save metrics table (.json or .csv).",
    )
    return parser.parse_args()