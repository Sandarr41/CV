import torch

from hw3.hw1.datasets import get_dataloaders

from .cli import parse_args
from .experiments import print_summary_table, run_single_experiment
from .results import save_results


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,
        labeled_fraction=args.labeled_fraction,
    )

    experiment_ids = [1, 2, 3] if args.experiment == "all" else [int(args.experiment)]
    rows = []
    for experiment_id in experiment_ids:
        rows.append(run_single_experiment(args, experiment_id, train_loader, test_loader, device))

    print_summary_table(rows)

    if args.results_path:
        save_results(rows, args.results_path)
        print(f"Saved results to: {args.results_path}")


if __name__ == "__main__":
    main()