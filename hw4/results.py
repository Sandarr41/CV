import csv
import json


def save_results(rows: list[dict], output_path: str):
    if output_path.endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(rows, file, ensure_ascii=False, indent=2)
    elif output_path.endswith(".csv"):
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(output_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("--results-path must end with .json or .csv")