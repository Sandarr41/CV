import csv
import os

class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])

    def log(self, epoch, loss):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss])
