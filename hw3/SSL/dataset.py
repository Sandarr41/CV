import torch
from augmentations import SimCLRAugmentation


class SSLDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset
        self.aug = SimCLRAugmentation()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # PIL image
        return self.aug(img)
