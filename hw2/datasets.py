import os
from PIL import Image
from torch.utils.data import Dataset

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        img = self.transform(img)
        return img


class TestDataset(Dataset):
    def __init__(self, images_dir, annotations_path, transform):
        self.images_dir = images_dir
        self.transform = transform

        self.samples = []
        with open(annotations_path, "r") as f:
            for line in f:
                name, label = line.strip().split()
                self.samples.append((name, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)

        img = Image.open(img_path).convert("L")
        img = self.transform(img)

        return img, label