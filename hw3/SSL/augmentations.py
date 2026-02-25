import torchvision.transforms as transforms


class SimCLRAugmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(96),  # << smaller crop
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)
