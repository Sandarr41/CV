import torch.nn as nn


class SimCLRModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = backbone
        self.encoder.reset_classifier(0)

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.projector(h)