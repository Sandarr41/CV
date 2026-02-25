import sys
import os
sys.path.append(os.path.abspath("."))

import torch
import torchvision
from torch.utils.data import DataLoader
import timm

from config import BACKBONE, DATA_ROOT, LR_SSL
from dataset import SSLDataset
from model import SimCLRModel
from loss import nt_xent_loss
from logger import CSVLogger

logger = CSVLogger("hw3/logs/ssl_loss.csv")

def pretrain_ssl(device, epochs=25, batch_size=128):
    # Dataset WITHOUT transforms
    base_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=None
    )

    ssl_dataset = SSLDataset(base_dataset)
    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    backbone = timm.create_model(BACKBONE, pretrained=False)
    model = SimCLRModel(backbone).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_SSL)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x1, x2 in ssl_loader:
            x1, x2 = x1.to(device), x2.to(device)

            z1 = model(x1)
            z2 = model(x2)
            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(ssl_loader)
        logger.log(epoch + 1, avg_loss)
        print(f"[SSL] Epoch {epoch+1}: loss={avg_loss:.4f}")

    return model.encoder