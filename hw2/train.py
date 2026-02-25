import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(model, train_ds, device, epochs=50, batch_size=128):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        losses = []

        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            noise = 0.02 * torch.randn_like(x)
            x_noisy = torch.clamp(x + noise, 0, 1)

            x_hat = model(x_noisy)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | loss = {np.mean(losses):.6f}")

    return model