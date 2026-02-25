import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)  # 2N x D

    # Cosine similarity matrix
    sim = torch.matmul(representations, representations.T)  # 2N x 2N
    sim = sim / temperature

    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    sim = sim.masked_fill(mask, -9e15)

    # Positive pairs
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels + batch_size, labels])  # Each i's positive is shifted by batch_size

    loss = nn.CrossEntropyLoss()(sim, labels)
    return loss

