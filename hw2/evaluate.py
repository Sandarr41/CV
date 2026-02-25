import numpy as np
import torch


def reconstruction_error(model, loader, device, std_weight=0.3, clip_max=0.01):
    model.eval()
    errors = []

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            x_hat = model(x)
            pixel_err = (x - x_hat) ** 2

            if clip_max:
                pixel_err = torch.clamp(pixel_err, max=clip_max)

            err = torch.mean(pixel_err, dim=[1,2,3]) + \
                  std_weight * torch.std(pixel_err, dim=[1,2,3])

            errors.extend(err.cpu().numpy())

    return np.array(errors)


def find_optimal_threshold(train_errors, proliv_errors):
    errors = np.concatenate([train_errors, proliv_errors])
    labels = np.concatenate([
        np.zeros(len(train_errors)),
        np.ones(len(proliv_errors))
    ])

    best_thr = 0
    best_score = 0

    for thr in np.linspace(errors.min(), errors.max(), 2000):
        preds = (errors > thr).astype(int)

        TP = np.sum((preds == 1) & (labels == 1))
        TN = np.sum((preds == 0) & (labels == 0))
        FP = np.sum((preds == 1) & (labels == 0))
        FN = np.sum((preds == 0) & (labels == 1))

        TPR = TP / (TP + FN + 1e-9)
        TNR = TN / (TN + FP + 1e-9)

        score = TPR * TNR

        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr


def evaluate_test(model, test_loader, device, threshold,
                  std_weight=0.3, clip_max=0.01):

    TP = TN = FP = FN = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pixel_err = (x - model(x)) ** 2

            if clip_max:
                pixel_err = torch.clamp(pixel_err, max=clip_max)

            errors = torch.mean(pixel_err, dim=[1,2,3]) + \
                     std_weight * torch.std(pixel_err, dim=[1,2,3])

            preds = (errors > threshold).long()

            TP += ((preds == 1) & (y == 1)).sum().item()
            TN += ((preds == 0) & (y == 0)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    return TPR, TNR