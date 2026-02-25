import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report


def get_all_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def compute_metrics(y_true, y_pred, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    for i, name in enumerate(class_names):
        print(
            f"Class: {name:10s} | "
            f"Precision: {precision[i]:.4f} | "
            f"Recall: {recall[i]:.4f} | "
            f"F1: {f1[i]:.4f} | "
            f"Support: {support[i]}"
        )

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    print("\n=== Macro Average ===")
    print(f"Precision: {p_macro:.4f}")
    print(f"Recall:    {r_macro:.4f}")
    print(f"F1:        {f1_macro:.4f}")

    print("\n", classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    ))
