from pathlib import Path

import timm
import torch


def get_model(
    num_classes=10,
    pretrained=True,
    freeze_features=False,
    ssl_encoder_path=None,
):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes
    )

    if ssl_encoder_path is not None:
        checkpoint_path = Path(ssl_encoder_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SSL encoder checkpoint not found: {ssl_encoder_path}")

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    if freeze_features:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    return model