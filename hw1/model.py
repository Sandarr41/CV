import timm


def get_model(num_classes=10, pretrained=True):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model