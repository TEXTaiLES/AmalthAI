import torch
import torchvision.models as models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ShuffleNet_V2_X1_0_Weights,
)


def _replace_input_conv(model, path, in_channels, pretrained):
    parent = model
    for component in path[:-1]:
        parent = parent[component] if isinstance(component, int) else getattr(parent, component)
    key = path[-1]
    old = parent[key] if isinstance(key, int) else getattr(parent, key)
    if in_channels == old.in_channels:
        return
    new = torch.nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    if pretrained:
        with torch.no_grad():
            # Preserve the approximate activation scale of the RGB stem.
            mean_weight = old.weight.mean(dim=1, keepdim=True)
            new.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1) * (3.0 / in_channels))
            if old.bias is not None:
                new.bias.copy_(old.bias)
    if isinstance(key, int):
        parent[key] = new
    else:
        setattr(parent, key, new)


def get_model(name, num_classes, in_channels, pretrained=True):
    if in_channels < 1:
        raise ValueError("in_channels must be at least 1")

    if name == "ResNet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        _replace_input_conv(model, ("conv1",), in_channels, pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == "ResNet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        _replace_input_conv(model, ("conv1",), in_channels, pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        _replace_input_conv(model, ("features", 0, 0), in_channels, pretrained)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "MobileNetV2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        _replace_input_conv(model, ("features", 0, 0), in_channels, pretrained)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "ShuffleNetV2":
        model = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)
        _replace_input_conv(model, ("conv1", 0), in_channels, pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == "ConvNeXt":
        model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        _replace_input_conv(model, ("features", 0, 0), in_channels, pretrained)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model
