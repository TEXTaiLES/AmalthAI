import torchvision.models as models
import torch
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights, MobileNet_V2_Weights, ShuffleNet_V2_X1_0_Weights

def get_model(name, num_classes, pretrained=True):
    if name == 'ResNet18':
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    elif name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif name == 'ShuffleNetV2':
        model = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    else:
        raise ValueError(f"Unknown model: {name}")
    
    return model