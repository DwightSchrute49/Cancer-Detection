import torch
import torch.nn as nn
from torchvision import models


class LungCancerCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone_name = backbone
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes),
            )
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(in_features, num_classes),
            )
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ValueError(
                "Unsupported backbone. Use one of: resnet18, resnet50, efficientnet_b0."
            )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
