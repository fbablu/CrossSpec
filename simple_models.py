"""
Simplified segmentation models for CrossSpec
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleUNet(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleUNet, self).__init__()
        # Use pretrained ResNet50 as encoder (fix deprecation warning)
        try:
            from torchvision.models import ResNet50_Weights

            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except ImportError:
            resnet = models.resnet50(pretrained=True)

        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Simple decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, 4, 2, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SimplePSPNet(nn.Module):
    def __init__(self, num_classes=7):
        super(SimplePSPNet, self).__init__()
        try:
            from torchvision.models import ResNet50_Weights

            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except ImportError:
            resnet = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.classifier = nn.Conv2d(2048, num_classes, 1)

    def forward(self, x):
        size = x.size()[2:]
        x = self.backbone(x)
        x = self.classifier(x)
        x = nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return x


class SimpleDeepLab(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleDeepLab, self).__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50

        try:
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

            self.model = deeplabv3_resnet50(
                weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            )
        except ImportError:
            self.model = deeplabv3_resnet50(pretrained=True)

        self.model.classifier[4] = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        return self.model(x)["out"]
