import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.base_model.features[0][0] = nn.Conv2d(1, self.base_model.features[0][0].out_channels,
                                                   kernel_size=self.base_model.features[0][0].kernel_size,
                                                   stride=self.base_model.features[0][0].stride,
                                                   padding=self.base_model.features[0][0].padding,
                                                   bias=False)
        self.fc = nn.Linear(self.base_model.classifier[1].in_features, 1)
        self.base_model.classifier = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x