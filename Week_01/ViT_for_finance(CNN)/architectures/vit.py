import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from torchvision import models
# class EmbeddingLayer(nn.Module):
#     pass
    
# class MultiHeadSelfAttention(nn.Module):
#     pass

    
# class MLP(nn.Module):
#     pass
    
# class Block(nn.Module):
#     pass 

# #여기까지 Encoder 구현 끝!!


# class VisionTransformer(nn.Module):
#     pass
    
class VisionTransformer(nn.Module):
    def __init__(self, model_name='swin_v2_t'):
        super(VisionTransformer, self).__init__()
        self.base_model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
        self.base_model.features[0][0] = nn.Conv2d(1, self.base_model.features[0][0].out_channels,
                                                   kernel_size=self.base_model.features[0][0].kernel_size,
                                                   stride=self.base_model.features[0][0].stride,
                                                   padding=self.base_model.features[0][0].padding,
                                                   bias=False)
        self.fc = nn.Linear(self.base_model.head.in_features, 1)
        self.base_model.head = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x