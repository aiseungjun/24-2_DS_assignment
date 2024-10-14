import torch
import torch.nn as nn

"""
GAN의 Discriminator를 자유롭게 개선해주세요!!
단순 논문 구현한 Discriminator에 이것 저것을 추가해도 좋고, 변경해도 좋습니다!

Hint:
1. Batch Normalization
2. Dropout
3. Deep Layer
4. etc...

Layer가 깊을수록 성능이 좋아질까요?? 

"""

class Discriminator(nn.Module):

    def __init__(self, in_features, feature_map_size=8):
        super().__init__()
        self.disc = nn.Sequential(
            *self.disc_block(3, feature_map_size, use_bn=False),  # (28, 28) -> (14, 14)
            *self.disc_block(feature_map_size, feature_map_size * 2),  # (14, 14) -> (7, 7)
            *self.disc_block(feature_map_size * 2, feature_map_size * 4),  # (7, 7) -> (3, 3)
            nn.Conv2d(feature_map_size * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),  # (3, 3) -> (1, 1)
            nn.Sigmoid()  # [0, 1] 범위로 정규화
        )

    def disc_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        return layers

    def forward(self, x):
        return self.disc(x).view(-1, 1)