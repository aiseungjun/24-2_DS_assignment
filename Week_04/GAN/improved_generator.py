import torch
import torch.nn as nn

"""
GAN의 Generator를 자유롭게 개선해주세요!!
단순 논문 구현한 Generator에 이것 저것을 추가해도 좋고, 변경해도 좋습니다!

Hint:
1. Batch Normalization
2. Dropout
3. Deep Layer
4. etc...

Layer가 깊을수록 성능이 좋아질까요??
 https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
"""

class Generator(nn.Module):

    def __init__(self, z_dim, img_dim, feature_map_size=8):
        super().__init__()
        self.gen = nn.Sequential(
            *self.gen_block(z_dim, feature_map_size * 8, kernel_size=4, stride=1, padding=0),  # (1, 1) -> (4, 4)
            *self.gen_block(feature_map_size * 8, feature_map_size * 4),  # (4, 4) -> (8, 8)
            *self.gen_block(feature_map_size * 4, feature_map_size * 2),  # (8, 8) -> (16, 16)
            nn.ConvTranspose2d(feature_map_size * 2, 3, kernel_size=4, stride=2, padding=1, bias=False),  # (16, 16) -> (28, 28)
            nn.Tanh()  # [-1, 1] 범위로 정규화
        )
        
        
    def gen_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
        ]


    def forward(self, z):
        return self.gen(z)