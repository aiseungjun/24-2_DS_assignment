import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        #self.scale = math.sqrt(d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) #* self.scale

class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()
        self.positional_encoding = torch.zeros(max_len, d_model)
        pos_term = torch.arange(0, max_len).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(torch.arange(0, max_len, 2) * -torch.log(Tensor([10000.0])) / d_model) # log version
        self.positional_encoding[:, 0::2] = math.sin(pos_term * div_term) # 2i
        self.positional_encoding[:, 1::2] = math.cos(pos_term * div_term) # 2i+1
        self.positional_encoding = self.positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', self.positional_encoding)
                
    def forward(self, x: Tensor) -> Tensor:
        return self.positional_encoding[:x.size(1), :] # (1,len,d_model)