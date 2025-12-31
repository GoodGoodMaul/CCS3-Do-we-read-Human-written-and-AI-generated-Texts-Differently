import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        # residual add; keeps GELU non-linearity while avoiding unstable multiplicative gates
        return out1 + out2




from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class CA_SA(nn.Module):
    def __init__(self, dim=32):
        super(CA_SA, self).__init__()
        self.dim = dim
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.Q = nn.Linear(dim, dim)
        self.attend = nn.Softmax(dim = -1)
        self.ln = nn.LayerNorm(dim)
    def forward(self, feat1, feat2, key_padding_mask=None):
        K = self.K(feat2)
        V = self.V(feat2)
        Q = self.Q(feat1)
        scale = self.dim ** -0.5
        dots = torch.bmm(Q, K.permute(0, 2, 1)) * scale
        if key_padding_mask is not None:
            # key_padding_mask: True for tokens to keep, False for tokens to mask out
            mask = key_padding_mask.unsqueeze(1).expand(-1, Q.shape[1], -1)
            dots = dots.masked_fill(~mask, float("-inf"))
        attn = self.attend(dots)
        # guard against potential NaNs/Infs introduced by masking
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        out = torch.bmm(attn, V)
        out = self.ln(out + feat1)  # residual + LayerNorm for stability
        return out


