import math

import torch
from torch import nn


class _GoLoraBlock(nn.Module):
    def __init__(self, embed_dim: int, rank: int, dropout: float):
        super().__init__()
        self.rank = rank
        self.scaling = 1.0 / math.sqrt(max(rank, 1))
        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.up = nn.Linear(rank, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.magnitude = nn.Parameter(torch.ones(embed_dim))
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.down(self.dropout(x))) * self.scaling
        base_norm_sq = (x * x).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        inner = (delta * x).sum(dim=-1, keepdim=True)
        delta_perp = delta - (inner / base_norm_sq) * x
        delta_perp_norm = delta_perp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        base_norm = base_norm_sq.sqrt()
        theta = (delta_perp_norm / base_norm).clamp_max(0.1)
        direction = delta_perp / delta_perp_norm
        geodesic = torch.cos(theta) * x + torch.sin(theta) * direction * base_norm
        geodesic_norm = geodesic.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = self.magnitude.view(1, 1, -1) / geodesic_norm.detach()
        return geodesic * scale


class GoLoraQKAdapter(nn.Module):
    def __init__(self, embed_dim: int, rank: int, dropout: float):
        super().__init__()
        self.q_block = _GoLoraBlock(embed_dim=embed_dim, rank=rank, dropout=dropout)
        self.k_block = _GoLoraBlock(embed_dim=embed_dim, rank=rank, dropout=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return self.q_block(q), self.k_block(k)

