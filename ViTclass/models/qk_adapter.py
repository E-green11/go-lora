import torch


def split_qkv(qkv: torch.Tensor, embed_dim: int):
    q, k, v = torch.split(qkv, embed_dim, dim=-1)
    return q, k, v


def merge_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return torch.cat([q, k, v], dim=-1)

