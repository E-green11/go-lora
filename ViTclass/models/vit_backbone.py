from importlib import import_module
from types import MethodType

import torch
from torch import nn

from .dora_qk import DoraQKAdapter
from .golora_qk import GoLoraQKAdapter
from .qk_adapter import merge_qkv, split_qkv


def _make_adapter(method: str, embed_dim: int, rank: int, dropout: float):
    if method == "dora":
        return DoraQKAdapter(embed_dim=embed_dim, rank=rank, dropout=dropout)
    if method == "golora":
        return GoLoraQKAdapter(embed_dim=embed_dim, rank=rank, dropout=dropout)
    raise ValueError(f"Unsupported method: {method}")


def _patch_attention_block(attn: nn.Module, method: str, rank: int, dropout: float):
    embed_dim = attn.qkv.out_features // 3
    attn.qk_adapter = _make_adapter(method=method, embed_dim=embed_dim, rank=rank, dropout=dropout)
    original_forward = attn.forward

    def patched_forward(self, x, attn_mask=None, is_causal: bool = False, **kwargs):
        batch_size, num_tokens, embed_dim_local = x.shape
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, embed_dim=embed_dim_local)
        q, k = self.qk_adapter(q, k)
        qkv = merge_qkv(q, k, v)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, embed_dim_local // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.fused_attn:
            x_out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            q = q * self.scale
            attn_scores = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask
            attn_scores = attn_scores.softmax(dim=-1)
            attn_scores = self.attn_drop(attn_scores)
            x_out = attn_scores @ v

        x_out = x_out.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim_local)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    attn.forward = MethodType(patched_forward, attn)
    attn._original_forward = original_forward


def _freeze_for_adapter_tuning(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False

    for block in model.blocks:
        for param in block.attn.qk_adapter.parameters():
            param.requires_grad = True

    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        for param in model.head.parameters():
            param.requires_grad = True


def build_vit_classifier(backbone_name: str, num_classes: int, method: str, rank: int, dropout: float, pretrained: bool = True):
    timm = import_module("timm")
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
    for block in model.blocks:
        _patch_attention_block(block.attn, method=method, rank=rank, dropout=dropout)
    _freeze_for_adapter_tuning(model)
    return model
