import torch

from ViTclass.models.dora_qk import DoraQKAdapter
from ViTclass.models.golora_qk import GoLoraQKAdapter
from ViTclass.models.qk_adapter import merge_qkv, split_qkv


def test_split_and_merge_qkv_preserve_shape():
    x = torch.randn(2, 197, 2304)
    q, k, v = split_qkv(x, embed_dim=768)
    y = merge_qkv(q, k, v)
    assert q.shape == (2, 197, 768)
    assert k.shape == (2, 197, 768)
    assert v.shape == (2, 197, 768)
    assert y.shape == x.shape


def test_split_and_merge_qkv_roundtrip_values():
    x = torch.randn(2, 5, 12)
    q, k, v = split_qkv(x, embed_dim=4)
    y = merge_qkv(q, k, v)
    assert torch.allclose(x, y)


def test_dora_qk_adapter_preserves_shape():
    adapter = DoraQKAdapter(embed_dim=8, rank=4, dropout=0.0)
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 3, 8)

    q_out, k_out = adapter(q, k)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_golora_qk_adapter_preserves_shape():
    adapter = GoLoraQKAdapter(embed_dim=8, rank=4, dropout=0.0)
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 3, 8)

    q_out, k_out = adapter(q, k)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
