import sys
from types import ModuleType

import torch
from torch import nn

from ViTclass.models.vit_backbone import build_vit_classifier


class FakeAttention(nn.Module):
    def __init__(self, dim=8, num_heads=2, fused_attn=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        raise AssertionError("attention block should be patched")


class FakeBlock(nn.Module):
    def __init__(self, dim=8, num_heads=2, fused_attn=True):
        super().__init__()
        self.attn = FakeAttention(dim=dim, num_heads=num_heads, fused_attn=fused_attn)


class FakeViT(nn.Module):
    def __init__(self, dim=8, num_heads=2, num_classes=47, fused_attn=True):
        super().__init__()
        self.embed_dim = dim
        self.blocks = nn.ModuleList([FakeBlock(dim=dim, num_heads=num_heads, fused_attn=fused_attn)])
        self.head = nn.Linear(dim, num_classes, bias=False)

    def forward(self, x):
        batch = x.shape[0]
        tokens = torch.ones(batch, 5, self.embed_dim)
        for block in self.blocks:
            tokens = block.attn(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)


def _install_fake_timm(monkeypatch, fused_attn=True):
    def create_model(backbone_name, pretrained, num_classes):
        assert backbone_name == "vit_base_patch16_224.augreg_in21k"
        assert pretrained is False
        return FakeViT(num_classes=num_classes, fused_attn=fused_attn)

    fake_timm = ModuleType("timm")
    fake_timm.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", fake_timm)


def test_build_vit_classifier_forward_runs_with_fused_attention(monkeypatch):
    _install_fake_timm(monkeypatch, fused_attn=True)

    model = build_vit_classifier(
        backbone_name="vit_base_patch16_224.augreg_in21k",
        num_classes=47,
        method="dora",
        rank=4,
        dropout=0.0,
        pretrained=False,
    )
    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    assert y.shape == (2, 47)
    assert hasattr(model.blocks[0].attn, "qk_adapter")


def test_build_vit_classifier_forward_runs_with_unfused_attention(monkeypatch):
    _install_fake_timm(monkeypatch, fused_attn=False)

    model = build_vit_classifier(
        backbone_name="vit_base_patch16_224.augreg_in21k",
        num_classes=47,
        method="golora",
        rank=4,
        dropout=0.0,
        pretrained=False,
    )
    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    assert y.shape == (2, 47)
    assert hasattr(model.blocks[0].attn, "_original_forward")


def test_patched_attention_accepts_timm_keyword_arguments(monkeypatch):
    _install_fake_timm(monkeypatch, fused_attn=True)

    model = build_vit_classifier(
        backbone_name="vit_base_patch16_224.augreg_in21k",
        num_classes=47,
        method="dora",
        rank=4,
        dropout=0.0,
        pretrained=False,
    )
    tokens = torch.randn(2, 5, 8)
    attn_mask = torch.zeros(1, 1, 5, 5)
    out = model.blocks[0].attn(tokens, attn_mask=attn_mask, is_causal=False)

    assert out.shape == tokens.shape


def test_only_adapters_and_head_are_trainable(monkeypatch):
    _install_fake_timm(monkeypatch, fused_attn=True)

    model = build_vit_classifier(
        backbone_name="vit_base_patch16_224.augreg_in21k",
        num_classes=47,
        method="dora",
        rank=4,
        dropout=0.0,
        pretrained=False,
    )
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]

    assert trainable
    assert all("qk_adapter" in name or name.startswith("head.") for name in trainable)
