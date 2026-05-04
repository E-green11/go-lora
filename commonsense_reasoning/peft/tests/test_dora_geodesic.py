# coding=utf-8
import math
import os
import unittest
from unittest import mock

import torch

from peft.tuners.dora import Linear


class DoraGeodesicTrigTest(unittest.TestCase):
    def _make_layer(self):
        return Linear(
            in_features=4,
            out_features=3,
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias=False,
            geodesic_direction=True,
        )

    def test_stable_geodesic_trig_uses_taylor_for_small_theta(self):
        layer = self._make_layer()
        theta = torch.tensor([[1e-8], [1e-6], [1e-4]], dtype=torch.float32)

        _, sin_over_theta, cos_theta = layer._get_stable_geodesic_trig(theta)

        self.assertTrue(torch.isfinite(sin_over_theta).all())
        self.assertTrue(torch.isfinite(cos_theta).all())
        self.assertAlmostEqual(float(sin_over_theta[0]), 1.0, places=6)
        self.assertAlmostEqual(float(cos_theta[0]), 1.0, places=6)

    def test_stable_geodesic_trig_applies_theta_clamp(self):
        layer = self._make_layer()
        layer.geo_theta_clamp = 0.1
        theta = torch.tensor([[0.5]], dtype=torch.float32)

        theta_for_trig, sin_over_theta, cos_theta = layer._get_stable_geodesic_trig(theta)

        self.assertAlmostEqual(float(theta_for_trig[0]), 0.1, places=6)
        self.assertAlmostEqual(float(sin_over_theta[0]), math.sin(0.1) / 0.1, places=6)
        self.assertAlmostEqual(float(cos_theta[0]), math.cos(0.1), places=6)

    def test_stable_geodesic_trig_supports_tanh_theta_trust_region(self):
        layer = self._make_layer()
        layer.geo_theta_clamp = 0.1
        layer.geo_theta_clamp_mode = "tanh"
        theta = torch.tensor([[0.5]], dtype=torch.float32)

        theta_for_trig, sin_over_theta, cos_theta = layer._get_stable_geodesic_trig(theta)

        expected_theta = 0.1 * math.tanh(0.5 / 0.1)
        self.assertAlmostEqual(float(theta_for_trig[0]), expected_theta, places=6)
        self.assertAlmostEqual(float(sin_over_theta[0]), math.sin(expected_theta) / expected_theta, places=6)
        self.assertAlmostEqual(float(cos_theta[0]), math.cos(expected_theta), places=6)

    def test_stable_geodesic_trig_supports_softband_theta_trust_region(self):
        layer = self._make_layer()
        layer.geo_theta_clamp = 0.1
        layer.geo_theta_clamp_mode = "softband"
        layer.geo_theta_softband_start = 0.08
        theta = torch.tensor([[0.05], [0.09], [0.2]], dtype=torch.float32)

        theta_for_trig, _, _ = layer._get_stable_geodesic_trig(theta)

        self.assertAlmostEqual(float(theta_for_trig[0]), 0.05, places=6)
        z = (0.09 - 0.08) / (0.1 - 0.08)
        expected_mid = 0.08 + (0.1 - 0.08) * (z + z * z - z * z * z)
        self.assertAlmostEqual(float(theta_for_trig[1]), expected_mid, places=6)
        self.assertAlmostEqual(float(theta_for_trig[2]), 0.1, places=6)

    def test_current_detached_anchor_stats_shift_reference_geometry(self):
        layer = self._make_layer()
        layer.geo_anchor_mode = "current_detached"

        base_norm_sq = torch.tensor([4.0, 9.0], dtype=torch.float32)
        inner_delta_base = torch.tensor([1.0, -2.0], dtype=torch.float32)
        delta_norm_sq = torch.tensor([0.5, 3.0], dtype=torch.float32)

        anchor_norm_sq, inner_delta_anchor = layer._resolve_geodesic_anchor_stats(
            base_norm_sq,
            inner_delta_base,
            delta_norm_sq,
        )

        self.assertTrue(torch.allclose(anchor_norm_sq, torch.tensor([6.5, 8.0])))
        self.assertTrue(torch.allclose(inner_delta_anchor, torch.tensor([1.5, 1.0])))

    def test_lagged_detached_anchor_buffers_initialize_to_zero(self):
        layer = self._make_layer()

        self.assertTrue(torch.count_nonzero(layer.geo_anchor_lora_A).item() == 0)
        self.assertTrue(torch.count_nonzero(layer.geo_anchor_lora_B).item() == 0)

    def test_lagged_detached_anchor_buffers_follow_current_weights_with_momentum(self):
        layer = self._make_layer()
        layer.geo_anchor_mode = "lagged_detached"
        layer.geo_anchor_momentum = 0.25
        layer.train(True)

        with torch.no_grad():
            layer.lora_A.weight.fill_(2.0)
            layer.lora_B.weight.fill_(4.0)

        layer._update_lagged_anchor_buffers()

        self.assertTrue(torch.allclose(layer.geo_anchor_lora_A, torch.full_like(layer.geo_anchor_lora_A, 0.5)))
        self.assertTrue(torch.allclose(layer.geo_anchor_lora_B, torch.full_like(layer.geo_anchor_lora_B, 1.0)))

    def test_lagged_detached_anchor_supports_forward_backward(self):
        layer = self._make_layer()
        layer.geo_anchor_mode = "lagged_detached"
        layer.orthogonal_direction = True
        layer.train(True)
        x = torch.randn(2, 4, dtype=torch.float32)

        loss = layer(x).sum()
        loss.backward()

        self.assertIsNotNone(layer.lora_A.weight.grad)
        self.assertIsNotNone(layer.lora_B.weight.grad)

    def test_checkpoint_memory_save_mode_matches_geodesic_forward_and_grads(self):
        baseline = self._make_layer()
        baseline.train(True)

        with mock.patch.dict(os.environ, {"DORA_GEO_MEMORY_SAVE_MODE": "checkpoint"}, clear=False):
            checkpointed = self._make_layer()
        checkpointed.load_state_dict(baseline.state_dict())
        checkpointed.train(True)

        x0 = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
        x1 = x0.detach().clone().requires_grad_(True)

        y0 = baseline(x0)
        y1 = checkpointed(x1)
        self.assertTrue(torch.allclose(y0, y1, atol=1e-6, rtol=1e-5))

        y0.sum().backward()
        y1.sum().backward()

        self.assertTrue(torch.allclose(x0.grad, x1.grad, atol=1e-6, rtol=1e-5))
        self.assertTrue(
            torch.allclose(
                baseline.lora_A.weight.grad,
                checkpointed.lora_A.weight.grad,
                atol=1e-6,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                baseline.lora_B.weight.grad,
                checkpointed.lora_B.weight.grad,
                atol=1e-6,
                rtol=1e-5,
            )
        )

    def test_checkpoint_mlp_only_mode_only_marks_mlp_modules(self):
        with mock.patch.dict(os.environ, {"DORA_GEO_MEMORY_SAVE_MODE": "checkpoint_mlp_only"}, clear=False):
            mlp_layer = Linear(
                in_features=4,
                out_features=3,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias=False,
                geodesic_direction=True,
                module_key="model.layers.0.mlp.up_proj",
            )
            attn_layer = Linear(
                in_features=4,
                out_features=3,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias=False,
                geodesic_direction=True,
                module_key="model.layers.0.self_attn.q_proj",
            )

        x = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
        mlp_layer.train(True)
        attn_layer.train(True)
        self.assertTrue(mlp_layer._should_checkpoint_geodesic(x))
        self.assertFalse(attn_layer._should_checkpoint_geodesic(x))

    def test_checkpoint_mlp_qkv_mode_marks_mlp_and_qkv_but_not_oproj(self):
        with mock.patch.dict(os.environ, {"DORA_GEO_MEMORY_SAVE_MODE": "checkpoint_mlp_qkv"}, clear=False):
            mlp_layer = Linear(
                in_features=4,
                out_features=3,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias=False,
                geodesic_direction=True,
                module_key="model.layers.0.mlp.up_proj",
            )
            q_layer = Linear(
                in_features=4,
                out_features=3,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias=False,
                geodesic_direction=True,
                module_key="model.layers.0.self_attn.q_proj",
            )
            o_layer = Linear(
                in_features=4,
                out_features=3,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias=False,
                geodesic_direction=True,
                module_key="model.layers.0.self_attn.o_proj",
            )

        x = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
        mlp_layer.train(True)
        q_layer.train(True)
        o_layer.train(True)
        self.assertTrue(mlp_layer._should_checkpoint_geodesic(x))
        self.assertTrue(q_layer._should_checkpoint_geodesic(x))
        self.assertFalse(o_layer._should_checkpoint_geodesic(x))

    def test_checkpoint_mlp_q_only_mode_marks_mlp_and_q_but_not_k_or_o(self):
        with mock.patch.dict(os.environ, {"DORA_GEO_MEMORY_SAVE_MODE": "checkpoint_mlp_q_only"}, clear=False):
            mlp_layer = Linear(
                in_features=4,
                out_features=3,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias=False,
                geodesic_direction=True,
                module_key="model.layers.0.mlp.up_proj",
            )
            q_layer = Linear(
                in_features=4,
                out_features=3,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias=False,
                geodesic_direction=True,
                module_key="model.layers.0.self_attn.q_proj",
            )
            k_layer = Linear(
                in_features=4,
                out_features=3,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias=False,
                geodesic_direction=True,
                module_key="model.layers.0.self_attn.k_proj",
            )
            o_layer = Linear(
                in_features=4,
                out_features=3,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias=False,
                geodesic_direction=True,
                module_key="model.layers.0.self_attn.o_proj",
            )

        x = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
        mlp_layer.train(True)
        q_layer.train(True)
        k_layer.train(True)
        o_layer.train(True)
        self.assertTrue(mlp_layer._should_checkpoint_geodesic(x))
        self.assertTrue(q_layer._should_checkpoint_geodesic(x))
        self.assertFalse(k_layer._should_checkpoint_geodesic(x))
        self.assertFalse(o_layer._should_checkpoint_geodesic(x))


if __name__ == "__main__":
    unittest.main()
