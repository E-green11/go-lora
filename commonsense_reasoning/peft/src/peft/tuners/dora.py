# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import math
import os
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose


try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except Exception:
    bnb = None
    _BNB_AVAILABLE = False


def is_bnb_available():
    return _BNB_AVAILABLE


@dataclass
class DoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    dora_simple: bool = field(
        default=True, metadata={"help": "Whether to apply simple dora ver to save up GPU memory"}
    )
    orthogonal_direction: bool = field(
        default=False,
        metadata={"help": "Project low-rank direction update to the orthogonal complement of the frozen weight direction."},
    )
    geodesic_direction: bool = field(
        default=False,
        metadata={"help": "Use geodesic (exponential map) update on the unit hypersphere for direction updates."},
    )
    geodesic_detach_trig: bool = field(
        default=True,
        metadata={"help": "Detach geodesic angle when evaluating trig factors to reduce backward graph overhead."},
    )
    Wdecompose_target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to only tune the magnitude part"
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.DORA


class DoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
            "dora_simple": self.peft_config.dora_simple,
            "orthogonal_direction": self.peft_config.orthogonal_direction,
            "geodesic_direction": self.peft_config.geodesic_direction,
            "geodesic_detach_trig": self.peft_config.geodesic_detach_trig,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)

            if isinstance(self.peft_config.Wdecompose_target_modules, str):
                wdecompose_target_module_found = re.fullmatch(self.peft_config.Wdecompose_target_modules, key)
            elif self.peft_config.Wdecompose_target_modules == None:
                wdecompose_target_module_found = False
            else:
                wdecompose_target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.Wdecompose_target_modules)


            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        raise NotImplementedError
                    
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    module_kwargs = dict(kwargs)
                    module_kwargs["module_key"] = key
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **module_kwargs)
                elif self.peft_config.enable_lora is not None:
                    raise NotImplementedError

                self._replace_module(parent, target_name, new_module, target)

            elif wdecompose_target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        raise NotImplementedError

                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    module_kwargs = dict(kwargs)
                    module_kwargs["module_key"] = key
                    new_module = Linear(target.in_features, target.out_features, bias=bias, Wdecompose= True, **module_kwargs)
                elif self.peft_config.enable_lora is not None:
                    raise NotImplementedError
                self._replace_module(parent, target_name, new_module, target)

 
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight

        # 
        with torch.no_grad():
            magnitude = (torch.linalg.norm(new_module.weight.detach(),dim=1)).unsqueeze(1).detach()
            new_module.weight_m_wdecomp.weight.copy_(magnitude)
        

        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name or "weight_m_wdecomp" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n and "weight_m_wdecomp" not in n:
            p.requires_grad = False
        else:
            print(f"{n} is trainable")
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        Wdecompose: bool = False,
        dora_simple: bool = True,
        orthogonal_direction: bool = False,
        geodesic_direction: bool = False,
        geodesic_detach_trig: bool = True,
        module_key: Optional[str] = None,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.weight_m_wdecomp = nn.Linear(1,out_features,bias=False) # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple # whether to use dora simple to save up GPU memory
        self.orthogonal_direction = orthogonal_direction
        self.geodesic_direction = geodesic_direction
        self.geodesic_detach_trig = geodesic_detach_trig
        self.module_key = module_key or ""
        self.geo_theta_clamp = None
        self.geo_theta_clamp_mode = "hard"
        self.geo_theta_softband_start = None
        self.geo_anchor_mode = "base"
        self.geo_memory_save_mode = "off"
        self.geo_anchor_momentum = 0.25
        _geo_theta_clamp_env = os.environ.get("DORA_GEO_THETA_CLAMP", "").strip()
        if _geo_theta_clamp_env:
            try:
                _parsed_theta_clamp = float(_geo_theta_clamp_env)
                if _parsed_theta_clamp > 0:
                    self.geo_theta_clamp = _parsed_theta_clamp
                else:
                    warnings.warn("DORA_GEO_THETA_CLAMP should be > 0. Ignoring non-positive value.")
            except ValueError:
                warnings.warn("Invalid DORA_GEO_THETA_CLAMP value. Expected a float, got: " + _geo_theta_clamp_env)
        _geo_theta_clamp_mode_env = os.environ.get("DORA_GEO_THETA_CLAMP_MODE", "").strip().lower()
        if _geo_theta_clamp_mode_env:
            if _geo_theta_clamp_mode_env in {"hard", "tanh", "softband"}:
                self.geo_theta_clamp_mode = _geo_theta_clamp_mode_env
            else:
                warnings.warn(
                    "Invalid DORA_GEO_THETA_CLAMP_MODE value. Expected one of {'hard', 'tanh', 'softband'}, got: "
                    + _geo_theta_clamp_mode_env
                )
        _geo_theta_softband_start_env = os.environ.get("DORA_GEO_THETA_SOFTBAND_START", "").strip()
        if _geo_theta_softband_start_env:
            try:
                _parsed_softband_start = float(_geo_theta_softband_start_env)
                if _parsed_softband_start > 0:
                    self.geo_theta_softband_start = _parsed_softband_start
                else:
                    warnings.warn("DORA_GEO_THETA_SOFTBAND_START should be > 0. Ignoring non-positive value.")
            except ValueError:
                warnings.warn(
                    "Invalid DORA_GEO_THETA_SOFTBAND_START value. Expected a float, got: "
                    + _geo_theta_softband_start_env
                )
        _geo_anchor_mode_env = os.environ.get("DORA_GEO_ANCHOR_MODE", "").strip().lower()
        if _geo_anchor_mode_env:
            if _geo_anchor_mode_env in {"base", "current_detached", "lagged_detached"}:
                self.geo_anchor_mode = _geo_anchor_mode_env
            else:
                warnings.warn(
                    "Invalid DORA_GEO_ANCHOR_MODE value. Expected one of {'base', 'current_detached', 'lagged_detached'}, got: "
                    + _geo_anchor_mode_env
                )
        _geo_anchor_momentum_env = os.environ.get("DORA_GEO_ANCHOR_MOMENTUM", "").strip()
        if _geo_anchor_momentum_env:
            try:
                _parsed_anchor_momentum = float(_geo_anchor_momentum_env)
                if 0.0 < _parsed_anchor_momentum <= 1.0:
                    self.geo_anchor_momentum = _parsed_anchor_momentum
                else:
                    warnings.warn("DORA_GEO_ANCHOR_MOMENTUM should be in (0, 1]. Ignoring invalid value.")
            except ValueError:
                warnings.warn(
                    "Invalid DORA_GEO_ANCHOR_MOMENTUM value. Expected a float, got: " + _geo_anchor_momentum_env
                )
        _geo_memory_save_mode_env = os.environ.get("DORA_GEO_MEMORY_SAVE_MODE", "").strip().lower()
        if _geo_memory_save_mode_env:
            if _geo_memory_save_mode_env in {"off", "checkpoint", "checkpoint_mlp_only", "checkpoint_mlp_q_only", "checkpoint_mlp_qkv"}:
                self.geo_memory_save_mode = _geo_memory_save_mode_env
            else:
                warnings.warn(
                    "Invalid DORA_GEO_MEMORY_SAVE_MODE value. Expected one of {'off', 'checkpoint', 'checkpoint_mlp_only', 'checkpoint_mlp_q_only', 'checkpoint_mlp_qkv'}, got: "
                    + _geo_memory_save_mode_env
                )
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                self.register_buffer("geo_anchor_lora_A", torch.zeros_like(self.lora_A.weight.data))
                self.register_buffer("geo_anchor_lora_B", torch.zeros_like(self.lora_B.weight.data))
                # Freezing the pre-trained weight matrix

        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def _project_to_orthogonal_complement(self, delta_v: torch.Tensor, base_v: torch.Tensor, denom: Optional[torch.Tensor] = None):
        eps = 1e-8
        if denom is None:
            denom = torch.sum(base_v * base_v, dim=1, keepdim=True)
        denom = denom.clamp_min(eps)
        proj_coeff = torch.sum(delta_v * base_v, dim=1, keepdim=True) / denom
        return delta_v - proj_coeff * base_v

    def _get_delta_v(self):
        delta_v = transpose(self.lora_B.weight @ self.lora_A.weight, fan_in_fan_out=self.fan_in_fan_out) * self.scaling
        delta_v = delta_v.to(self.weight.dtype)
        if self.orthogonal_direction:
            base_v = self.weight
            if self.geo_anchor_mode == "lagged_detached":
                base_v = self.weight + self._get_lagged_anchor_delta()
            delta_v = self._project_to_orthogonal_complement(delta_v, base_v)
        return delta_v

    def _apply_theta_trust_region(self, theta: torch.Tensor):
        if self.geo_theta_clamp is None:
            return theta
        if self.geo_theta_clamp_mode == "tanh":
            return self.geo_theta_clamp * torch.tanh(theta / self.geo_theta_clamp)
        if self.geo_theta_clamp_mode == "softband":
            start = self.geo_theta_softband_start
            if start is None:
                start = self.geo_theta_clamp * 0.8
            start = min(start, self.geo_theta_clamp)
            if start <= 0 or start >= self.geo_theta_clamp:
                return torch.clamp(theta, max=self.geo_theta_clamp)

            width = self.geo_theta_clamp - start
            z = ((theta - start) / width).clamp(0.0, 1.0)
            softband_theta = start + width * (z + z * z - z * z * z)
            return torch.where(theta <= start, theta, softband_theta)
        return torch.clamp(theta, max=self.geo_theta_clamp)

    def _resolve_geodesic_anchor_stats(
        self,
        base_norm_sq: torch.Tensor,
        inner_delta_base: torch.Tensor,
        delta_norm_sq: torch.Tensor,
    ):
        eps = 1e-8
        if self.geo_anchor_mode != "current_detached":
            if self.geo_anchor_mode != "lagged_detached":
                return base_norm_sq.clamp_min(eps), inner_delta_base

            lag_inner_base, lag_delta_norm_sq = self._lagged_anchor_stats()
            anchor_a, anchor_b = self._get_lagged_anchor_factors()
            cross_inner = self._low_rank_cross_inner(
                self.lora_A.weight,
                self.lora_B.weight,
                anchor_a,
                anchor_b,
            )
            anchor_norm_sq = (base_norm_sq + 2 * lag_inner_base + lag_delta_norm_sq).clamp_min(eps)
            inner_delta_anchor = inner_delta_base + cross_inner
            return anchor_norm_sq, inner_delta_anchor

        anchor_norm_sq = (base_norm_sq + 2 * inner_delta_base + delta_norm_sq).clamp_min(eps)
        inner_delta_anchor = inner_delta_base + delta_norm_sq
        return anchor_norm_sq, inner_delta_anchor

    def _get_detached_lora_output(self, x: torch.Tensor):
        with torch.no_grad():
            return self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))) * self.scaling

    def _get_stable_geodesic_trig(self, theta: torch.Tensor, detach_theta: Optional[bool] = None):
        if detach_theta is None:
            detach_theta = self.geodesic_detach_trig

        theta_for_trig = theta.detach() if detach_theta else theta
        theta_for_trig = self._apply_theta_trust_region(theta_for_trig)

        small_theta = theta_for_trig <= 1e-4
        theta_sq = theta_for_trig * theta_for_trig
        sin_over_theta = torch.where(
            small_theta,
            1.0 - theta_sq / 6.0 + (theta_sq * theta_sq) / 120.0,
            torch.sin(theta_for_trig) / theta_for_trig,
        )
        cos_theta = torch.where(
            small_theta,
            1.0 - theta_sq / 2.0 + (theta_sq * theta_sq) / 24.0,
            torch.cos(theta_for_trig),
        )
        return theta_for_trig, sin_over_theta, cos_theta

    def _should_checkpoint_geodesic(self, x: torch.Tensor) -> bool:
        if self.geo_memory_save_mode == "checkpoint_mlp_only":
            save_mode_enabled = ".mlp." in self.module_key
        elif self.geo_memory_save_mode == "checkpoint_mlp_q_only":
            save_mode_enabled = (
                ".mlp." in self.module_key
                or ".self_attn.q_proj" in self.module_key
            )
        elif self.geo_memory_save_mode == "checkpoint_mlp_qkv":
            save_mode_enabled = (
                ".mlp." in self.module_key
                or ".self_attn.q_proj" in self.module_key
                or ".self_attn.k_proj" in self.module_key
                or ".self_attn.v_proj" in self.module_key
            )
        else:
            save_mode_enabled = self.geo_memory_save_mode == "checkpoint"
        return (
            save_mode_enabled
            and self.training
            and torch.is_grad_enabled()
            and self.geo_anchor_mode == "base"
            and x.requires_grad
        )

    def _geodesic_forward_impl(self, x: torch.Tensor):
        dropout_x = self.lora_dropout(x)
        base_out = F.linear(x, transpose(self.weight, self.fan_in_fan_out))
        lora_out = self.lora_B(self.lora_A(dropout_x.to(self.lora_A.weight.dtype))) * self.scaling
        base_drop = F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out))

        base_norm_sq, inner_delta_base, delta_norm_sq = self._low_rank_row_stats()
        anchor_norm_sq, inner_delta_anchor = self._resolve_geodesic_anchor_stats(
            base_norm_sq,
            inner_delta_base,
            delta_norm_sq,
        )
        coeff = (inner_delta_anchor / anchor_norm_sq).to(lora_out.dtype)

        if self.geo_anchor_mode == "current_detached":
            base_out = base_out + self._get_detached_lora_output(x).to(base_out.dtype)
            base_drop = base_drop + self._get_detached_lora_output(dropout_x).to(base_drop.dtype)
        elif self.geo_anchor_mode == "lagged_detached":
            base_out = base_out + self._get_lagged_lora_output(x).to(base_out.dtype)
            base_drop = base_drop + self._get_lagged_lora_output(dropout_x).to(base_drop.dtype)

        delta_perp_out = lora_out - coeff.view(1, -1) * base_drop.to(lora_out.dtype)
        delta_perp_norm_sq = (
            delta_norm_sq - (inner_delta_anchor * inner_delta_anchor) / anchor_norm_sq
        ).clamp_min(0.0)

        eps = 1e-8
        eps_grad = 1e-12
        base_norm = torch.sqrt(anchor_norm_sq).clamp_min(eps)
        theta = torch.sqrt(delta_perp_norm_sq + eps_grad) / base_norm
        _, sin_over_theta, cos_theta = self._get_stable_geodesic_trig(theta)

        weight_mag = self.weight_m_wdecomp.weight.view(-1).to(base_out.dtype)
        result = weight_mag.view(1, -1) * (
            (cos_theta / base_norm).view(1, -1) * base_out
            + (sin_over_theta / base_norm).view(1, -1) * delta_perp_out.to(base_out.dtype)
        )
        if self.bias is not None:
            result += self.bias.view(1, -1).expand_as(result)
        self._update_lagged_anchor_buffers()
        return result

    def _low_rank_row_stats_from_factors(self, a: torch.Tensor, b: torch.Tensor, base_v: torch.Tensor):
        eps = 1e-8
        base_v = base_v.to(a.dtype)
        base_norm_sq = torch.sum(base_v * base_v, dim=1).clamp_min(eps)
        va = torch.matmul(base_v, a.t())
        inner_delta_base = torch.sum(b * va.to(b.dtype), dim=1) * self.scaling
        gram = torch.matmul(a, a.t()).to(b.dtype)
        b_gram = torch.matmul(b, gram)
        delta_norm_sq = torch.sum(b_gram * b, dim=1) * (self.scaling ** 2)
        return base_norm_sq, inner_delta_base, delta_norm_sq

    def _low_rank_cross_inner(
        self,
        current_a: torch.Tensor,
        current_b: torch.Tensor,
        anchor_a: torch.Tensor,
        anchor_b: torch.Tensor,
    ):
        cross_gram = torch.matmul(current_a, anchor_a.t()).to(current_b.dtype)
        return torch.sum(torch.matmul(current_b, cross_gram) * anchor_b.to(current_b.dtype), dim=1) * (self.scaling ** 2)

    def _get_lagged_anchor_factors(self):
        return self.geo_anchor_lora_A.detach().clone(), self.geo_anchor_lora_B.detach().clone()

    def _get_lagged_anchor_delta(self):
        anchor_a, anchor_b = self._get_lagged_anchor_factors()
        return transpose(anchor_b @ anchor_a, fan_in_fan_out=self.fan_in_fan_out).to(self.weight.dtype) * self.scaling

    def _get_lagged_lora_output(self, x: torch.Tensor):
        anchor_a, anchor_b = self._get_lagged_anchor_factors()
        with torch.no_grad():
            hidden = F.linear(x.to(anchor_a.dtype), anchor_a)
            return F.linear(hidden, anchor_b) * self.scaling

    def _lagged_anchor_stats(self):
        anchor_a, anchor_b = self._get_lagged_anchor_factors()
        _, lag_inner_base, lag_delta_norm_sq = self._low_rank_row_stats_from_factors(
            anchor_a,
            anchor_b,
            transpose(self.weight, self.fan_in_fan_out),
        )
        return lag_inner_base, lag_delta_norm_sq

    def _update_lagged_anchor_buffers(self):
        if (
            self.geo_anchor_mode != "lagged_detached"
            or self.r <= 0
            or self.Wdecompose
            or not self.training
        ):
            return
        with torch.no_grad():
            self.geo_anchor_lora_A.lerp_(self.lora_A.weight.detach(), self.geo_anchor_momentum)
            self.geo_anchor_lora_B.lerp_(self.lora_B.weight.detach(), self.geo_anchor_momentum)

    def _geodesic_weight_update(self, delta_v: torch.Tensor):
        eps = 1e-8
        # Base weight is frozen; keep these tensors out of autograd graph to lower memory.
        with torch.no_grad():
            base_v = self.weight
            if self.geo_anchor_mode == "current_detached":
                base_v = base_v + delta_v.detach()
            elif self.geo_anchor_mode == "lagged_detached":
                base_v = base_v + self._get_lagged_anchor_delta()
            base_norm = torch.linalg.norm(base_v, dim=1, keepdim=True).clamp_min(eps)
            base_dir = base_v / base_norm
            base_dir_norm_sq = torch.sum(base_dir * base_dir, dim=1, keepdim=True).clamp_min(eps)

        tangent = delta_v / base_norm
        # If Ortho-DoRA is enabled, delta_v is already orthogonal to base_v.
        if not self.orthogonal_direction:
            tangent = self._project_to_orthogonal_complement(tangent, base_dir, base_dir_norm_sq)
        theta = torch.linalg.norm(tangent, dim=1, keepdim=True).clamp_min(eps)

        _, sin_over_theta, cos_theta = self._get_stable_geodesic_trig(theta, detach_theta=False)
        new_dir = base_dir * cos_theta + tangent * sin_over_theta

        # Keep a numerically safe unit direction in non-simple mode.
        if not self.dora_simple:
            new_dir = new_dir / torch.linalg.norm(new_dir, dim=1, keepdim=True).clamp_min(eps)
        return self.weight_m_wdecomp.weight * new_dir

    def _low_rank_row_stats(self):
        # Compute per-output-row statistics without constructing BA.
        return self._low_rank_row_stats_from_factors(
            self.lora_A.weight,
            self.lora_B.weight,
            transpose(self.weight, self.fan_in_fan_out),
        )

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.weight_m_wdecomp.train(mode)

        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.Wdecompose:
                norm_scale = ( self.weight_m_wdecomp.weight / (torch.linalg.norm(self.weight,dim=1)).unsqueeze(1) )
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    delta_v = self._get_delta_v()
                    if self.geodesic_direction:
                        weight = self._geodesic_weight_update(delta_v)
                    else:
                        new_weight_v = self.weight + delta_v
                        weight = ( self.weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True
        elif self.merge_weights and self.merged:
            raise NotImplementedError

    def eval(self):
        nn.Linear.eval(self)
        if self.Wdecompose == False:
            self.lora_A.eval()
            self.lora_B.eval()
        self.weight_m_wdecomp.eval()


    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype

        if self.disable_adapters:
            raise NotImplementedError
        
        elif self.Wdecompose and not self.merged:


            norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight,dim=1))

            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))

            result = org_result + (norm_scale-1) * (F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))

            if not self.bias is None:
                    result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:
            if self.geodesic_direction:
                if self._should_checkpoint_geodesic(x):
                    result = checkpoint(self._geodesic_forward_impl, x, use_reentrant=False)
                else:
                    result = self._geodesic_forward_impl(x)
            else:
                if self.orthogonal_direction:
                    base_norm_sq, inner_delta_base, delta_norm_sq = self._low_rank_row_stats()
                    delta_perp_norm_sq = (delta_norm_sq - (inner_delta_base * inner_delta_base) / base_norm_sq).clamp_min(0.0)
                    new_weight_norm = torch.sqrt(base_norm_sq + delta_perp_norm_sq)
                else:
                    # Keep original DoRA baseline path when Ortho/Geo are both disabled.
                    new_weight_v = self.weight + transpose(self.lora_B.weight @ self.lora_A.weight, fan_in_fan_out=self.fan_in_fan_out) * self.scaling
                    new_weight_norm = torch.linalg.norm(new_weight_v, dim=1)

                if self.dora_simple:
                    norm_scale = self.weight_m_wdecomp.weight.view(-1) / new_weight_norm.detach()
                else:
                    norm_scale = self.weight_m_wdecomp.weight.view(-1) / new_weight_norm

                org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))

                dropout_x = self.lora_dropout(x)

                result = org_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))

                if not self.bias is None:
                        result += self.bias.view(1, -1).expand_as(result)

                if self.orthogonal_direction:
                    # Memory-efficient projected LoRA output without materializing BA in activation path.
                    raw_lora_result = self.lora_B(self.lora_A(dropout_x.to(self.lora_A.weight.dtype))) * self.scaling
                    coeff = (inner_delta_base / base_norm_sq).to(raw_lora_result.dtype)
                    base_drop = F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out))
                    projected_lora_result = raw_lora_result - coeff.view(1, -1) * base_drop.to(raw_lora_result.dtype)
                    result += norm_scale * projected_lora_result
                else:
                    # Keep original DoRA/LoRA path for strict baseline behavior and lower memory pressure.
                    result += (norm_scale * (self.lora_B(self.lora_A(dropout_x.to(self.lora_A.weight.dtype))))) * self.scaling
            
        else:
             result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result


class MergedLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        raise NotImplementedError

if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            Wdecompose: bool = False,
            **kwargs,
        ):
            raise NotImplementedError

    class MergedLinear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            enable_lora: List[bool] = [False],
            **kwargs,
        ):
            raise NotImplementedError
