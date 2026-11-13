"""ConvRot INT8 weights for the Forge IP-Adapter runtime.

HSWQ-converted IP-Adapter packs store their Linear weights as INT8 with a
per-output-channel ``weight_scale`` and a ``comfy_quant`` sidecar describing
``int8_tensorwise`` with ``convrot``. The stock loader hands every key of the
state dict to ``load_state_dict``, so the sidecar keys are reported as
unexpected and the load aborts before a single sample runs.

This module keeps those weights INT8 and evaluates them with the ConvRot INT8
GEMM in ``comfy_kitchen.int8_linear``, which applies the online Hadamard
rotation to the activation. Nothing is dequantized in the runtime path.

Float (non-quantized) IP-Adapter packs never enter this code: every entry point
returns immediately unless the state dict actually carries ConvRot INT8
sidecars.
"""

from __future__ import annotations

import json

import torch
from torch import nn

_WEIGHT = ".weight"
_SCALE = ".weight_scale"
_SIDECAR = ".comfy_quant"


def parse_sidecar(value) -> dict:
    """Decode a ``comfy_quant`` sidecar tensor/bytes/str into a dict."""
    if value is None:
        return {}
    try:
        if isinstance(value, torch.Tensor):
            return json.loads(value.numpy().tobytes())
        if isinstance(value, (bytes, bytearray)):
            return json.loads(bytes(value))
        if isinstance(value, str):
            return json.loads(value)
    except Exception:
        return {}
    return {}


def has_convrot_int8(state_dict) -> bool:
    """True when the state dict carries INT8 sidecars at all."""
    if not isinstance(state_dict, dict):
        return False
    return any(isinstance(k, str) and k.endswith(_SCALE) for k in state_dict.keys())


class ConvRotInt8Linear(nn.Module):
    """``nn.Linear`` replacement that keeps its weight in INT8 ConvRot form.

    ``int8_weight`` is ``[out_features, in_features]`` and ``weight_scale`` is
    ``[out_features]`` (float32). ``forward`` runs the fused INT8 GEMM; the
    activation rotation for the ConvRot groups happens inside the kernel.
    """

    def __init__(self, in_features, out_features, bias=True, convrot_groupsize=256):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.convrot_groupsize = int(convrot_groupsize)
        self.register_buffer("int8_weight", torch.empty(0, dtype=torch.int8), persistent=False)
        self.register_buffer("weight_scale", torch.empty(0, dtype=torch.float32), persistent=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_weight(cls, weight: torch.Tensor, weight_scale: torch.Tensor, convrot_groupsize: int = 256,
                    bias: torch.Tensor | None = None):
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        module = cls(in_features, out_features, bias=bias is not None, convrot_groupsize=convrot_groupsize)
        module.int8_weight = weight.detach().to(torch.int8).contiguous()
        module.weight_scale = weight_scale.detach().reshape(-1).to(torch.float32).contiguous()
        if bias is not None:
            module.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        return module

    @classmethod
    def from_linear(cls, linear: nn.Linear, weight: torch.Tensor, weight_scale: torch.Tensor,
                    convrot_groupsize: int = 256):
        bias = linear.bias.detach() if linear.bias is not None else None
        if bias is not None and weight.dtype == torch.int8:
            bias = bias.to(torch.float32)
        return cls.from_weight(weight, weight_scale, convrot_groupsize, bias)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        # ``Module.to(dtype=...)`` casts float buffers; the scale must stay
        # float32 for the INT8 kernel, so coerce it back.
        if self.weight_scale.dtype != torch.float32:
            self.weight_scale = self.weight_scale.to(torch.float32)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.int8_weight.numel() == 0:
            raise RuntimeError("ConvRotInt8Linear used before its INT8 weight was loaded")
        if x.shape[-1] != self.in_features:
            raise RuntimeError(
                f"ConvRotInt8Linear expected {self.in_features} input features, got {x.shape[-1]}"
            )
        import comfy_kitchen

        bias = None if self.bias is None else self.bias
        return comfy_kitchen.int8_linear(
            x.contiguous(),
            self.int8_weight,
            self.weight_scale,
            bias,
            out_dtype=x.dtype,
            convrot=True,
            convrot_groupsize=self.convrot_groupsize,
        )


def split_convrot_state_dict(state_dict: dict):
    """Split an IP-Adapter state dict into (quantized layers, float remainder).

    Returns ``({base: {"weight", "weight_scale", "groupsize"}}, remaining)``.
    ``remaining`` keeps biases and every non-quantized tensor, so a normal
    ``load_state_dict`` can consume it unchanged.
    """
    if not has_convrot_int8(state_dict):
        return {}, state_dict

    bases = sorted({k[: -len(_SCALE)] for k in state_dict if isinstance(k, str) and k.endswith(_SCALE)})
    quantized = {}
    for base in bases:
        conf = parse_sidecar(state_dict.get(base + _SIDECAR))
        quant_format = conf.get("format", "int8_tensorwise")
        if quant_format != "int8_tensorwise":
            raise RuntimeError(
                f"[IPAdapter ConvRot INT8] unsupported quantization format {quant_format!r} for layer {base}"
            )
        if not bool(conf.get("convrot", False)):
            raise RuntimeError(
                f"[IPAdapter ConvRot INT8] layer {base} is INT8 without ConvRot; this path only implements ConvRot"
            )
        weight = state_dict.get(base + _WEIGHT)
        if weight is None or weight.dtype != torch.int8:
            raise RuntimeError(f"[IPAdapter ConvRot INT8] layer {base} has no INT8 weight")
        groupsize = int(conf.get("convrot_groupsize", 256) or 256)
        if weight.dim() != 2 or weight.shape[1] % groupsize != 0:
            raise RuntimeError(
                f"[IPAdapter ConvRot INT8] layer {base} input features {tuple(weight.shape)} "
                f"are not divisible by the ConvRot groupsize {groupsize}"
            )
        quantized[base] = {
            "weight": weight,
            "weight_scale": state_dict[base + _SCALE],
            "groupsize": groupsize,
        }

    remaining = {}
    for key, value in state_dict.items():
        if isinstance(key, str):
            if key.endswith(_SCALE) or key.endswith(_SIDECAR):
                continue
            if key.endswith(_WEIGHT) and key[: -len(_WEIGHT)] in quantized:
                continue
        remaining[key] = value
    return quantized, remaining


def install_convrot_linears(root: nn.Module, quantized: dict):
    """Replace ``root``'s Linear submodules with ``ConvRotInt8Linear``.

    ``quantized`` is the first element returned by ``split_convrot_state_dict``.
    Returns the list of replaced layer names.
    """
    replaced = []
    for base, entry in quantized.items():
        try:
            target = root.get_submodule(base)
        except AttributeError:
            raise RuntimeError(
                f"[IPAdapter ConvRot INT8] layer {base} does not exist in {type(root).__name__}"
            )
        if not isinstance(target, nn.Linear):
            raise RuntimeError(
                f"[IPAdapter ConvRot INT8] layer {base} is {type(target).__name__}, expected nn.Linear"
            )
        module = ConvRotInt8Linear.from_linear(target, entry["weight"], entry["weight_scale"], entry["groupsize"])
        parent_path, _, name = base.rpartition(".")
        parent = root.get_submodule(parent_path) if parent_path else root
        setattr(parent, name, module)
        replaced.append(base)
    return replaced
