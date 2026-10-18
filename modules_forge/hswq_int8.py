# Forge INT8 Low Bits — UI-only separate branch.
# Entry: Low Bits "int8" or "int8 (fp16 LoRA)" → forge_unet_storage_dtype == int8_tensorwise.
# Never auto-detect from checkpoint / Automatic. Never share float8 / bnb construct.
# Imports comfy_quant patches from modules/ComfyUI-HSWQ-Loader-and-Tools only.

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager

INT8_UNET_STORAGE = "int8_tensorwise"
_LOG = "[HSWQ INT8]"

_LOADER_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "modules",
    "ComfyUI-HSWQ-Loader-and-Tools",
)


def _ensure_loader_path() -> None:
    if _LOADER_ROOT not in sys.path:
        sys.path.insert(0, _LOADER_ROOT)


def _plog(msg: str) -> None:
    line = f"{_LOG} {msg}"
    print(line)
    logging.info(line)


@contextmanager
def int8_load_scope():
    _ensure_loader_path()
    from patches.comfy_quant_int8 import _int8_quant_conv_scope

    with _int8_quant_conv_scope():
        yield


_HADAMARD_GLOBAL_CACHE: dict[tuple[int, str], object] = {}


def _apply_forge_hadamard_cache_patches() -> None:
    """Runtime monkeypatch for Hadamard device caching without modifying ComfyUI-HSWQ-Loader-and-Tools."""
    import patches.comfy_quant_int8 as cqi

    if not getattr(cqi, "_forge_hadamard_cache_patched", False):
        orig_regular_hadamard = cqi._regular_hadamard_global

        def _cached_regular_hadamard_global(size: int, device=None):
            key = (int(size), str(device))
            cached = _HADAMARD_GLOBAL_CACHE.get(key)
            if cached is not None:
                return cached
            res = orig_regular_hadamard(size, device=device)
            _HADAMARD_GLOBAL_CACHE[key] = res
            return res

        cqi._regular_hadamard_global = _cached_regular_hadamard_global
        cqi._forge_hadamard_cache_patched = True

    try:
        nc = cqi._load_native_convert_int8_helpers()
        if nc is not None and not getattr(nc, "_forge_unrotate_conv2d_patched", False):
            orig_unrotate_conv2d = nc.unrotate_weight_conv2d

            def _device_cached_unrotate_weight_conv2d(weight, h_matrix, group_size: int):
                if getattr(weight, "device", None) is not None and (
                    getattr(h_matrix, "device", None) != weight.device
                    or getattr(h_matrix, "dtype", None) != weight.dtype
                ):
                    h_matrix = nc.get_hadamard_on_device(
                        int(group_size), device=weight.device, dtype=weight.dtype
                    )
                return orig_unrotate_conv2d(weight, h_matrix, group_size)

            nc.unrotate_weight_conv2d = _device_cached_unrotate_weight_conv2d
            nc._forge_unrotate_conv2d_patched = True
    except Exception as exc:
        _plog(f"warn: _apply_forge_hadamard_cache_patches nc hook skipped: {exc}")


def ensure_int8_patches() -> bool:
    """Apply INT8 patches only when the INT8 load branch runs."""
    _ensure_loader_path()
    from patches.comfy_quant_int8 import apply_comfy_quant_int8_patches

    ok = apply_comfy_quant_int8_patches()
    _apply_forge_hadamard_cache_patches()
    _plog(f"ensure_int8_patches → {ok}")
    return ok


def is_int8_unet_storage(storage_dtype) -> bool:
    return storage_dtype == INT8_UNET_STORAGE


def _conv2d_is_quant_capable(conv_cls) -> bool:
    return bool(getattr(conv_cls, "_hswq_quant_conv2d", False)) or "Quantized" in getattr(
        conv_cls, "__name__", ""
    ) or hasattr(conv_cls, "quant_format")


def build_int8_operations(compute_dtype):
    """MixedPrecisionOps for INT8 only (+ Forge online LoRA on those ops).

    CRITICAL: mixed_precision_ops(quant_config={}) must run INSIDE int8_load_scope,
    otherwise patched mixed_precision_ops will NOT inject Quantized Conv2d and
    comfy_quant / weight_scale keys become Unexpected → NaN / black images.
    """
    import torch
    import comfy.ops
    from backend.operations import ForgeOperations
    from backend.patcher.lora import merge_lora_to_weight

    ensure_int8_patches()

    _plog(f"build_int8_operations: compute_dtype={compute_dtype}")
    with int8_load_scope():
        ops = comfy.ops.mixed_precision_ops(quant_config={}, compute_dtype=compute_dtype)

    conv_ok = _conv2d_is_quant_capable(ops.Conv2d) or getattr(ops.Conv2d, "_hswq_quant_conv2d", False)
    # After inject, class is named Conv2d but has _hswq_quant_conv2d on the class created inside patch.
    inject_flag = getattr(ops.Conv2d, "_hswq_quant_conv2d", False)
    _plog(
        f"MixedPrecisionOps.Conv2d={ops.Conv2d!r} "
        f"_hswq_quant_conv2d={inject_flag} mro={[c.__name__ for c in ops.Conv2d.__mro__[:4]]}"
    )
    if not inject_flag:
        _plog(
            "ERROR: Quantized Conv2d was NOT injected. "
            "Expect Unexpected *.comfy_quant keys and black output."
        )
    else:
        _plog("OK: Quantized Conv2d injected (comfy_quant load path ready)")

    for name in (
        "Linear",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
        "GroupNorm",
        "LayerNorm",
        "RMSNorm",
        "Embedding",
    ):
        if not hasattr(ops, name):
            fo = getattr(ForgeOperations, name, None)
            setattr(ops, name, fo if fo is not None else getattr(torch.nn, name))
            _plog(f"ops.{name} missing on MixedPrecisionOps → filled from Forge/torch")

    def _dequant_weight(layer, weight):
        convert = getattr(layer, "convert_weight", None)
        if callable(convert):
            try:
                out = convert(weight)
                data = out.data if hasattr(out, "data") else out
                try:
                    from comfy.quant_ops import QuantizedTensor

                    if isinstance(out, QuantizedTensor):
                        return out.dequantize()
                    if isinstance(data, QuantizedTensor):
                        return data.dequantize()
                except Exception:
                    pass
                return data
            except Exception:
                pass
        try:
            from comfy.quant_ops import QuantizedTensor

            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()
            data = weight.data if hasattr(weight, "data") else weight
            if isinstance(data, QuantizedTensor):
                return data.dequantize()
            return data
        except Exception:
            return weight.data if hasattr(weight, "data") else weight

    def _forward_with_forge_online(layer, input, core_forward):
        patches = getattr(layer, "forge_online_loras", None)
        if not patches:
            return core_forward(input)

        weight = getattr(layer, "weight", None)
        if weight is None:
            return core_forward(input)

        w = _dequant_weight(layer, weight)
        w = w.to(device=input.device, dtype=input.dtype)
        weight_patches = patches.get("weight")
        if weight_patches:
            w = merge_lora_to_weight(
                patches=weight_patches,
                weight=w,
                key="online weight lora",
                computation_dtype=input.dtype,
            )

        bias = getattr(layer, "bias", None)
        if bias is not None:
            bias = bias.to(device=input.device, dtype=input.dtype)
            bias_patches = patches.get("bias")
            if bias_patches:
                bias = merge_lora_to_weight(
                    patches=bias_patches,
                    weight=bias,
                    key="online bias lora",
                    computation_dtype=input.dtype,
                )

        if getattr(layer, "_hswq_quant_conv2d", False) or layer.__class__.__name__ == "Conv2d":
            # Prefer F.conv2d when online LoRA replaced weight; keep stride/padding from layer.
            if hasattr(layer, "stride"):
                return torch.nn.functional.conv2d(
                    input,
                    w,
                    bias,
                    getattr(layer, "stride", 1),
                    getattr(layer, "padding", 0),
                    getattr(layer, "dilation", 1),
                    getattr(layer, "groups", 1),
                )
        return torch.nn.functional.linear(input, w, bias)

    _OrigLinear = ops.Linear
    _OrigConv2d = ops.Conv2d

    class Linear(_OrigLinear):
        def forward(self, input, *args, **kwargs):
            return _forward_with_forge_online(
                self, input, lambda x: super(Linear, self).forward(x, *args, **kwargs)
            )

    class Conv2d(_OrigConv2d):
        # Preserve inject marker through online-LoRA wrapper subclass.
        _hswq_quant_conv2d = getattr(_OrigConv2d, "_hswq_quant_conv2d", False)

        def forward(self, input, *args, **kwargs):
            return _forward_with_forge_online(
                self, input, lambda x: super(Conv2d, self).forward(x, *args, **kwargs)
            )

    ops.Linear = Linear
    ops.Conv2d = Conv2d
    _plog(
        f"online-LoRA wrap applied; Conv2d._hswq_quant_conv2d="
        f"{getattr(ops.Conv2d, '_hswq_quant_conv2d', None)}"
    )
    return ops


def _count_sd_int8_keys(state_dict: dict) -> dict:
    cq = sum(1 for k in state_dict if k.endswith(".comfy_quant"))
    ws = sum(1 for k in state_dict if k.endswith(".weight_scale"))
    wi8 = 0
    try:
        import torch

        for k, v in state_dict.items():
            if k.endswith(".weight") and hasattr(v, "dtype") and v.dtype == torch.int8:
                wi8 += 1
    except Exception:
        pass
    return {"comfy_quant": cq, "weight_scale": ws, "int8_weight": wi8}


def _probe_model_quant(model) -> dict:
    try:
        from comfy.quant_ops import QuantizedTensor
    except Exception:
        QuantizedTensor = tuple()  # type: ignore

    n_qt = 0
    n_conv_quant = 0
    n_conv_plain = 0
    sample = None
    for name, mod in model.named_modules():
        is_conv = "Conv2d" in type(mod).__name__ or getattr(mod, "_hswq_quant_conv2d", False)
        if getattr(mod, "_hswq_quant_conv2d", False) or (
            is_conv and getattr(mod, "quant_format", None) is not None
        ):
            n_conv_quant += 1
        elif is_conv and hasattr(mod, "weight"):
            n_conv_plain += 1
        w = getattr(mod, "weight", None)
        if w is not None and QuantizedTensor and isinstance(w, QuantizedTensor):
            n_qt += 1
            if sample is None:
                sample = f"{name}: QuantizedTensor format={getattr(mod, 'quant_format', None)}"
        elif w is not None and sample is None and name.endswith("op"):
            sample = f"{name}: class={type(mod).__name__} wdtype={getattr(w, 'dtype', None)}"
    return {
        "quantized_tensor_weights": n_qt,
        "conv_quant_modules": n_conv_quant,
        "conv_plain_modules": n_conv_plain,
        "sample": sample,
    }


def load_unet_int8_branch(
    *,
    model_loader,
    unet_config: dict,
    state_dict: dict,
    state_dict_parameters,
    guess,
    cls_name: str,
    _nz: bool,
    precision=None,
    rank=None,
):
    """UNet load for UI Low Bits int8 / int8 (fp16 LoRA) only (no auto-detect)."""
    from backend import memory_management
    from backend.operations import using_forge_operations
    from backend.state_dict import load_state_dict

    _plog("=" * 60)
    _plog("PATH enter load_unet_int8_branch (UI int8 / int8 (fp16 LoRA) only)")
    _plog(f"cls_name={cls_name} _nz={_nz} state_dict_keys={len(state_dict)}")
    key_stats = _count_sd_int8_keys(state_dict)
    _plog(f"checkpoint INT8 key stats: {key_stats}")

    load_device = memory_management.get_torch_device()
    computation_dtype = memory_management.get_computation_dtype(
        load_device,
        parameters=state_dict_parameters,
        supported_dtypes=guess.supported_inference_dtypes,
    )
    offload_device = memory_management.unet_offload_device()
    initial_device = memory_management.unet_initial_load_device(
        parameters=state_dict_parameters, dtype=computation_dtype
    )
    _plog(
        f"devices: load={load_device} initial={initial_device} offload={offload_device} "
        f"computation_dtype={computation_dtype}"
    )

    # MUST build ops inside Conv-inject scope (see build_int8_operations docstring).
    int8_ops = build_int8_operations(computation_dtype)
    if cls_name in ("Lumina2Transformer2DModel", "ZImageTransformer2DModel") and not _nz:
        unet_config = dict(unet_config)
        unet_config["operations"] = int8_ops
        _plog("ZIT/Lumina2: unet_config['operations']=int8_ops")

    _plog("construct UNet under using_forge_operations(operations=int8_ops)")
    with using_forge_operations(
        operations=int8_ops,
        device=initial_device,
        dtype=computation_dtype,
        manual_cast_enabled=False,
    ):
        # torch.nn.Conv2d is temporarily int8_ops.Conv2d (Quantized).
        import torch

        _plog(
            f"during construct: torch.nn.Conv2d={torch.nn.Conv2d!r} "
            f"_hswq={getattr(torch.nn.Conv2d, '_hswq_quant_conv2d', None)}"
        )
        model = model_loader(unet_config)

    probe_before = _probe_model_quant(model)
    _plog(f"after construct (before load_state_dict): {probe_before}")

    if _nz:
        from backend.nn.svdq import patch_nunchaku_zimage

        model = patch_nunchaku_zimage(model, precision, rank)
        _plog("patched nunchaku zimage")
    elif cls_name in ("Lumina2Transformer2DModel", "ZImageTransformer2DModel"):
        from backend.nn.svdq import patch_standard_zimage

        model = patch_standard_zimage(model)
        _plog("patched standard zimage")

    _plog("load_state_dict …")
    load_state_dict(model, state_dict, log_name="IntegratedUNet2DConditionModel[INT8]")

    probe_after = _probe_model_quant(model)
    _plog(f"after load_state_dict: {probe_after}")
    if probe_after["quantized_tensor_weights"] == 0:
        _plog(
            "ERROR: zero QuantizedTensor weights after load — "
            "INT8 path failed; output will likely be NaN/black."
        )
    elif probe_after["conv_quant_modules"] == 0 and key_stats["comfy_quant"] > 0:
        _plog(
            "WARN: checkpoint has comfy_quant keys but no quant Conv modules — "
            "Conv layers may still be Unexpected / wrong."
        )
    else:
        _plog(
            f"OK path summary: qt_weights={probe_after['quantized_tensor_weights']} "
            f"conv_quant={probe_after['conv_quant_modules']}"
        )

    if hasattr(model, "_internal_dict"):
        model._internal_dict = unet_config
    else:
        model.config = unet_config

    model.storage_dtype = INT8_UNET_STORAGE
    model.computation_dtype = computation_dtype
    model.load_device = load_device
    model.initial_device = initial_device
    model.offload_device = offload_device
    _plog(
        f"PATH exit load_unet_int8_branch storage_dtype={model.storage_dtype} "
        f"computation_dtype={model.computation_dtype}"
    )
    _plog("=" * 60)
    return model
