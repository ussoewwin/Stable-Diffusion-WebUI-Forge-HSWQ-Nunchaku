"""Forge-side fallback when comfy_kitchen layouts are unavailable.

Issue #3 (TimothyDudorov): Krea2 quantized TE load crashes in
``comfy.ops._load_quantized_module``:

    layout_cls = get_layout_class(...)  # None when kitchen missing/stub
    params = layout_cls.Params(...)     # AttributeError

Do not edit ``ComfyUI-master/``. Patch ``comfy.ops`` at runtime so MixedPrecision
Linear (and Embedding) dequantize to float Parameters when layout registration
is missing. TE path uses ``full_precision_mm=True`` already, so float weights
are correct for inference.
"""

from __future__ import annotations

import json
import logging

import torch

_PATCHED = False
_LOG = logging.getLogger("forge.quant_layout_fallback")


def _dequant_to_compute(weight: torch.Tensor, quant_format: str, scales: dict, compute_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize stored weight to ``compute_dtype`` without kitchen layouts."""
    w = weight
    if quant_format in ("float8_e4m3fn", "float8_e5m2"):
        out = w.to(dtype=torch.float32)
        scale = scales.get("scale")
        if scale is not None:
            out = out * scale.to(device=out.device, dtype=torch.float32)
        return out.to(dtype=compute_dtype)

    if quant_format == "int8_tensorwise":
        out = w.to(dtype=torch.float32)
        scale = scales.get("scale")
        if scale is None:
            raise ValueError("Missing INT8 weight_scale for Forge dequant fallback")
        s = scale.to(device=out.device, dtype=torch.float32)
        if s.ndim == 0:
            out = out * s
        elif s.ndim == 1:
            out = out * s.reshape(-1, *([1] * (out.ndim - 1)))
        else:
            out = out * s
        if scales.get("convrot", False):
            gs = int(scales.get("convrot_groupsize", 256) or 256)
            if out.dim() == 2 and out.shape[1] < gs and (out.shape[1] in (16, 64, 128)):
                gs = out.shape[1]
            from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard, _rotate_weight
            h = _build_hadamard(gs, device=out.device, dtype=torch.float32)
            out = _rotate_weight(out, h, gs)
        return out.to(dtype=compute_dtype)

    raise RuntimeError(
        f"[Forge] comfy_kitchen layout unavailable for format={quant_format!r}. "
        "Install/update comfy_kitchen, or use a non-quantized checkpoint for this TE/UNet."
    )


def _forge_dequant_load_quantized_module(
    module,
    super_load,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
    load_extra_params=False,
):
    """Same key consumption as ``_load_quantized_module``, but float Parameter output."""
    from comfy.quant_ops import QUANT_ALGOS

    device = module.factory_kwargs["device"]
    compute_dtype = module.factory_kwargs["dtype"]
    layer_name = prefix.rstrip(".")

    weight = state_dict.pop(f"{prefix}weight", None)
    if weight is None:
        logging.warning("Missing weight for layer %s", layer_name)
        module.weight = None
        return
    manually_loaded_keys = [f"{prefix}weight"]

    def pop_scale(name, dtype=None):
        key = f"{prefix}{name}"
        v = state_dict.pop(key, None)
        if v is not None:
            v = v.to(device=device)
            if dtype is not None:
                v = v.view(dtype=dtype)
            manually_loaded_keys.append(key)
        return v

    layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
    if layer_conf is None:
        module.weight = torch.nn.Parameter(weight.to(device=device, dtype=compute_dtype), requires_grad=False)
        super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        for key in manually_loaded_keys:
            if key in missing_keys:
                missing_keys.remove(key)
        return

    layer_conf = json.loads(layer_conf.numpy().tobytes())
    quant_format = layer_conf.get("format", None)
    if quant_format is None:
        raise ValueError(f"Unknown quantization format for layer {layer_name}")
    if quant_format not in QUANT_ALGOS:
        raise ValueError(f"Unsupported quantization format: {quant_format}")

    scales = {}
    if quant_format in ("float8_e4m3fn", "float8_e5m2"):
        scales = {"scale": pop_scale("weight_scale")}
    elif quant_format == "int8_tensorwise":
        scale = pop_scale("weight_scale")
        if scale is None:
            raise ValueError(f"Missing INT8 weight scale for layer {layer_name}")
        scales = {
            "scale": scale,
            "convrot": bool(layer_conf.get("convrot", False)),
            "convrot_groupsize": int(layer_conf.get("convrot_groupsize", 256) or 256),
        }
        # drop unused scale keys that may exist for other formats
        pop_scale("weight_scale_2")
        pop_scale("input_scale")
    elif quant_format == "mxfp8":
        raise RuntimeError(
            f"[Forge] comfy_kitchen required for mxfp8 layer {layer_name} (no float dequant fallback)."
        )
    elif quant_format == "nvfp4":
        raise RuntimeError(
            f"[Forge] comfy_kitchen required for nvfp4 layer {layer_name} (no float dequant fallback)."
        )
    elif quant_format == "convrot_w4a4":
        raise RuntimeError(
            f"[Forge] comfy_kitchen required for convrot_w4a4 layer {layer_name} (no float dequant fallback)."
        )
    else:
        raise ValueError(f"Unsupported quantization format: {quant_format}")

    if load_extra_params:
        qconfig = QUANT_ALGOS[quant_format]
        for param_name in qconfig["parameters"]:
            if param_name in {"weight_scale", "weight_scale_2"}:
                continue
            param_key = f"{prefix}{param_name}"
            _v = state_dict.pop(param_key, None)
            if _v is None:
                continue
            module.register_parameter(param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False))
            manually_loaded_keys.append(param_key)

    print(
        f"[Forge] comfy_kitchen layout missing for {quant_format} — "
        f"dequantizing {layer_name} to {compute_dtype} (Issue #3 fallback)"
    )
    dequant = _dequant_to_compute(weight, quant_format, scales, compute_dtype)
    module.weight = torch.nn.Parameter(dequant.to(device=device), requires_grad=False)
    module.layout_type = None
    module.quant_format = None
    module._full_precision_mm = True
    module._full_precision_mm_config = True

    super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    for key in manually_loaded_keys:
        if key in missing_keys:
            missing_keys.remove(key)


def _wrap_load_quantized_module(orig):
    def _load_quantized_module(
        module,
        super_load,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        load_extra_params=False,
    ):
        from comfy.quant_ops import QUANT_ALGOS, get_layout_class

        quant_key = f"{prefix}comfy_quant"
        if quant_key in state_dict:
            try:
                conf = json.loads(state_dict[quant_key].numpy().tobytes())
                fmt = conf.get("format")
                layout_name = QUANT_ALGOS.get(fmt, {}).get("comfy_tensor_layout")
                if layout_name is not None and get_layout_class(layout_name) is None:
                    return _forge_dequant_load_quantized_module(
                        module,
                        super_load,
                        state_dict,
                        prefix,
                        local_metadata,
                        strict,
                        missing_keys,
                        unexpected_keys,
                        error_msgs,
                        load_extra_params=load_extra_params,
                    )
            except Exception as e:
                _LOG.debug("layout peek failed for %s: %s", prefix, e)

        return orig(
            module,
            super_load,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            load_extra_params=load_extra_params,
        )

    return _load_quantized_module


def _wrap_embedding_load(orig_load, compute_dtype: torch.dtype):
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        from comfy.quant_ops import QUANT_ALGOS, get_layout_class

        weight_key = f"{prefix}weight"
        quant_key = f"{prefix}comfy_quant"
        if quant_key in state_dict and weight_key in state_dict:
            try:
                layer_conf = json.loads(state_dict[quant_key].numpy().tobytes())
                quant_format = layer_conf.get("format") if layer_conf is not None else None
                if quant_format in ("float8_e4m3fn", "float8_e5m2", "int8_tensorwise"):
                    layout_name = QUANT_ALGOS[quant_format]["comfy_tensor_layout"]
                    if get_layout_class(layout_name) is None:
                        state_dict.pop(quant_key)
                        weight = state_dict.pop(weight_key)
                        scale_key = f"{prefix}weight_scale"
                        scale = state_dict.pop(scale_key, None)
                        device = weight.device
                        scales = {
                            "scale": scale.to(device=device) if scale is not None else None,
                            "convrot": bool(layer_conf.get("convrot", False)),
                            "convrot_groupsize": int(layer_conf.get("convrot_groupsize", 256) or 256),
                        }
                        dequant = _dequant_to_compute(weight, quant_format, scales, compute_dtype)
                        self.weight = torch.nn.Parameter(dequant, requires_grad=False)
                        self.layout_type = None
                        self.quant_format = None
                        print(
                            f"[Forge] comfy_kitchen layout missing for {quant_format} — "
                            f"dequantizing embedding {prefix.rstrip('.')} to {compute_dtype} (Issue #3 fallback)"
                        )
                        torch.nn.Module._load_from_state_dict(
                            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
                        )
                        for k in (weight_key, scale_key, quant_key):
                            if k in missing_keys:
                                missing_keys.remove(k)
                        return
            except Exception as e:
                _LOG.debug("embedding layout peek failed for %s: %s", prefix, e)

        return orig_load(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    return _load_from_state_dict


def _wrap_mixed_precision_ops(orig_mp):
    def mixed_precision_ops(*args, **kwargs):
        MP = orig_mp(*args, **kwargs)
        emb = getattr(MP, "Embedding", None)
        if emb is not None and not getattr(emb, "_forge_layout_fallback", False):
            cd = getattr(MP, "_compute_dtype", torch.bfloat16)
            emb._load_from_state_dict = _wrap_embedding_load(emb._load_from_state_dict, cd)
            emb._forge_layout_fallback = True
        return MP

    return mixed_precision_ops


def ensure_comfy_quant_layout_fallback_patch() -> bool:
    """Idempotent. Returns True when patch is active."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        import comfy.ops as ops
    except ImportError as e:
        print(f"[Forge] quant layout fallback: comfy.ops not importable yet ({e})")
        return False

    if getattr(ops, "_forge_layout_fallback_patched", False):
        _PATCHED = True
        return True

    if not hasattr(ops, "_load_quantized_module"):
        print("[Forge] quant layout fallback: _load_quantized_module missing")
        return False

    ops._load_quantized_module = _wrap_load_quantized_module(ops._load_quantized_module)
    if hasattr(ops, "mixed_precision_ops"):
        ops.mixed_precision_ops = _wrap_mixed_precision_ops(ops.mixed_precision_ops)

    ops._forge_layout_fallback_patched = True
    _PATCHED = True
    print("[Forge] comfy_quant layout fallback patch active (dequant when kitchen layout is None)")
    return True


def dequantize_state_dict_inplace(state_dict: dict, compute_dtype=None) -> dict:
    """Dequantize any quantized weights (*.weight_scale, *.comfy_quant) in state_dict in-place.

    Supports:
      - int8_tensorwise (with or without ConvRot Hadamard rotation)
      - float8_e4m3fn / float8_e5m2
    """
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict

    has_quant = any(
        isinstance(k, str) and (k.endswith(".weight_scale") or k.endswith(".comfy_quant"))
        for k in state_dict.keys()
    )
    if not has_quant:
        return state_dict

    import time
    t0 = time.time()
    count = 0

    from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard, _rotate_weight

    h_cache = {}

    use_cuda = torch.cuda.is_available()
    has_ck_cuda = False
    if use_cuda:
        try:
            import comfy_kitchen
            if hasattr(torch.ops.comfy_kitchen, "dequantize_int8_convrot_weight_dtype"):
                has_ck_cuda = True
        except Exception:
            pass

    if compute_dtype is None:
        for v in state_dict.values():
            if isinstance(v, torch.Tensor) and v.dtype in (torch.bfloat16, torch.float16):
                compute_dtype = v.dtype
                break
        if compute_dtype is None:
            compute_dtype = torch.float16

    scale_keys = [k for k in list(state_dict.keys()) if isinstance(k, str) and k.endswith(".weight_scale")]

    for scale_key in scale_keys:
        base = scale_key[: -len(".weight_scale")]
        w_key = base + ".weight"
        q_key = base + ".comfy_quant"

        if w_key not in state_dict:
            continue

        weight = state_dict[w_key]
        scale = state_dict.pop(scale_key)
        q_tensor = state_dict.pop(q_key, None)

        target_dtype = compute_dtype

        layer_conf = {}
        if q_tensor is not None:
            try:
                if hasattr(q_tensor, "numpy"):
                    layer_conf = json.loads(q_tensor.numpy().tobytes())
                elif isinstance(q_tensor, (bytes, bytearray)):
                    layer_conf = json.loads(bytes(q_tensor))
                elif isinstance(q_tensor, str):
                    layer_conf = json.loads(q_tensor)
            except Exception:
                layer_conf = {}

        quant_format = layer_conf.get("format", "int8_tensorwise")
        has_convrot = bool(layer_conf.get("convrot", False))
        gs = int(layer_conf.get("convrot_groupsize", 256) or 256)
        if weight.dim() == 2 and weight.shape[1] < gs and (weight.shape[1] in (16, 64, 128)):
            gs = weight.shape[1]

        if quant_format in ("float8_e4m3fn", "float8_e5m2"):
            out = weight.to(dtype=torch.float32)
            if scale is not None:
                out = out * scale.to(device=out.device, dtype=torch.float32)
            state_dict[w_key] = out.to(dtype=target_dtype)
            count += 1
        elif quant_format == "int8_tensorwise" or weight.dtype == torch.int8:
            if has_convrot and weight.dim() == 2:
                dequantized = False
                if has_ck_cuda:
                    try:
                        w_cu = weight.cuda()
                        s_cu = scale.to(device=w_cu.device, dtype=torch.float32).reshape(-1).contiguous()
                        code = 1 if target_dtype == torch.float16 else (2 if target_dtype == torch.bfloat16 else 0)
                        out_cu = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(w_cu, s_cu, gs, code)
                        state_dict[w_key] = out_cu.to(dtype=target_dtype).cpu()
                        del w_cu, s_cu, out_cu
                        dequantized = True
                    except Exception:
                        dequantized = False

                if not dequantized:
                    rot_dev = "cuda" if use_cuda else weight.device
                    h_key = (gs, str(rot_dev))
                    if h_key not in h_cache:
                        h_cache[h_key] = _build_hadamard(gs, device=rot_dev, dtype=torch.float32)
                    w_f = weight.to(device=rot_dev, dtype=torch.float32)
                    s = scale.to(device=rot_dev, dtype=torch.float32)
                    if s.ndim == 1:
                        s = s.reshape(-1, *([1] * (weight.ndim - 1)))
                    scaled = w_f * s
                    rotated = _rotate_weight(scaled, h_cache[h_key], gs)
                    state_dict[w_key] = rotated.to(dtype=target_dtype).cpu()
                    del w_f, s, scaled, rotated
            else:
                s = scale.to(device=weight.device, dtype=torch.float32)
                if s.ndim == 1:
                    s = s.reshape(-1, *([1] * (weight.ndim - 1)))
                state_dict[w_key] = (weight.to(dtype=torch.float32) * s).to(dtype=target_dtype)
            count += 1

        state_dict.pop(base + ".weight_scale_2", None)
        state_dict.pop(base + ".input_scale", None)

    for k in list(state_dict.keys()):
        if isinstance(k, str) and k.endswith(".comfy_quant"):
            state_dict.pop(k, None)

    if count > 0:
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print(f"[Forge] Dequantized {count} ConvRot/INT8 weights to {target_dtype} in {time.time() - t0:.2f}s")

    return state_dict

