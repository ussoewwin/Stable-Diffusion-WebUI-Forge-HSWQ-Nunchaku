# Qwen Image Fun ControlNet: control_img_in + control_blocks.0..4 only (174 keys).
# Uses ComfyUI's QwenImageFunControlNetModel directly for correct attention/modulation.
# Injects via transformer_options["control"]["input"], NOT double_block patches.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from modules_forge.supported_controlnet import ControlModelPatcher

# Import ComfyUI's own model classes directly — these inherit from
# QwenImageTransformerBlock and have correct attention, modulation, LayerNorm etc.
import comfy.ldm.qwen_image.controlnet as comfy_qwen_cn
import comfy.ops


def _build_raw_hint_latent(cond, process=None) -> Optional[torch.Tensor]:
    """
    Convert the control image condition into a raw latent tensor suitable for
    ComfyUI's QwenImageFunControlNetModel.forward(hint=...).

    ComfyUI's _process_hint_tokens expects 4D (B,C,H,W) or 5D (B,C,T,H,W) raw latent.
    If cond is already a latent (C>=16), return it directly.
    If cond is an RGB image (C<=4), encode through VAE first.
    """
    if cond is None or not hasattr(cond, "shape"):
        return None

    if cond.ndim == 3:
        # (B, seq, dim) — already patchified, can't use directly
        print(f"[Qwen Fun ControlNet] cond is 3D (patchified), cannot use as raw hint: {cond.shape}")
        return None

    if cond.ndim == 4:
        B, C, H, W = cond.shape
        if C <= 4:
            # RGB/RGBA image — needs VAE encoding
            if process is not None:
                vae = getattr(getattr(process.sd_model, "forge_objects", None), "vae", None)
                if vae is not None:
                    if C == 1:
                        cond = cond.repeat(1, 3, 1, 1)
                    x = cond.movedim(1, -1)  # B,C,H,W -> B,H,W,C for VAE
                    try:
                        latent = vae.encode(x)
                    except Exception as e:
                        print(f"[Qwen Fun ControlNet] VAE encode failed: {e}")
                        return None
                    if latent.ndim == 4:
                        if latent.shape[-1] < latent.shape[1]:
                            latent = latent.movedim(-1, 1)
                    print(f"[Qwen Fun ControlNet] VAE encoded: {cond.shape} -> {latent.shape}")
                    return latent
            print(f"[Qwen Fun ControlNet] cond is RGB (C={C}) but no VAE available")
            return None
        else:
            # Already a latent (C>=16), return as-is
            print(f"[Qwen Fun ControlNet] Using raw latent hint: {cond.shape}")
            return cond

    if cond.ndim == 5:
        # 5D video latent (B,C,T,H,W), return as-is
        print(f"[Qwen Fun ControlNet] Using raw 5D latent hint: {cond.shape}")
        return cond

    print(f"[Qwen Fun ControlNet] Unexpected cond ndim={cond.ndim}: {cond.shape}")
    return None


def _build_comfyui_fun_model(state_dict: dict, dtype=None):
    """Instantiate ComfyUI's QwenImageFunControlNetModel and load weights."""
    try:
        from backend.patch_comfy_quant_layout_fallback import dequantize_state_dict_inplace
        state_dict = dequantize_state_dict_inplace(state_dict, compute_dtype=torch.bfloat16)
    except Exception as e:
        print(f"[Qwen Fun ControlNet] dequantize_state_dict_inplace note: {e}")

    in_features = state_dict["control_img_in.weight"].shape[1]  # 132
    inner_dim = state_dict["control_img_in.weight"].shape[0]  # 3072

    block_weight = state_dict["control_blocks.0.attn.to_q.weight"]
    attention_head_dim = state_dict["control_blocks.0.attn.norm_q.weight"].shape[0]  # 128
    num_attention_heads = max(1, block_weight.shape[0] // max(1, attention_head_dim))  # 24

    if dtype is None:
        dtype = block_weight.dtype
        if getattr(dtype, "is_floating_point", None) is False:
            dtype = torch.bfloat16

    operations = comfy.ops.disable_weight_init

    model = comfy_qwen_cn.QwenImageFunControlNetModel(
        control_in_features=in_features,
        inner_dim=inner_dim,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        num_control_blocks=5,
        main_model_double=60,
        injection_layers=(0, 12, 24, 36, 48),
        operations=operations,
        device="cpu",
        dtype=dtype,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        real_missing = [k for k in missing if "norm" not in k]
        if real_missing:
            print(f"[Qwen Fun ControlNet] Missing keys (non-norm): {real_missing}")
    return model


class QwenFunControlNetPatcher(ControlModelPatcher):
    """
    Patcher that injects ControlNet signals via transformer_options["control"]["input"].
    Uses ComfyUI's QwenImageFunControlNetModel.forward() with the Nunchaku model as base_model.
    """
    def __init__(self, fun_model: QwenFunControlNetModel, control_context: torch.Tensor, strength: float = 1.0):
        self.fun_model = fun_model
        self.control_context = control_context  # (B, 1, 132) or (B, L, 132)
        self.strength = strength
        self._state: dict = {"c": None}
        self._logged_run = False

    def __call__(self, inp: dict) -> dict:
        img = inp["img"]
        txt = inp["txt"]
        vec = inp.get("vec")
        pe = inp.get("pe")
        i = inp.get("block_index", -1)
        if vec is None or pe is None:
            return {"img": img, "txt": txt}
        if i not in self.CONTROL_BLOCK_INDICES:
            return {"img": img, "txt": txt}
        if not self._logged_run:
            self._logged_run = True
        dev = img.device
        if next(self.fun_model.parameters()).device != dev:
            self.fun_model.to(dev)
        cc = self.control_context.to(device=dev)
        if cc.shape[0] != img.shape[0]:
            cc = cc.expand(img.shape[0], -1, -1)
        idx_in_fun = self.CONTROL_BLOCK_INDICES.index(i)
        if idx_in_fun == 0:
            self._state["c"] = self.fun_model.control_img_in(cc)
            if self._state["c"].dim() == 2:
                self._state["c"] = self._state["c"].unsqueeze(1).expand(-1, img.shape[1], -1)
        # Transformer hidden_dim may differ from Fun's 3072 (e.g. 5776); feed 3072-dim views and pad output
        hidden_dim = img.shape[-1]
        txt_dim = txt.shape[-1]
        if hidden_dim >= FUN_INNER_DIM:
            img_in = img[:, :, :FUN_INNER_DIM].contiguous()
            if txt_dim >= FUN_INNER_DIM:
                txt_in = txt[:, :, :FUN_INNER_DIM].contiguous()
            else:
                txt_in = torch.nn.functional.pad(txt, (0, FUN_INNER_DIM - txt_dim), value=0)
        else:
            pad_img = FUN_INNER_DIM - hidden_dim
            img_in = torch.nn.functional.pad(img, (0, pad_img), value=0)
            if txt_dim >= FUN_INNER_DIM:
                txt_in = txt[:, :, :FUN_INNER_DIM].contiguous()
            else:
                txt_in = torch.nn.functional.pad(txt, (0, FUN_INNER_DIM - txt_dim), value=0)
        # blk(c, img, ...) does c + img inside; c and img_in must match shape (seq, feature).
        c_in = self._state["c"]
        if c_in is None:
            return {"img": img, "txt": txt}
        while c_in.dim() < 3:
            c_in = c_in.unsqueeze(1)
        seq_len, feat_len = img_in.shape[1], img_in.shape[2]
        if c_in.shape[1] != seq_len:
            if c_in.shape[1] < seq_len:
                n = (seq_len + c_in.shape[1] - 1) // max(1, c_in.shape[1])
                c_in = c_in.repeat(1, n, 1)[:, :seq_len, :]
            else:
                c_in = c_in[:, :seq_len, :]
        if c_in.shape[2] != feat_len:
            if c_in.shape[2] < feat_len:
                c_in = torch.nn.functional.pad(c_in, (0, feat_len - c_in.shape[2]), value=0)
            else:
                c_in = c_in[:, :, :feat_len]
        blk = self.fun_model.control_blocks[idx_in_fun]
        add = blk(c_in, img_in, txt_in, vec, pe, None)
        self._state["c"] = add
        t = min(img.shape[1], add.shape[1])
        if t > 0:
            if hidden_dim > FUN_INNER_DIM:
                add_full = torch.nn.functional.pad(add[:, :t], (0, hidden_dim - FUN_INNER_DIM), value=0)
            elif hidden_dim < FUN_INNER_DIM:
                add_full = add[:, :t, :hidden_dim]
            else:
                add_full = add[:, :t]
            img = img + self.strength * add_full
        return {"img": img, "txt": txt}


def _is_qwen_image_unet(unet) -> bool:
    """Force-apply: detect by diffusion_model type, not path. Accepts Nunchaku/Std/comfy_ldm Qwen Image."""
    diffusion_model = getattr(getattr(unet, "model", None), "diffusion_model", None)
    if diffusion_model is None:
        return False
    if type(diffusion_model).__name__ == "QwenImageTransformer2DModel":
        return True
    try:
        from backend.nn.svdq import NunchakuQwenImageTransformer2DModel
        from backend.nn.qwen import QwenImageTransformer2DModel as StdQwenImageTransformer2DModel
        from backend.nn.comfy_ldm.qwen_image.model import QwenImageTransformer2DModel as ComfyLdmQwenImageTransformer2DModel
        return isinstance(diffusion_model, (NunchakuQwenImageTransformer2DModel, StdQwenImageTransformer2DModel, ComfyLdmQwenImageTransformer2DModel))
    except Exception:
        return False


def _encode_image_to_132(cond: torch.Tensor, vae, proj_cache: dict, device, dtype) -> Optional[torch.Tensor]:
    """Encode control image (B,C,H,W) to (B,1,132) via VAE + pool + learned projection. proj_cache is on patcher, not on model (keeps model = checkpoint only)."""
    if cond.ndim != 4:
        return None
    B, C, H, W = cond.shape
    if C == 1:
        cond = cond.repeat(1, 3, 1, 1)
    x = cond.to(device=device, dtype=dtype).movedim(1, -1)
    try:
        latent = vae.encode(x)
    except Exception:
        return None
    if latent.ndim != 4:
        return None
    latent_c = latent.shape[1]
    pooled = latent.mean(dim=(2, 3))
    if pooled.shape[1] != latent_c:
        return None
    if latent_c not in proj_cache:
        proj_cache[latent_c] = torch.randn(latent_c, 132, device=device, dtype=dtype) * 0.02
    proj = proj_cache[latent_c]
    ctx = torch.nn.functional.linear(pooled, proj)
    return ctx.unsqueeze(1)


def _build_fun_control_context(cond, fun_model, process=None, proj_cache=None) -> Optional[torch.Tensor]:
    if cond is None or not hasattr(cond, "shape"):
        return None
    sh = cond.shape
    dtype = next(fun_model.parameters()).dtype
    if sh[-1] >= 132:
        cond = cond.to(dtype=dtype)
        control_context = cond[..., :132].contiguous()
        if control_context.dim() == 2:
            control_context = control_context.unsqueeze(1)
        return control_context
    if process is not None and cond.ndim == 4 and proj_cache is not None:
        vae = getattr(getattr(process.sd_model, "forge_objects", None), "vae", None)
        if vae is not None:
            if not hasattr(vae, "spacial_compression_encode"):
                from modules_forge.supported_controlnet_qwen_image import QwenImageVAEWrapper
                vae = QwenImageVAEWrapper(vae)
            device = next(fun_model.parameters()).device
            out = _encode_image_to_132(cond, vae, proj_cache, device, dtype)
            if out is not None:
                return out
    return None


class QwenFunControlNetPatcher(ControlModelPatcher):
    """Patcher that registers QwenFunControlPatch via set_model_patch (Z Image style).
    Compatible with existing load/apply flow: same try_build_from_state_dict, same process_before_every_sampling(process, cond, mask), strength."""


    @staticmethod
    def try_build_from_state_dict(controlnet_data: dict, ckpt_path: str) -> Optional["ControlModelPatcher"]:
        _marker = "control_img_in.weight"
        _keys = list(controlnet_data.keys())
        _found = [k for k in _keys if _marker in str(k)]
        if not _found:
            return None
        _prefix = str(_found[0]).split(_marker)[0]
        if _prefix:
            _data = {str(k)[len(_prefix):] if str(k).startswith(_prefix) else str(k): v for k, v in controlnet_data.items()}
        else:
            _data = {str(k): v for k, v in controlnet_data.items()}
        try:
            loaded = {k: v for k, v in _data.items() if k.startswith(("control_img_in.", "control_blocks."))}
            if not loaded:
                return None
            model = _build_comfyui_fun_model(loaded)
            print(f"[Qwen Fun ControlNet] Successfully loaded Qwen Image Fun ControlNet: {ckpt_path}")
            print(f"[Qwen Fun ControlNet] Using ComfyUI QwenImageFunControlNetModel (direct import)")
            return QwenFunControlNetPatcher(model)
        except Exception as e:
            print(f"[Qwen Fun ControlNet] Error loading: {e}")
            import traceback
            traceback.print_exc()
            return None

    def __init__(self, fun_model):
        super().__init__(model_patcher=None)
        self.fun_model = fun_model

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        # Verify this is a Qwen Image model
        diffusion_model = getattr(getattr(unet, "model", None), "diffusion_model", None)
        if diffusion_model is None:
            return

        is_qwen_image = False
        try:
            from backend.nn.svdq import NunchakuQwenImageTransformer2DModel
            from backend.nn.qwen import QwenImageTransformer2DModel as StdQwenImageTransformer2DModel
            if isinstance(diffusion_model, (NunchakuQwenImageTransformer2DModel, StdQwenImageTransformer2DModel)):
                is_qwen_image = True
        except ImportError:
            pass
        if not is_qwen_image:
            dn = type(diffusion_model).__name__
            if "QwenImage" in dn:
                is_qwen_image = True
        if not is_qwen_image:
            print(f"[Qwen Fun ControlNet] REJECTED: This ControlNet is for Qwen Image models ONLY!")
            print(f"[Qwen Fun ControlNet] Current model: {type(diffusion_model).__name__}")
            return

        # Build raw hint latent from the control image (NOT patchified)
        # ComfyUI's _process_hint_tokens handles patchification internally
        raw_hint = _build_raw_hint_latent(cond, process=process)
        if raw_hint is None:
            print(f"[Qwen Fun ControlNet] REJECTED: could not build raw hint latent from cond")
            print(f"[Qwen Fun ControlNet] cond type={type(cond)}, shape={getattr(cond, 'shape', 'N/A')}")
            return

        strength = getattr(self, "strength", 1.0)
        print(f"[Qwen Fun ControlNet] Processing ControlNet for Qwen Image")
        print(f"[Qwen Fun ControlNet] Strength: {strength}, hint shape: {raw_hint.shape}")

        # Move fun_model to the same device and floating-point dtype as diffusion_model
        try:
            target_dev = next(diffusion_model.parameters()).device
        except StopIteration:
            target_dev = torch.device("cuda:0")
        target_dtype = torch.bfloat16
        for p in diffusion_model.parameters():
            if p.dtype in (torch.bfloat16, torch.float16, torch.float32):
                target_dtype = p.dtype
                break
        fun_param = next(self.fun_model.parameters())
        if fun_param.device != target_dev or fun_param.dtype != target_dtype:
            self.fun_model.to(device=target_dev, dtype=target_dtype)
            print(f"[Qwen Fun ControlNet] Moved Fun model from {fun_param.device}/{fun_param.dtype} to {target_dev}/{target_dtype}")

        # Create a model_function_wrapper that injects control at each step
        fun_model = self.fun_model
        hint_latent = raw_hint
        ctrl_strength = strength
        base_model_ref = diffusion_model
        step_counter = [0]

        def qwen_fun_controlnet_wrapper(apply_model_func, args):
            """Wrapper that calls Fun ControlNet per step and injects via transformer_options['control']."""
            input_x = args["input"]
            timestep = args["timestep"]
            c = args["c"]

            # Get transformer_options from the conditioning
            transformer_options = c.get("transformer_options", {})

            # Get the text context from c_crossattn
            context = c.get("c_crossattn", None)

            # Get guidance if available
            guidance = c.get("guidance", None)

            step_counter[0] += 1

            try:
                # Determine fun_model's floating-point dtype for casting inputs
                try:
                    model_dtype = next(fun_model.parameters()).dtype
                except StopIteration:
                    model_dtype = torch.bfloat16

                # Cast input_x to model dtype (samplers may operate in float64)
                x_cast = input_x.to(dtype=model_dtype)

                # Move hint to match input device/dtype
                hint = hint_latent.to(device=x_cast.device, dtype=model_dtype)

                # Ensure batch size matches (may differ for CFG)
                if hint.shape[0] != x_cast.shape[0]:
                    hint = hint.expand(x_cast.shape[0], *hint.shape[1:])

                # Cast context to model dtype too
                ctx = context.to(dtype=model_dtype) if context is not None else None

                # Cast timestep (keep as-is, it's typically int or float)
                ts = timestep.to(dtype=model_dtype) if timestep is not None else None

                # Run the Fun ControlNet forward pass
                # This calls QwenImageFunControlNetModel.forward(x, timesteps, context, hint=..., base_model=...)
                # which returns {"input": [None]*60 with tensors at injection layers}
                with torch.no_grad():
                    control_output = fun_model(
                        x=x_cast,
                        timesteps=ts,
                        context=ctx,
                        guidance=guidance,
                        hint=hint,
                        transformer_options=transformer_options,
                        base_model=base_model_ref,
                    )

                # Inject control output into c["control"] — this is the standard
                # Forge/ComfyUI path: _apply_model passes it as the `control`
                # parameter to diffusion_model._forward(), where the per-block
                # residual add happens.
                if control_output is not None:
                    control_list = control_output.get("input", [])
                    scaled_control_list = []
                    for ct in control_list:
                        if ct is not None:
                            scaled_control_list.append(ct * ctrl_strength)
                        else:
                            scaled_control_list.append(None)
                            
                    control_dict = {
                        "input": scaled_control_list,
                    }
                    c["control"] = control_dict
                    if step_counter[0] <= 2:
                        n_active = sum(1 for x in scaled_control_list if x is not None)
                        print(f"[Qwen Fun ControlNet] Step {step_counter[0]}: Injected control into c['control'] ({n_active} active layers, strength={ctrl_strength})")
                        for idx, ct in enumerate(scaled_control_list):
                            if ct is not None:
                                print(f"  [ctrl] layer {idx}: shape={ct.shape}, dtype={ct.dtype}, norm={ct.norm().item():.4f}")
                else:
                    print(f"[Qwen Fun ControlNet] Step {step_counter[0]}: control_output is None!")

            except Exception as e:
                print(f"[Qwen Fun ControlNet] Error in per-step control: {e}")
                import traceback
                traceback.print_exc()

            return apply_model_func(input_x, timestep, **c)

        unet.set_model_unet_function_wrapper(qwen_fun_controlnet_wrapper)
        print(f"[Qwen Fun ControlNet] ControlNet wrapper registered (per-step control injection)")
