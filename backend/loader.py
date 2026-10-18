import importlib
import logging
import os

import huggingface_guess
import torch
from diffusers import DiffusionPipeline
from transformers import modeling_utils

import backend.args
from backend import memory_management
from backend.comfy_te_glue import comfy_te_model_options, is_comfy_sd_clip
from backend.diffusion_engine.chroma import Chroma
from backend.diffusion_engine.flux import Flux
from backend.diffusion_engine.anima import Anima
from backend.diffusion_engine.krea2 import Krea2
from backend.diffusion_engine.lumina import Lumina2
from backend.diffusion_engine.qwen import QwenImage
from backend.diffusion_engine.sd15 import StableDiffusion
from backend.diffusion_engine.sdxl import StableDiffusionXL, StableDiffusionXLRefiner
from backend.diffusion_engine.zimage import ZImage
from backend.nn.clip import IntegratedCLIP
from backend.nn.unet import IntegratedUNet2DConditionModel
from backend.nn.vae import IntegratedAutoencoderKL
from backend.nn.wan_vae import WanVAE
from backend.operations import using_forge_operations
from backend.state_dict import load_state_dict, try_filter_state_dict
from backend.utils import (
    beautiful_print_gguf_state_dict_statics,
    load_torch_file,
    read_arbitrary_config,
)
from modules_forge.packages.huggingface_guess import model_list

possible_models = [StableDiffusion, StableDiffusionXLRefiner, StableDiffusionXL, Chroma, Flux, QwenImage, Anima, Lumina2, ZImage, Krea2]


logging.getLogger("diffusers").setLevel(logging.ERROR)
dir_path = os.path.dirname(__file__)

_ANIMA_GUESS_NAMES = frozenset({"Anima", "AnimaBase", "AnimaWai68"})


def _is_anima_guess(guess) -> bool:
    """Anima guess comes from ``modules_forge`` model_list; do not use cross-module isinstance."""
    return type(guess).__name__ in _ANIMA_GUESS_NAMES


def _is_krea2_guess(guess) -> bool:
    """Krea2 guess comes from ``modules_forge`` model_list; do not use cross-module isinstance."""
    return type(guess).__name__ == "Krea2"


def _state_dict_has_comfy_quant(state_dict) -> bool:
    return any(isinstance(k, str) and k.endswith(".comfy_quant") for k in state_dict)


def _load_krea2_mixed_precision_unet(
    model_loader,
    unet_config,
    state_dict,
    state_dict_parameters,
    guess,
):
    """Krea2-only MixedPrecision path for Comfy ``comfy_quant`` / ``weight_scale`` checkpoints.

    Early return used only when:
      - ``cls_name == "SingleStreamDiT"``
      - ``_is_krea2_guess(guess)``
      - state_dict contains ``*.comfy_quant``

    Must not alter Flux / Anima / Qwen / ZImage / INT8 / shared float8 construct.
    Forge Neo equivalent: detect comfy_quant → mixed_precision_ops + bf16 storage
    (UI float8_e4m3fn must not cast-destroy QuantizedTensor → noise).
    """
    from backend.patch_comfy_quant_layout_fallback import ensure_comfy_quant_layout_fallback_patch

    ensure_comfy_quant_layout_fallback_patch()
    import comfy.ops
    import comfy.model_management as cmm

    load_device = memory_management.get_torch_device()
    computation_dtype = memory_management.get_computation_dtype(
        load_device,
        parameters=state_dict_parameters,
        supported_dtypes=guess.supported_inference_dtypes,
    )
    # Neo: quant_config present → storage stays bf16 (ignore float8 UI overwrite).
    storage_dtype = torch.bfloat16
    offload_device = memory_management.unet_offload_device()
    initial_device = memory_management.unet_initial_load_device(
        parameters=state_dict_parameters, dtype=computation_dtype
    )

    disabled = set()
    if not cmm.supports_nvfp4_compute(load_device):
        disabled.add("nvfp4")
    if not cmm.supports_mxfp8_compute(load_device):
        disabled.add("mxfp8")
    if not cmm.supports_fp8_compute(load_device):
        disabled.add("float8_e4m3fn")
        disabled.add("float8_e5m2")

    mixed_ops = comfy.ops.mixed_precision_ops(
        {}, compute_dtype=computation_dtype, disabled=disabled
    )
    unet_config = dict(unet_config)
    unet_config["operations"] = mixed_ops

    print(
        f"[Krea2 MixedPrecision] comfy_quant detected — "
        f"storage={storage_dtype} compute={computation_dtype} "
        f"(UI float8 overwrite ignored; other models untouched)"
    )
    try:
        from backend.attention_backend_info import log_comfy_attention_backend

        log_comfy_attention_backend(tag="[Krea2]", when="mixed_precision_load")
    except Exception as e:
        print(f"[Krea2][Attention] visibility log failed: {e}")

    # operations=False: do not monkeypatch torch.nn. SingleStreamDiT uses
    # unet_config["operations"] for Linear / QuantizedTensor load.
    with using_forge_operations(
        operations=False,
        device=initial_device,
        dtype=computation_dtype,
        manual_cast_enabled=False,
    ):
        model = model_loader(unet_config)

    model = model.to(device=initial_device)
    load_state_dict(model, state_dict)

    if hasattr(model, "_internal_dict"):
        model._internal_dict = unet_config
    else:
        model.config = unet_config

    model.storage_dtype = storage_dtype
    model.computation_dtype = computation_dtype
    model.load_device = load_device
    model.initial_device = initial_device
    model.offload_device = offload_device
    return model


def _matches_guess_config(estimated_config, matched_guesses) -> bool:
    """Match engine to guess config.

    ``huggingface_guess.guess()`` returns instances of ``huggingface_guess.model_list.*``.
    Diffusion engines may import the same classes via ``huggingface_guess`` or
    ``modules_forge.packages.huggingface_guess`` (different module objects, same source file).
    ``type(x) is y`` fails across those imports; compare class names as well.
    """
    cfg_type = type(estimated_config)
    cfg_name = cfg_type.__name__
    for guess_cls in matched_guesses:
        if cfg_type is guess_cls or cfg_name == guess_cls.__name__:
            return True
    return False


def _te_filter_prefixes(guess, clip_key: str) -> list[str]:
    if hasattr(guess, "te_filter_prefixes"):
        prefixes = list(guess.te_filter_prefixes(clip_key))
    elif hasattr(guess, "anima_te_filter_prefixes"):
        prefixes = list(guess.anima_te_filter_prefixes(clip_key))
    else:
        pref = guess.text_encoder_key_prefix[0]
        prefixes = [clip_key + ".", pref + clip_key + "."]
    root = clip_key.split(".")[0]
    prefixes.extend([root + ".", "transformer."])
    seen: set[str] = set()
    out: list[str] = []
    for p in prefixes:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _comfy_load_te(guess, state_dict, clip_key: str, layer_probe: str, spiece_from_guess: bool = False, clip_type=None):
    from backend.patch_comfy_quant_layout_fallback import ensure_comfy_quant_layout_fallback_patch
    from comfy.sd import load_text_encoder_state_dicts

    # Issue #3: quantized TE (*.comfy_quant) crashes when get_layout_class → None
    ensure_comfy_quant_layout_fallback_patch()

    if not isinstance(state_dict, dict) or len(state_dict) <= 16:
        return None

    te_sd = dict(state_dict)
    if layer_probe not in te_sd:
        te_sd = guess.process_clip_state_dict(dict(te_sd))
        for prefix in _te_filter_prefixes(guess, clip_key):
            part = try_filter_state_dict(dict(te_sd), [prefix])
            if part:
                te_sd = part
                break

    if spiece_from_guess:
        spiece = getattr(guess, "forge_spiece_model", None)
        if spiece is not None:
            te_sd["spiece_model"] = spiece

    te_opts = comfy_te_model_options(te_sd)
    if clip_type is not None:
        return load_text_encoder_state_dicts([te_sd], clip_type=clip_type, model_options=te_opts)
    return load_text_encoder_state_dicts([te_sd], model_options=te_opts)


def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
    config_path = os.path.join(repo_path, component_name)

    if component_name in ["feature_extractor", "safety_checker"]:
        return None

    if lib_name in ["transformers", "diffusers"]:
        if component_name == "scheduler":
            cls = getattr(importlib.import_module(lib_name), cls_name)
            return cls.from_pretrained(os.path.join(repo_path, component_name))
        if component_name.startswith("tokenizer"):
            cls = getattr(importlib.import_module(lib_name), cls_name)
            comp = cls.from_pretrained(os.path.join(repo_path, component_name))
            comp._eventual_warn_about_too_long_sequence = lambda *args, **kwargs: None
            return comp
        if cls_name == "AutoencoderKL":
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have VAE state dict!"

            config = IntegratedAutoencoderKL.load_config(config_path)

            with using_forge_operations(device=memory_management.cpu, dtype=memory_management.vae_dtype()):
                model = IntegratedAutoencoderKL.from_config(config)

            if "decoder.up_blocks.0.resnets.0.norm1.weight" in state_dict.keys():  # diffusers format
                state_dict = huggingface_guess.diffusers_convert.convert_vae_state_dict(state_dict)
            load_state_dict(model, state_dict, ignore_start="loss.")
            return model
        if cls_name in ["AutoencoderKLWan", "AutoencoderKLQwenImage"]:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have VAE state dict!"

            config = WanVAE.load_config(config_path)

            with using_forge_operations(device=memory_management.cpu, dtype=memory_management.vae_dtype()):
                model = WanVAE.from_config(config)

            load_state_dict(model, state_dict)
            return model
        if component_name.startswith("text_encoder") and cls_name in ["CLIPTextModel", "CLIPTextModelWithProjection"]:
            if not isinstance(state_dict, dict) or len(state_dict) <= 16:
                return None
            
            from transformers import CLIPTextConfig, CLIPTextModel

            config = CLIPTextConfig.from_pretrained(config_path)

            to_args = dict(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype())

            with modeling_utils.no_init_weights():
                with using_forge_operations(**to_args, manual_cast_enabled=True):
                    model = IntegratedCLIP(CLIPTextModel, config, add_text_projection=True).to(**to_args)

            from backend.patch_comfy_quant_layout_fallback import dequantize_state_dict_inplace
            state_dict = dequantize_state_dict_inplace(state_dict, compute_dtype=to_args["dtype"])

            # transformers 5.x flattened CLIPTextModel: text_model.X -> X
            # SD checkpoint keys use "text_model.X", Nunchaku keys use "transformer.text_model.X"
            # Strip all known wrapper prefixes, then re-add "transformer." for IntegratedCLIP
            new_state_dict = {}
            for k, v in state_dict.items():
                clean = k
                clean = clean.removeprefix("transformer.")
                clean = clean.removeprefix("text_model.")
                new_state_dict[f"transformer.{clean}"] = v
            load_state_dict(model, new_state_dict, ignore_errors=[
                "transformer.text_projection.weight",
                "transformer.embeddings.position_ids",       # transformers 5.x
                "transformer.text_model.embeddings.position_ids",  # transformers 4.x compat
                "logit_scale",
            ], log_name=cls_name)

            return model
        if cls_name == "Qwen2_5_VLForConditionalGeneration":
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have Qwen 2.5 state dict!"

            from backend.nn.llm.llama import Qwen25_7BVLI

            config = read_arbitrary_config(config_path)

            storage_dtype = memory_management.text_encoder_dtype()
            state_dict_dtype = memory_management.state_dict_dtype(state_dict)

            if state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, "nf4", "fp4", "gguf"]:
                print(f"Using Detected Qwen2.5 Data Type: {state_dict_dtype}")
                storage_dtype = state_dict_dtype
                if state_dict_dtype in ["nf4", "fp4", "gguf"]:
                    print("Using pre-quant state dict!")
                    if state_dict_dtype in ["gguf"]:
                        beautiful_print_gguf_state_dict_statics(state_dict)
            else:
                print(f"Using Default Qwen2.5 Data Type: {storage_dtype}")

            if storage_dtype in ["nf4", "fp4", "gguf"]:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype(), manual_cast_enabled=False, bnb_dtype=storage_dtype):
                        model = Qwen25_7BVLI(config)
            else:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=storage_dtype, manual_cast_enabled=True):
                        model = Qwen25_7BVLI(config)

            load_state_dict(model, state_dict, log_name=cls_name, ignore_errors=["lm_head.weight"])

            return model
        if cls_name == "Gemma2Model":
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have Gemma2 state dict!"

            return _comfy_load_te(
                guess,
                state_dict,
                clip_key="gemma2_2b.transformer",
                layer_probe="model.layers.0.post_feedforward_layernorm.weight",
                spiece_from_guess=True,
            )
        if cls_name == "Qwen3Model":
            if not isinstance(state_dict, dict) or len(state_dict) <= 16:
                print(f"[Loader] Qwen3Model skipped: state_dict has {len(state_dict) if isinstance(state_dict, dict) else 0} keys. Wrong checkpoint? (SDXL may be misdetected as Qwen)")
                return None

            _anima_te = _is_anima_guess(guess)

            if _anima_te:
                return _comfy_load_te(
                    guess,
                    state_dict,
                    clip_key="qwen3_06b.transformer",
                    layer_probe="model.layers.0.post_attention_layernorm.weight",
                )

            return _comfy_load_te(
                guess,
                state_dict,
                clip_key="qwen3_4b.transformer",
                layer_probe="model.layers.0.post_attention_layernorm.weight",
            )
        if cls_name == "Qwen3VLModel" and _is_krea2_guess(guess):
            if not isinstance(state_dict, dict) or len(state_dict) <= 16:
                print(f"[Loader] Qwen3VLModel (krea2) skipped: state_dict has {len(state_dict) if isinstance(state_dict, dict) else 0} keys.")
                return None

            from comfy.sd import CLIPType

            return _comfy_load_te(
                guess,
                state_dict,
                clip_key="qwen3vl_4b.transformer",
                layer_probe="model.visual.deepstack_merger_list.0.norm.weight",
                clip_type=CLIPType.KREA2,
            )
        if cls_name in ["T5EncoderModel", "UMT5EncoderModel"]:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have T5 state dict!"

            if filename := state_dict.get("transformer.filename", None):
                if memory_management.is_device_cpu(memory_management.text_encoder_device()):
                    raise SystemError("nunchaku T5 does not support CPU!")

                from backend.nn.svdq import SVDQT5

                print("Using Nunchaku T5")
                model = SVDQT5(filename)
                return model

            from backend.nn.t5 import IntegratedT5

            config = read_arbitrary_config(config_path)

            storage_dtype = memory_management.text_encoder_dtype()
            state_dict_dtype = memory_management.state_dict_dtype(state_dict)

            if state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, "nf4", "fp4", "gguf"]:
                print(f"Using Detected T5 Data Type: {state_dict_dtype}")
                storage_dtype = state_dict_dtype
                if state_dict_dtype in ["nf4", "fp4", "gguf"]:
                    print("Using pre-quant state dict!")
                    if state_dict_dtype in ["gguf"]:
                        beautiful_print_gguf_state_dict_statics(state_dict)
            else:
                print(f"Using Default T5 Data Type: {storage_dtype}")

            from backend.patch_comfy_quant_layout_fallback import dequantize_state_dict_inplace
            state_dict = dequantize_state_dict_inplace(state_dict, compute_dtype=storage_dtype)

            if storage_dtype in ["nf4", "fp4", "gguf"]:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=memory_management.text_encoder_dtype(), manual_cast_enabled=False, bnb_dtype=storage_dtype):
                        model = IntegratedT5(config)
            else:
                with modeling_utils.no_init_weights():
                    with using_forge_operations(device=memory_management.cpu, dtype=storage_dtype, manual_cast_enabled=True):
                        model = IntegratedT5(config)

            load_state_dict(model, state_dict, log_name=cls_name, ignore_errors=["transformer.encoder.embed_tokens.weight", "logit_scale"])

            return model
        if cls_name in [
            "UNet2DConditionModel",
            "FluxTransformer2DModel",
            "ChromaTransformer2DModel",
            "QwenImageTransformer2DModel",
            "Lumina2Transformer2DModel",
            "ZImageTransformer2DModel",
            "CosmosTransformer3DModel",
            "SingleStreamDiT",
        ]:
            assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have model state dict!"

            model_loader = None
            _nz = False  # Nunchaku Z-Image
            _nf = False  # Nunchaku Flux (disable Forge operations)

            if cls_name == "UNet2DConditionModel":
                model_loader = lambda c: IntegratedUNet2DConditionModel.from_config(c)
            elif cls_name == "FluxTransformer2DModel":
                if guess.nunchaku:
                    from backend.nn.svdq import SVDQFluxTransformer2DModel

                    model_loader = lambda c: SVDQFluxTransformer2DModel(c)
                    _nf = True  # Disable Forge operations for Nunchaku Flux
                else:
                    from backend.nn.flux import IntegratedFluxTransformer2DModel

                    model_loader = lambda c: IntegratedFluxTransformer2DModel(**c)
            elif cls_name == "ChromaTransformer2DModel":
                from backend.nn.chroma import IntegratedChromaTransformer2DModel

                model_loader = lambda c: IntegratedChromaTransformer2DModel(**c)
            elif cls_name == "QwenImageTransformer2DModel":
                if guess.nunchaku:
                    from backend.nn.svdq import NunchakuQwenImageTransformer2DModel

                    model_loader = lambda c: NunchakuQwenImageTransformer2DModel(**c)
                else:
                    from backend.nn.qwen import QwenImageTransformer2DModel

                    model_loader = lambda c: QwenImageTransformer2DModel(**c)
            elif cls_name in ("Lumina2Transformer2DModel", "ZImageTransformer2DModel"):
                if guess.nunchaku:
                    from backend.nn.svdq import patch_nunchaku_zimage

                    guess.unet_config.pop("filename")
                    precision = guess.unet_config.pop("precision")
                    rank = guess.unet_config.pop("rank")
                    _nz = True

                from comfy.ldm.lumina.model import NextDiT
                import comfy.ops

                # NextDiT requires operations parameter (cannot be None)
                # For standard ZIT: use ComfyUI's manual_cast operations (same as ComfyUI's Lumina2)
                # For Nunchaku ZIT: use disable_weight_init (torch.nn wrapper) since using_forge_operations(operations=False) uses torch.nn
                if _nz:
                    # Nunchaku ZIT: use disable_weight_init (torch.nn wrapper) since operations=False uses torch.nn directly
                    # CRITICAL: NextDiT.__init__ requires operations parameter (cannot be None)
                    # using_forge_operations(operations=False) only affects context, not NextDiT.__init__
                    guess.unet_config["operations"] = comfy.ops.disable_weight_init
                    model_loader = lambda c: NextDiT(**c)
                else:
                    # Standard ZIT: use ComfyUI's manual_cast operations (same as ComfyUI BaseModel)
                    # Same as ComfyUI BaseModel.__init__: operations = comfy.ops.pick_operations(...)
                    # For ZIT/Lumina2, ComfyUI uses manual_cast operations
                    # CRITICAL: Set operations in unet_config before creating model_loader, same as ComfyUI
                    # ComfyUI BaseModel sets operations in unet_config before calling unet_model(**unet_config, ...)
                    guess.unet_config["operations"] = comfy.ops.manual_cast
                    model_loader = lambda c: NextDiT(**c)
            elif cls_name == "CosmosTransformer3DModel" and isinstance(
                guess, (model_list.Anima, model_list.AnimaBase, model_list.AnimaWai68)
            ):
                from comfy.ldm.anima.model import Anima
                from backend.nn.comfy_anima import install_anima_rms_rope_fuse, remap_anima_state_dict
                import comfy.ops

                install_anima_rms_rope_fuse()
                if isinstance(state_dict, dict):
                    state_dict = remap_anima_state_dict(state_dict)
                for _k in ("dim", "n_layers", "rope_axis_dim"):
                    guess.unet_config.pop(_k, None)
                guess.unet_config["operations"] = comfy.ops.manual_cast
                model_loader = lambda c: Anima(**c)
            elif cls_name == "SingleStreamDiT" and _is_krea2_guess(guess):
                # Krea2 (K2): Comfy SingleStreamDiT. Checkpoint keys are Comfy-native
                # (model.diffusion_model.*) so no remap is needed. Config comes verbatim
                # from comfy.model_detection.detect_unet_config (image_model + features/
                # channels/patch/layers/heads/kvheads/txtlayers/txtdim). SingleStreamDiT
                # accepts image_model + **kwargs, so extra keys are tolerated.
                from comfy.ldm.krea2.model import SingleStreamDiT
                import comfy.ops

                guess.unet_config["operations"] = comfy.ops.manual_cast
                model_loader = lambda c: SingleStreamDiT(**c)
                try:
                    from backend.attention_backend_info import log_comfy_attention_backend

                    log_comfy_attention_backend(tag="[Krea2]", when="unet_construct")
                except Exception as e:
                    print(f"[Krea2][Attention] visibility log failed: {e}")

            unet_config = guess.unet_config.copy()
            
            # CRITICAL: Ensure operations is set in unet_config for standard ZIT (same as ComfyUI BaseModel)
            # ComfyUI BaseModel.__init__ sets operations before calling unet_model(**unet_config, ...)
            if cls_name in ("Lumina2Transformer2DModel", "ZImageTransformer2DModel") and not _nz:
                import comfy.ops
                # Ensure operations is set in unet_config for standard ZIT (same as ComfyUI Lumina2)
                unet_config["operations"] = comfy.ops.manual_cast
            
            state_dict_parameters = memory_management.state_dict_parameters(state_dict)
            state_dict_dtype = memory_management.state_dict_dtype(state_dict)

            storage_dtype = memory_management.unet_dtype(model_params=state_dict_parameters, supported_dtypes=guess.supported_inference_dtypes)

            unet_storage_dtype_overwrite = backend.args.dynamic_args.get("forge_unet_storage_dtype")

            # INT8: UI Low Bits "int8" / "int8 (fp16 LoRA)" only (token int8_tensorwise).
            # Never from Automatic, never from checkpoint / state_dict auto-detect.
            # Early return — does not enter float8 / bnb / Automatic construct below.
            if unet_storage_dtype_overwrite == "int8_tensorwise":
                from modules_forge.hswq_int8 import load_unet_int8_branch

                print(
                    f"[HSWQ INT8] loader.py branch: overwrite={unet_storage_dtype_overwrite!r} "
                    f"cls={cls_name} (not Automatic/float8/bnb)"
                )
                return load_unet_int8_branch(
                    model_loader=model_loader,
                    unet_config=unet_config,
                    state_dict=state_dict,
                    state_dict_parameters=state_dict_parameters,
                    guess=guess,
                    cls_name=cls_name,
                    _nz=_nz,
                    precision=precision if _nz else None,
                    rank=rank if _nz else None,
                )

            # Krea2 MixedPrecision (comfy_quant): early return — Krea2 + SingleStreamDiT only.
            # Before float8 UI overwrite / shared construct (those drop scales → noise).
            if (
                cls_name == "SingleStreamDiT"
                and _is_krea2_guess(guess)
                and _state_dict_has_comfy_quant(state_dict)
            ):
                return _load_krea2_mixed_precision_unet(
                    model_loader=model_loader,
                    unet_config=unet_config,
                    state_dict=state_dict,
                    state_dict_parameters=state_dict_parameters,
                    guess=guess,
                )

            if unet_storage_dtype_overwrite is not None:
                storage_dtype = unet_storage_dtype_overwrite
            elif state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, "nf4", "fp4", "gguf"]:
                print(f"Using Detected UNet Type: {state_dict_dtype}")
                storage_dtype = state_dict_dtype
                if state_dict_dtype in ["nf4", "fp4", "gguf"]:
                    print("Using pre-quant state dict!")
                    if state_dict_dtype in ["gguf"]:
                        beautiful_print_gguf_state_dict_statics(state_dict)

            load_device = memory_management.get_torch_device()
            computation_dtype = memory_management.get_computation_dtype(load_device, parameters=state_dict_parameters, supported_dtypes=guess.supported_inference_dtypes)
            offload_device = memory_management.unet_offload_device()

            # CRITICAL: Ensure operations is set in unet_config for standard ZIT before calling model_loader
            # Same as ComfyUI BaseModel.__init__: operations is set before calling unet_model(**unet_config, ...)
            if cls_name in ("Lumina2Transformer2DModel", "ZImageTransformer2DModel") and not _nz:
                import comfy.ops
                # Ensure operations is set in unet_config for standard ZIT (same as ComfyUI Lumina2)
                unet_config["operations"] = comfy.ops.manual_cast
            
            if storage_dtype in ["nf4", "fp4", "gguf"]:
                initial_device = memory_management.unet_initial_load_device(parameters=state_dict_parameters, dtype=computation_dtype)
                # CRITICAL: Ensure operations is set in unet_config for standard ZIT before calling model_loader
                if cls_name in ("Lumina2Transformer2DModel", "ZImageTransformer2DModel") and not _nz:
                    import comfy.ops
                    unet_config["operations"] = comfy.ops.manual_cast
                with using_forge_operations(device=initial_device, dtype=computation_dtype, manual_cast_enabled=False, bnb_dtype=storage_dtype):
                    model = model_loader(unet_config)
            else:
                initial_device = memory_management.unet_initial_load_device(parameters=state_dict_parameters, dtype=storage_dtype)
                need_manual_cast = storage_dtype != computation_dtype
                to_args = dict(device=initial_device, dtype=storage_dtype)

                # CRITICAL: Ensure operations is set in unet_config for standard ZIT before calling model_loader
                if cls_name in ("Lumina2Transformer2DModel", "ZImageTransformer2DModel") and not _nz:
                    import comfy.ops
                    unet_config["operations"] = comfy.ops.manual_cast

                with using_forge_operations(operations=False if (_nz or _nf) else None, **to_args, manual_cast_enabled=need_manual_cast):
                    model = model_loader(unet_config).to(**to_args)

            if _nz:
                model = patch_nunchaku_zimage(model, precision, rank)
            elif cls_name in ("Lumina2Transformer2DModel", "ZImageTransformer2DModel"):
                # Standard ZIT: Apply LoRA support (same as Nunchaku ZIT but without Nunchaku-specific patching)
                from backend.nn.svdq import patch_standard_zimage
                model = patch_standard_zimage(model)
            load_state_dict(model, state_dict)

            if hasattr(model, "_internal_dict"):
                model._internal_dict = unet_config
            else:
                model.config = unet_config

            model.storage_dtype = storage_dtype
            model.computation_dtype = computation_dtype
            model.load_device = load_device
            model.initial_device = initial_device
            model.offload_device = offload_device

            return model

    print(f"Skipped: {component_name} = {lib_name}.{cls_name}")
    return None


def replace_state_dict(sd: dict[str, torch.Tensor], asd: dict[str, torch.Tensor], guess, path: os.PathLike):
    vae_key_prefix = guess.vae_key_prefix[0]
    text_encoder_key_prefix = guess.text_encoder_key_prefix[0]

    # SD1系/SDXL系のレガシーアーキテクチャ判定
    is_legacy_model = False
    legacy_test_key = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
    if legacy_test_key in sd:
        match sd[legacy_test_key].shape[1]:
            case 768 | 2048:
                is_legacy_model = True
    
    # 追加モジュールがLLM（Qwen3, Gemma等）であるかの判定
    is_llm_asd = any(k.startswith("qwen3_") or k.startswith("gemma2_") for k in asd.keys())
    
    # レガシーモデルに対してLLMを混入させようとした場合は合体をブロックする
    if is_legacy_model and is_llm_asd:
        print(f"Skipping incompatible LLM text encoder '{path}' for legacy SD1/SDXL model.")
        return sd


    if "enc.blk.0.attn_k.weight" in asd:
        gguf_t5_format = {  # city96
            "enc.": "encoder.",
            ".blk.": ".block.",
            "token_embd": "shared",
            "output_norm": "final_layer_norm",
            "attn_q": "layer.0.SelfAttention.q",
            "attn_k": "layer.0.SelfAttention.k",
            "attn_v": "layer.0.SelfAttention.v",
            "attn_o": "layer.0.SelfAttention.o",
            "attn_norm": "layer.0.layer_norm",
            "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
            "ffn_up": "layer.1.DenseReluDense.wi_1",
            "ffn_down": "layer.1.DenseReluDense.wo",
            "ffn_gate": "layer.1.DenseReluDense.wi_0",
            "ffn_norm": "layer.1.layer_norm",
        }
        asd_new = {}
        for k, v in asd.items():
            for s, d in gguf_t5_format.items():
                k = k.replace(s, d)
            asd_new[k] = v
        for k in ("shared.weight",):
            asd_new[k] = asd_new[k].dequantize_as_pytorch_parameter()
        asd.clear()
        asd = asd_new

    if "blk.0.attn_norm.weight" in asd:
        gguf_llm_format = {  # city96
            "blk.": "model.layers.",
            "attn_norm": "input_layernorm",
            "attn_q_norm.": "self_attn.q_norm.",
            "attn_k_norm.": "self_attn.k_norm.",
            "attn_v_norm.": "self_attn.v_norm.",
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_gate": "mlp.gate_proj",
            "ffn_norm": "post_attention_layernorm",
            "token_embd": "model.embed_tokens",
            "output_norm": "model.norm",
            "output.weight": "lm_head.weight",
        }
        asd_new = {}
        for k, v in asd.items():
            for s, d in gguf_llm_format.items():
                k = k.replace(s, d)
            asd_new[k] = v
        for k in ("model.embed_tokens.weight",):
            asd_new[k] = asd_new[k].dequantize_as_pytorch_parameter()
        asd.clear()
        asd = asd_new

    #   sd / sdxl
    if "decoder.conv_in.weight" in asd or "decoder.middle.0.residual.0.gamma" in asd:
        keys_to_delete = [k for k in sd if k.startswith(vae_key_prefix)]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[vae_key_prefix + k] = v

    ##  identify model type
    flux_test_key = "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"
    svdq_test_key = "model.diffusion_model.single_transformer_blocks.0.mlp_fc1.qweight"
    legacy_test_key = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"

    model_type = "-"
    if legacy_test_key in sd:
        match sd[legacy_test_key].shape[1]:
            case 768:
                model_type = "sd1"
            case 1280:
                model_type = "xlrf"  # sdxl refiner model
            case 2048:
                model_type = "sdxl"
    elif flux_test_key in sd or svdq_test_key in sd:
        model_type = "flux"

    ##  prefixes used by various model types for CLIP-L
    prefix_L = {
        "-": None,
        "sd1": "cond_stage_model.transformer.",
        "xlrf": None,
        "sdxl": "conditioner.embedders.0.transformer.",
        "flux": "text_encoders.clip_l.transformer.",
    }
    ##  prefixes used by various model types for CLIP-G
    prefix_G = {
        "-": None,
        "sd1": None,
        "xlrf": "conditioner.embedders.0.model.transformer.",
        "sdxl": "conditioner.embedders.1.model.transformer.",
        "flux": None,
    }

    ##  VAE format 0 (extracted from model, could be sd1/sdxl)
    if "first_stage_model.decoder.conv_in.weight" in asd:
        if model_type in ("sd1", "xlrf", "sdxl"):
            assert asd["first_stage_model.decoder.conv_in.weight"].shape[1] == 4
            for k, v in asd.items():
                sd[k] = v

    ##  CLIP-G
    CLIP_G = {"conditioner.embedders.1.model.transformer.resblocks.0.ln_1.bias": "conditioner.embedders.1.model.transformer.", "text_encoders.clip_g.transformer.text_model.encoder.layers.0.layer_norm1.bias": "text_encoders.clip_g.transformer.", "text_model.encoder.layers.0.layer_norm1.bias": "", "transformer.resblocks.0.ln_1.bias": "transformer."}  #   key to identify source model                                                old_prefix
    for CLIP_key in CLIP_G.keys():
        if CLIP_key in asd and asd[CLIP_key].shape[0] == 1280:
            new_prefix = prefix_G[model_type]
            old_prefix = CLIP_G[CLIP_key]

            if new_prefix is not None:
                if "resblocks" not in CLIP_key:  # need to convert

                    def convert_transformers(statedict, prefix_from, prefix_to, number):
                        keys_to_replace = {
                            "{}text_model.embeddings.position_embedding.weight": "{}positional_embedding",
                            "{}text_model.embeddings.token_embedding.weight": "{}token_embedding.weight",
                            "{}text_model.final_layer_norm.weight": "{}ln_final.weight",
                            "{}text_model.final_layer_norm.bias": "{}ln_final.bias",
                            "text_projection.weight": "{}text_projection",
                        }
                        resblock_to_replace = {
                            "layer_norm1": "ln_1",
                            "layer_norm2": "ln_2",
                            "mlp.fc1": "mlp.c_fc",
                            "mlp.fc2": "mlp.c_proj",
                            "self_attn.out_proj": "attn.out_proj",
                        }

                        for x in keys_to_replace:  #   remove trailing 'transformer.' from new prefix
                            k = x.format(prefix_from)
                            statedict[keys_to_replace[x].format(prefix_to[:-12])] = statedict.pop(k)

                        for resblock in range(number):
                            for y in ["weight", "bias"]:
                                for x in resblock_to_replace:
                                    k = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, x, y)
                                    k_to = "{}resblocks.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                                    statedict[k_to] = statedict.pop(k)

                                k_from = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, "self_attn.q_proj", y)
                                weightsQ = statedict.pop(k_from)
                                k_from = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, "self_attn.k_proj", y)
                                weightsK = statedict.pop(k_from)
                                k_from = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_from, resblock, "self_attn.v_proj", y)
                                weightsV = statedict.pop(k_from)

                                k_to = "{}resblocks.{}.attn.in_proj_{}".format(prefix_to, resblock, y)

                                statedict[k_to] = torch.cat((weightsQ, weightsK, weightsV))
                        return statedict

                    asd = convert_transformers(asd, old_prefix, new_prefix, 32)
                    for k, v in asd.items():
                        sd[k] = v

                elif old_prefix == "":
                    for k, v in asd.items():
                        new_k = new_prefix + k
                        sd[new_k] = v
                else:
                    for k, v in asd.items():
                        new_k = k.replace(old_prefix, new_prefix)
                        sd[new_k] = v

    ##  CLIP-L
    CLIP_L = {"cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.bias": "cond_stage_model.transformer.", "conditioner.embedders.0.transformer.text_model.encoder.layers.0.layer_norm1.bias": "conditioner.embedders.0.transformer.", "text_encoders.clip_l.transformer.text_model.encoder.layers.0.layer_norm1.bias": "text_encoders.clip_l.transformer.", "text_model.encoder.layers.0.layer_norm1.bias": "", "transformer.resblocks.0.ln_1.bias": "transformer."}  #   key to identify source model                                                    old_prefix

    for CLIP_key in CLIP_L.keys():
        if CLIP_key in asd and asd[CLIP_key].shape[0] == 768:
            new_prefix = prefix_L[model_type]
            old_prefix = CLIP_L[CLIP_key]

            if new_prefix is not None:
                if "resblocks" in CLIP_key:  # need to convert

                    def transformers_convert(statedict, prefix_from, prefix_to, number):
                        keys_to_replace = {
                            "positional_embedding": "{}text_model.embeddings.position_embedding.weight",
                            "token_embedding.weight": "{}text_model.embeddings.token_embedding.weight",
                            "ln_final.weight": "{}text_model.final_layer_norm.weight",
                            "ln_final.bias": "{}text_model.final_layer_norm.bias",
                            "text_projection": "text_projection.weight",
                        }
                        resblock_to_replace = {
                            "ln_1": "layer_norm1",
                            "ln_2": "layer_norm2",
                            "mlp.c_fc": "mlp.fc1",
                            "mlp.c_proj": "mlp.fc2",
                            "attn.out_proj": "self_attn.out_proj",
                        }

                        for k in keys_to_replace:
                            statedict[keys_to_replace[k].format(prefix_to)] = statedict.pop(k)

                        for resblock in range(number):
                            for y in ["weight", "bias"]:
                                for x in resblock_to_replace:
                                    k = "{}resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                                    k_to = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                                    statedict[k_to] = statedict.pop(k)

                                k_from = "{}resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
                                weights = statedict.pop(k_from)
                                shape_from = weights.shape[0] // 3
                                for x in range(3):
                                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                                    k_to = "{}text_model.encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                                    statedict[k_to] = weights[shape_from * x : shape_from * (x + 1)]
                        return statedict

                    asd = transformers_convert(asd, old_prefix, new_prefix, 12)
                    for k, v in asd.items():
                        sd[k] = v

                elif old_prefix == "":
                    for k, v in asd.items():
                        new_k = new_prefix + k
                        sd[new_k] = v
                else:
                    for k, v in asd.items():
                        new_k = k.replace(old_prefix, new_prefix)
                        sd[new_k] = v

    if model_type in ("sd1", "xlrf", "sdxl"):
        return sd

    if "encoder.block.0.layer.0.SelfAttention.k.weight" in asd:
        _key = "umt5xxl" if asd["shared.weight"].size(0) == 256384 else "t5xxl"
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}{_key}.")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            if k == "spiece_model":
                continue
            sd[f"{text_encoder_key_prefix}{_key}.transformer.{k}"] = v

    elif "encoder.block.0.layer.0.SelfAttention.k.qweight" in asd:
        keys_to_delete = [k for k in sd if k.startswith(f"{text_encoder_key_prefix}t5xxl.")]
        for k in keys_to_delete:
            del sd[k]
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}t5xxl.transformer.{k}"] = True
        sd[f"{text_encoder_key_prefix}t5xxl.transformer.filename"] = str(path)

    if "model.layers.0.post_feedforward_layernorm.weight" in asd:
        assert "model.layers.0.self_attn.q_norm.weight" not in asd
        for k, v in asd.items():
            if k == "spiece_model":
                sd[f"{text_encoder_key_prefix}spiece_model"] = v
                continue
            sd[f"{text_encoder_key_prefix}gemma2_2b.{k}"] = v

    elif "model.visual.deepstack_merger_list.0.norm.weight" in asd:  # Krea2 TE: Qwen3-VL-4B (DeepStack is unique to Qwen3-VL)
        assert asd["model.visual.merger.linear_fc2.weight"].shape[0] == 2560  # 4B (8B == 3584)
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}qwen3vl_4b.transformer.{k}"] = v

    elif "model.layers.0.self_attn.k_proj.bias" in asd:
        weight = asd["model.layers.0.self_attn.k_proj.bias"]
        assert weight.shape[0] == 512
        for k, v in asd.items():
            sd[f"{text_encoder_key_prefix}qwen25_7b.{k}"] = v

    elif "model.layers.0.post_attention_layernorm.weight" in asd:
        assert "model.layers.0.self_attn.q_norm.weight" in asd
        weight = asd["model.layers.0.post_attention_layernorm.weight"]
        if weight.shape[0] == 1024 and "Anima" in getattr(guess, "huggingface_repo", ""):
            for k, v in asd.items():
                if k == "spiece_model":
                    continue
                sd[f"{text_encoder_key_prefix}qwen3_06b.transformer.{k}"] = v
        else:
            for k, v in asd.items():
                sd[f"{text_encoder_key_prefix}qwen3_4b.transformer.{k}"] = v

    return sd


def preprocess_state_dict(sd):
    if not any(k.startswith("model.diffusion_model") for k in sd.keys()):
        sd = {f"model.diffusion_model.{k}": v for k, v in sd.items()}

    return sd


def split_state_dict(sd, additional_state_dicts: list = None):
    sd, metadata = load_torch_file(sd, return_metadata=True)
    sd = preprocess_state_dict(sd)

    from modules_forge.packages.huggingface_guess import model_list
    from modules_forge.packages.huggingface_guess.detection import detect_unet_config, unet_prefix_from_state_dict

    prefix = unet_prefix_from_state_dict(sd)
    anima_cfg = detect_unet_config(sd, prefix)
    if anima_cfg.get("image_model") == "anima" and "in_channels" in anima_cfg:
        in_ch = int(anima_cfg["in_channels"])
        model_channels = int(anima_cfg.get("model_channels", anima_cfg.get("dim", 2048)))
        if in_ch == 68 and model_channels == 2048:
            guess = model_list.AnimaWai68(anima_cfg)
        elif model_channels == 2048:
            guess = model_list.AnimaBase(anima_cfg)
        else:
            guess = model_list.Anima(anima_cfg)
    else:
        guess = huggingface_guess.guess(sd)

    if getattr(guess, "nunchaku", False) and ("Z-Image" in guess.huggingface_repo or "Qwen" in guess.huggingface_repo):
        import json

        from nunchaku.utils import get_precision_from_quantization_config

        quantization_config = json.loads(metadata["quantization_config"])
        guess.unet_config.update(
            {
                "precision": get_precision_from_quantization_config(quantization_config),
                "rank": quantization_config.get("rank", 32),
            }
        )

    if isinstance(additional_state_dicts, list):
        for asd in additional_state_dicts:
            _asd = load_torch_file(asd)
            sd = replace_state_dict(sd, _asd, guess, asd)
            del _asd

    guess.clip_target = guess.clip_target(sd)
    guess.model_type = guess.model_type(sd)
    guess.ztsnr = "ztsnr" in sd

    sd = guess.process_vae_state_dict(sd)

    state_dict = {guess.unet_target: try_filter_state_dict(sd, guess.unet_key_prefix), guess.vae_target: try_filter_state_dict(sd, guess.vae_key_prefix)}

    from backend.patch_comfy_quant_layout_fallback import dequantize_state_dict_inplace
    sd = dequantize_state_dict_inplace(sd, compute_dtype=memory_management.text_encoder_dtype())

    sd = guess.process_clip_state_dict(sd)

    for k, v in guess.clip_target.items():
        if hasattr(guess, "anima_te_filter_prefixes"):
            prefixes = guess.anima_te_filter_prefixes(k)
        elif hasattr(guess, "te_filter_prefixes"):
            prefixes = guess.te_filter_prefixes(k)
        else:
            prefixes = [k + ".", f"{guess.text_encoder_key_prefix[0]}{k}."]
        state_dict[v] = try_filter_state_dict(sd, prefixes)

    state_dict["ignore"] = sd

    print_dict = {k: len(v) for k, v in state_dict.items()}
    print(f"StateDict Keys: {print_dict}")

    del state_dict["ignore"]

    return state_dict, guess


@torch.inference_mode()
def forge_loader(sd: os.PathLike, additional_state_dicts: list[os.PathLike] = None):
    try:
        state_dicts, estimated_config = split_state_dict(sd, additional_state_dicts=additional_state_dicts)
    except Exception as e:
        from modules.errors import display

        display(e, "forge_loader")
        raise ValueError("Failed to recognize model type!") from e

    # Issue #3: before any MixedPrecision TE/UNet load (kitchen layout may be None)
    try:
        from backend.patch_comfy_quant_layout_fallback import ensure_comfy_quant_layout_fallback_patch

        ensure_comfy_quant_layout_fallback_patch()
    except Exception as e:
        print(f"[Forge] quant layout fallback patch skipped: {e}")

    repo_name = estimated_config.huggingface_repo

    backend.args.dynamic_args["kontext"] = "kontext" in str(sd).lower()
    backend.args.dynamic_args["edit"] = "qwen" in str(sd).lower() and "edit" in str(sd).lower()
    backend.args.dynamic_args["nunchaku"] = getattr(estimated_config, "nunchaku", False)

    if getattr(estimated_config, "nunchaku", False):
        estimated_config.unet_config["filename"] = str(sd)

    local_path = os.path.join(dir_path, "huggingface", repo_name)
    config: dict = DiffusionPipeline.load_config(local_path)
    huggingface_components = {}
    comfy_te_loaded = False
    for component_name, v in config.items():
        if isinstance(v, list) and len(v) == 2:
            lib_name, cls_name = v
            if component_name == "tokenizer" and comfy_te_loaded:
                continue
            component_sd = state_dicts.pop(component_name, None)

            component = load_huggingface_component(estimated_config, component_name, lib_name, cls_name, local_path, component_sd)
            if component_sd is not None:
                del component_sd
            if component is not None:
                huggingface_components[component_name] = component
                if component_name == "text_encoder" and is_comfy_sd_clip(component):
                    comfy_te_loaded = True

    del state_dicts

    yaml_config = None
    yaml_config_prediction_type = None

    try:
        from pathlib import Path

        import yaml

        config_filename = os.path.splitext(sd)[0] + ".yaml"
        if Path(config_filename).is_file():
            with open(config_filename, "r") as stream:
                yaml_config = yaml.safe_load(stream)
    except ImportError:
        pass

    prediction_types = {
        "EPS": "epsilon",
        "V_PREDICTION": "v_prediction",
        "FLUX": "const",
        "FLOW": "const",
    }

    has_prediction_type = "scheduler" in huggingface_components and hasattr(huggingface_components["scheduler"], "config") and "prediction_type" in huggingface_components["scheduler"].config

    if yaml_config is not None:
        yaml_config_prediction_type: str = yaml_config.get("model", {}).get("params", {}).get("parameterization", "") or yaml_config.get("model", {}).get("params", {}).get("denoiser_config", {}).get("params", {}).get("scaling_config", {}).get("target", "")
        if yaml_config_prediction_type == "v" or yaml_config_prediction_type.endswith(".VScaling"):
            yaml_config_prediction_type = "v_prediction"
        else:
            # Use estimated prediction config if no suitable prediction type found
            yaml_config_prediction_type = ""

    # Fallback: detect v-prediction from safetensors metadata (Pony/Illustrious/ISDXL)
    if not yaml_config_prediction_type:
        try:
            if isinstance(sd, (str, os.PathLike)):
                metadata = load_torch_file(sd, return_metadata=True)[1]
                if metadata and "modelspec.prediction_type" in metadata:
                    pt = metadata["modelspec.prediction_type"].lower()
                    if pt == "v":
                        yaml_config_prediction_type = "v_prediction"
                        print(f"[V-Prediction] Detected from metadata: {os.path.basename(sd)}")
        except Exception:
            pass

    if has_prediction_type:
        if yaml_config_prediction_type:
            huggingface_components["scheduler"].config.prediction_type = yaml_config_prediction_type
        else:
            huggingface_components["scheduler"].config.prediction_type = prediction_types.get(estimated_config.model_type.name, huggingface_components["scheduler"].config.prediction_type)

    for M in possible_models:
        if _matches_guess_config(estimated_config, M.matched_guesses):
            return M(estimated_config=estimated_config, huggingface_components=huggingface_components)

    print(f"Failed to recognize model type! (guess={type(estimated_config).__name__}, repo={getattr(estimated_config, 'huggingface_repo', None)})")
    return None
