import os
import sys
import torch

# Use ComfyUI path from modules.paths (COMFYUI_FOLDER_NAME in paths_internal)
from modules.paths import comfyui_master_path
project_root = os.path.dirname(comfyui_master_path)
if os.path.exists(comfyui_master_path):
    if comfyui_master_path not in sys.path:
        sys.path.insert(0, comfyui_master_path)
    project_comfy_path = os.path.normpath(os.path.join(project_root, "comfy"))
    if project_comfy_path in sys.path:
        sys.path.remove(project_comfy_path)

from huggingface_guess.detection import unet_config_from_diffusers_unet, model_config_from_unet
from huggingface_guess.utils import unet_to_diffusers
from backend import memory_management
from backend.operations import using_forge_operations
from backend.nn.cnets import cldm
from backend.patcher.controlnet import ControlLora, ControlNet, load_t2i_adapter, apply_controlnet_advanced
from modules_forge.shared import add_supported_control_model


# =============================================================================
# ComfyUI ControlNet patches for Forge compatibility
# Uses ComfyUI-master/comfy package (located in ComfyUI-master directory)
# =============================================================================
try:
    import comfy.controlnet
    
    # Patch ControlNet.get_control for transformer_options compatibility
    _original_controlnet_get_control = comfy.controlnet.ControlNet.get_control
    
    def _patched_controlnet_get_control(self, x_noisy, t, cond, batched_number, transformer_options=None):
        """Patched get_control that accepts transformer_options for Forge compatibility."""
        # transformer_options is required in newer ComfyUI versions, default to empty dict if None
        if transformer_options is None:
            transformer_options = {}
        return _original_controlnet_get_control(self, x_noisy, t, cond, batched_number, transformer_options)
    
    comfy.controlnet.ControlNet.get_control = _patched_controlnet_get_control
    
    # Patch T2IAdapter.get_control if available
    if hasattr(comfy.controlnet, "T2IAdapter"):
        _original_t2i_get_control = comfy.controlnet.T2IAdapter.get_control
        
        def _patched_t2i_get_control(self, x_noisy, t, cond, batched_number, transformer_options=None):
            try:
                return _original_t2i_get_control(self, x_noisy, t, cond, batched_number, transformer_options)
            except TypeError:
                return _original_t2i_get_control(self, x_noisy, t, cond, batched_number)
        
        comfy.controlnet.T2IAdapter.get_control = _patched_t2i_get_control
except ImportError:
    pass  # comfy.controlnet not available



class ControlModelPatcher:
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        return None

    def __init__(self, model_patcher=None):
        self.model_patcher = model_patcher
        self.strength = 1.0
        self.start_percent = 0.0
        self.end_percent = 1.0
        self.positive_advanced_weighting = None
        self.negative_advanced_weighting = None
        self.advanced_frame_weighting = None
        self.advanced_sigma_weighting = None
        self.advanced_mask_weighting = None

    def process_after_running_preprocessors(self, process, params, *args, **kwargs):
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        return

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        return


class ControlNetPatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(controlnet_data, ckpt_path):
        try:
            from backend.patch_comfy_quant_layout_fallback import dequantize_state_dict_inplace
            controlnet_data = dequantize_state_dict_inplace(controlnet_data, compute_dtype=None)
        except Exception as e:
            print(f"[ControlNet] dequantize_state_dict_inplace note: {e}")

        # Nunchaku Qwen Image ControlNet ONLY (detected by transformer_blocks.0.img_mlp.net.0.proj.weight)
        # This is for NunchakuQwenImageTransformer2DModel ONLY - NOT for standard QwenImageTransformer2DModel
        _ix_marker = "transformer_blocks.0.img_mlp.net.0.proj.weight"
        _ix_keys = [k for k in controlnet_data.keys() if _ix_marker in str(k)]
        if _ix_keys:
            try:
                from modules_forge.supported_controlnet_qwen_image import QwenImageControlNetPatcher
                _pfx = str(_ix_keys[0]).split(_ix_marker)[0]
                _data = {str(k)[len(_pfx):] if str(k).startswith(_pfx) else str(k): v for k, v in controlnet_data.items()} if _pfx else controlnet_data
                result = QwenImageControlNetPatcher.try_build_from_state_dict(_data, ckpt_path)
                if result is not None:
                    print(f"[ControlNet] Loaded Nunchaku Qwen Image ControlNet patcher (for NunchakuQwenImageTransformer2DModel ONLY)")
                return result
            except Exception as e:
                print(f"Failed to load Nunchaku Qwen Image ControlNet: {e}")
                return None
        
        # ZIT ControlNet (from model_patches folder)
        # Detected by 'control_all_x_embedder.2-1.weight' key
        if 'control_all_x_embedder.2-1.weight' in controlnet_data:
            try:
                from modules_forge.supported_controlnet_zit import ZImageControlNetPatcher
                result = ZImageControlNetPatcher.try_build_from_state_dict(controlnet_data, ckpt_path)
                if result is not None:
                    print(f"[ControlNet] Loaded ZIT ControlNet (model patch): {ckpt_path}")
                return result
            except Exception as e:
                print(f"Failed to load ZIT ControlNet: {e}")
                return None
        
        # Qwen Image Fun ControlNet (control_img_in + control_blocks; same folder as InstantX)
        # Keys may have a prefix (e.g. "controlnet."), so match by substring
        if any("control_img_in.weight" in str(k) for k in controlnet_data.keys()):
            try:
                from modules_forge.supported_controlnet_qwen_fun import QwenFunControlNetPatcher
                result = QwenFunControlNetPatcher.try_build_from_state_dict(controlnet_data, ckpt_path)
                if result is not None:
                    print(f"[ControlNet] Loaded Qwen Image Fun ControlNet patcher (double_block patch)")
                return result
            except Exception as e:
                print(f"[Qwen Fun ControlNet] Error loading Qwen Image Fun ControlNet: {e}")
                return None

        # Flux ControlNet (InstantX format)
        if "controlnet_x_embedder.weight" in controlnet_data:
            try:
                from modules_forge.supported_controlnet_flux import FluxControlNetPatcher
                return FluxControlNetPatcher.try_build_from_state_dict(controlnet_data, ckpt_path)
            except Exception as e:
                print(f"Failed to load Flux ControlNet: {e}")
                return None
        
        if "lora_controlnet" in controlnet_data:
            return ControlNetPatcher(ControlLora(controlnet_data))

        controlnet_config = None
        if "controlnet_cond_embedding.conv_in.weight" in controlnet_data:  # diffusers format
            unet_dtype = memory_management.unet_dtype()
            controlnet_config = unet_config_from_diffusers_unet(controlnet_data, unet_dtype)
            diffusers_keys = unet_to_diffusers(controlnet_config)
            diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
            diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

            count = 0
            loop = True
            while loop:
                suffix = [".weight", ".bias"]
                for s in suffix:
                    k_in = "controlnet_down_blocks.{}{}".format(count, s)
                    k_out = "zero_convs.{}.0{}".format(count, s)
                    if k_in not in controlnet_data:
                        loop = False
                        break
                    diffusers_keys[k_in] = k_out
                count += 1

            count = 0
            loop = True
            while loop:
                suffix = [".weight", ".bias"]
                for s in suffix:
                    if count == 0:
                        k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                    else:
                        k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                    k_out = "input_hint_block.{}{}".format(count * 2, s)
                    if k_in not in controlnet_data:
                        k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                        loop = False
                    diffusers_keys[k_in] = k_out
                count += 1

            new_sd = {}
            for k in diffusers_keys:
                if k in controlnet_data:
                    new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

            leftover_keys = controlnet_data.keys()
            if len(leftover_keys) > 0:
                print("leftover keys:", leftover_keys)
            controlnet_data = new_sd

        pth_key = 'control_model.zero_convs.0.0.weight'
        pth = False
        key = 'zero_convs.0.0.weight'
        if pth_key in controlnet_data:
            pth = True
            key = pth_key
            prefix = "control_model."
        elif key in controlnet_data:
            prefix = ""
        else:
            net = load_t2i_adapter(controlnet_data)
            if net is None:
                return None
            return ControlNetPatcher(net)

        if controlnet_config is None:
            unet_dtype = memory_management.unet_dtype()
            controlnet_config = model_config_from_unet(controlnet_data, prefix, True).unet_config
            controlnet_config['dtype'] = unet_dtype

        load_device = memory_management.get_torch_device()
        computation_dtype = memory_management.get_computation_dtype(load_device)

        controlnet_config.pop("out_channels")
        controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]

        with using_forge_operations(dtype=unet_dtype, manual_cast_enabled=computation_dtype != unet_dtype):
            control_model = cldm.ControlNet(**controlnet_config).to(dtype=unet_dtype)

        if pth:
            if 'difference' in controlnet_data:
                print("WARNING: Your controlnet model is diff version rather than official float16 model. "
                      "Please use an official float16/float32 model for robust performance.")

            class WeightsLoader(torch.nn.Module):
                pass

            w = WeightsLoader()
            w.control_model = control_model
            missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
        else:
            missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)
        print(missing, unexpected)

        global_average_pooling = False
        filename = os.path.splitext(ckpt_path)[0]
        if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"):
            # TODO: smarter way of enabling global_average_pooling
            global_average_pooling = True

        control = ControlNet(control_model, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=computation_dtype)
        return ControlNetPatcher(control)

    def __init__(self, model_patcher):
        super().__init__(model_patcher)

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        unet = apply_controlnet_advanced(
            unet=unet,
            controlnet=self.model_patcher,
            image_bchw=cond,
            strength=self.strength,
            start_percent=self.start_percent,
            end_percent=self.end_percent,
            positive_advanced_weighting=self.positive_advanced_weighting,
            negative_advanced_weighting=self.negative_advanced_weighting,
            advanced_frame_weighting=self.advanced_frame_weighting,
            advanced_sigma_weighting=self.advanced_sigma_weighting,
            advanced_mask_weighting=self.advanced_mask_weighting
        )

        process.sd_model.forge_objects.unet = unet
        return


add_supported_control_model(ControlNetPatcher)

# Register ZIT ControlNet patcher
try:
    from modules_forge.supported_controlnet_zit import ZImageControlNetPatcher
    add_supported_control_model(ZImageControlNetPatcher)
except ImportError:
    pass  # ZIT ControlNet support may not be available
