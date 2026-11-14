import math

import torch

from backend import memory_management
from backend.modules.k_prediction import k_prediction_from_diffusers_scheduler

# Import comfy modules for WrapperExecutor
import comfy.patcher_extension
from backend.modules.lumina_k_glue import (
    forward_nextdit,
    is_nextdit_model,
    merge_kmodel_transformer_options,
)


def _is_anima_diffusion_model(diffusion_model) -> bool:
    """ComfyUI ``comfy.ldm.anima.model.Anima`` (MiniTrainDIT + llm_adapter)."""
    try:
        from comfy.ldm.anima.model import Anima as ComfyAnima

        return isinstance(diffusion_model, ComfyAnima)
    except ImportError:
        return False


def reshape_sigma(sigma, noise_dim):
    if sigma.nelement() == 1:
        return sigma.view(())
    else:
        return sigma.view(sigma.shape[:1] + (1,) * (noise_dim - 1))


class CONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = reshape_sigma(sigma, model_output.ndim)
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = reshape_sigma(sigma, noise.ndim)
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        sigma = reshape_sigma(sigma, latent.ndim)
        return latent / (1.0 - sigma)


def flux_time_shift(mu: float, sigma: float, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class ModelSamplingFlux(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        self.set_parameters(shift=sampling_settings.get("shift", 1.15))

    def set_parameters(self, shift=1.15, timesteps=10000):
        self.shift = shift
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps))
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return flux_time_shift(self.shift, 1.0, timestep)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return flux_time_shift(self.shift, 1.0, 1.0 - percent)


class ModelSamplingFluxWithConst(ModelSamplingFlux, CONST):
    pass


class KModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, diffusers_scheduler, k_predictor=None, config=None):
        super().__init__()

        self.config = config

        self.storage_dtype = model.storage_dtype
        self.computation_dtype = model.computation_dtype

        print(f"K-Model Created: {dict(storage_dtype=self.storage_dtype, computation_dtype=self.computation_dtype)}")

        self.diffusion_model = model
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)

        if k_predictor is None:
            self.predictor = k_prediction_from_diffusers_scheduler(diffusers_scheduler)
        else:
            self.predictor = k_predictor

        # Add ComfyUI-style model_sampling (ControlNet.pre_run() expects model.model_sampling)
        if config is not None and hasattr(config, "model_type"):
            model_type = config.model_type
            if hasattr(model_type, "name"):
                import comfy.model_base
                from comfy.model_base import ModelType
                if model_type.name == "FLUX":
                    self.model_sampling = ModelSamplingFluxWithConst(config)
                elif model_type.name == "FLOW":
                    # Qwen Image uses ModelType.FLOW
                    # Use ComfyUI's model_sampling function to create the correct ModelSampling class
                    model_type_enum = ModelType.FLOW
                    self.model_sampling = comfy.model_base.model_sampling(config, model_type_enum)
                elif model_type.name == "EPS":
                    # SDXL and other EPS models use ModelType.EPS
                    # Use ComfyUI's model_sampling function to create the correct ModelSampling class
                    model_type_enum = ModelType.EPS
                    self.model_sampling = comfy.model_base.model_sampling(config, model_type_enum)
                else:
                    # For other model types, try to use ComfyUI's model_sampling function
                    # This handles V_PREDICTION, V_PREDICTION_EDM, STABLE_CASCADE, EDM, etc.
                    try:
                        model_type_enum = getattr(ModelType, model_type.name, None)
                        if model_type_enum is not None:
                            self.model_sampling = comfy.model_base.model_sampling(config, model_type_enum)
                        else:
                            self.model_sampling = None
                    except (AttributeError, TypeError):
                        self.model_sampling = None
            else:
                self.model_sampling = None
        else:
            self.model_sampling = None
        
        # Initialize model_options for transformer_options (used by ControlNet patches, etc.)
        self.model_options = {"transformer_options": {}}

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        transformer_options = merge_kmodel_transformer_options(
            self.model_options,
            transformer_options,
            self.diffusion_model,
        )

        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._apply_model,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.predictor.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.computation_dtype

        xc = xc.to(dtype)
        t = self.predictor.timestep(t).float()
        context = context.to(dtype)
        anima_temporal = _is_anima_diffusion_model(self.diffusion_model) and xc.ndim == 4
        if anima_temporal:
            xc = xc.unsqueeze(2)
        extra_conds = {}
        for o in kwargs:
            # Skip transformer_options as it's passed explicitly
            if o == "transformer_options":
                continue
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        # Remove transformer_options from extra_conds to avoid duplicate argument
        extra_conds_clean = {k: v for k, v in extra_conds.items() if k != "transformer_options"}

        if is_nextdit_model(self.diffusion_model):
            model_output = forward_nextdit(
                self.diffusion_model,
                xc,
                t,
                context,
                control,
                extra_conds_clean,
                transformer_options,
            ).float()
        else:
            # Nunchaku Qwen Image ONLY: forward uses patches["double_block"] (Fun ControlNet), so pass transformer_options.
            # Other models (SD1.5, SDXL, Flux1, ComfyUI Qwen Image): do NOT pass transformer_options (no side effects).
            # Use __name__ so any NunchakuQwenImageTransformer2DModel (e.g. from svdq or nunchaku_sdxl_unet) is recognized.
            dm_type_name = type(self.diffusion_model).__name__
            is_nunchaku_qwen_image = dm_type_name == "NunchakuQwenImageTransformer2DModel"
            if is_nunchaku_qwen_image:
                model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds_clean).float()
            elif "patches_replace" in (transformer_options or {}):
                # IP-Adapter (attn2 patches_replace) needs transformer_options to reach the
                # UNet forward; otherwise patches_replace is registered but never consumed
                # and the conditioning has no effect on the generated image.
                # "patches" (attn1_patch family, e.g. ToMeSD token merging) stays excluded:
                # this forward never consumed it before, and ToMeSD re-registers its patch
                # every generation (apply_token_merging), so passing it would merge tokens
                # twice into the same tensor and crash (backend/misc/tomesd.py size mismatch).
                forward_options = {k: v for k, v in (transformer_options or {}).items() if k != "patches"}
                model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=forward_options, **extra_conds_clean).float()
            else:
                model_output = self.diffusion_model(xc, t, context=context, control=control, **extra_conds_clean).float()

        if anima_temporal and model_output.ndim == 5:
            model_output = model_output.squeeze(2)

        return self.predictor.calculate_denoised(sigma, model_output, x)

    def memory_required(self, input_shape: list[int]) -> float:
        """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/model_base.py#L354"""
        input_shapes = [input_shape]
        area = sum(map(lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes))

        if memory_management.xformers_enabled():
            return (area * memory_management.dtype_size(self.computation_dtype) * 0.01 * self.config.memory_usage_factor) * (1024 * 1024)
        else:
            return (area * 0.15 * self.config.memory_usage_factor) * (1024 * 1024)

    def cleanup(self):
        del self.config
        del self.predictor
        del self.diffusion_model
