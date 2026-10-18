"""
Forge integration for RES4LYF samplers and schedulers
"""

import functools
import importlib
import logging
import os
import sys
from contextlib import contextmanager
from typing import Callable

import k_diffusion.sampling

# Import sd_samplers at module level to avoid circular import
import modules.sd_samplers as sd_samplers
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_schedulers

logger = logging.getLogger(__name__)


@contextmanager
def res4lyf_data_prev_zeros_patch():
    """Grow RES4LYF ``data_prev_*`` buffers from size 4 to 8 for hybrid ``*4h4s``.

    RES4LYF allocates ``data_prev_`` / ``data_prev_x_`` / ``data_prev_y_`` as
    ``torch.zeros(4, *x.shape, ...)`` (``rk_sampler_beta.py``). For hybrid
    samplers like ``lawson45-gen-mod_4h4s`` where ``hybrid_stages = 4``,
    ``recycled_stages`` becomes 4 and the rotation loop writes
    ``data_prev_[4]`` → ``IndexError`` (size 4, valid 0..3).

    Same approach as A1111 ``a1111_res4lyf_shim.res4lyf_shim_context`` §3:
    do not edit ``modules/RES4LYF/``; temporarily replace ``torch`` only on
    the ``rk_sampler_beta`` module so leading-dim ``4`` zeros become ``8``.
    """
    rk_sampler_beta_mod = None
    try:
        for _modname in (
            "modules.RES4LYF.beta.rk_sampler_beta",
            "RES4LYF.beta.rk_sampler_beta",
            "beta.rk_sampler_beta",
        ):
            _m = sys.modules.get(_modname)
            if _m is not None:
                rk_sampler_beta_mod = _m
                break
        if rk_sampler_beta_mod is None:
            for _modname in (
                "modules.RES4LYF.beta.rk_sampler_beta",
                "RES4LYF.beta.rk_sampler_beta",
                "beta.rk_sampler_beta",
            ):
                try:
                    rk_sampler_beta_mod = importlib.import_module(_modname)
                    break
                except Exception:
                    continue
    except Exception:
        rk_sampler_beta_mod = None

    zeros_patched = False
    _rk_torch = None
    if rk_sampler_beta_mod is not None:
        try:
            _rk_torch = rk_sampler_beta_mod.torch
            _orig_torch_zeros = _rk_torch.zeros

            def _patched_zeros(*args, **kwargs):
                if args and isinstance(args[0], int) and args[0] == 4:
                    args = (8,) + args[1:]
                return _orig_torch_zeros(*args, **kwargs)

            class _TorchProxy:
                def __getattr__(self, name):
                    return getattr(_rk_torch, name)

                def zeros(self, *args, **kwargs):
                    return _patched_zeros(*args, **kwargs)

            rk_sampler_beta_mod.torch = _TorchProxy()
            zeros_patched = True
        except Exception:
            logger.debug(
                "[RES4LYF] Failed to patch rk_sampler_beta.torch.zeros",
                exc_info=True,
            )

    try:
        yield
    finally:
        if zeros_patched and rk_sampler_beta_mod is not None:
            try:
                rk_sampler_beta_mod.torch = _rk_torch
            except Exception:
                logger.debug(
                    "[RES4LYF] restore rk_sampler_beta.torch failed",
                    exc_info=True,
                )


class RES4LYFSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    """Wrapper for RES4LYF samplers"""

    def __init__(self, sd_model, sampler_name):
        sampler_function: Callable = getattr(k_diffusion.sampling, f"sample_{sampler_name}", None)
        if sampler_function is None:
            raise ValueError(f"Unknown RES4LYF sampler: {sampler_name}")

        def wrapped_sampler(*args, **kwargs):
            with res4lyf_data_prev_zeros_patch():
                return sampler_function(*args, **kwargs)

        functools.update_wrapper(wrapped_sampler, sampler_function)
        super().__init__(wrapped_sampler, sd_model, None)


def build_res4lyf_constructor(sampler_key: str) -> Callable:
    """Build constructor for RES4LYF sampler"""
    def constructor(model):
        return RES4LYFSampler(model, sampler_key)

    return constructor


def register_res4lyf_samplers():
    """Register RES4LYF samplers with Forge"""
    try:
        # Install required packages if not already installed
        try:
            from modules import launch_utils
            is_installed = launch_utils.is_installed
            run_pip = launch_utils.run_pip
            
            # Install pywavelets if not installed (required by RES4LYF)
            if not is_installed("pywavelets"):
                try:
                    run_pip("install pywavelets", "pywavelets")
                    logger.info("[RES4LYF] Installed pywavelets")
                except Exception as e:
                    logger.warning(f"[RES4LYF] Failed to install pywavelets: {e}")
            
            # Install comfy-kitchen if not installed
            if not is_installed("comfy-kitchen") and not is_installed("comfy_kitchen"):
                try:
                    run_pip("install comfy-kitchen", "comfy-kitchen")
                    logger.info("[RES4LYF] Installed comfy-kitchen")
                except Exception as e:
                    logger.warning(f"[RES4LYF] Failed to install comfy-kitchen: {e}")
        except ImportError:
            # launch_utils not available, skip automatic installation
            pass
        
        # Mock ComfyUI dependencies before importing RES4LYF to avoid import errors
        # RES4LYF requires folder_paths and server (PromptServer) which are ComfyUI dependencies
        import types
        
        # Mock folder_paths before any RES4LYF imports
        # RES4LYF's __init__.py imports loaders.py which imports folder_paths at module level
        if 'folder_paths' not in sys.modules:
            # Try to get folder_paths from comfy.cmd first (ComfyUI standard location)
            try:
                from comfy.cmd import folder_paths
                sys.modules['folder_paths'] = folder_paths
            except (ImportError, ModuleNotFoundError):
                # Create a minimal mock for folder_paths
                class MockFolderPaths:
                    @staticmethod
                    def get_filename_list(folder_type):
                        return []
                    
                    @staticmethod
                    def get_full_path_or_raise(folder_type, filename):
                        raise FileNotFoundError(f"Mock folder_paths: {folder_type}/{filename}")
                    
                    @staticmethod
                    def get_input_directory():
                        return ""
                    
                    @staticmethod
                    def get_output_directory():
                        return ""
                    
                    @staticmethod
                    def get_save_image_path(*args, **kwargs):
                        return ("", "", 0, "", "")
                
                # Inject mock into sys.modules as a proper module object
                mock_module = types.ModuleType('folder_paths')
                mock_folder_paths = MockFolderPaths()
                for attr_name in dir(mock_folder_paths):
                    if not attr_name.startswith('_'):
                        setattr(mock_module, attr_name, getattr(mock_folder_paths, attr_name))
                sys.modules['folder_paths'] = mock_module
        
        # Mock server module (contains PromptServer) - required by res4lyf.py
        if 'server' not in sys.modules:
            # Create a minimal mock for server module with PromptServer class
            # PromptServer.instance is used as a decorator and for routes
            class MockPromptServerInstance:
                def __init__(self):
                    # Create a mock routes object with post method that returns a no-op decorator
                    class MockRoutes:
                        def post(self, path):
                            # Return a decorator that does nothing
                            def decorator(func):
                                return func
                            return decorator
                    self.routes = MockRoutes()
                    self.client_id = None
                    self.supports = set()
                
                def send_sync(self, event_type, data):
                    pass
                
                async def send(self, event_type, data):
                    pass
            
            class MockPromptServer:
                instance = MockPromptServerInstance()
            
            server_module = types.ModuleType('server')
            server_module.PromptServer = MockPromptServer
            sys.modules['server'] = server_module
        
        # Add ComfyUI directory to sys.path (path from modules.paths / COMFYUI_FOLDER_NAME)
        from modules.paths import comfyui_master_path
        if os.path.exists(comfyui_master_path):
            if comfyui_master_path not in sys.path:
                sys.path.insert(0, comfyui_master_path)
                logger.info(f"[RES4LYF] Added ComfyUI to sys.path: {comfyui_master_path}")
            nodes_py = os.path.join(comfyui_master_path, "nodes.py")
            latent_preview_py = os.path.join(comfyui_master_path, "latent_preview.py")
            logger.info(f"[RES4LYF] nodes.py exists: {os.path.exists(nodes_py)}, latent_preview.py exists: {os.path.exists(latent_preview_py)}")
        else:
            logger.warning(f"[RES4LYF] ComfyUI not found at: {comfyui_master_path}")
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from modules.paths_internal import COMFYUI_FOLDER_NAME
            comfyui_master_path_alt = os.path.join(parent_dir, COMFYUI_FOLDER_NAME)
            if os.path.exists(comfyui_master_path_alt) and comfyui_master_path_alt not in sys.path:
                sys.path.insert(0, comfyui_master_path_alt)
                logger.info(f"[RES4LYF] Added ComfyUI to sys.path (from parent): {comfyui_master_path_alt}")
        
        # Import RES4LYF module - this will automatically call add_samplers() in __init__.py
        # Note: RES4LYF/__init__.py imports helper_sigma_preview_image_preproc and nodes_latents
        # which require nodes and latent_preview modules to be available
        # Note: RES4LYF also requires optional dependencies like matplotlib, pywavelets
        # but these are only needed for certain nodes, not for sampler registration
        from modules import RES4LYF
        sys.modules['res4lyf'] = RES4LYF
        sys.modules['RES4LYF'] = RES4LYF
        import comfy.k_diffusion.sampling as comfy_k_diffusion_sampling
        
        # Get the extra_samplers dictionary (populated by beta/legacy/zampler modules)
        extra_samplers = getattr(RES4LYF, 'extra_samplers', {})
        
        if not extra_samplers:
            logger.info("[RES4LYF] No samplers found in extra_samplers")
            return
        
        # Copy sampler functions from comfy.k_diffusion.sampling to k_diffusion.sampling
        # RES4LYF registers samplers in comfy.k_diffusion.sampling, but Forge uses k_diffusion.sampling
        # Check if k_diffusion.sampling and comfy.k_diffusion.sampling are different modules
        if k_diffusion.sampling is not comfy_k_diffusion_sampling:
            # Copy all RES4LYF sampler functions from comfy.k_diffusion.sampling to k_diffusion.sampling
            logger.info("[RES4LYF] Syncing RES4LYF samplers from comfy.k_diffusion.sampling to k_diffusion.sampling")
            for sampler_name in extra_samplers:
                sampler_func = getattr(comfy_k_diffusion_sampling, f"sample_{sampler_name}", None)
                if sampler_func is not None:
                    setattr(k_diffusion.sampling, f"sample_{sampler_name}", sampler_func)
                    logger.debug(f"[RES4LYF] Synced sampler function: sample_{sampler_name}")
        
        # Register each sampler with Forge
        registered_count = 0
        existing_sampler_names = {x.name for x in sd_samplers.all_samplers}
        
        for sampler_name in extra_samplers:
            if sampler_name in existing_sampler_names:
                logger.debug(f"[RES4LYF] Sampler '{sampler_name}' already registered in Forge, skipping")
                continue
            
            # Verify the sampler function exists in k_diffusion.sampling
            sampler_func = getattr(k_diffusion.sampling, f"sample_{sampler_name}", None)
            if sampler_func is None:
                logger.warning(f"[RES4LYF] Sampler function 'sample_{sampler_name}' not found in k_diffusion.sampling after sync")
                continue
            
            try:
                # Create SamplerData for this sampler
                sampler_data = sd_samplers_common.SamplerData(
                    sampler_name,
                    build_res4lyf_constructor(sampler_key=sampler_name),
                    [sampler_name],
                    {}
                )
                
                # Add to all_samplers
                sd_samplers.add_sampler(sampler_data)
                registered_count += 1
                logger.info(f"[RES4LYF] Registered sampler: {sampler_name}")
            except Exception as e:
                logger.warning(f"[RES4LYF] Failed to register sampler '{sampler_name}': {e}")
        
        if registered_count > 0:
            logger.info(f"[RES4LYF] Successfully registered {registered_count} RES4LYF samplers")
        else:
            logger.info("[RES4LYF] No new RES4LYF samplers were registered (they may already be registered)")
    
    except ImportError as e:
        logger.warning(f"[RES4LYF] Failed to import RES4LYF module: {e}")
    except Exception as e:
        logger.error(f"[RES4LYF] Error registering samplers: {e}", exc_info=True)


def register_res4lyf_schedulers():
    """Register RES4LYF schedulers with Forge"""
    try:
        from modules import RES4LYF
        from comfy.samplers import SCHEDULER_HANDLERS, SCHEDULER_NAMES
        
        # Check if RES4LYF has added any schedulers
        registered_count = 0
        
        # Check for beta57 scheduler
        if "beta57" not in [s.name for s in sd_schedulers.schedulers]:
            try:
                # Try to get the handler from RES4LYF's res4lyf module
                from modules.RES4LYF import res4lyf
                if hasattr(res4lyf, 'calculate_sigmas_RES4LYF'):
                    # beta57 scheduler function
                    def beta57_scheduler(n, sigma_min, sigma_max, inner_model, device):
                        from comfy import samplers as comfy_samplers
                        return comfy_samplers.beta_scheduler(inner_model, n, alpha=0.5, beta=0.7)
                    
                    scheduler = sd_schedulers.Scheduler(
                        "beta57",
                        "Beta57",
                        beta57_scheduler,
                        need_inner_model=True
                    )
                    sd_schedulers.schedulers.append(scheduler)
                    sd_schedulers.schedulers_map["beta57"] = scheduler
                    sd_schedulers.schedulers_map["Beta57"] = scheduler
                    registered_count += 1
                    logger.info("[RES4LYF] Registered scheduler: beta57")
            except Exception as e:
                logger.debug(f"[RES4LYF] Could not register beta57 scheduler: {e}")
        
        # bong_tangent should already be registered by RES4LYF's __init__.py
        # but let's check if it's in our schedulers list
        if "bong_tangent" in SCHEDULER_NAMES:
            # Check if it's already in sd_schedulers
            if "bong_tangent" not in [s.name for s in sd_schedulers.schedulers]:
                # bong_tangent is already registered in sd_schedulers.py, so we skip
                pass
        
        if registered_count > 0:
            logger.info(f"[RES4LYF] Successfully registered {registered_count} schedulers")
    
    except ImportError as e:
        logger.warning(f"[RES4LYF] Failed to import RES4LYF module: {e}")
    except Exception as e:
        logger.error(f"[RES4LYF] Error registering schedulers: {e}", exc_info=True)
