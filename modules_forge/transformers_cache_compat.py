"""
Compatibility shims for transformers 5+ and diffusers.

Applied lazily via __import__ hook:
  1. HybridCache -> DynamicCache alias (for peft 0.17.x)
  2. modeling_utils.no_init_weights -> local shim (backend/loader.py, etc.)
  3. Re-validate diffusers _flash_attn_available with a real import
     (find_spec alone cannot detect broken C extensions / DLL mismatch)

Important: apply() must not import transformers or diffusers at module load time.
All patches run inside the import hook.
"""
import builtins
from contextlib import contextmanager

_original_import = None
_no_init_weights_patched = False
_diffusers_import_utils_patched = False


@contextmanager
def _no_init_weights():
    """Shim compatible with transformers 4.x/5.x modeling_utils.no_init_weights().
    Disables torch.nn.init ops and PreTrainedModel.init_weights inside the
    context so weight initialisation is skipped."""
    import torch

    _skip = {
        "uniform_": torch.nn.init.uniform_,
        "normal_": torch.nn.init.normal_,
        "trunc_normal_": torch.nn.init.trunc_normal_,
        "constant_": torch.nn.init.constant_,
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "xavier_normal_": torch.nn.init.xavier_normal_,
        "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
        "kaiming_normal_": torch.nn.init.kaiming_normal_,
        "orthogonal_": torch.nn.init.orthogonal_,
        "sparse_": torch.nn.init.sparse_,
        "zeros_": torch.nn.init.zeros_,
        "ones_": torch.nn.init.ones_,
        "dirac_": torch.nn.init.dirac_,
        "eye_": torch.nn.init.eye_,
    }

    def _noop(*args, **kwargs):
        pass

    # Also patch PreTrainedModel.init_weights (transformers 5.x calls this
    # automatically during construction; the native no_init_weights does the
    # same).
    _original_init_weights = None
    try:
        from transformers.modeling_utils import PreTrainedModel
        _original_init_weights = PreTrainedModel.init_weights
        PreTrainedModel.init_weights = _noop
    except Exception:
        pass

    try:
        for name in _skip:
            setattr(torch.nn.init, name, _noop)
        yield
    finally:
        for name, fn in _skip.items():
            setattr(torch.nn.init, name, fn)
        if _original_init_weights is not None:
            PreTrainedModel.init_weights = _original_init_weights


def _patch_no_init_weights(mod):
    """Inject no_init_weights into modeling_utils if missing."""
    global _no_init_weights_patched
    if _no_init_weights_patched:
        return
    _no_init_weights_patched = True
    if not hasattr(mod, "no_init_weights"):
        mod.no_init_weights = _no_init_weights


def _patch_diffusers_flash_attn_flag(mod):
    """Re-validate diffusers _flash_attn_available with a real import.

    diffusers uses importlib.util.find_spec() for optional packages; that does not
    detect DLL/symbol mismatch in flash_attn_2_cuda. After a torch upgrade an old
    .pyd may remain: find_spec is True but import fails and startup crashes.

    This runs once right after diffusers.utils.import_utils loads. If
    _flash_attn_available is True, try import flash_attn; on failure set the flag
    to False.
    """
    global _diffusers_import_utils_patched
    if _diffusers_import_utils_patched:
        return
    _diffusers_import_utils_patched = True

    if not getattr(mod, "_flash_attn_available", False):
        return

    try:
        _original_import("flash_attn")
    except Exception:
        mod._flash_attn_available = False
        mod._flash_attn_version = "N/A"
        print("[transformers_cache_compat] flash_attn native extension failed to load; "
              "marked unavailable (will be reinstalled on next launch with --flash)")


def _patch_gradio_client(mod):
    orig_json_schema = getattr(mod, "_json_schema_to_python_type", None)
    if orig_json_schema is not None and not getattr(orig_json_schema, "_is_patched", False):
        def safe_json_schema_to_python_type(schema, defs=None):
            if isinstance(schema, bool):
                return "Any" if schema else "None"
            if not isinstance(schema, dict):
                return "Any"
            return orig_json_schema(schema, defs)
        safe_json_schema_to_python_type._is_patched = True
        mod._json_schema_to_python_type = safe_json_schema_to_python_type

    orig_get_type = getattr(mod, "get_type", None)
    if orig_get_type is not None and not getattr(orig_get_type, "_is_patched", False):
        def safe_get_type(schema):
            if isinstance(schema, bool):
                return "Any" if schema else "None"
            if not isinstance(schema, dict):
                return "Any"
            return orig_get_type(schema)
        safe_get_type._is_patched = True
        mod.get_type = safe_get_type


def _hooked_import(name, *args, **kwargs):
    if name == "torchaudio" or name.startswith("torchaudio."):
        import sys
        if name in sys.modules:
            return sys.modules[name]
        try:
            return _original_import(name, *args, **kwargs)
        except Exception:
            from modules_forge import torchaudio_compat
            torchaudio_compat.apply()
            if name in sys.modules:
                return sys.modules[name]
            return torchaudio_compat._make_stub(name)

    if name == "facexlib.version" or (name == "version" and args and isinstance(args[0], dict) and args[0].get("__package__") == "facexlib"):
        import sys
        if "facexlib.version" in sys.modules:
            return sys.modules["facexlib.version"]
        try:
            return _original_import(name, *args, **kwargs)
        except Exception:
            import types
            vmod = types.ModuleType("facexlib.version")
            vmod.__version__ = "0.3.0"
            vmod.__gitsha__ = "unknown"
            sys.modules["facexlib.version"] = vmod
    if name in ("res4lyf", "RES4LYF") or name.startswith(("res4lyf.", "RES4LYF.")):
        import sys
        if name in sys.modules:
            return sys.modules[name]
        try:
            return _original_import(name, *args, **kwargs)
        except (ModuleNotFoundError, ImportError):
            if name in ("res4lyf", "RES4LYF"):
                real_mod = _original_import("modules.RES4LYF", *args, **kwargs)
            else:
                sub = name.split(".", 1)[1]
                real_mod = _original_import(f"modules.RES4LYF.{sub}", *args, **kwargs)
            sys.modules[name] = real_mod
            return real_mod

    try:
        mod = _original_import(name, *args, **kwargs)
    except ModuleNotFoundError as e:
        if "facexlib.version" in str(e) or name in ("version", "facexlib.version"):
            import sys
            import types
            vmod = types.ModuleType("facexlib.version")
            vmod.__version__ = "0.3.0"
            vmod.__gitsha__ = "unknown"
            sys.modules["facexlib.version"] = vmod
            return vmod
        raise

    # HybridCache patch
    if name == "transformers":
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if fromlist and "HybridCache" in fromlist and not hasattr(mod, "HybridCache"):
            from transformers.cache_utils import DynamicCache
            mod.__dict__["HybridCache"] = DynamicCache

    # no_init_weights patch
    if name == "transformers.modeling_utils":
        _patch_no_init_weights(mod)
    elif name == "transformers":
        fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
        if fromlist and "modeling_utils" in fromlist:
            import sys
            mu = sys.modules.get("transformers.modeling_utils")
            if mu is not None:
                _patch_no_init_weights(mu)

    # diffusers flash_attn availability patch
    if name == "diffusers.utils.import_utils":
        _patch_diffusers_flash_attn_flag(mod)
        if hasattr(mod, "is_torchaudio_available"):
            mod.is_torchaudio_available = lambda: False

    # transformers import_utils torchaudio flag patch
    if name == "transformers.utils.import_utils":
        mod.is_torchaudio_available = lambda: False

    # gradio_client pydantic v2 boolean additionalProperties schema patch
    if name == "gradio_client.utils":
        _patch_gradio_client(mod)

    # numpy >= 2.4 compatibility: _blas_supports_fpe stub for scipy < 1.18
    if name in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        if not hasattr(mod, "_blas_supports_fpe"):
            mod._blas_supports_fpe = lambda x: False

    return mod


def apply():
    """Call before importing nunchaku / peft / diffusers; does not import transformers."""
    global _original_import

    from modules_forge import torchaudio_compat
    torchaudio_compat.apply()

    import sys
    if "gradio_client.utils" in sys.modules:
        _patch_gradio_client(sys.modules["gradio_client.utils"])

    if _original_import is not None:
        return  # idempotent

    _original_import = builtins.__import__
    builtins.__import__ = _hooked_import
