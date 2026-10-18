# Technical Documentation: ComfyUI v0.40.0 AIMDO Compatibility Stubs and RES4LYF Sampler Restoration

---

## Part 1: ComfyUI v0.40.0 AIMDO Import Failure Resolution

### 1. Error Overview

#### Symptom & Stack Trace
During initialization of the bundled ComfyUI core (v0.40.0) or when importing core modules such as `comfy.ops`, `comfy.storage`, or `comfy.model_prefetch`, startup fails unconditionally with `ModuleNotFoundError`:

```text
Traceback (most recent call last):
  File "modules/initialize.py", line 49, in imports
    from modules import processing
  File "modules/processing.py", line 18, in <module>
    import comfy.samplers
  File "ComfyUI-master/comfy/samplers.py", line 16, in <module>
    import comfy.sampler_helpers
  File "ComfyUI-master/comfy/sampler_helpers.py", line 8, in <module>
    import comfy.model_prefetch
  File "ComfyUI-master/comfy/model_prefetch.py", line 12, in <module>
    import comfy.storage
  File "ComfyUI-master/comfy/storage.py", line 10, in <module>
    from comfy_aimdo.storage import fast_disk
ModuleNotFoundError: No module named 'comfy_aimdo.storage'
```

A subsequent failure on execution of stream-aware memory allocation blocks in `comfy.ops` / `comfy.model_management`:

```text
Traceback (most recent call last):
  File "ComfyUI-master/comfy/ops.py", line 22, in <module>
    from comfy_aimdo.malloc_graph import MallocGraph, use_stream
ModuleNotFoundError: No module named 'comfy_aimdo.malloc_graph'
```

---

### 2. Root Cause Analysis

Upstream ComfyUI introduced optional dependencies on the proprietary hardware acceleration / memory manager library `comfy_aimdo` (AIMDO). Specifically:
1. `comfy.storage` calls `from comfy_aimdo.storage import fast_disk` to query whether storage volumes support direct hardware bypass.
2. `comfy.ops` and model compilation paths call `from comfy_aimdo.malloc_graph import MallocGraph, use_stream` to manage CUDA graph allocations and stream synchronization.

In standard environments where the proprietary binary wheel of `comfy_aimdo` is not installed, these unshielded top-level imports abort initialization. While Forge previously provided top-level compatibility stubs for `comfy_aimdo.control`, `comfy_aimdo.vram_buffer`, `comfy_aimdo.host_buffer`, and `comfy_aimdo.model_mmap`, ComfyUI v0.40.0 introduced `storage` and `malloc_graph` as new module entry points that were not yet stubbed.

---

### 3. Created & Modified Files

- **`comfy_aimdo/storage.py`** *(Newly Created)*
- **`comfy_aimdo/malloc_graph.py`** *(Newly Created)*

---

### 4. Full Source Code of New / Modified Implementations

#### File: `comfy_aimdo/storage.py`
```python
# Stub: real comfy_aimdo.storage checks fast disk on Windows. Forge uses its own loader.


def fast_disk(path):
    """Stub: return False so ComfyUI falls back to standard disk loading."""
    return False
```

#### File: `comfy_aimdo/malloc_graph.py`
```python
# Stub: real comfy_aimdo.malloc_graph provides CUDA graph memory compilation. Forge uses its own loader.

from contextlib import contextmanager


class MallocGraph:
    def __init__(self, *args, **kwargs):
        self._scopes = []

    def push(self, name=None):
        pass

    def pop(self):
        return False

    def abort(self):
        pass

    def pause(self, sync=False):
        pass

    def resume(self, sync=False):
        pass

    @contextmanager
    def use_stream(self, stream):
        yield

    def iterate(self, name=None):
        return False
```

---

### 5. Technical Rationale & Architectural Function

1. **`fast_disk(path) -> False`**:
   Returning `False` explicitly signals to ComfyUI's internal model prefetcher and storage subsystem that the underlying disk is standard filesystem storage. This forces ComfyUI to bypass AIMDO direct kernel I/O paths and seamlessly use standard Python/PyTorch `safetensors.torch.load_file` and file descriptors, preserving full compatibility without external binary dependencies.

2. **`MallocGraph` & `use_stream`**:
   Provides an exact drop-in structural interface for the expected AIMDO memory graph compiler. `use_stream(stream)` acts as a transparent, no-overhead context manager (`yield`), allowing any stream management in ComfyUI ops to execute normally on PyTorch's native CUDA stream stack without requiring proprietary memory pooling binaries.

---

## Part 2: RES4LYF Sampler Restoration & Import Architecture

### 1. Error Overview

#### Symptom & Stack Trace
During WebUI startup in `initialize_rest()`, registering the RES4LYF sampler and scheduler suite failed with a `NameError`:

```text
WARNING:root:Failed to register RES4LYF samplers: name 'res4lyf' is not defined
Traceback (most recent call last):
  File "d:\USERFILES\GitHub\Stable-Diffusion-WebUI-Forge-Nunchaku\modules\initialize.py", line 110, in initialize_rest
    forge_res4lyf_samplers.register_res4lyf_samplers()
  File "d:\USERFILES\GitHub\Stable-Diffusion-WebUI-Forge-Nunchaku\modules_forge\forge_res4lyf_samplers.py", line 248, in register_res4lyf_samplers
    from modules import RES4LYF
  File "d:\USERFILES\GitHub\Stable-Diffusion-WebUI-Forge-Nunchaku\modules\RES4LYF\__init__.py", line 34, in <module>
    res4lyf.init()
NameError: name 'res4lyf' is not defined
```

Furthermore, importing `res4lyf` or `modules.RES4LYF.res4lyf` directly from external scripts failed with:

```text
Traceback (most recent call last):
  File "d:\USERFILES\GitHub\Stable-Diffusion-WebUI-Forge-Nunchaku\modules\RES4LYF\res4lyf.py", line 13, in <module>
    from server import PromptServer
ModuleNotFoundError: No module named 'server'
```

And importing via `import res4lyf` or `import RES4LYF` from outside the `modules` package failed with:

```text
ModuleNotFoundError: No module named 'res4lyf'
```

---

### 2. Root Cause Analysis

1. **Unimported Module Reference in `modules/RES4LYF/__init__.py`**:
   Line 28 contained `from .res4lyf import RESplain`, which bound only the function `RESplain` to the module namespace. When Line 34 subsequently invoked `res4lyf.init()`, the symbol `res4lyf` did not exist in the globals of `__init__.py`, immediately raising `NameError: name 'res4lyf' is not defined` and crashing the import of the entire RES4LYF package.

2. **Unshielded Server Dependency in `modules/RES4LYF/res4lyf.py`**:
   `res4lyf.py` included a direct, unshielded `from server import PromptServer` at line 13 to register REST API routes (`@PromptServer.instance.routes.post(...)`). While `modules_forge/forge_res4lyf_samplers.py` created a mock server instance prior to importing `RES4LYF`, any independent or direct import of `res4lyf.py` in standalone contexts failed with `ModuleNotFoundError: No module named 'server'`.

3. **Top-Level Package Namespace Mismatch**:
   In Forge, RES4LYF is vendored inside `modules/RES4LYF`. Nodes, extensions, and user scripts written against ComfyUI custom node conventions frequently execute `import res4lyf` or `import RES4LYF` directly. Without import hook resolution, Python failed to locate `res4lyf` on the root `sys.path`.

---

### 3. Created & Modified Files

- **`modules/RES4LYF/__init__.py`** *(Modified)*
- **`modules/RES4LYF/res4lyf.py`** *(Modified)*
- **`modules_forge/transformers_cache_compat.py`** *(Modified)*
- **`modules_forge/forge_res4lyf_samplers.py`** *(Modified)*

---

### 4. Full Source Code of New / Modified Implementations

#### File: `modules/RES4LYF/__init__.py` (Lines 20–38)
```python
from comfy.samplers import SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES
new_scheduler_name = "bong_tangent"
if new_scheduler_name not in SCHEDULER_HANDLERS:
    bong_tangent_handler = SchedulerHandler(handler=sigmas.bong_tangent_scheduler, use_ms=True)
    SCHEDULER_HANDLERS[new_scheduler_name] = bong_tangent_handler
    SCHEDULER_NAMES.append(new_scheduler_name)


from . import res4lyf
from .res4lyf import RESplain

#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

res4lyf.init()

discard_penultimate_sigma_samplers = set((
))
```

#### File: `modules/RES4LYF/res4lyf.py` (Lines 10–35)
```python
import comfy.model_sampling
import comfy.samplers
from aiohttp import web
try:
    from server import PromptServer
except (ImportError, ModuleNotFoundError):
    class _MockPromptServerInstance:
        def __init__(self):
            class _MockRoutes:
                def post(self, path):
                    def decorator(func):
                        return func
                    return decorator
            self.routes = _MockRoutes()
            self.client_id = None
            self.supports = set()

        def send_sync(self, event_type, data):
            pass

        async def send(self, event_type, data):
            pass

    class PromptServer:
        instance = _MockPromptServerInstance()
from tqdm import tqdm
```

#### File: `modules_forge/transformers_cache_compat.py` (Lines 150–175)
```python
        try:
            return _original_import(name, *args, **kwargs)
        except Exception:
            import types
            vmod = types.ModuleType("facexlib.version")
            vmod.__version__ = "0.3.0"
            vmod.__gitsha__ = "unknown"
            sys.modules["facexlib.version"] = vmod
            return vmod

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
```

#### File: `modules_forge/forge_res4lyf_samplers.py` (Lines 243–257)
```python
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
```

---

### 5. Technical Rationale & Architectural Function

1. **`from . import res4lyf`**:
   Ensures the submodule `res4lyf` is explicitly imported and bound to the package namespace of `modules.RES4LYF` before calling `res4lyf.init()`. This satisfies the symbol resolution and allows `res4lyf.init()` to register custom sigmas and monkey-patch `comfy.samplers.calculate_sigmas` seamlessly.

2. **`PromptServer` Standalone Mock**:
   Provides an embedded fallback mock implementing `.instance.routes.post()` and event delivery handlers. If `res4lyf.py` is imported in an environment where ComfyUI's HTTP server is not active or `server` is not in `sys.modules`, route decorators degrade to no-op wrappers rather than terminating the process with an import exception.

3. **`_hooked_import` Transparent Namespace Aliasing**:
   Intercepts top-level `res4lyf` / `RES4LYF` import calls via the global import hook. If a caller requests `import res4lyf` or `import res4lyf.beta`, the hook resolves the actual module from `modules.RES4LYF`, caches the resulting module object in `sys.modules`, and returns it. This guarantees zero breakage regardless of how external nodes reference the library.

4. **Bi-directional `sys.modules` Sync in Forge Loader**:
   When `forge_res4lyf_samplers.py` loads `modules.RES4LYF`, it synchronizes `sys.modules['res4lyf']` and `sys.modules['RES4LYF']` to the same module reference. All samplers (`rk_beta`, `res_multistep`, ODE variants) are safely synced to Forge's `k_diffusion.sampling` and registered into `sd_samplers.all_samplers`.
