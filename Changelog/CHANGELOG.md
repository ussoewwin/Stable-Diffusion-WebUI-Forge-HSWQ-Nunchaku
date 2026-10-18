# Changelog

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../zhmd/CHANGELOG.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## Version 2.0.1

- **Deprecation & Removal: SDXL ConvRot NVFP4 Support**
  - Officially removed all references and support for **SDXL ConvRot NVFP4** in accordance with upstream [Hybrid-Sensitivity-Weighted-Quantization (HSWQ)](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization) updates.
  - **Technical Rationale**: Deterministic latent-trajectory divergence measurements revealed that 4-bit precision (NVFP4) fundamentally fails to preserve latent trajectory fidelity on SDXL (final latent trajectory cosine fails to reach 0.90, suffering severe trajectory drift and collapse). Legacy decoded SSIM (~0.95) scores were masked by the VAE decoder rather than true latent-space preservation.
  - **SDXL Precision Policy**: SDXL is supported strictly in **ConvRot INT8 (reverse hybrid)** only (`int8` / `int8 (fp16 LoRA)`).
  - Cleaned up documentation, READMEs, and loader specifications across the repository.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v2.0.1) for details.

## Version 2.0.0

- **Python 3.14 Exclusive Support & Platform Compatibility Update**
  - Upgraded runtime environment exclusively to **Python 3.14** across Windows and Linux (CUDA 13.2 / PyTorch 2.13 stack).
  - Bundled automated universal wheel installation for Python 3.14 runtime requirements:
    - **`distutils`** (`distutils-3.14.0-py3-none-any.whl`)
    - **`pyaudioop`** (`pyaudioop-1.0.0-py3-none-any.whl`)
    - **`imghdr`** (`imghdr-1.0.0-py3-none-any.whl`)
    - **`facexlib`** (`facexlib-0.3.0-py3-none-any.whl`, pure Python dependency-stripped build with embedded `version.py`)
    - **`gradio 4.40.0`** (`gradio-4.40.0-py3-none-any.whl` with unpinned dependencies and OpenAPI boolean schema compatibility)
  - Dedicated **Nunchaku 1.3.0** prebuilt wheels for Python 3.14 on CUDA 13.2:
    - Windows: `nunchaku-1.3.0.dev20260619+cu13.2torch2.12-cp314-cp314-win_amd64.whl`
    - Linux: `nunchaku-1.3.0.dev20260829+cu13.2torch2.13-cp314-cp314-linux_x86_64.whl`
  - Integrated **SageAttention 3 (SA3 / Blackwell FP4)**, **SageAttention 2 (SA2)**, and **FlashAttention 2 (FA2)** attention backends.
  - Added built-in `torchaudio` compatibility module with standard ModuleSpec stub for Python 3.14.
  - Stabilized `StableDiffusionProcessing.sd_model` synchronization and eliminated module shadowing in `modules/shared.py`.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v2.0.0) for details.

## Version 1.8.6

- **Fix: root `comfy-kitchen` pin bumped to `>=0.2.22` (honest correction of the v1.8.5 miss)**
  - v1.8.5 only raised the kitchen pin in **`ComfyUI-master/requirements.txt`**, but the Forge runtime installs from the **root `requirements.txt`**, which was left at **`comfy-kitchen>=0.2.21`**. The runtime kept resolving to `0.2.21`, so the intended fix never took effect.
  - Under `0.2.21`, `from comfy_kitchen import TensorCoreConvRotW4A4Layout` fails in `ComfyUI-master/comfy/quant_ops.py`; a single `try` block then sets `_CK_AVAILABLE = False`, disabling every kitchen layout and forcing the dequant fallback in `backend/patch_comfy_quant_layout_fallback.py`. Quantized KREA2 (int8/fp8 ConvRot) weights get dequantized to bf16, exhausting RAM (up to ~64 GB).
  - This release bumps the **root `requirements.txt`** to **`comfy-kitchen>=0.2.22`** so the runtime installs the correct version and the quantized layouts import successfully.
  - **User action:** run **`pip install -U comfy-kitchen`** once — a plain `pip install -r requirements.txt` will not upgrade, because `0.2.21` already satisfies `>=0.2.21`.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.8.6) for details.

## Version 1.8.5

- **ComfyUI-master sync and comfy-kitchen bump**
  - Synced vendored **`ComfyUI-master`** to the latest upstream tree used for Forge imports (JoyImage, Anima LLLite, API nodes, model detection, and related tests).
  - Raised the kitchen pin used with that tree to **`comfy-kitchen==0.2.22`** in `ComfyUI-master/requirements.txt`; Forge runtime continues to install from root **`requirements.txt`** (`comfy-kitchen>=0.2.21`).
  - Ensures remote installs can pick up kitchen layouts such as **`TensorCoreConvRotW4A4Layout`** / INT8 ConvRot support instead of failing on an older kitchen import.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.8.5) for details.

## Version 1.8.4

- **Krea2: quantized TE load fallback when `layout_cls` is None**
  - Forge-side runtime patch dequantizes MixedPrecision weights to float Parameters when `get_layout_class(...)` returns `None`, fixing `AttributeError: 'NoneType' object has no attribute 'Params'` on quantized Krea2 text-encoder load.
  - Issue: https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/issues/3
  - Dependency: `safetensors>=0.8.0` (compatible with `diffusers>=0.39.0`).
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.8.4) for details.

## Version 1.8.3

- **Anima: Q/K RMSNorm + split-half RoPE fuse**
  - On Anima SelfAttn with RoPE, fuse Q/K RMSNorm and split-half RoPE via **`comfy_kitchen.rms_rope_split_half`** (requires **`comfy-kitchen>=0.2.21`**); CrossAttn and no-rope paths stay stock.
  - Wired from the Anima loader path (`install_anima_rms_rope_fuse` on `comfy.ldm.cosmos.predict2.Attention.compute_qkv`).
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.8.3) for details.

## Version 1.8.2

- **Krea2 Model Support**
  - Native Forge support for **Krea2** (`SingleStreamDiT` UNet, Qwen3-VL-4B text encoder, Qwen-Image VAE); e.g. `moodyKreaMix_V33.safetensors`.
  - Dedicated **UI Preset: Krea2** in the Forge checkpoint panel; load **Additional modules** `qwen_image_vae.safetensors` (Qwen-Image VAE) and Qwen3-VL-4B TE (e.g. `qwen3vl_4b_bf16.safetensors`); LoRA key mapping via **`krea2_to_diffusers`**; MixedPrecision **`comfy_quant`** load path for quantized Krea2 checkpoints.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.8.2) for details.

## Version 1.8.1

- **Cleanup: Nunchaku SDXL code removal (phase 1)**
  - Removed dedicated Nunchaku SDXL files (**`nunchaku_sdxl_unet.py`**, **`nunchaku_sdxl_clip.py`**, **`nunchaku_sdxl_unet_append.py`**) and SDXL-specific branches from **`backend/loader.py`**, **`extensions-builtin/sd_forge_lora/networks.py`**, and the four RES4LYF beta samplers.
  - No impact on normal SDXL, fp16 / fp8 / int8, or Nunchaku for other models.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.8.1) for details.

## Version 1.8.0

- **INT8 (Low Bits) UNet quantization for SDXL and Z-Image**
  - New **`int8`** / **`int8 (fp16 LoRA)`** entries in the **`Diffusion in Low Bits`** dropdown enable tensor-wise INT8 (`int8_tensorwise`) UNet storage for reduced VRAM usage; selected explicitly, never auto-detected.
  - **`int8`** bakes LoRA into the INT8 weights, while **`int8 (fp16 LoRA)`** keeps LoRA in FP16 for online (bake-free) application.
  - INT8 loading is fully separated from the float8 / bnb / Automatic paths; every non-`int8` dropdown entry uses a different loading branch.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.8.0) for details.

## Version 1.7.9

- **Anima Hires Fix: tiled second pass for high-resolution upscale**
  - On the pixel-upscale Hires path, **Anima-only** detection (`anima` in checkpoint name) routes large upscaled latents through **tiled img2img** instead of a single full-image pass, mirroring ComfyUI USDU and keeping each tile within Anima’s training resolution (~1024-class).
  - **96×96 latent tiles** with **50% overlap** (`pad_latent=48`), **Gaussian blend** stitching, and a **seam-fix pass** (narrow boundary strips at half denoising strength).
  - **Anima-only** dynamic VAE decode normalization before pixel upscaling; **5D decoded latent** normalization and **5D truncate** cropping on the shared Hires path.
  - Hires **sigma schedule** aligned with ComfyUI for Anima img2img. Non-Anima models and **latent-upscale** Hires unchanged aside from v1.7.8 5D helpers.
  - Scoped to **`modules/processing.py`**; SD 1.5 / SDXL / Flux / Wan behavior on latent-upscale Hires is unchanged.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.9) for details.

## Version 1.7.8

- **Hires Fix: 5D tensor compatibility for video/DiT models (Wan / Flux)**
  - Fixed **`ValueError: Input and output must have the same number of spatial dimensions`** during latent-space upscale by locally squeezing/unsqueezing the extra temporal dimension (dim 2) of 5D latents `(N, C, 1, H, W)` around `torch.nn.functional.interpolate`.
  - Fixed **`TypeError: Cannot handle this data type: (1, 1, 1, 1024)`** during pixel-space upscale by squeezing the extra temporal dimension (dim 1) of 5D VAE-decoded outputs `(N, 1, C, H, W)` before processing with NumPy/PIL.
  - Squeezing logic is scoped strictly to 5D tensors; standard 4D latent models (SD 1.5, SDXL, anime models) bypass these blocks entirely and remain unaffected.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.8) for details.

## Version 1.7.7

- **Anima + ADetailer: img2img inpaint flag fix**
  - **`Anima`** engine sets **`is_inpaint = False`** after load so WebUI SD-style latent mask/image concat is not used on Qwen-Image VAE **5D** latents.
  - Fixes **`RuntimeError: Tensors must have same number of dimensions: got 4 and 5`** when ADetailer post-processes **`AnimaWai68`** checkpoints (e.g. **`waiANIMA_pw3.safetensors`**) after main txt2img; regional masks still apply via existing **5D** mask blend in img2img **`sample()`**.
  - Change is scoped to **`backend/diffusion_engine/anima.py`** only (same pattern as Qwen / Flux / Lumina); other models unchanged. Commit **`9c85472`**.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.7) for details.

## Version 1.7.6

- **IoPaint Launcher Fix:**
  - Changed default port for IOPaint to **`8081`** to avoid conflicts with ComfyUI's `AiHelper` (which uses `8080`).
  - Implemented a readiness check in `iopaint-cuda.bat` using PowerShell to wait for IOPaint's `/api/v1/server-config` endpoint to return HTTP 200 before launching the browser.
  - Corrected the working directory (`cd`) in `iopaint-cuda.bat` to ensure proper execution.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.6) for details.

## Version 1.7.5

- **ADetailer (built-in extension):** Face detection no longer uses InsightFace (YOLO only).

## Version 1.7.4

- **Pony / SDXL LoRA (CLIP-L): IntegratedCLIP key mapping**
  - **`model_lora_keys_clip()`** in **`modules_forge/packages/comfy/lora.py`** now tries both **`transformer.text_model.encoder.layers.*`** (Transformers 4.x–5.5) and **`transformer.encoder.layers.*`** (Transformers 5.6+ / IntegratedCLIP) so **`lora_te1_*`** keys bind on Pony / SDXL checkpoints.
  - Reduces **`[LORA] LoRA mismatch for CLIP`** when unmatched CLIP keys exceed the 50% gate in **`extensions-builtin/sd_forge_lora/networks.py`** (whole-file skip even if UNet keys match).
  - Commits `3a8ea0c` / `3d1950d` (clip_layer_paths) and Pony SDXL LoRA CLIP fix (`08e6e70`).
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.4) for details.

## Version 1.7.3

- **Anima: ComfyUI text encoder import (Step 5a)**
  - Anima TE load and encode now use Comfy **`load_text_encoder_state_dicts`** and **`CLIP.encode_from_tokens`** instead of Forge duplicate **`Qwen3_06B`** and hand-written **`_encode_qwen`**.
  - **`forge_objects.clip`** holds Comfy **`sd.CLIP`** (`AnimaTEModel` + `AnimaTokenizer`) directly; Forge **`CLIP`** wrapper and HF **`tokenizer`** / **`tokenizer_2`** wiring removed from the Anima diffusion engine.
  - Removed Forge **`Qwen3_06BConfig`** / **`Qwen3_06B`** from **`llama.py`** (Anima only; **`Qwen3_4B`** unchanged for other models).
  - **`split_state_dict`**: extract TE keys **before** **`process_clip_state_dict`**, using **`anima_te_filter_prefixes`** for HF **`text_encoders.*`**, Comfy **`cond_stage_model.*`**, and bare **`qwen3_06b.*`** checkpoint layouts.
  - **`AnimaBase`**: expanded **`clip_target`** (six layout patterns) and **`process_clip_state_dict`**; large Anima **`class Anima(AnimaBase)`** inherits TE extraction fixes (previously **`Anima(BASE)`**).
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.3) for details.

## Version 1.7.2

- **Nunchaku Z-Image Turbo: Lumina detection regression fix**
  - Anima v1.7.1 Lumina entry required `noise_refiner.k_norm`, but Nunchaku Z-Image Turbo exports use `norm_k` (remapped later in `svdq.py`), so checkpoints were mis-detected as Nunchaku SDXL and failed with `You do not have CLIP state dict!`.
  - Lumina entry now accepts **`k_norm` or `norm_k`** in `modules_forge/packages/huggingface_guess/detection.py` only; Anima guard, Lumina block body, and fuzzy SDXL detection unchanged.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.2) for details.

## Version 1.7.1

- **Anima: ComfyUI import refactor**
  - Replaced the native Forge DiT (`backend/nn/anima.py`) with **`comfy.ldm.anima.model.Anima`** from bundled ComfyUI-master so checkpoint layout, `llm_adapter` placement, and UNet config detection match upstream Comfy.
  - Text path: **`comfy.text_encoders.anima.AnimaTokenizer`** + Forge **`Qwen3_06B`** (embeddings only); **`preprocess_text_embeds`** runs once on the Comfy UNet in `get_learned_conditioning` (no TE-side `llm_adapter`).
  - Loader/detection: **`remap_anima_state_dict`** for key-name fixes only; **`comfy.model_detection.detect_unet_config`** delegate; **`k_model`** 4D↔5D wrap for still-image latents; **`compile_conditions`** omits pooled `y` when absent.
  - UI preset and Additional modules (`qwen_3_06b_base`, `qwen_image_vae`) unchanged from v1.7.0.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.1) for details.

## Version 1.7.0

- **Anima model support**
  - Native Forge support for [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) and compatible single-file checkpoints (e.g. `anima-base-v1.0.safetensors`, community merges such as `waiANIMA_pw3.safetensors`).
  - New **UI Preset: Anima** in the checkpoint manager; load **Additional modules** `qwen_3_06b_base.safetensors` (Qwen3 text encoder) and `qwen_image_vae.safetensors` (Qwen-Image VAE). T5 tokenizer vocabulary is used for `llm_adapter` cross-attention without a separate T5/UMT5 weight file.
  - Native `backend.nn.anima` UNet with flow-matching sampling; **Shift** in the UI applies to the noise schedule.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.0) for details.

## Version 1.6.1

- **UI Preset control: radio buttons to dropdown**
  - Replaced the horizontal SD1.5 / SDXL / Flux / Qwen / Lumina radio preset selector with a **UI Preset** dropdown in the checkpoint manager row (same row as Checkpoint, VAE, and Diffusion in Low Bits).
  - Dropdown labels: SD1.5 (`sd`), SDXL (`xl`), Flux (`flux`), Qwen (`qwen`), Lumina (`lumina`). Existing preset-change behavior (resolution, CFG, sampler, VAE modules, Clip Skip, checkpoint paths) is unchanged.
  - **LoRA tab:** Updated Extra Networks filtering in `javascript/extraNetworks.js` to read the dropdown value instead of legacy radio `checked` state (with a fallback if the old radio UI is still present).

## Version 1.6.0

- **ComfyUI-master update and comfy_aimdo compatibility fixes**
  - Updated the stub packages of `comfy_aimdo` by creating `vram_buffer.py` (`VRAMBuffer` class) and extending `control.py` (`init_devices` method) to ensure compatibility with updated ComfyUI-master startup sequences without the proprietary AIMDO library.
  - Resolved `ModuleNotFoundError: No module named 'comfy_aimdo.vram_buffer'` during startup.
  - Release notes detail error traceback, root cause, and explanation of modifications.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.6.0) for details.

## Version 1.5.1


- **NumPy 2.4.4 startup compatibility fix**
  - Fixed startup crashes caused by SciPy/diffusers import chains when running with `numpy==2.4.4` by adding `_blas_supports_fpe` compatibility stubs in `launch.py` and `modules_forge/transformers_cache_compat.py`.
  - Updated runtime dependency pins in `requirements.txt` (`Pillow`, `accelerate`, `numpy`, `diffusers`, `protobuf`) to align with the current tested environment.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.5.1) for details.

## Version 1.5.0

- **Built-in Aspect Ratio extension with resolution calculator**
  - Added a new built-in extension at `extensions-builtin/forge_aspect_ratio` for one-click aspect ratio presets in txt2img/img2img.
  - Added a calculator panel (`Calc`) to compute target resolution from megapixels, aspect ratio, and divisibility (8/16/32/64), then apply directly to width/height sliders.
  - Added configurable presets via `aspect_ratios.txt` and `resolutions.txt`.
  - Integrated MIT-attributed calculation logic adaptation from `ControlAltAI-Nodes` into the Forge extension structure.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.5.0) for details.

## Version 1.4.9

- **transformers 5.6.2 CLIP compatibility fixes**
  - Fixed transformers 5.6.2 CLIP loading path for SDXL/Nunchaku by normalizing checkpoint key prefixes in `backend/loader.py` (`transformer.` / `text_model.` cleanup then re-map).
  - Updated CLIP text processing access paths for transformers 5.x flattening in `backend/text_processing/classic_engine.py` (`text_model.*` -> direct model attributes).
  - Improved no-init compatibility shim for transformers 5.x in `modules_forge/transformers_cache_compat.py` by patching `PreTrainedModel.init_weights` inside context and restoring it safely.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.4.9) for details.

## Version 1.4.8

- **ComfyUI-master 0.19.3 sync and AIMDO import compatibility**
  - Updated the bundled `ComfyUI-master` tree to match ComfyUI 0.19.3; upstream now imports optional AIMDO modules (`comfy_aimdo.*`) during startup.
  - Extended the repository `comfy_aimdo` stub package (root and `ComfyUI-master/comfy_aimdo/`) with `host_buffer`, `model_mmap`, and related APIs so Forge starts without the real AIMDO wheel; Forge continues to use its own loader and memory paths.
  - Release notes (English) cover root cause, full traceback, import-flow sequence diagrams, and file-level notes.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.4.8) for details.

## Version 1.4.7

- **Fresh install and setup bug fixes for Python 3.13, ADetailer initialization, and triton-windows environment setup.** See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.7) for details.

## Version 1.4.6

- **PyTorch 2.11.0+cu130 defaults, Windows FA2/SA2 wheel URLs, and fixes for `flash_attn` / diffusers startup after torch upgrades.** See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.6) for details.

## Version 1.4.5

- **Qwen-Image-2512-Fun-Controlnet-Union support**
  - Native support for Qwen Image Fun ControlNet (e.g. Qwen-Image-2512-Fun-Controlnet, Union variants). Uses ComfyUI's `QwenImageFunControlNetModel`; injects control via per-step wrapper into `c["control"]` for Nunchaku and standard Qwen Image backends.
  - Control image is encoded with Forge VAE and passed as raw latent; strength is applied before injection. See Release Notes for implementation details.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.5) for details.

## Version 1.4.4

- **ComfyUI-Master update compatibility**
  - Aligned Forge with latest ComfyUI core: removed dependency on deleted `comfy.checkpoint_pickle`; checkpoint loading now uses `torch.load(..., weights_only=True)` only (same as upstream).
  - Added stub package `comfy_aimdo` so ComfyUI code that imports the optional PyPI package `comfy_aimdo` (AIMDO) runs without it; Forge keeps using its own memory/loader.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.4) for details.

## Version 1.4.3

- **Float8 + fp16 LoRA: fix crash when switching model across categories**
  - Fixed `"addmm_cuda" not implemented for 'Float8_e4m3fn'` when using "Diffusion in Low Bits" (float8-e4m3fn + fp16 LoRA) and switching to a different model category (e.g. Z-Image ↔ Flux1). LoRA B@A is now computed in fp16 and cast back to Float8 after adding to weights.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.3) for details.

## Version 1.4.2

- **wd14-tagger as built-in extension**
  - WD14 Tagger integrated into `extensions-builtin` for image captioning/tagging without a separate install.
- **ComfyUI-Master core refresh**
  - Bundled ComfyUI updated to track upstream; compatible with current ComfyUI-Master for node and model loading.

## Version 1.4.1

- **transformers 5+ compatibility**
  - Shims for `HybridCache` and `no_init_weights` (removed in transformers 5.x) so that the peft / diffusers / nunchaku import chain does not fail at startup.
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.1) for details.

## Version 1.4.0

- **Improved PEFT format LoRA detection (Nunchaku Qwen Image)**
  - LoRA files created with Hugging Face PEFT (e.g. `.lora_A.default.weight`) are no longer incorrectly skipped as "unsupported"
  - Format detection uses partial match so both standard and PEFT key patterns are recognized as Standard LoRA
  - Log which LoRA files had weights skipped when applicable (e.g. modulation layers apply only to Nunchaku Qwen Image)
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.0) for details

## Version 1.3.9

- **Config-Presets as built-in extension**
  - Integrated [Zyin055/Config-Presets](https://github.com/Zyin055/Config-Presets) (MIT) as a built-in extension
  - Configurable dropdown to switch txt2img/img2img settings in bulk; create/delete presets; custom fields via `*custom-tracked-components.txt`
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.3.9) for details

## Version 1.3.8

- **IoPaint integration**
  - Auto-install iopaint and imghdr (Python 3.13) in `prepare_environment()`; requirements (python-socketio, typer-config, loguru, rembg); `iopaint-cuda.bat` launcher
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.3.8) for details

## Version 1.3.7

- **Model detection / Qwen3**
  - When `detect_unet_config` returns None, strip `unet_key_prefix` before diffusers fallback; reduces SDXL–Flux misdetection and downstream VAE shape mismatch (avoids incorrect Qwen3 load path for SDXL)

## Version 1.3.6

- **ADetailer (built-in extension)**
  - Prevent duplicate face passes per image: only the first face detector tab runs per image; subsequent face tabs are skipped so the same face is not processed twice (hand and other detector tabs unchanged; multiple faces in one image still get one pass each)

## Version 1.3.5

- **Enhanced Nunchaku LoRA implementation and documentation**
  - Improved LoRA support for Nunchaku Qwen Image with AWQ quantization layer handling
  - Enhanced Manual Planar Injection for AWQ modulation layers (img_mod/txt_mod) in Qwen Image models
  - Strict model type detection to ensure AWQ modifications only apply to Nunchaku Qwen Image models
  - Complete separation of LoRA paths for Qwen Image, Flux1, SDXL, and Z-Image models
  - Fixed LoRA application flow with proper model detection and routing
  - Comprehensive documentation of the modification architecture and processing flow
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.3.5) for complete technical details

## Version 1.3.1

- **Added Diffsynth Union ControlNet support for ZIT (standard and Nunchaku)**
  - Full Diffsynth Union ControlNet support for both standard Z-Image Turbo (ZIT) and Nunchaku ZIT models
  - Multiple ControlNet models can be used simultaneously (Union ControlNet)
  - Supports ZIT ControlNet models (e.g., `z-image-turbo-controlnet.safetensors`)
  - Automatic model detection for ZIT models via NextDiT model type
  - Strict model type checking to ensure compatibility only with ZIT models
  - VAE wrapper for seamless Forge VAE integration with ComfyUI ControlNet interface
  - Complete implementation based on ComfyUI's nodes_model_patch.py
  - Fixed double patching and stale patches issues causing RecursionError
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.3.1) for details

## Version 1.3.0

- **Added RES4LYF Sampler Support**
  - Full support for RES4LYF (RES4) samplers for all model types
  - Support for Nunchaku Qwen Image, Nunchaku Flux1, Nunchaku SDXL, standard SDXL, and standard Flux1 models
  - Comprehensive sampler collection including multistep (res_2m, res_3m, etc.) and exponential (res_2s, res_3s, etc.) variants
  - ODE version support for non-implicit samplers
  - Robust model detection and handling for both Forge and ComfyUI model structures
  - Automatic CONST and EPS model type detection for proper sampling behavior
  - Fixed model_sampling access for Forge models via KModel wrapper
  - Improved compatibility with ComfyUI-master directory structure
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.3.0) for details

## Version 1.2.1

- **Added Union ControlNet support for Nunchaku Qwen Image**
  - Full Union ControlNet support for Nunchaku Qwen Image (QI) models
  - Multiple ControlNet models can be used simultaneously (Union ControlNet)
  - Supports Qwen Image Union ControlNet models (e.g., `Qwen-Image-InstantX-ControlNet-Union.safetensors`)
  - Automatic model detection via `transformer_blocks.0.img_mlp.net.0.proj.weight` key
  - Strict model type checking to ensure compatibility only with Nunchaku Qwen Image models
  - VAE wrapper for seamless Forge VAE integration with ComfyUI ControlNet interface
  - Complete and independent implementation separate from Flux ControlNet
  - Fixed device placement issues for ControlNet model loading
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.2.1) for details

## Version 1.2.0

- **Added LoRA support for Nunchaku Qwen Image and Z-Image models**
  - Full LoRA support for Nunchaku Qwen Image (QI) models
  - Full LoRA support for Nunchaku Z-Image (ZIT) models
  - Completely separated implementations for Qwen Image and Z-Image
  - Comprehensive logging with format detection for all LoRAs
  - Robust change detection to handle model reloads correctly
  - Support for standard LoRA formats (lora_A/lora_B, lora_up/lora_down)
  - AWQ quantization layer handling with safety switch
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.2.0) for details

## Version 1.1.0

- **Added Union ControlNet support for Flux1 and Nunchaku Flux1**
  - Full Union ControlNet support for both Flux1 and Nunchaku Flux1 models
  - Multiple ControlNet models can be used simultaneously
  - Automatic model detection via `controlnet_x_embedder.weight` key
  - VAE wrapper for seamless Forge VAE integration
  - See [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.1.0) for details

## Version 1.0.7

- **Added ADetailer as built-in extension**
  - Integrated [ADetailer](https://github.com/Bing-su/adetailer) as a standard built-in feature
  - Python 3.13 compatible with InsightFace instead of MediaPipe
  - Includes YOLOv8, YOLOv11, and InsightFace hybrid detection system
  - Fixed ControlNet preprocessor initialization issue
  - Added `extensions-builtin/adetailer/models/` to `.gitignore`

## Version 1.0.6

- Nunchaku SDXL loader, LoRA loader, and ControlNet support completed
