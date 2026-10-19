<h1 align="center">Stable Diffusion WebUI Forge HSWQ-Nunchaku</h1>

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="zhmd/README.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

⚠️ **This project requires Python 3.14 exclusively.**

<p align="center">
  <img src="png/top.png" alt="Stable Diffusion WebUI Forge - Nunchaku" width="400">
</p>

## About

**Stable Diffusion WebUI Forge — Nunchaku** integrates Nunchaku into Forge, including LoRA and ControlNet support for Nunchaku models.

## Features

### 🎯 Key Features

- **Common Attention Backend Switch (Quicksettings)**
  - ✅ **Common runtime Attention backend** next to `Diffusion in Low Bits` in the Forge quicksettings bar
  - Dropdown choices: `Default` / `SA2` / `SA3` / `FA2`
    - `Default` — pytorch SDPA
    - `SA2` — SageAttention 2
    - `SA3` — SageAttention3 (Blackwell FP4), e.g. `3.0.0.b1` when installed
    - `FA2` — Flash-Attention 2 called directly
  - Common Attention path (runtime switch; no UNet reload)
  - When using **Nunchaku**, keep Attention on **`Default`**
  - Logs the resolved backend with the full installed version string (not a short token only)

  <p align="left">
    <img src="png/attention.png" alt="Attention quicksettings dropdown: Default / SA2 / SA3 / FA2" width="400">
  </p>

  *`Attention` dropdown in quicksettings (next to Diffusion in Low Bits) — common UI*

- **LoRA Support for Nunchaku Qwen Image and Z-Image**
  - ✅ **Full LoRA support for Nunchaku Qwen Image (QI) models**
  - ✅ **Full LoRA support for Nunchaku Z-Image (ZIT) models**
  - Completely separated implementations for Qwen Image and Z-Image
  - Support for standard LoRA formats (lora_A/lora_B, lora_up/lora_down)
  - Comprehensive logging with format detection for all LoRAs
  - Robust change detection to handle model reloads correctly
  - AWQ quantization layer handling with safety switch

- **ConvRot INT8 (Low Bits) UNet / DiT Quantization**
  - ✅ **ConvRot INT8 UNet / DiT support for SDXL, Z-Image, and Krea2**
  - Selected explicitly from the **`Diffusion in Low Bits` dropdown** in the Forge settings panel, never auto-detected. The full menu is:
    - `Automatic` — no forced storage dtype (default; **not** ConvRot INT8)
    - `Automatic (fp16 LoRA)` — same as `Automatic`, LoRA kept in FP16
    - `float8-e4m3fn` / `float8-e4m3fn (fp16 LoRA)` — FP8 storage paths
    - **`int8`** — ConvRot INT8 UNet / DiT, LoRA baked into the INT8 weights
    - **`int8 (fp16 LoRA)`** — ConvRot INT8 UNet / DiT, LoRA kept in FP16 for online (bake-free) application
    - `bnb-nf4` / `bnb-fp4` (and their `(fp16 LoRA)` variants) — only shown when bitsandbytes is available
  - Only the two **`int8`** / **`int8 (fp16 LoRA)`** entries enable ConvRot INT8; every other entry uses a different loading path
  - Tensor-wise ConvRot INT8 storage (`int8_tensorwise`) with Hadamard rotation handling for reduced VRAM usage
  - `int8 (fp16 LoRA)` keeps LoRA in FP16 for online (bake-free) application
  - Fully separated from the float8 / bnb / Automatic loading paths

  <p align="left">
    <img src="png/int8.png" alt="Diffusion in Low Bits dropdown with int8 options" width="400">
  </p>

  *`Diffusion in Low Bits` dropdown: `int8` / `int8 (fp16 LoRA)` (ConvRot INT8) selection*

- **ConvRot INT8 Text Encoder (CLIP) & ControlNet Support**
  - ✅ **Full ConvRot INT8 CLIP / Text Encoder support across major architectures**
    - **SDXL**: In-place dequantization with Hadamard unrotation for CLIP-L and CLIP-G (OpenCLIP bigG) prior to key remapping, eliminating NaN and black image artifacts
    - **Flux1 / Nunchaku Flux1**: Native support for `clip_l_hswq_convrot_int8` and `flan_t5_xxl_hswq_convrot_int8`
    - **Z-Image (ZIT)**: Full ConvRot INT8 text encoder loading
    - **Qwen Image**: Seamless support for ConvRot INT8 text encoders
    - **Krea2 & Anima**: Direct INT8 `QuantizedTensor` native loading via `comfy_quant` / `MixedPrecisionOps` for Qwen3-VL-4B, Qwen3-4B, and Gemma2-2B (loads in ~0.4s @ 4.24 GB without uncompressed FP16 memory bloat)
  - ✅ **ConvRot INT8 ControlNet Support**
    - In-place dequantization supporting both `bfloat16` and `float16` computation dtypes
    - Hardware-accelerated Hadamard rotation matrix cache per `(group_size, device)` to maximize throughput
    - Safe model construction under non-inference mode to prevent autograd version counter conflicts
  - Zero manual conversion needed — simply place ConvRot INT8 CLIP / ControlNet weights into the corresponding model directories or select them under **Additional modules**

- **Krea2 Model Support**
  - ✅ **Native Forge support for Krea2** (SingleStreamDiT; e.g. `moodyKreaMix_V33.safetensors`)
  - Dedicated **UI Preset: Krea2** in the Forge checkpoint panel
  - ✅ **Full ConvRot INT8 DiT (`int8` / `int8 (fp16 LoRA)`) support with dynamic layer swapping** for smooth generation on 16GB GPUs
  - Load **Additional modules**: `qwen_image_vae.safetensors` (Qwen-Image VAE), Qwen3-VL-4B text encoder (e.g. `qwen3vl_4b_bf16.safetensors`)
  - LoRA key mapping via `krea2_to_diffusers`

  <p align="left">
    <img src="png/krea2.png" alt="Krea2 UI preset and txt2img example" width="400">
  </p>

  *Krea2 preset workflow example*

- **Anima Model Support**
  - ✅ **Native Forge support for Anima** (e.g. `waiANIMA_pw3.safetensors`)
  - Dedicated **UI Preset: Anima** in the Forge checkpoint panel
  - Load **Additional modules**: `qwen_3_06b_base.safetensors` (Qwen3 text encoder), `qwen_image_vae.safetensors` (Qwen-Image VAE)
  - T5 tokenizer vocabulary for `llm_adapter` cross-attention (no separate T5/UMT5 weight file required)
  - Do **not** use Attention **`SA2` / `SA3`** with Anima — they break the image. Use **`Default`** or **`FA2`** instead.

  <p align="left">
    <img src="png/anima.png" alt="Anima UI preset and txt2img example" width="400">
  </p>

  *Anima preset workflow example*

- **Union ControlNet for Flux1, Nunchaku Flux1, and Nunchaku Qwen Image**
  - ✅ **Union ControlNet support for Flux1 and Nunchaku Flux1 models**
  - ✅ **Union ControlNet support for Nunchaku Qwen Image (QI) models**
  - Multiple ControlNet models can be used simultaneously (Union ControlNet)
  - Supports Flux Union ControlNet models:
    - `flux_shakker_labs_union_pro-2-fp8.safetensors`
  - Supports Qwen Image Union ControlNet models:
    - `Qwen-Image-InstantX-ControlNet-Union.safetensors` (InstantX)
    - `Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors` (Fun)
  - Automatic model detection and loading:
    - Flux models: via `controlnet_x_embedder.weight` key detection
    - Qwen Image models:
      - InstantX: via `transformer_blocks.0.img_mlp.net.0.proj.weight` key detection
      - Fun: via `control_img_in.weight` key detection
  - VAE wrapper for seamless Forge VAE integration with ComfyUI ControlNet interface
  - Strict model type checking to ensure compatibility with correct model types
  - Complete and independent implementation for each model type

  <img src="png/f1cn.png" alt="Flux1 ControlNet Union" width="400">

  *Flux1 Union ControlNet workflow example*

- **Diffsynth Union ControlNet for Z-Image Turbo (ZIT) - Standard and Nunchaku**
  - ✅ **Diffsynth Union ControlNet support for standard Z-Image Turbo (ZIT) and Nunchaku Z-Image Turbo (ZIT) models**
  - Multiple ControlNet models can be used simultaneously (Union ControlNet)
  - Supports ZIT ControlNet models (e.g., `z-image-turbo-controlnet.safetensors`)
  - **Note:** ZIT Diffsynth ControlNet uses a different mechanism than standard ControlNet
  - ZIT ControlNet works as a "model patch" rather than a traditional ControlNet
  - Automatic model detection for ZIT models via NextDiT model type
  - Strict model type checking to ensure compatibility only with ZIT models
  - VAE wrapper for seamless Forge VAE integration with ComfyUI ControlNet interface
  - Complete implementation based on ComfyUI's nodes_model_patch.py
  - Fixed double patching and stale patches issues causing RecursionError

- **Built-in ADetailer**
  - Python 3.14 compatible face detection and enhancement
  - YOLOv8, YOLOv11, and InsightFace hybrid detection system
  - Enhanced face detection accuracy with complementary detection
  - Automatic model download and management
  - SDXL/Pony optimized detection thresholds

- **Python 3.14 Exclusive**
  - Latest Python features and performance improvements
  - Optimized for modern hardware and workflows
  - Future-proof architecture

- **RES4LYF Sampler Support**
  - ✅ **Full support for RES4LYF (RES4) samplers**
  - Support for all model types including Nunchaku Qwen Image, Nunchaku Flux1, Nunchaku SDXL, standard SDXL, and standard Flux1
  - Comprehensive sampler collection with multistep and exponential variants
  - ODE version support for non-implicit samplers
  - Robust model detection and handling for both Forge and ComfyUI model structures
  - Automatic CONST and EPS model type detection for proper sampling behavior

## Known Limitations

### LoRA Format Support for Nunchaku Models

**⚠️ Important:** The following limitations apply **only to Nunchaku quantization models** (Nunchaku Qwen Image, Nunchaku Z-Image, Nunchaku SDXL). Standard (non-quantized) models may support additional LoRA formats.

### LoKR (Lycoris) LoRA Support

**Status:** ❌ Not Supported for Nunchaku Models

**Issue:** LoRAs in LoKR format (created by Lycoris) are not supported for Nunchaku quantization models.

**Note:** LoKR format LoRAs may work with standard (non-quantized) Qwen Image, Z-Image, or SDXL models, but this implementation is designed specifically for Nunchaku quantization models.

⚠️ **Development History:** We have spent considerable time analyzing the internal structure of LoKR format and performing extensive mapping tests. Despite these efforts, we have not been able to find a successful way to apply LoKR weights to Nunchaku quantization models. The experimental conversion code has been tested but ultimately disabled due to incompatibility issues.

- LoKR weights are automatically skipped when detected for Nunchaku models (experimental conversion code is disabled).
- Converting to Standard LoRA using SVD approximation (via external tools or scripts) has also been tested and found to result in noise/artifacts when applied to Nunchaku quantization models.

**Conclusion:** At this time, we have not found a way to successfully apply LoKR weights to Nunchaku models. Please use Standard LoRA formats for Nunchaku models.

### Supported LoRA Formats for Nunchaku Models

**✅ Standard LoRA (Rank-Decomposed):**

Supported weight keys:

- `lora_up.weight` / `lora_down.weight`
- `lora.up.weight` / `lora.down.weight`
- `lora_A.weight` / `lora_B.weight`
- `lora.A.weight` / `lora.B.weight`

These are the standard formats produced by Kohya-ss, Diffusers, and most training scripts.

**❌ LoKR (Lycoris):** Not supported for Nunchaku models (Keys like `lokr_w1`, `lokr_w2`)

**❌ LoHa:** Not supported for Nunchaku models (Keys like `hada_w1`, `hada_w2`)

**❌ IA3:** Not supported for Nunchaku models

### Non-Quantized Qwen Image (QI) Models

**Status:** ❌ LoRA and ControlNet Not Implemented

**Issue:** Non-quantized Qwen Image (QI) models do not have LoRA or ControlNet support implemented. The performance is too heavy for development and testing in our environment.

**Note:** This implementation is designed specifically for Nunchaku quantization models. For non-quantized Qwen Image models, please use alternative implementations or consider using Nunchaku quantized models instead.

## Changelog

For full changelog, see [CHANGELOG.md](Changelog/CHANGELOG.md).

## Base Repositories

This project is built upon the following repositories:

- **[stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)**  
  Original Forge implementation by [@lllyasviel](https://github.com/lllyasviel)

- **[sd-webui-forge-classic (neo branch)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**  
  Classic Forge implementation by [@Haoming02](https://github.com/Haoming02)

- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)**  
  The most powerful and modular diffusion model GUI, API and backend with a graph/nodes interface by [@Comfy-Org](https://github.com/Comfy-Org)

- **[ComfyUI-nunchaku-unofficial-loader](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader)**  
  Nunchaku integration reference implementation by [@ussoewwin](https://github.com/ussoewwin)

- **[ComfyUI-QwenImageLoraLoader](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader)**  
  LoRA Loader & Diffsynth ControlNet Loader for Nunchaku Qwen Image & Nunchaku Z-Image Turbo by [@ussoewwin](https://github.com/ussoewwin)

- **[ADetailer](https://github.com/Bing-su/adetailer)**  
  Auto detecting, masking and inpainting with detection model by [@Bing-su](https://github.com/Bing-su)

- **[Config-Presets](https://github.com/Zyin055/Config-Presets)**  
  Extension for Automatic1111 - configurable dropdown to change settings in txt2img and img2img tabs by [@Zyin055](https://github.com/Zyin055)

- **[stable-diffusion-webui-wd14-tagger](https://github.com/picobyte/stable-diffusion-webui-wd14-tagger)**  
  Labeling extension for Web UI (DeepDanbooru etc.) by [@picobyte](https://github.com/picobyte)

- **[ControlAltAI-Nodes](https://github.com/gseth/ControlAltAI-Nodes)**  
  Source reference for the built-in aspect ratio calculator logic adaptation by [@gseth](https://github.com/gseth)

- **[RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF)**  
  RES/Bongmath samplers and schedulers by [@ClownsharkBatwing](https://github.com/ClownsharkBatwing)

## License

This project is licensed under **AGPL-3.0** (GNU Affero General Public License v3.0).

### Licenses of Base Repositories

This project is built upon the following repositories, each with their respective licenses:

- **[stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)** - AGPL-3.0
- **[sd-webui-forge-classic (neo branch)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** - AGPL-3.0
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** - GPL-3.0
- **[ComfyUI-nunchaku-unofficial-loader](https://github.com/ussoewwin/ComfyUI-nunchaku-unofficial-loader)** - Apache 2.0
- **[ComfyUI-QwenImageLoraLoader](https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader)** - Apache 2.0
- **[ADetailer](https://github.com/Bing-su/adetailer)** - AGPL-3.0
- **[Config-Presets](https://github.com/Zyin055/Config-Presets)** - MIT
- **[stable-diffusion-webui-wd14-tagger](https://github.com/picobyte/stable-diffusion-webui-wd14-tagger)** - Public domain, except borrowed parts (e.g. dbimutils.py)
- **[ControlAltAI-Nodes](https://github.com/gseth/ControlAltAI-Nodes)** - MIT
- **[RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF?tab=License-1-ov-file)** - AGPL-3.0 - Commercial AI image generation service use requires separate permission and/or a commercial license from the copyright holder.

See [LICENSE](LICENSE) file for details.
