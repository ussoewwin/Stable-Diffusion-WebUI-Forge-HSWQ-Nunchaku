# 更新日志

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../Changelog/CHANGELOG.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

## Version 2.0.3

- **修复：SDXL ConvRot INT8 CLIP-G 逆量化与黑图/NaN 缺陷**
  - 彻底解决了在 `int8` / `int8 (fp16 LoRA)` 模式下载入带有 ConvRot INT8 文本编码器的 SDXL 模型时，生成结果输出全黑图像（黑图或 NaN 崩溃）的问题。
  - **技术根本原因**：在 `backend/loader.py` 中，`guess.process_clip_state_dict()` 在执行 `comfy_quant` 逆量化之前就重命名了 CLIP-G（`text_encoder_2` / OpenCLIP bigG）的层键名（例如将 `resblocks.*.weight` 转换为 `text_model.encoder.layers.*.weight`）。这导致 CLIP-G 的 `.weight_scale` 与 `.comfy_quant` 键无法匹配（成为 Unexpected keys），使得 63 个经过 Hadamard 旋转的 Linear 层未经逆量化与逆旋转便以原始 INT8 权重视为浮点数送入文本编码，导致文本特征输出全为极端离群值/NaN。
  - **`backend/loader.py` 修复方案**：在分离 `unet` 和 `vae` 的 state_dict 后、且在 `guess.process_clip_state_dict()` 重命名键名之前，立即就地执行逆量化与 Hadamard 逆旋转（`dequantize_state_dict_inplace`），确保 CLIP-L（70层）与 CLIP-G（63层）均在标准 Float16 下完成完整逆旋转后再被载入。
- **架构优化：Hadamard 矩阵显存缓存与无侵入式运行时 Hook**
  - 将 Hadamard 矩阵的设备缓存与 GPU 原生逆旋转优化移入 Forge 自身的 `modules_forge/hswq_int8.py`，通过运行时 Hook（`_apply_forge_hadamard_cache_patches`）动态挂载。
  - 保持 `modules/ComfyUI-HSWQ-Loader-and-Tools` 零改动，避免后续上游更新时发生覆盖或代码冲突。
- 详情请参阅 [Release Notes（中文）](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/zhmd/v2.0.3.md)。

## Version 2.0.2

- **修复：ComfyUI v0.40.0 AIMDO 导入异常与兼容性 Stub 实现**
  - 彻底解决了在未安装专有 `comfy_aimdo` C++ 扩展的环境下，ComfyUI 核心模块（`comfy.storage`、`comfy.ops`、`comfy.model_prefetch`）在启动时无条件抛出 `ModuleNotFoundError` 的严重缺陷。
  - 构建了无缝兼容的结构性 Stub 模块：
    - **`comfy_aimdo/storage.py`**：实现 `fast_disk(path) -> False`，明确告知存储子系统回退到标准文件系统 I/O（`safetensors.torch.load_file`），绕过专有硬件直通通道。
    - **`comfy_aimdo/malloc_graph.py`**：提供 `MallocGraph` 类与 `use_stream(stream)` 上下文管理器，使 CUDA Graph 内存调度透明运行在 PyTorch 原生 CUDA Stream 堆栈上。
- **修复：RES4LYF 采样器套件恢复与全局导入架构优化**
  - 修复 `modules/RES4LYF/__init__.py` 中因缺失 `from . import res4lyf` 导致调用 `res4lyf.init()` 时崩溃的 `NameError: name 'res4lyf' is not defined`。
  - 在 `modules/RES4LYF/res4lyf.py` 中内置 `PromptServer` 独立回退 Mock，消除在非 ComfyUI HTTP 服务环境下直接导入时抛出的 `ModuleNotFoundError: No module named 'server'`。
  - 在 `modules_forge/transformers_cache_compat.py` 中通过 `_hooked_import` 拦截顶层 `res4lyf` / `RES4LYF` 导入，自动映射至内置的 `modules.RES4LYF`。
  - 在 `forge_res4lyf_samplers.py` 中双向同步 `sys.modules`，确保所有高级采样器（`rk_beta`、`res_multistep` 等）无损注册至 Forge 原生采样调度器列表中。
- 详情请参阅 [Release Notes（中文）](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/blob/main/zhmd/v2.0.2.md)。

## Version 2.0.1

- **废除并完全移除：SDXL ConvRot NVFP4 支持**
  - 同步上游 [Hybrid-Sensitivity-Weighted-Quantization (HSWQ)](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization) 的最新调整，全面移除 **SDXL ConvRot NVFP4** 的所有支持与相关文档说明。
  - **技术原因**：确定性潜在空间轨迹发散测量（deterministic latent-trajectory divergence measurements）表明，4-bit 精度（NVFP4）从根本上无法在 SDXL 上维持潜在轨迹保真度（最终潜在轨迹 cosine 无法达到 0.90，出现严重的轨迹漂移与发散崩溃）。早期的解码后 SSIM（~0.95）是因 VAE 解码器的平滑滤波效应掩盖了误差假象，并非真正的潜在空间高保真保留。
  - **SDXL 精度支持规范**：SDXL 的量化支持严格收敛并专注于 **ConvRot INT8（reverse hybrid）**（`int8` / `int8 (fp16 LoRA)`）。
  - 同步清理了仓库内的相关文档、README 及加载器规范。
  - 详情请参阅 [Release Notes（中文）](v2.0.1.md)。

## Version 2.0.0

- **全面升级为仅支持 Python 3.14 与多平台兼容性支持**
  - 将 Windows 与 Linux 运行环境全面升级为 **Python 3.14 专属支持**（基于 CUDA 13.2 / PyTorch 2.13 架构）。
  - 集成 Python 3.14 运行时所需的 Universal Wheel 自动安装与依赖适配：
    - **`distutils`**（`distutils-3.14.0-py3-none-any.whl`）
    - **`pyaudioop`**（`pyaudioop-1.0.0-py3-none-any.whl`）
    - **`imghdr`**（`imghdr-1.0.0-py3-none-any.whl`）
    - **`facexlib`**（`facexlib-0.3.0-py3-none-any.whl`，剥离 numba/filterpy 强绑定并内置 `version.py` 的 Pure Python 定制版）
    - **`gradio 4.40.0`**（`gradio-4.40.0-py3-none-any.whl`，解除 numpy 等版本约束并适配 OpenAPI 布尔 schema）
  - 提供 Python 3.14 / CUDA 13.2 专用的 **Nunchaku 1.3.0** 预编译 Wheel：
    - Windows：`nunchaku-1.3.0.dev20260619+cu13.2torch2.12-cp314-cp314-win_amd64.whl`
    - Linux：`nunchaku-1.3.0.dev20260829+cu13.2torch2.13-cp314-cp314-linux_x86_64.whl`
  - 完整集成 **SageAttention 3（SA3 / Blackwell FP4）**、**SageAttention 2（SA2）** 与 **FlashAttention 2（FA2）** Attention 后端。
  - 新增 `torchaudio` 运行时兼容层（基于 ModuleSpec Stub），消除 Python 3.14 导入异常。
  - 修复 `StableDiffusionProcessing.sd_model` 属性同步机制与 `modules/shared.py` 变量遮蔽问题。
  - 详情请参阅 [Release Notes（中文）](v2.0.0.md)。

## Version 1.8.6

- **修复：根目录 `comfy-kitchen` 固定升级至 `>=0.2.22`（如实修正 v1.8.5 的失误）**
  - v1.8.5 仅在 **`ComfyUI-master/requirements.txt`** 中提升了 kitchen 固定版本，但 Forge 运行时是从**仓库根目录 `requirements.txt`** 安装的，而该文件仍停留在 **`comfy-kitchen>=0.2.21`**。因此运行时仍解析到 `0.2.21`，本应生效的修复完全没有生效。
  - 在 `0.2.21` 下，`ComfyUI-master/comfy/quant_ops.py` 中的 `from comfy_kitchen import TensorCoreConvRotW4A4Layout` 会导入失败；由于只有一个 `try` 块，`_CK_AVAILABLE` 被置为 `False`，导致所有 kitchen layout 失效，并触发 `backend/patch_comfy_quant_layout_fallback.py` 的 dequant 回退。量化的 KREA2（int8/fp8 ConvRot）权重被 dequant 为 bf16，耗尽内存（最高约 64 GB）。
  - 本次发布将**根目录 `requirements.txt`** 提升为 **`comfy-kitchen>=0.2.22`**，使运行时安装正确版本，量化 layout 得以正常导入。
  - **用户操作：** 执行一次 **`pip install -U comfy-kitchen`** —— 直接 `pip install -r requirements.txt` 不会升级，因为 `0.2.21` 已满足 `>=0.2.21`。
  - 详情请参阅 [Release Notes（中文）](v1.8.6.md)。

## Version 1.8.5

- **ComfyUI-master 同步与 comfy-kitchen 升级**
  - 将内嵌的 **`ComfyUI-master`** 同步到 Forge 导入所用的最新上游树（JoyImage、Anima LLLite、API 节点、模型检测及相关测试等）。
  - 将该树配套的 kitchen 固定为 **`comfy-kitchen==0.2.22`**（`ComfyUI-master/requirements.txt`）；Forge 运行时仍由仓库根目录 **`requirements.txt`** 安装（`comfy-kitchen>=0.2.21`）。
  - 使远程安装可获得 **`TensorCoreConvRotW4A4Layout`** / INT8 ConvRot 等 kitchen layout，避免因旧版 kitchen 导入失败而整段量化路径失效。
  - 详情请参阅 [Release Notes（中文）](v1.8.5.md)。

## Version 1.8.4

- **Krea2：`layout_cls` 为 None 时的量化 TE 加载回退**
  - Forge 侧运行时补丁：当 `get_layout_class(...)` 返回 `None` 时，将 MixedPrecision 权重 dequant 为 float Parameter，修复量化 Krea2 文本编码器加载时的 `AttributeError: 'NoneType' object has no attribute 'Params'`。
  - Issue：https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/issues/3
  - 依赖：`safetensors>=0.8.0`（与 `diffusers>=0.39.0` 兼容）。
  - 详情请参阅 [Release Notes（中文）](v1.8.4.md)。

## Version 1.8.3

- **Anima：Q/K RMSNorm + split-half RoPE 融合**
  - Anima SelfAttn 且带 RoPE 时，通过 **`comfy_kitchen.rms_rope_split_half`** 融合 Q/K RMSNorm 与 split-half RoPE（需 **`comfy-kitchen>=0.2.21`**）；CrossAttn / 无 RoPE 路径保持原样。
  - 在 Anima 加载路径挂接（`install_anima_rms_rope_fuse` → `comfy.ldm.cosmos.predict2.Attention.compute_qkv`）。
  - 详情请参阅 [Release Notes（中文）](v1.8.3.md)。

## Version 1.8.2

- **Krea2 模型支持**
  - 为 **Krea2** 提供原生 Forge 支持（`SingleStreamDiT` UNet、Qwen3-VL-4B 文本编码器、Qwen-Image VAE）；例如 `moodyKreaMix_V33.safetensors`。
  - Forge checkpoint 面板中的专用 **UI Preset: Krea2**；加载 **Additional modules** `qwen_image_vae.safetensors`（Qwen-Image VAE）与 Qwen3-VL-4B TE（例如 `qwen3vl_4b_bf16.safetensors`）；通过 **`krea2_to_diffusers`** 进行 LoRA 键映射；量化 Krea2 checkpoint 的 MixedPrecision **`comfy_quant`** 加载路径。
  - 详情请参阅 [Release Notes（中文）](v1.8.2.md)。

## Version 1.8.1

- **清理：移除 Nunchaku SDXL 代码（第一阶段）**
  - 删除 Nunchaku SDXL 专用文件（**`nunchaku_sdxl_unet.py`**、**`nunchaku_sdxl_clip.py`**、**`nunchaku_sdxl_unet_append.py`**），并从 **`backend/loader.py`**、**`extensions-builtin/sd_forge_lora/networks.py`** 以及四个 RES4LYF beta 采样器中移除 SDXL 专用分支。
  - 不影响常规 SDXL、fp16 / fp8 / int8，以及其他模型的 Nunchaku。
  - 详情请参阅 [Release Notes（中文）](v1.8.1.md)。

## Version 1.8.0

- **面向 SDXL 与 Z-Image 的 INT8（Low Bits）UNet 量化**
  - 在 **`Diffusion in Low Bits`** 下拉菜单中新增 **`int8`** / **`int8 (fp16 LoRA)`** 选项，启用 tensor-wise INT8（`int8_tensorwise`）UNet 存储以降低显存占用；需显式选择，绝不自动检测。
  - **`int8`** 将 LoRA 烘焙进 INT8 权重，而 **`int8 (fp16 LoRA)`** 将 LoRA 保留为 FP16 以进行在线（免烘焙）应用。
  - INT8 加载与 float8 / bnb / Automatic 路径完全分离；每个非 `int8` 下拉选项都使用不同的加载分支。
  - 详情请参阅 [Release Notes（中文）](v1.8.0.md)。

## Version 1.7.9

- **Anima Hires Fix：高分辨率放大时的分块二次采样**
  - 在**像素放大** Hires 路径上，仅当 checkpoint 名含 **`anima`** 时，将放大后的 latent 走**分块 img2img**（对标 ComfyUI USDU），避免单次全图 img2img 在超出 Anima 训练分辨率时噪声化、画面崩坏。
  - **96×96 latent 分块**、**50% 重叠**（`pad_latent=48`）、**高斯权重融合**拼接，并对瓦片边界做**接缝修复**（窄条带、约一半 denoising strength）。
  - **仅 Anima**：VAE 解码后动态 min/max 归一化；共用路径上支持 **5D 解码张量**归一化与 **5D truncate** 裁剪。
  - Anima Hires **sigma 调度**与 ComfyUI 对齐。非 Anima 及 **latent 放大** Hires 不变（v1.7.8 的 5D 辅助逻辑除外）。
  - 变更限于 **`modules/processing.py`**。
  - 详情请参阅 [Release Notes（中文）](v1.7.9.md)。

## Version 1.7.8

- **Hires Fix：对视频/DiT模型（Wan / Flux）的 5D 张量兼容性支持**
  - 修复 latent 空间超分时的 **`ValueError: Input and output must have the same number of spatial dimensions`** 错误：在 `torch.nn.functional.interpolate` 调用前后，对 5D latent `(N, C, 1, H, W)` 的额外时间轴维度（dim 2）进行局部 squeeze / unsqueeze 处理。
  - 修复像素空间超分（pixel-space upscale）时的 **`TypeError: Cannot handle this data type: (1, 1, 1, 1024)`** 错误：在进行 NumPy / PIL 转换前，对 5D VAE 解码输出 `(N, 1, C, H, W)` 的额外时间轴维度（dim 1）进行 squeeze 处理。
  - 这些 squeeze 处理仅在检测到 5D 张量时生效。标准 4D latent 模型（SD 1.5, SDXL, 动漫模型等）会直接跳过该处理，因此完全不受影响。
  - 详情请参阅 [Release Notes（中文）](v1.7.8.md)。

## Version 1.7.7

- **Anima + ADetailer：img2img inpaint 标志修复**
  - **`Anima`** 引擎加载后将 **`is_inpaint = False`**，避免在 Qwen-Image VAE **5D** latent 上走 WebUI SD 风格的 latent mask/图像 concat。
  - 修复主 txt2img 成功后 ADetailer 后处理 **`AnimaWai68`** checkpoint（如 **`waiANIMA_pw3.safetensors`**）时出现的 **`RuntimeError: Tensors must have same number of dimensions: got 4 and 5`**；区域 mask 仍通过 img2img **`sample()`** 中已有的 **5D** mask blend 生效。
  - 变更仅限 **`backend/diffusion_engine/anima.py`**（与 Qwen / Flux / Lumina 相同做法），其他模型不受影响。提交 **`9c85472`**。
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.7.7)。

## Version 1.7.6

- **IoPaint 启动脚本修复：**
  - 将 IOPaint 默认端口改为 **`8081`**，避免与 ComfyUI 的 `AiHelper`（占用 `8080`）冲突。
  - 在 `iopaint-cuda.bat` 中增加就绪检测：使用 PowerShell 轮询 `/api/v1/server-config`，返回 HTTP 200 后再打开浏览器。
  - 修正 `iopaint-cuda.bat` 的工作目录（`cd`），确保在正确路径下启动。
  - 详情请参阅 [Release Notes（中文）](v1.7.6.md)。

## Version 1.7.5

- **ADetailer（内置扩展）：** 人脸检测不再使用 InsightFace（仅 YOLO）。
  - 详情请参阅 [Release Notes（中文）](v1.7.5.md)。

## Version 1.7.4

- **Pony / SDXL LoRA（CLIP-L）：IntegratedCLIP 键映射**
  - **`modules_forge/packages/comfy/lora.py`** 中的 **`model_lora_keys_clip()`** 现同时尝试 **`transformer.text_model.encoder.layers.*`**（Transformers 4.x–5.5）与 **`transformer.encoder.layers.*`**（Transformers 5.6+ / IntegratedCLIP），使 Pony / SDXL 上的 **`lora_te1_*`** 键正确绑定。
  - 降低 CLIP 未匹配键超过 **`extensions-builtin/sd_forge_lora/networks.py`** 中 50% 门限时出现 **`[LORA] LoRA mismatch for CLIP`**、整包 LoRA 被跳过的情况（即使 UNet 键可匹配）。
  - 提交 `3a8ea0c` / `3d1950d`（clip_layer_paths）及 Pony SDXL LoRA CLIP 修复（`08e6e70`）。
  - 详情请参阅 [Release Notes（中文）](v1.7.4.md)。

## Version 1.7.3

- **Anima：ComfyUI 文本编码器 import（Step 5a）**
  - Anima TE 的加载与编码现使用 Comfy **`load_text_encoder_state_dicts`** 和 **`CLIP.encode_from_tokens`**，不再使用 Forge 复刻 **`Qwen3_06B`** 及手写 **`_encode_qwen`**。
  - **`forge_objects.clip`** 直接持有 Comfy **`sd.CLIP`**（`AnimaTEModel` + `AnimaTokenizer`）；Anima diffusion engine 中已移除 Forge **`CLIP`** 包装器及 HF **`tokenizer`** / **`tokenizer_2`** 接线。
  - 从 **`llama.py`** 移除 Forge **`Qwen3_06BConfig`** / **`Qwen3_06B`**（仅 Anima；其他模型仍用 **`Qwen3_4B`**，未改动）。
  - **`split_state_dict`**：在 **`process_clip_state_dict`** **之前** 切出 TE 键；通过 **`anima_te_filter_prefixes`** 支持 HF **`text_encoders.*`**、Comfy **`cond_stage_model.*`** 及裸 **`qwen3_06b.*`** checkpoint 布局。
  - **`AnimaBase`**：扩展 **`clip_target`**（六种布局）与 **`process_clip_state_dict`**；大 Anima **`class Anima(AnimaBase)`** 继承 TE 切出修复（此前为 **`Anima(BASE)`**）。
  - 详情请参阅 [Release Notes（中文）](v1.7.3.md)。

## Version 1.7.2

- **Nunchaku Z-Image Turbo：Lumina 检测回退修复**
  - Anima v1.7.1 的 Lumina 入口要求 `noise_refiner.k_norm`，但 Nunchaku Z-Image Turbo 导出使用 `norm_k`（稍后在 `svdq.py` 中重映射），导致 checkpoint 被误检为 Nunchaku SDXL 并报错 `You do not have CLIP state dict!`。
  - 仅在 `modules_forge/packages/huggingface_guess/detection.py` 的 Lumina 入口接受 **`k_norm` 或 `norm_k`**；Anima 守卫、Lumina 块主体和 fuzzy SDXL 检测未改动。
  - 详情请参阅 [Release Notes（中文）](v1.7.2.md)。

## Version 1.7.1

- **Anima：ComfyUI import 重构**
  - 将 native Forge DiT（`backend/nn/anima.py`）替换为 bundled ComfyUI-master 中的 **`comfy.ldm.anima.model.Anima`**，使 checkpoint 布局、`llm_adapter` 放置和 UNet 配置检测与上游 Comfy 一致。
  - 文本路径：**`comfy.text_encoders.anima.AnimaTokenizer`** + Forge **`Qwen3_06B`**（仅 embeddings）；**`preprocess_text_embeds`** 在 `get_learned_conditioning` 中对 Comfy UNet 运行一次（TE 侧无 `llm_adapter`）。
  - Loader/检测：**`remap_anima_state_dict`** 仅用于键名修复；**`comfy.model_detection.detect_unet_config`** 委托；**`k_model`** 对静止图像 latent 进行 4D↔5D 包装；**`compile_conditions`** 在缺少 pooled `y` 时省略。
  - UI preset 和 Additional modules（`qwen_3_06b_base`、`qwen_image_vae`）与 v1.7.0 相同。
  - 详情请参阅 [Release Notes（中文）](v1.7.1.md)。

## Version 1.7.0

- **Anima 模型支持**
  - 对 [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) 及兼容单文件 checkpoint（例如 `anima-base-v1.0.safetensors`、社区合并模型如 `waiANIMA_pw3.safetensors`）的 Native Forge 支持。
  - checkpoint 管理器中新增 **UI Preset: Anima**；加载 **Additional modules** `qwen_3_06b_base.safetensors`（Qwen3 文本编码器）和 `qwen_image_vae.safetensors`（Qwen-Image VAE）。T5 tokenizer 词汇表用于 `llm_adapter` cross-attention，无需单独的 T5/UMT5 权重文件。
  - Native `backend.nn.anima` UNet 与 flow-matching 采样；UI 中的 **Shift** 应用于噪声 schedule。
  - 详情请参阅 [Release Notes（中文）](v1.7.0.md)。

## Version 1.6.1

- **UI Preset 控件：单选按钮改为下拉菜单**
  - 将水平 SD1.5 / SDXL / Flux / Qwen / Lumina 单选 preset 选择器替换为 checkpoint 管理器行中的 **UI Preset** 下拉菜单（与 Checkpoint、VAE、Diffusion in Low Bits 同一行）。
  - 下拉标签：SD1.5（`sd`）、SDXL（`xl`）、Flux（`flux`）、Qwen（`qwen`）、Lumina（`lumina`）。现有 preset 切换行为（分辨率、CFG、sampler、VAE 模块、Clip Skip、checkpoint 路径）不变。
  - **LoRA 标签页：** 更新 `javascript/extraNetworks.js` 中的 Extra Networks 过滤，读取下拉值而非旧版单选 `checked` 状态（旧单选 UI 仍存在时有回退）。

## Version 1.6.0

- **ComfyUI-master 更新与 comfy_aimdo 兼容性修复**
  - 通过创建 `vram_buffer.py`（`VRAMBuffer` 类）并扩展 `control.py`（`init_devices` 方法）更新 `comfy_aimdo` stub 包，确保与更新后的 ComfyUI-master 启动序列兼容，无需专有 AIMDO 库。
  - 修复启动时的 `ModuleNotFoundError: No module named 'comfy_aimdo.vram_buffer'`。
  - 发行说明详述错误 traceback、根本原因和修改说明。
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.6.0)。

## Version 1.5.1

- **NumPy 2.4.4 启动兼容性修复**
  - 在 `launch.py` 和 `modules_forge/transformers_cache_compat.py` 中添加 `_blas_supports_fpe` 兼容性 stub，修复在 `numpy==2.4.4` 下 SciPy/diffusers 导入链导致的启动崩溃。
  - 更新 `requirements.txt` 中的运行时依赖 pin（`Pillow`、`accelerate`、`numpy`、`diffusers`、`protobuf`）以匹配当前测试环境。
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.5.1)。

## Version 1.5.0

- **内置宽高比扩展与分辨率计算器**
  - 在 `extensions-builtin/forge_aspect_ratio` 新增内置扩展，用于 txt2img/img2img 一键宽高比 preset。
  - 添加计算器面板（`Calc`），根据百万像素、宽高比和整除性（8/16/32/64）计算目标分辨率，并直接应用到 width/height 滑块。
  - 通过 `aspect_ratios.txt` 和 `resolutions.txt` 添加可配置 preset。
  - 将来自 `ControlAltAI-Nodes` 的 MIT 署名计算逻辑改编集成到 Forge 扩展结构中。
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.5.0)。

## Version 1.4.9

- **transformers 5.6.2 CLIP 兼容性修复**
  - 在 `backend/loader.py` 中规范化 checkpoint 键前缀（`transformer.` / `text_model.` 清理后重新映射），修复 SDXL/Nunchaku 的 transformers 5.6.2 CLIP 加载路径。
  - 在 `backend/text_processing/classic_engine.py` 中更新 transformers 5.x 扁平化后的 CLIP 文本处理访问路径（`text_model.*` → 直接模型属性）。
  - 在 `modules_forge/transformers_cache_compat.py` 中改进 transformers 5.x 的 no-init 兼容性 shim，在 context 内 patch `PreTrainedModel.init_weights` 并安全恢复。
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.4.9)。

## Version 1.4.8

- **ComfyUI-master 0.19.3 同步与 AIMDO 导入兼容性**
  - 更新 bundled `ComfyUI-master` 树以匹配 ComfyUI 0.19.3；上游现在在启动时导入可选 AIMDO 模块（`comfy_aimdo.*`）。
  - 扩展仓库 `comfy_aimdo` stub 包（根目录和 `ComfyUI-master/comfy_aimdo/`），添加 `host_buffer`、`model_mmap` 及相关 API，使 Forge 无需真实 AIMDO wheel 即可启动；Forge 继续使用自己的 loader 和内存路径。
  - 发行说明（英文）涵盖根本原因、完整 traceback、导入流程序列图和文件级说明。
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.4.8)。

## Version 1.4.7

- **Python 3.13 全新安装与设置、ADetailer 初始化以及 triton-windows 环境设置的 bug 修复。** 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.7)。

## Version 1.4.6

- **PyTorch 2.11.0+cu130 默认值、Windows FA2/SA2 wheel URL，以及 torch 升级后 `flash_attn` / diffusers 启动修复。** 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.6)。

## Version 1.4.5

- **Qwen-Image-2512-Fun-Controlnet-Union 支持**
  - 原生支持 Qwen Image Fun ControlNet（例如 Qwen-Image-2512-Fun-Controlnet、Union 变体）。使用 ComfyUI 的 `QwenImageFunControlNetModel`；通过每步 wrapper 注入到 `c["control"]`，适用于 Nunchaku 和标准 Qwen Image backend。
  - 控制图像使用 Forge VAE 编码并作为 raw latent 传递；注入前应用 strength。实现细节见发行说明。
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.5)。

## Version 1.4.4

- **ComfyUI-Master 更新兼容性**
  - 与最新 ComfyUI 核心对齐：移除对已删除 `comfy.checkpoint_pickle` 的依赖；checkpoint 加载现在仅使用 `torch.load(..., weights_only=True)`（与上游相同）。
  - 添加 stub 包 `comfy_aimdo`，使导入可选 PyPI 包 `comfy_aimdo`（AIMDO）的 ComfyUI 代码可在无该包时运行；Forge 继续使用自己的 memory/loader。
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.4)。

## Version 1.4.3

- **Float8 + fp16 LoRA：修复跨类别切换模型时的崩溃**
  - 修复使用「Diffusion in Low Bits」（float8-e4m3fn + fp16 LoRA）并切换到不同模型类别（例如 Z-Image ↔ Flux1）时的 `"addmm_cuda" not implemented for 'Float8_e4m3fn'`。LoRA B@A 现在在 fp16 中计算，加到权重后再 cast 回 Float8。
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.3)。

## Version 1.4.2

- **wd14-tagger 作为内置扩展**
  - WD14 Tagger 集成到 `extensions-builtin`，无需单独安装即可进行图像标注/打 tag。
- **ComfyUI-Master 核心刷新**
  - 更新 bundled ComfyUI 以跟踪上游；与当前 ComfyUI-Master 兼容，用于节点和模型加载。

## Version 1.4.1

- **transformers 5+ 兼容性**
  - 为 `HybridCache` 和 `no_init_weights`（在 transformers 5.x 中移除）添加 shim，使 peft / diffusers / nunchaku 导入链在启动时不失败。
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.1)。

## Version 1.4.0

- **改进 PEFT 格式 LoRA 检测（Nunchaku Qwen Image）**
  - 使用 Hugging Face PEFT 创建的 LoRA 文件（例如 `.lora_A.default.weight`）不再被错误地跳过为「unsupported」
  - 格式检测使用部分匹配，标准键和 PEFT 键模式均识别为标准 LoRA
  - 在适用时记录哪些 LoRA 文件的权重被跳过（例如 modulation 层仅适用于 Nunchaku Qwen Image）
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.4.0)

## Version 1.3.9

- **Config-Presets 作为内置扩展**
  - 集成 [Zyin055/Config-Presets](https://github.com/Zyin055/Config-Presets)（MIT）作为内置扩展
  - 可配置下拉菜单批量切换 txt2img/img2img 设置；创建/删除 preset；通过 `*custom-tracked-components.txt` 自定义字段
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.3.9)

## Version 1.3.8

- **IoPaint 集成**
  - 在 `prepare_environment()` 中自动安装 iopaint 和 imghdr（Python 3.13）；依赖（python-socketio、typer-config、loguru、rembg）；`iopaint-cuda.bat` 启动器
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.3.8)

## Version 1.3.7

- **模型检测 / Qwen3**
  - 当 `detect_unet_config` 返回 None 时，在 diffusers 回退前剥离 `unet_key_prefix`；减少 SDXL–Flux 误检测和下游 VAE shape 不匹配（避免 SDXL 错误走 Qwen3 加载路径）

## Version 1.3.6

- **ADetailer（内置扩展）**
  - 防止每张图像重复人脸处理：每张图像仅第一个 face detector 标签页运行；后续 face 标签页跳过，避免同一张脸处理两次（hand 和其他 detector 标签页不变；一张图像中的多张脸仍各处理一次）

## Version 1.3.5

- **增强 Nunchaku LoRA 实现与文档**
  - 改进 Nunchaku Qwen Image 的 LoRA 支持，含 AWQ 量化层处理
  - 增强 Qwen Image 模型 AWQ 调制层（img_mod/txt_mod）的 Manual Planar Injection
  - 严格模型类型检测，确保 AWQ 修改仅适用于 Nunchaku Qwen Image 模型
  - Qwen Image、Flux1、SDXL 和 Z-Image 模型的 LoRA 路径完全分离
  - 修复 LoRA 应用流程，正确处理模型检测和路由
  - 修改架构和处理流程的全面文档
  - 完整技术详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.3.5)

## Version 1.3.1

- **新增 ZIT（标准版和 Nunchaku 版）的 Diffsynth Union ControlNet 支持**
  - 对标准 Z-Image Turbo（ZIT）和 Nunchaku ZIT 模型的完整 Diffsynth Union ControlNet 支持
  - 可同时使用多个 ControlNet 模型（Union ControlNet）
  - 支持 ZIT ControlNet 模型（例如 `z-image-turbo-controlnet.safetensors`）
  - 通过 NextDiT 模型类型自动检测 ZIT 模型
  - 严格模型类型检查，确保仅与 ZIT 模型兼容
  - VAE 包装器，实现 Forge VAE 与 ComfyUI ControlNet 接口的无缝集成
  - 基于 ComfyUI 的 nodes_model_patch.py 的完整实现
  - 修复导致 RecursionError 的双重补丁和过期补丁问题
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.3.1)

## Version 1.3.0

- **新增 RES4LYF Sampler 支持**
  - 对所有模型类型的 RES4LYF（RES4）sampler 完整支持
  - 支持 Nunchaku Qwen Image、Nunchaku Flux1、Nunchaku SDXL、标准 SDXL 和标准 Flux1 模型
  - 全面的 sampler 集合，含 multistep（res_2m、res_3m 等）和 exponential（res_2s、res_3s 等）变体
  - 非 implicit sampler 的 ODE 版本支持
  - 针对 Forge 和 ComfyUI 模型结构的健壮模型检测与处理
  - 自动 CONST 和 EPS 模型类型检测，确保正确的采样行为
  - 通过 KModel wrapper 修复 Forge 模型的 model_sampling 访问
  - 改进与 ComfyUI-master 目录结构的兼容性
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.3.0)

## Version 1.2.1

- **新增 Nunchaku Qwen Image 的 Union ControlNet 支持**
  - 对 Nunchaku Qwen Image（QI）模型的完整 Union ControlNet 支持
  - 可同时使用多个 ControlNet 模型（Union ControlNet）
  - 支持 Qwen Image Union ControlNet 模型（例如 `Qwen-Image-InstantX-ControlNet-Union.safetensors`）
  - 通过 `transformer_blocks.0.img_mlp.net.0.proj.weight` 键自动检测模型
  - 严格模型类型检查，确保仅与 Nunchaku Qwen Image 模型兼容
  - VAE 包装器，实现 Forge VAE 与 ComfyUI ControlNet 接口的无缝集成
  - 与 Flux ControlNet 完全独立的实现
  - 修复 ControlNet 模型加载的设备放置问题
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.2.1)

## Version 1.2.0

- **新增 Nunchaku Qwen Image 和 Z-Image 模型的 LoRA 支持**
  - 对 Nunchaku Qwen Image（QI）模型的完整 LoRA 支持
  - 对 Nunchaku Z-Image（ZIT）模型的完整 LoRA 支持
  - Qwen Image 与 Z-Image 的实现完全分离
  - 对所有 LoRA 提供格式检测的全面日志
  - 健壮的变更检测，正确处理模型重载
  - 支持标准 LoRA 格式（lora_A/lora_B、lora_up/lora_down）
  - AWQ 量化层处理及安全开关
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/v1.2.0)

## Version 1.1.0

- **新增 Flux1 和 Nunchaku Flux1 的 Union ControlNet 支持**
  - 对 Flux1 和 Nunchaku Flux1 模型的完整 Union ControlNet 支持
  - 可同时使用多个 ControlNet 模型
  - 通过 `controlnet_x_embedder.weight` 键自动检测模型
  - VAE 包装器，实现 Forge VAE 无缝集成
  - 详情请参阅 [Release Notes](https://github.com/ussoewwin/Stable-Diffusion-WebUI-Forge-Nunchaku/releases/tag/1.1.0)

## Version 1.0.7

- **新增 ADetailer 作为内置扩展**
  - 集成 [ADetailer](https://github.com/Bing-su/adetailer) 作为标准内置功能
  - 使用 InsightFace 替代 MediaPipe，兼容 Python 3.13
  - 包含 YOLOv8、YOLOv11 和 InsightFace 混合检测系统
  - 修复 ControlNet 预处理器初始化问题
  - 将 `extensions-builtin/adetailer/models/` 添加到 `.gitignore`

## Version 1.0.6

- Nunchaku SDXL loader、LoRA loader 和 ControlNet 支持完成
