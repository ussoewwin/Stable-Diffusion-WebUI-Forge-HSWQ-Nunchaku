"""
Forge-only checkpoint key remap for Comfy ``comfy.ldm.anima.model.Anima``.

Do not edit ComfyUI-master; upstream updates frequently. All Anima glue lives here
and under backend/ / modules_forge/ (import comfy.* only).
"""

from __future__ import annotations

import logging
from typing import Optional

_logger = logging.getLogger(__name__)

_RMS_ROPE_FUSE_INSTALLED = False
_RMS_ROPE_FUSE_WARNED = False


def remap_anima_state_dict(state_dict: dict) -> dict:
    """
    Main DiT blocks (MiniTrainDIT / predict2) use cross_attn.output_proj.
    llm_adapter blocks (ldm.anima Attention) use cross_attn.o_proj.
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith("blocks.") and ".cross_attn.o_proj." in k:
            nk = k.replace(".cross_attn.o_proj.", ".cross_attn.output_proj.")
        elif "llm_adapter.blocks." in k and ".cross_attn.output_proj." in k:
            nk = k.replace(".cross_attn.output_proj.", ".cross_attn.o_proj.")
        else:
            nk = k
        out[nk] = v
    return out


def _patched_compute_qkv(
    self,
    x,
    context: Optional[object] = None,
    rope_emb: Optional[object] = None,
    transformer_options: Optional[dict] = None,
):
    """Fuse Q/K RMSNorm + split-half RoPE on SelfAttn via ``rms_rope_split_half``.

    Scale cast: plan B — cast ``q_norm.weight`` / ``k_norm.weight`` to activation
    dtype/device (Comfy ``operations.RMSNorm`` has no Forge ``weights_manual_cast``).
    """
    from einops import rearrange
    import comfy.quant_ops as quant_ops

    q = self.q_proj(x)
    context = x if context is None else context
    k = self.k_proj(context)
    v = self.v_proj(context)
    q, k, v = map(
        lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
        (q, k, v),
    )

    if self.is_selfattn and rope_emb is not None:
        q_scale = self.q_norm.weight.to(dtype=q.dtype, device=q.device)
        k_scale = self.k_norm.weight.to(dtype=k.dtype, device=k.device)
        eps = float(getattr(self.q_norm, "eps", 1e-6))
        q, k = quant_ops.ck.rms_rope_split_half(q, k, rope_emb, q_scale, k_scale, eps)
    else:
        q = self.q_norm(q)
        k = self.k_norm(k)

    v = self.v_norm(v)
    return q, k, v


def install_anima_rms_rope_fuse() -> bool:
    """Monkey-patch ``comfy.ldm.cosmos.predict2.Attention.compute_qkv`` once.

    Install from the Anima loader path only. The same ``Attention`` class is shared
    by Cosmos Predict2 MiniTrainDIT; the fuse branch is SelfAttn+rope only, so
    CrossAttn / no-rope paths keep stock ``q_norm`` / ``k_norm``.

    Returns True if applied (or already applied), False if skipped (API missing).
    """
    global _RMS_ROPE_FUSE_INSTALLED, _RMS_ROPE_FUSE_WARNED

    if _RMS_ROPE_FUSE_INSTALLED:
        return True

    try:
        import comfy.quant_ops as quant_ops

        ck = getattr(quant_ops, "ck", None)
        if ck is None or not hasattr(ck, "rms_rope_split_half"):
            if not _RMS_ROPE_FUSE_WARNED:
                msg = (
                    "[Anima] comfy_kitchen.rms_rope_split_half unavailable "
                    "(need comfy-kitchen>=0.2.21); keeping stock compute_qkv."
                )
                print(msg)
                _logger.warning(msg)
                _RMS_ROPE_FUSE_WARNED = True
            return False

        from comfy.ldm.cosmos.predict2 import Attention

        if getattr(Attention.compute_qkv, "_anima_rms_rope_fuse", False):
            _RMS_ROPE_FUSE_INSTALLED = True
            return True

        _patched_compute_qkv._anima_rms_rope_fuse = True  # type: ignore[attr-defined]
        Attention.compute_qkv = _patched_compute_qkv
        _RMS_ROPE_FUSE_INSTALLED = True
        msg = (
            "[Anima] Installed rms_rope_split_half fuse on "
            "comfy.ldm.cosmos.predict2.Attention.compute_qkv "
            "(SelfAttn+rope only; CrossAttn unchanged)."
        )
        # Match FA-2 / attention_backend_info: Forge console shows print, not only logger.
        print(msg)
        _logger.info(msg)
        return True
    except Exception as e:
        if not _RMS_ROPE_FUSE_WARNED:
            msg = f"[Anima] Failed to install rms_rope_split_half fuse: {e}"
            print(msg)
            _logger.warning(msg)
            _RMS_ROPE_FUSE_WARNED = True
        return False
