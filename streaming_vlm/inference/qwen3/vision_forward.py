"""Vision forward passes for Qwen3-VL streaming"""
import torch
from typing import Optional, Tuple
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)

# Try to import from Qwen3-VL model
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_rotary_pos_emb_flashatt,
        flash_attn_varlen_func
    )
    HAS_FLASH_UTILS = True
except ImportError:
    HAS_FLASH_UTILS = False
    logger.warning("Could not import flash attention utilities from transformers. Falling back to standard attention.")

def streaming_visual_attention_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
    """Qwen3-VL visual attention with streaming support"""
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    else:
        cos, sin = position_embeddings
    
    # Try to use flash attention utilities if available
    if HAS_FLASH_UTILS:
        try:
            q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
            q = q.squeeze(0)
            k = k.squeeze(0)
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
                seq_length, -1
            )
        except Exception as e:
            logger.warning(f"Flash attention failed: {e}. Falling back to standard attention.")
            # Fallback to standard attention
            q, k, v = _apply_rope_and_standard_attn(q, k, v, cos, sin)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None).reshape(seq_length, -1)
    else:
        # Fallback: use standard attention with RoPE
        q, k, v = _apply_rope_and_standard_attn(q, k, v, cos, sin)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None).reshape(seq_length, -1)

    attn_output = self.proj(attn_output)
    return attn_output

def _apply_rope_and_standard_attn(q, k, v, cos, sin):
    """Apply RoPE and compute attention using standard PyTorch"""
    # Apply RoPE manually
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_rot = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_rot = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
    return q_rot, k_rot, v

def streaming_visual_block_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Qwen3-VL visual block with streaming support"""
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states

def streaming_visual_encoder_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Qwen3-VL visual encoder with streaming support"""
    for block in self.blocks:
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
    hidden_states = self.norm(hidden_states)
    return hidden_states

__all__ = [
    "streaming_visual_attention_forward",
    "streaming_visual_block_forward", 
    "streaming_visual_encoder_forward",
]
