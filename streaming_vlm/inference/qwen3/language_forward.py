"""Language forward passes for Qwen3 streaming"""
# Qwen3 language model architecture is compatible with Qwen2.5
# Import and re-export the implementations
from streaming_vlm.inference.qwen2_5.language_forward import (
    streaming_language_model_forward,
    streaming_text_flash_attn_forward,
    streaming_text_decoder_layer_forward,
    _update_causal_mask,
)

__all__ = [
    "streaming_language_model_forward",
    "streaming_text_flash_attn_forward",
    "streaming_text_decoder_layer_forward",
    "_update_causal_mask",
]
