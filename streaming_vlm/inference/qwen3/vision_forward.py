"""Vision forward passes for Qwen3-VL streaming"""
# Qwen3-VL vision architecture is compatible with Qwen2.5-VL
# Import and re-export the implementations
from streaming_vlm.inference.qwen2_5.vision_forward import (
    streaming_visual_attention_forward,
    streaming_visual_block_forward,
    streaming_visual_encoder_forward,
)

__all__ = [
    "streaming_visual_attention_forward",
    "streaming_visual_block_forward", 
    "streaming_visual_encoder_forward",
]
