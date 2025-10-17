"""
Qwen3 Streaming VLM Patches

Patches Qwen3-VL models for streaming video understanding with KV cache management.
Qwen3-VL has a native language model, so we only patch the vision encoder.
"""

from streaming_vlm.inference.qwen3.vision_forward import (
    streaming_visual_attention_forward,
    streaming_visual_block_forward, 
    streaming_visual_encoder_forward
)
from transformers import Qwen3VLForConditionalGeneration
from types import MethodType
import os

os.makedirs('./output_image', exist_ok=True)

DURATION_SEC = 4
WINDOW_SIZE = 4  # Ensure WINDOW_SIZE is divisible by DURATION_SEC
OUT_FPS = 1
MAX_TOKEN_PER_DURATION = 30
MAX_PIXELS = 360 * 640


def convert_qwen3_to_streaming(model: Qwen3VLForConditionalGeneration):
    """
    Convert Qwen3-VL model to streaming mode by patching vision encoder.
    
    Qwen3-VL has a native language model that works with standard transformers,
    so we only patch the vision encoder for streaming support.
    
    Args:
        model: Qwen3VLForConditionalGeneration model instance
        
    Returns:
        Model with streaming vision encoder patches
    """
    # Patch vision encoder for streaming
    model.model.visual.forward = MethodType(streaming_visual_encoder_forward, model.model.visual)
    for blk in model.model.visual.blocks:
        blk.forward = MethodType(streaming_visual_block_forward, blk)
        blk.attn.forward = MethodType(streaming_visual_attention_forward, blk.attn)
    
    return model
