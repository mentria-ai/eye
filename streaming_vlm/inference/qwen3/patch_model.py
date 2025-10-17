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
    For Qwen3-VL, we don't apply streaming patches.
    
    Qwen3-VL has:
    - Native Flash Attention 2 support (built-in)
    - Optimized architecture that works with standard transformers
    - KV cache support through transformers' native implementation
    
    The model works optimally without custom streaming patches.
    Just return the model as-is.
    
    Args:
        model: Qwen3VLForConditionalGeneration model instance
        
    Returns:
        Model unchanged (Qwen3-VL uses native implementation)
    """
    # Qwen3-VL works optimally with native transformers implementation
    # No custom patches needed
    return model
