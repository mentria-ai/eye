"""Model forward passes for Qwen3-VL streaming"""
# Qwen3-VL model architecture is compatible with Qwen2.5-VL
# Import and re-export the implementations with Qwen3 naming
from streaming_vlm.inference.qwen2_5.model_forward import (
    model_forward,
    qwen2_5_vl_forward as qwen3_vl_forward,
    prepare_inputs_for_streaming_generation,
)

__all__ = [
    "model_forward",
    "qwen3_vl_forward",
    "prepare_inputs_for_streaming_generation",
]
