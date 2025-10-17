"""
Qwen3 Streaming VLM Patches

Patches Qwen3-VL models for streaming video understanding with KV cache management.
Qwen3-VL architecture is similar to Qwen2.5-VL but with improvements to vision and language components.
"""

from streaming_vlm.inference.qwen3.vision_forward import (
    streaming_visual_attention_forward,
    streaming_visual_block_forward, 
    streaming_visual_encoder_forward
)
from streaming_vlm.inference.qwen3.language_forward import (
    streaming_language_model_forward,
    streaming_text_flash_attn_forward,
    streaming_text_decoder_layer_forward,
    _update_causal_mask
)
from streaming_vlm.inference.qwen3.model_forward import (
    model_forward,
    qwen3_vl_forward,
    prepare_inputs_for_streaming_generation
)
from streaming_vlm.inference.qwen3.pos_emb import get_rope_index
from transformers import Qwen3VLForConditionalGeneration
from types import MethodType
from streaming_vlm.inference.generate.streaming_generate_qwen import streaming_generate, _sample
from streaming_vlm.inference.generate.prepare_generation import prepare_multiturn_multimodal_inputs_for_generation
import os

os.makedirs('./output_image', exist_ok=True)

DURATION_SEC = 4
WINDOW_SIZE = 4  # Ensure WINDOW_SIZE is divisible by DURATION_SEC
OUT_FPS = 1
MAX_TOKEN_PER_DURATION = 30
MAX_PIXELS = 360 * 640


def convert_qwen3_to_streaming(model: Qwen3VLForConditionalGeneration):
    """
    Convert Qwen3-VL model to streaming mode by patching forward passes.
    
    Args:
        model: Qwen3VLForConditionalGeneration model instance
        
    Returns:
        Patched model with streaming capabilities
    """
    # Patch generation and input preparation
    model.generate = MethodType(streaming_generate, model)
    model.prepare_inputs_for_generation = MethodType(prepare_multiturn_multimodal_inputs_for_generation, model)
    model._sample = MethodType(_sample, model)

    # Patch main forward passes
    model.forward = MethodType(qwen3_vl_forward, model)
    model.model.forward = MethodType(model_forward, model.model)
    
    # Patch language model
    model.model.language_model.forward = MethodType(streaming_language_model_forward, model.model.language_model)
    model.model.language_model._update_causal_mask = MethodType(_update_causal_mask, model.model.language_model)
    
    # Patch language decoder layers
    for layer in model.model.language_model.layers:
        layer.forward = MethodType(streaming_text_decoder_layer_forward, layer)
        layer.self_attn.forward = MethodType(streaming_text_flash_attn_forward, layer.self_attn)

    # Patch vision encoder
    model.model.visual.forward = MethodType(streaming_visual_encoder_forward, model.model.visual)
    for blk in model.model.visual.blocks:
        blk.forward = MethodType(streaming_visual_block_forward, blk)
        blk.attn.forward = MethodType(streaming_visual_attention_forward, blk.attn)

    # Patch positional embeddings
    model.model.get_rope_index = MethodType(get_rope_index, model.model)
    
    return model
