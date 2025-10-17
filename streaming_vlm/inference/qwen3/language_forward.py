"""Language forward passes for Qwen3 streaming"""
import logging
from typing import Optional, List, Tuple
import torch

logger = logging.getLogger(__name__)

# Try to import utilities from transformers
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        logger as transformers_logger,
        rotate_half,
    )
    HAS_TRANSFORMERS_UTILS = True
except ImportError:
    HAS_TRANSFORMERS_UTILS = False
    logger.warning("Could not import some utilities from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")

# Try to import streaming components from Qwen2.5
try:
    from streaming_vlm.inference.qwen2_5.language_forward import (
        streaming_language_model_forward,
        streaming_text_flash_attn_forward,
        streaming_text_decoder_layer_forward,
        _update_causal_mask,
    )
    HAS_QWEN2_5_LANGUAGE = True
except ImportError as e:
    HAS_QWEN2_5_LANGUAGE = False
    logger.warning(f"Could not import Qwen2.5 language streaming components: {e}")

# Export the imports if they worked
if HAS_QWEN2_5_LANGUAGE:
    __all__ = [
        "streaming_language_model_forward",
        "streaming_text_flash_attn_forward",
        "streaming_text_decoder_layer_forward",
        "_update_causal_mask",
    ]
else:
    # Provide stub implementations if imports fail
    def streaming_language_model_forward(*args, **kwargs):
        raise NotImplementedError("Qwen2.5 language streaming not available. Ensure transformers>=4.51.0 is installed.")
    
    def streaming_text_flash_attn_forward(*args, **kwargs):
        raise NotImplementedError("Qwen2.5 language streaming not available. Ensure transformers>=4.51.0 is installed.")
    
    def streaming_text_decoder_layer_forward(*args, **kwargs):
        raise NotImplementedError("Qwen2.5 language streaming not available. Ensure transformers>=4.51.0 is installed.")
    
    def _update_causal_mask(*args, **kwargs):
        raise NotImplementedError("Qwen2.5 language streaming not available. Ensure transformers>=4.51.0 is installed.")
    
    __all__ = [
        "streaming_language_model_forward",
        "streaming_text_flash_attn_forward",
        "streaming_text_decoder_layer_forward",
        "_update_causal_mask",
    ]
