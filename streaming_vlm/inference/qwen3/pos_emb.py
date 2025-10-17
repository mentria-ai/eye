"""Positional embeddings for Qwen3 streaming"""
from typing import Optional, Tuple
import torch

# Gracefully handle FPS import
try:
    from qwen_vl_utils.vision_process import FPS
except ImportError:
    FPS = 2.0  # Default FPS if not available

# Import the base get_rope_index from Qwen2.5 (same architecture)
from streaming_vlm.inference.qwen2_5.pos_emb import get_rope_index

__all__ = ["get_rope_index"]
