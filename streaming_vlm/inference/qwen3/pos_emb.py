"""Positional embeddings for Qwen3 streaming"""
# Qwen3 uses the same positional embedding approach as Qwen2.5
# Import and re-export the implementation
from streaming_vlm.inference.qwen2_5.pos_emb import get_rope_index

__all__ = ["get_rope_index"]
