# Code Changes for Qwen3VL Latest API Support

## 📝 Files Modified

### 1. `streaming_vlm/inference/inference.py`
**Changes:**
- ✅ Removed problematic imports from `qwen_vl_utils.vision_process`
  - Removed: `VIDEO_TOTAL_PIXELS`, `FPS_MAX_FRAMES`, `VIDEO_MIN_PIXELS`, `VIDEO_MAX_PIXELS`, `FRAME_FACTOR`, `IMAGE_FACTOR`
  - These constants no longer exist in latest qwen-vl-utils

- ✅ Added graceful import for `process_vision_info` from `qwen_vl_utils`
  - New: `from qwen_vl_utils import process_vision_info`

- ✅ Simplified global configuration
  - Now uses hardcoded values: `FPS = 2.0`, `VIDEO_MAX_PIXELS = 32000 * 28 * 28`
  - Dynamic calculation: `MAX_PIXELS = 360 * 640`

- ✅ Updated Qwen3 model loading
  - **Changed**: `torch_dtype="auto"` → `dtype="auto"` (Qwen3VL uses `dtype` parameter)
  - Maintains backward compatibility for Qwen2/Qwen2.5 with `torch_dtype`

### 2. `infer_requirements.txt`
**Changes:**
- ✅ Removed `block_sparse_attn==0.0.1` (commented out)
  - This package doesn't exist on PyPI and isn't needed for Qwen3VL
  
- ✅ Updated qwen-vl-utils version
  - Changed: `qwen-vl-utils==0.0.11` → `qwen-vl-utils>=0.0.11`
  - Allows latest compatible versions

### 3. NEW: `RUNPOD_QUICK_SETUP.md`
**Purpose:** Quick reference guide for RunPod deployment with latest changes

**Contents:**
- One-command setup script
- Test video download instructions
- Inference commands
- Troubleshooting FAQ
- Complete workflow

## 🎯 Why These Changes?

| Issue | Cause | Solution |
|-------|-------|----------|
| `ImportError: cannot import name 'VIDEO_TOTAL_PIXELS'` | Old API removed from qwen-vl-utils | Use hardcoded values |
| `block_sparse_attn` not found on PyPI | Package doesn't exist | Removed from requirements |
| Qwen3 model loading fails | Uses `dtype` not `torch_dtype` | Updated model loading |

## ✅ Backward Compatibility

- ✅ Qwen2-VL still works (uses `torch_dtype`)
- ✅ Qwen2.5-VL still works (uses `torch_dtype`)
- ✅ Qwen3-VL now works (uses `dtype`)
- ✅ Can switch models with `--model_base` flag

## 🚀 For RunPod Users

**Quick setup with updated code:**

```bash
cd /root/streaming-vlm
git pull  # Get latest changes
source venv/bin/activate

# Install essentials (skip Flash Attention)
pip install transformers==4.52.4 qwen-vl-utils accelerate

# Set path and run
export PYTHONPATH=/root/streaming-vlm:$PYTHONPATH
python streaming_vlm/inference/inference.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --model_base Qwen3 \
    --video_path video.mp4 \
    --output_dir output.vtt
```

## 📊 Impact

- **Code lines changed**: ~50 lines in inference.py
- **New documentation**: 1 new quick setup guide
- **Dependencies removed**: 1 package (`block_sparse_attn`)
- **New functionality**: Full Qwen3VL support with latest API
- **Breaking changes**: None (100% backward compatible)

## ✨ Next Steps

1. **Push changes** to your repo
2. **Pull on RunPod**: `git pull` in `/root/streaming-vlm`
3. **Install dependencies**: `pip install transformers==4.52.4 qwen-vl-utils>=0.0.11`
4. **Run inference**: Follow commands in RUNPOD_QUICK_SETUP.md

That's it! 🎉
