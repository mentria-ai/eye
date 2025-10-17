# Quick Reference: Qwen3 Upgrade

## 🎯 TL;DR

You now have Qwen3-VL support! Here's how to use it:

```bash
python streaming_vlm/inference/inference.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --model_base Qwen3 \
    --video_path your_video.mp4 \
    --output_dir output.vtt
```

---

## What Changed?

| Component | Before | After |
|-----------|--------|-------|
| **Vision Model** | Qwen2-VL (7B) | Qwen3-VL (8B) ✨ |
| **Text Model** | Qwen2.5-VL | Integrated in Qwen3-VL |
| **Context Length** | 8K tokens | 256K tokens (32x longer!) |
| **Code Changes** | - | New `qwen3/` module |
| **Inference Script** | inference.py | Updated with Qwen3 support |
| **Requirements** | - | Already compatible |

---

## Local Testing

### 1. Upgrade Transformers (if needed)
```bash
pip install --upgrade "transformers>=4.51.0"
# Already at 4.52.4 in infer_requirements.txt ✓
```

### 2. Run Test
```bash
cd /root/streaming-vlm
python streaming_vlm/inference/inference.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --model_base Qwen3 \
    --video_path test.mp4
```

### 3. Check Output
- Subtitles saved to: `output/` directory
- Logs show FPS and VRAM usage
- Compare with previous model output

---

## RunPod Deployment (3 Steps)

### Step 1: Create Pod
1. Go to runpod.io
2. Create GPU Pod (A100 recommended: ~$0.60/hour)
3. Copy SSH command

### Step 2: Run Setup
```bash
ssh root@your-pod-id.runpod.io
bash <(curl -s https://raw.githubusercontent.com/your-org/streaming-vlm/main/scripts/setup_runpod.sh)
```

### Step 3: Inference
```bash
cd /root/streaming-vlm
./run_inference.sh your_video.mp4 output.vtt
```

**Done!** Full guide: See `RUNPOD_DEPLOYMENT.md`

---

## Expected Performance

| GPU | FPS | Cost/Hour | Best For |
|-----|-----|-----------|----------|
| H100 | 5-8 | $0.90 | Production |
| A100 | 2-4 | $0.60 | Most use cases |
| L40S | 3-5 | $0.30 | Budget |

---

## Key Improvements

✨ **Unified Model**: No separate text model needed
✨ **Better Vision**: Improved spatial reasoning
✨ **Longer Context**: 256K tokens (from 8K)
✨ **Better OCR**: 32 languages (from 19)
✨ **Same Patches**: Reuses proven Qwen2.5 streaming code

---

## Common Commands

### Reduce Quality (faster)
```bash
--window_size 8 --chunk_duration 1
```

### Increase Quality
```bash
--window_size 32 --temperature 0.5
```

### Long Videos
```bash
--text_sink 256 --text_sliding_window 256
```

### JSON Output (streaming)
```bash
--emit_json
```

---

## File Summary

```
NEW:
  streaming_vlm/inference/qwen3/          (Qwen3 streaming patches)
  RUNPOD_DEPLOYMENT.md                     (deployment guide)
  UPGRADE_SUMMARY.md                       (detailed explanation)
  scripts/setup_runpod.sh                  (automated setup)

UPDATED:
  streaming_vlm/inference/inference.py     (Qwen3 support added)
```

---

## Support Matrix

| Model | Status | Supported |
|-------|--------|-----------|
| Qwen3-VL-8B-Instruct | ✅ New | Yes |
| Qwen2.5-VL | ✅ Existing | Yes |
| Qwen2-VL | ✅ Existing | Yes |
| Qwen3-4B-Instruct-2507-FP8 | ⚠️ Optional | As standalone |

---

## Troubleshooting

**ImportError for Qwen3?**
```bash
pip install --upgrade transformers>=4.51.0
```

**Out of Memory?**
```bash
--window_size 8 --chunk_duration 1
```

**Slow on RunPod?**
```bash
# Check GPU:
nvidia-smi

# Check Flash Attention:
python -c "from transformers import is_flash_attn2_available; print(is_flash_attn2_available())"
```

---

## Migration Path

1. ✅ Code is ready (you have it)
2. ⏭️ Test locally (you)
3. ⏭️ Test on RunPod (you)
4. ⏭️ Benchmark quality (you)
5. ⏭️ Update production (you)

---

## Documents to Read

- **Quick Start**: This file (you're reading it! ✓)
- **Full Details**: UPGRADE_SUMMARY.md
- **Deployment**: RUNPOD_DEPLOYMENT.md
- **Paper**: https://arxiv.org/abs/2510.09608

---

## One-Liner to Remember

```bash
python streaming_vlm/inference/inference.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --model_base Qwen3 \
    --video_path video.mp4 --output_dir output.vtt
```

---

**That's it!** You're ready to go. Good luck! 🚀
