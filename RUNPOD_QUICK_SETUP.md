# RunPod Quick Setup - Qwen3VL Latest

This is the fastest way to get Qwen3-VL-8B running on RunPod with the latest API.

## ✅ One-Command Setup

```bash
cd /root/streaming-vlm
source venv/bin/activate

# Update code from repo
git pull

# Install only essential dependencies
pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --no-deps
pip install transformers==4.52.4 qwen-vl-utils accelerate Pillow numpy opencv-python

# Set Python path
export PYTHONPATH=/root/streaming-vlm:$PYTHONPATH

# Verify setup
python3 -c "from transformers import Qwen3VLForConditionalGeneration; print('✓ Ready!')"
```

## 🎬 Run Inference

### Get a test video:
```bash
cd /root/streaming-vlm
wget "https://commondatastorage.googleapis.com/gtv-videos-library/sample/ForBiggerBlazes.mp4" -O video.mp4
```

### Run inference:
```bash
export PYTHONPATH=/root/streaming-vlm:$PYTHONPATH

python streaming_vlm/inference/inference.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --model_base Qwen3 \
    --video_path video.mp4 \
    --output_dir output.vtt
```

### Check results:
```bash
cat output/output.vtt
```

## 📝 Key Changes Made

1. ✅ Removed deprecated `block_sparse_attn` package (not on PyPI)
2. ✅ Updated to latest `qwen-vl-utils>=0.0.11` API
3. ✅ Changed Qwen3 model loading to use `dtype` instead of `torch_dtype`
4. ✅ Simplified imports for better compatibility
5. ✅ Removed hardcoded old constants, using dynamic config

## 🚀 New Requirements

No need for Flash Attention on RunPod (eager attention works fine).

Just install:
```bash
pip install transformers==4.52.4 qwen-vl-utils accelerate
```

## 💡 Tips

- If model download is slow, it's just the first time (17.5GB)
- Model is cached in `~/.cache/huggingface/`
- Output subtitles go to `output/` directory
- Use `nvidia-smi` in another terminal to monitor GPU

## ❓ FAQ

**Q: Do I need Flash Attention?**  
A: No, not required. Eager attention works on A40 and other RunPod GPUs.

**Q: Where's my video file?**  
A: Upload via SFTP or use a downloaded sample (see above).

**Q: How do I download results?**  
A: Use SFTP from your local machine:
```bash
sftp root@your-pod-id
get /root/streaming-vlm/output.vtt
quit
```

**Q: Can I use a URL for the video?**  
A: Yes, use `--video_path "http://example.com/video.mp4"`

## 🔄 Workflow

```bash
# Step 1: Setup (one time)
cd /root/streaming-vlm && source venv/bin/activate
git pull
pip install -r infer_requirements.txt  # or just essentials: see above

# Step 2: Get video
wget "https://example.com/video.mp4" -O video.mp4

# Step 3: Run
export PYTHONPATH=/root/streaming-vlm:$PYTHONPATH
python streaming_vlm/inference/inference.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --model_base Qwen3 \
    --video_path video.mp4 \
    --output_dir output.vtt

# Step 4: Check results
cat output/output.vtt

# Step 5: Download (from local machine)
sftp root@your-pod-id
get /root/streaming-vlm/output.vtt
quit
```

That's it! 🚀
