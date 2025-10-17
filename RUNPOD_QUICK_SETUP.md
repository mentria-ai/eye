# RunPod Quick Setup - Qwen3VL Latest

This is the fastest way to get Qwen3-VL-8B running on RunPod with the latest API.

## ✅ One-Command Setup (No More Manual Paths!)

```bash
cd /root/streaming-vlm
source venv/bin/activate

# Update code from repo
git pull

# Install local packages FIRST
pip install -e streaming_vlm/livecc_utils/

# Install core dependencies (skip Flash Attention)
pip install torch==2.7.1 torchvision==0.22.1 --no-deps
pip install transformers==4.52.4 qwen-vl-utils accelerate Pillow numpy opencv-python

# Verify setup
python3 -c "from transformers import Qwen3VLForConditionalGeneration; print('✓ Ready!')"
```

## 🎬 Run Inference (No PYTHONPATH Needed!)

### Get a test video:
```bash
cd /root/streaming-vlm
wget "https://commondatastorage.googleapis.com/gtv-videos-library/sample/ForBiggerBlazes.mp4" -O video.mp4
```

### Run inference (clean command, no path setup):
```bash
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

1. ✅ Installed local `livecc_utils` package with `pip install -e`
2. ✅ No need for `export PYTHONPATH` anymore
3. ✅ All imports work automatically
4. ✅ Removed deprecated `block_sparse_attn` package
5. ✅ Updated to latest `qwen-vl-utils>=0.0.11` API
6. ✅ Changed Qwen3 model loading to use `dtype` instead of `torch_dtype`

## 🚀 Complete Setup (Copy & Paste)

```bash
#!/bin/bash
cd /root/streaming-vlm
source venv/bin/activate

# Update
git pull

# Install local packages
pip install -e streaming_vlm/livecc_utils/

# Core dependencies
pip install torch==2.7.1 torchvision==0.22.1 --no-deps
pip install transformers==4.52.4 qwen-vl-utils accelerate Pillow numpy opencv-python

# Get test video
wget "https://commondatastorage.googleapis.com/gtv-videos-library/sample/ForBiggerBlazes.mp4" -O video.mp4

# Run inference
python streaming_vlm/inference/inference.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --model_base Qwen3 \
    --video_path video.mp4 \
    --output_dir output.vtt

# Check results
cat output/output.vtt
```

## 💡 Tips

- `pip install -e` means "install in editable mode" (it links to the local folder)
- No more environment variable setup needed
- Model is cached in `~/.cache/huggingface/` (~17.5GB first time)
- Output subtitles go to `output/` directory
- Use `nvidia-smi` in another terminal to monitor GPU

## ❓ FAQ

**Q: Do I need PYTHONPATH anymore?**  
A: No! With `pip install -e streaming_vlm/livecc_utils/`, paths are automatic.

**Q: Do I need Flash Attention?**  
A: No, not required. Eager attention works fine on A40 and other RunPod GPUs.

**Q: Where's my video file?**  
A: Upload via SFTP or use the downloaded sample (see above).

**Q: How do I download results?**  
A: Use SFTP from your local machine:
```bash
sftp root@your-pod-id
get /root/streaming-vlm/output.vtt
quit
```

**Q: Can I use a URL for the video?**  
A: Yes, use `--video_path "http://example.com/video.mp4"`

**Q: What if I run setup multiple times?**  
A: Safe! `pip install -e` is idempotent (does nothing if already installed).

## 🔄 Full Workflow (After Setup)

```bash
# Always starts like this:
cd /root/streaming-vlm
source venv/bin/activate

# Then either:

# Option 1: Process a local video
python streaming_vlm/inference/inference.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --model_base Qwen3 \
    --video_path video.mp4 \
    --output_dir output.vtt

# Option 2: Process a URL
python streaming_vlm/inference/inference.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --model_base Qwen3 \
    --video_path "http://example.com/video.mp4" \
    --output_dir output.vtt

# Check results
cat output/output.vtt
```

## 🎉 That's It!

No more path headaches. Just activate venv and run! 🚀
