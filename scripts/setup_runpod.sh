#!/bin/bash
# RunPod Setup Script for StreamingVLM with Qwen3
# Run this script on a fresh RunPod instance

set -e  # Exit on error

echo "🚀 StreamingVLM RunPod Setup Script"
echo "=================================="

# Step 1: System Updates
echo "📦 Updating system packages..."
apt-get update
apt-get install -y git git-lfs curl wget build-essential

# Step 2: Clone Repository
echo "📥 Cloning StreamingVLM repository..."
cd /root
if [ ! -d "streaming-vlm" ]; then
    git clone https://github.com/your-org/streaming-vlm.git
    cd streaming-vlm
else
    cd streaming-vlm
    git pull
fi

# Step 3: Create Virtual Environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3.10 -m venv venv
fi
source venv/bin/activate

# Step 4: Install Dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r infer_requirements.txt
pip install --upgrade "transformers>=4.51.0"

# Step 5: Verify Installation
echo "✅ Verifying installation..."
python3 << 'EOF'
import torch
print(f"\n✓ PyTorch Version: {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
print(f"✓ CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

try:
    from transformers import Qwen3VLForConditionalGeneration
    print("✓ Qwen3 support available\n")
except ImportError:
    print("❌ Qwen3 not available - upgrade transformers\n")
EOF

# Step 6: Create quick run script
echo "📄 Creating quick run script..."
cat > run_inference.sh << 'SCRIPT'
#!/bin/bash
source /root/streaming-vlm/venv/bin/activate
cd /root/streaming-vlm

python streaming_vlm/inference/inference.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --model_base Qwen3 \
    --video_path "${1:?Error: video path required}" \
    --output_dir "${2:-output.vtt}" \
    "${@:3}"
SCRIPT
chmod +x run_inference.sh

echo ""
echo "✨ Setup Complete!"
echo "=================================="
echo ""
echo "📝 Quick Commands:"
echo "  1. Activate environment:  source venv/bin/activate"
echo "  2. Run inference:         ./run_inference.sh your_video.mp4 output.vtt"
echo "  3. View logs:             tail -f inference.log"
echo ""
echo "📖 Full guide:  See RUNPOD_DEPLOYMENT.md"
echo ""
echo "🎬 Next Steps:"
echo "  1. Upload or download a test video"
echo "  2. Run: ./run_inference.sh test_video.mp4"
echo "  3. Check output in output.vtt"
echo ""
