#!/bin/bash
# M1 Mac VLM 环境设置脚本

echo "🚀 设置 M1 Mac Pipeline + VLM 环境..."

# 检查是否是 M1 Mac
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  警告: 此脚本为 M1/M2 Mac 优化"
fi

# 更新 pip
echo "📦 更新 pip..."
python -m pip install --upgrade pip

# 安装 PyTorch MPS 支持版本
echo "🔥 安装 PyTorch (MPS 支持)..."
pip install torch torchvision torchaudio

# 安装 Transformers 和相关依赖
echo "🤖 安装 Transformers..."
pip install transformers>=4.35.0
pip install accelerate>=0.20.0
pip install sentencepiece
pip install protobuf

# 安装图像处理库
echo "🖼️  安装图像处理库..."
pip install pillow opencv-python

# 可选: 安装 MLX (Apple Silicon 专用)
echo "🍎 安装 MLX (可选，Apple Silicon 专用)..."
pip install mlx mlx-lm || echo "MLX 安装失败，将使用 PyTorch 后端"

# 安装其他有用的库
echo "📊 安装额外依赖..."
pip install numpy pandas matplotlib seaborn
pip install tqdm loguru

# 验证安装
echo "✅ 验证安装..."
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'MPS 可用: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
print(f'MPS 已构建: {torch.backends.mps.is_built() if hasattr(torch.backends, \"mps\") else False}')

try:
    import transformers
    print(f'Transformers 版本: {transformers.__version__}')
except ImportError:
    print('Transformers 未安装')

try:
    import mlx.core as mx
    print('MLX 可用')
except ImportError:
    print('MLX 不可用 (可选)')
"

echo "🎉 环境设置完成！"
echo ""
echo "📝 下一步:"
echo "1. 运行: python m1_pipeline_vlm_hybrid.py"
echo "2. 或者运行测试脚本验证功能"