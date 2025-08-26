#!/bin/bash
# MinerU VLM Production Deployment Script for Ubuntu A100
# 用于Ubuntu服务器A100显卡的生产环境部署

set -e

echo "🚀 MinerU VLM Production Deployment for Ubuntu A100"
echo "=================================================="

# 检查系统环境
echo "📋 检查系统环境..."
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"

# 检查NVIDIA驱动和CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA Driver:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "❌ NVIDIA Driver not found! Please install NVIDIA drivers first."
    exit 1
fi

# 检查CUDA
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA Version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
else
    echo "❌ CUDA not found! Please install CUDA toolkit."
    exit 1
fi

# 安装系统依赖
echo "📦 安装系统依赖..."
sudo apt-get update
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    libgomp1 \
    build-essential

# 创建Python虚拟环境
echo "🐍 设置Python环境..."
if [ ! -d "venv_mineru_prod" ]; then
    python3 -m venv venv_mineru_prod
fi
source venv_mineru_prod/bin/activate

# 升级pip
pip install --upgrade pip

# 安装PyTorch（CUDA版本）
echo "🔥 安装PyTorch (CUDA支持)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装Transformers和VLM依赖
echo "🤖 安装VLM依赖..."
pip install transformers>=4.35.0
pip install accelerate>=0.20.0
pip install sentencepiece
pip install protobuf
pip install timm  # Table Transformer需要

# 安装MinerU依赖
echo "⛏️  安装MinerU依赖..."
pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装其他生产环境依赖
echo "📊 安装生产环境优化包..."
pip install \
    flash-attn \
    xformers \
    bitsandbytes \
    optimum \
    ninja  # 编译优化

# 验证安装
echo "✅ 验证安装..."
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA设备数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'GPU {i} 显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB')

try:
    import transformers
    print(f'Transformers版本: {transformers.__version__}')
except ImportError:
    print('❌ Transformers未安装')

try:
    import timm
    print(f'TIMM版本: {timm.__version__}')
except ImportError:
    print('❌ TIMM未安装')
"

# 设置环境变量
echo "🔧 设置环境变量..."
cat > ~/.mineru_env << 'EOF'
# MinerU Production Environment Variables
export CUDA_VISIBLE_DEVICES=0  # 使用第一块GPU，根据需要调整
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/huggingface_cache

# 优化设置
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDNN_BENCHMARK=1

# VLM特定设置
export MINERU_VLM_ENABLED=true
export MINERU_DEVICE_MODE=cuda
export MINERU_MIN_BATCH_INFERENCE_SIZE=512  # A100优化
EOF

# 创建生产环境配置脚本
echo "⚙️  创建生产环境配置..."
cat > run_mineru_production.py << 'EOF'
#!/usr/bin/env python3
"""
MinerU VLM Production Runner
针对Ubuntu A100优化的生产环境运行脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def setup_production_env():
    """设置生产环境"""
    # GPU内存管理
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步执行提高性能
    
    # 优化设置
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    os.environ['CUDNN_BENCHMARK'] = '1'
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description='MinerU VLM Production')
    parser.add_argument('-p', '--path', required=True, help='Input PDF/image path')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('--vlm', action='store_true', help='Enable VLM enhancement')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_production_env()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # 构建mineru命令
    cmd_parts = [
        'mineru',
        '-p', args.path,
        '-o', args.output,
        '-d', f'cuda:{args.gpu_id}',
        '--source', 'huggingface'
    ]
    
    if args.vlm:
        cmd_parts.append('--vlm')
    
    # 执行命令
    import subprocess
    cmd = ' '.join(cmd_parts)
    print(f"执行命令: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 处理完成!")
        print(result.stdout)
    else:
        print("❌ 处理失败!")
        print(result.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
EOF

chmod +x run_mineru_production.py

# 创建systemd服务文件（可选）
echo "🔧 创建系统服务..."
sudo tee /etc/systemd/system/mineru-vlm.service << EOF
[Unit]
Description=MinerU VLM Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
Environment=PATH=$PWD/venv_mineru_prod/bin:$PATH
ExecStart=$PWD/venv_mineru_prod/bin/python $PWD/run_mineru_production.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "🎉 生产环境部署完成!"
echo "==============================="
echo ""
echo "📋 使用说明:"
echo "1. 激活环境: source venv_mineru_prod/bin/activate"
echo "2. 加载环境变量: source ~/.mineru_env"
echo "3. 运行示例:"
echo "   # 基础模式"
echo "   mineru -p document.pdf -o output/"
echo "   # VLM增强模式"
echo "   mineru -p document.pdf -o output/ --vlm"
echo "   # 生产脚本"
echo "   python run_mineru_production.py -p document.pdf -o output/ --vlm"
echo ""
echo "🔧 系统服务:"
echo "   sudo systemctl enable mineru-vlm"
echo "   sudo systemctl start mineru-vlm"
echo ""
echo "📊 监控GPU使用: watch -n 1 nvidia-smi"