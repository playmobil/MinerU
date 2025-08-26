# MinerU VLM 集成文档

🚀 **MinerU Pipeline + VLM 混合增强表格识别系统**

本文档详细介绍如何在 MinerU 中使用 VLM（Vision Language Model）增强功能，特别针对生产环境部署进行了优化。

## 📖 目录

- [概述](#概述)
- [核心特性](#核心特性)
- [系统要求](#系统要求)
- [安装配置](#安装配置)
- [使用方法](#使用方法)
- [生产环境部署](#生产环境部署)
- [性能优化](#性能优化)
- [故障排除](#故障排除)
- [API 参考](#api-参考)

## 📋 概述

MinerU VLM 集成采用**智能混合架构**，结合 MinerU Pipeline 的高性能专业处理和 VLM 的语义理解能力：

- **Pipeline 基础处理**: 保持 MinerU 原有的高速、专业化文档解析
- **VLM 智能增强**: 使用 Table Transformer 等模型增强表格结构识别
- **无缝集成**: 通过简单的 `--vlm` 参数启用，无需修改现有工作流

## ✨ 核心特性

### 🎯 智能增强
- **Table Transformer 支持**: Microsoft 专业表格识别模型
- **自适应处理**: 根据表格复杂度自动选择最优策略
- **语义理解**: 提升复杂表格的结构识别准确性

### ⚡ 性能优化
- **A100/V100 专门优化**: 检测高端 GPU 并自动优化配置
- **Flash Attention 2**: A100 环境下自动启用
- **混合精度**: FP16 优化，减少 50% 显存占用
- **批处理优化**: 智能批处理大小调整

### 🛡️ 生产就绪
- **优雅降级**: VLM 不可用时自动回退到 Pipeline 模式
- **内存管理**: 低内存占用和内存优化模式
- **设备自适应**: 自动检测并适配不同硬件环境
- **日志监控**: 详细的处理过程日志

## 💻 系统要求

### 基础要求
- **Python**: 3.8+
- **操作系统**: Linux (Ubuntu 18.04+), macOS, Windows
- **内存**: 最低 8GB RAM，推荐 16GB+

### GPU 要求
- **NVIDIA GPU**: 支持 CUDA 11.0+
- **显存**: 最低 6GB，推荐 8GB+
- **优化支持**: A100, V100, H100, A6000

### 软件依赖
```bash
# 必需依赖
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
timm>=0.9.0

# 可选优化依赖
flash-attn>=2.0.0        # A100 Flash Attention 2
xformers>=0.0.20         # 内存优化
bitsandbytes>=0.41.0     # 量化支持
```

## 🔧 安装配置

### 1. 环境准备

#### Linux/Ubuntu（推荐生产环境）
```bash
# 更新系统
sudo apt-get update && sudo apt-get upgrade -y

# 安装系统依赖
sudo apt-get install -y python3 python3-pip python3-venv \
    git wget curl build-essential \
    libgl1-mesa-glx libglib2.0-0 libgomp1

# 创建虚拟环境
python3 -m venv venv_mineru_vlm
source venv_mineru_vlm/bin/activate
```

#### macOS
```bash
# 使用 Homebrew 安装依赖
brew install python git

# 创建虚拟环境
python3 -m venv venv_mineru_vlm
source venv_mineru_vlm/bin/activate
```

### 2. 安装核心依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装 PyTorch (选择适合的版本)
# CUDA 版本 (Linux/Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# MPS 版本 (Apple Silicon Mac)
pip install torch torchvision torchaudio

# CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装 VLM 依赖

```bash
# Transformers 生态
pip install transformers>=4.35.0
pip install accelerate>=0.20.0
pip install sentencepiece protobuf

# Table Transformer 支持
pip install timm

# 生产环境优化（可选）
pip install flash-attn xformers bitsandbytes optimum
```

### 4. 安装 MinerU

```bash
# 从源码安装最新版本
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
pip install -e .

# 或者使用包管理器
pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com
```

### 5. 验证安装

```bash
python -c "
import torch
import transformers
import timm
from mineru.backend.pipeline.model_init import VLM_AVAILABLE

print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
print(f'TIMM: {timm.__version__}')
print(f'VLM Available: {VLM_AVAILABLE}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
"
```

## 🚀 使用方法

### 基础使用

```bash
# 不启用 VLM（默认 Pipeline 模式）
mineru -p document.pdf -o output/

# 启用 VLM 增强
mineru -p document.pdf -o output/ --vlm

# 完整参数示例
mineru -p document.pdf -o output/ \
    --vlm \
    -d cuda:0 \
    -l ch \
    -m auto \
    --vram 16
```

### Python API 使用

```python
from mineru.cli.common import do_parse
from pathlib import Path

# 读取文档
pdf_path = "document.pdf"
with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()

# VLM 增强处理
result = do_parse(
    output_dir="output/",
    pdf_file_names=["document"],
    pdf_bytes_list=[pdf_bytes],
    p_lang_list=["ch"],
    backend="pipeline",
    parse_method="auto",
    formula_enable=True,
    table_enable=True,
    enable_vlm=True,  # 启用 VLM
    device_mode="cuda:0"
)
```

### 高级配置

```python
# 自定义 VLM 处理器
from mineru.backend.pipeline.model_init import TableVLMProcessor

# 使用不同的 VLM 模型
vlm_processor = TableVLMProcessor(
    model_name="microsoft/table-transformer-structure-recognition",
    device="cuda:0"
)

# 手动处理表格图像
from PIL import Image
image = Image.open("table.png")
result = vlm_processor.enhance_table_result(
    image=image,
    table_result={"html": "<table>...</table>"},
    context="financial report analysis"
)
```

## 🏭 生产环境部署

### 自动化部署脚本

我们提供了完整的 Ubuntu A100 生产环境部署脚本：

```bash
# 下载部署脚本
wget https://raw.githubusercontent.com/your-repo/MinerU/main/deploy_production_ubuntu_a100.sh

# 执行部署
chmod +x deploy_production_ubuntu_a100.sh
sudo ./deploy_production_ubuntu_a100.sh
```

### 手动部署步骤

#### 1. 环境检查

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA
nvcc --version

# 检查系统信息
lsb_release -a
uname -a
```

#### 2. 创建生产环境

```bash
# 创建专用用户（推荐）
sudo useradd -m -s /bin/bash mineru
sudo usermod -aG sudo mineru
sudo su - mineru

# 设置工作目录
mkdir -p /opt/mineru
cd /opt/mineru

# 克隆代码
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
```

#### 3. 环境变量配置

```bash
# 创建环境配置文件
cat > ~/.mineru_production << 'EOF'
# MinerU Production Environment
export MINERU_HOME=/opt/mineru/MinerU
export MINERU_VLM_ENABLED=true
export MINERU_DEVICE_MODE=cuda
export MINERU_MIN_BATCH_INFERENCE_SIZE=512

# CUDA 优化
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDNN_BENCHMARK=1

# 缓存目录
export TRANSFORMERS_CACHE=/tmp/transformers_cache
export HF_HOME=/tmp/huggingface_cache

# Python Path
export PYTHONPATH=$MINERU_HOME:$PYTHONPATH
export PATH=$MINERU_HOME/venv/bin:$PATH
EOF

# 加载环境变量
source ~/.mineru_production
```

#### 4. 创建生产服务

```bash
# 创建 systemd 服务
sudo tee /etc/systemd/system/mineru-vlm.service << EOF
[Unit]
Description=MinerU VLM Processing Service
After=network.target nvidia-persistenced.service

[Service]
Type=forking
User=mineru
Group=mineru
WorkingDirectory=/opt/mineru/MinerU
Environment=PATH=/opt/mineru/MinerU/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
EnvironmentFile=/home/mineru/.mineru_production
ExecStart=/opt/mineru/MinerU/scripts/start_production.sh
ExecStop=/opt/mineru/MinerU/scripts/stop_production.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mineru-vlm

[Install]
WantedBy=multi-user.target
EOF

# 启用服务
sudo systemctl daemon-reload
sudo systemctl enable mineru-vlm
sudo systemctl start mineru-vlm
```

### 容器化部署

#### Dockerfile 示例

```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 \
    libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制代码
COPY . .

# 安装 Python 依赖
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install transformers>=4.35.0 accelerate timm && \
    pip install -e .

# 设置权限
RUN chmod +x scripts/*.sh

# 暴露端口（如果有 API 服务）
EXPOSE 8000

# 启动命令
CMD ["./scripts/start_container.sh"]
```

#### Docker Compose 配置

```yaml
version: '3.8'

services:
  mineru-vlm:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_VISIBLE_DEVICES=0
      - MINERU_VLM_ENABLED=true
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - /tmp/model_cache:/tmp/transformers_cache
    ports:
      - "8000:8000"
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## ⚡ 性能优化

### 硬件优化配置

#### A100/V100 高端 GPU
```python
# 在代码中自动检测并应用
# 或手动设置环境变量
export MINERU_GPU_TYPE=A100
export MINERU_FLASH_ATTENTION=true
export MINERU_FP16=true
export MINERU_BATCH_SIZE=32
```

#### 中端 GPU (RTX 3080/4080)
```bash
export MINERU_FP16=true
export MINERU_BATCH_SIZE=16
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

#### 入门级 GPU
```bash
export MINERU_FP16=false
export MINERU_BATCH_SIZE=8
export MINERU_LOW_MEM_MODE=true
```

### 软件优化配置

#### 模型选择优化
```python
# 根据硬件自动选择最优模型
from mineru.backend.pipeline.model_init import get_optimal_table_model_type

# A100 环境 - 优先精度
model_type = get_optimal_table_model_type(
    device="cuda:0", 
    image_count=100,
    performance_priority="accuracy"
)
# 返回: "unitable"

# 批量处理 - 优先速度
model_type = get_optimal_table_model_type(
    device="cuda:0", 
    image_count=1000,
    performance_priority="speed"
)
# 返回: "slanet_plus"
```

#### 批处理优化
```bash
# 根据显存大小设置批处理
# 32GB+ 显存
export MINERU_MIN_BATCH_INFERENCE_SIZE=1024

# 16-24GB 显存
export MINERU_MIN_BATCH_INFERENCE_SIZE=512

# 8-12GB 显存
export MINERU_MIN_BATCH_INFERENCE_SIZE=256
```

### 性能基准测试

```bash
# 运行基准测试
python scripts/benchmark_vlm.py \
    --test-dir ./test_data \
    --output-dir ./benchmark_results \
    --gpu-id 0 \
    --vlm-enabled true
```

预期性能指标：

| 硬件配置 | 模式 | 处理速度 | 准确率 | 显存占用 |
|---------|------|----------|---------|----------|
| **A100 80GB** | Pipeline + VLM | ~0.5s/页 | 98%+ | ~12GB |
| **V100 32GB** | Pipeline + VLM | ~0.8s/页 | 97%+ | ~10GB |
| **RTX 4080** | Pipeline + VLM | ~1.2s/页 | 96%+ | ~8GB |
| **RTX 3080** | Pipeline Only | ~0.6s/页 | 94%+ | ~6GB |

## 🔍 故障排除

### 常见问题

#### 1. VLM 模型加载失败
```bash
# 错误信息
ERROR: Failed to load VLM model: Insufficient GPU memory

# 解决方案
export MINERU_LOW_MEM_MODE=true
export MINERU_VLM_FP16=true
# 或者使用更小的模型
export MINERU_VLM_MODEL=microsoft/table-transformer-detection
```

#### 2. CUDA 内存不足
```bash
# 错误信息
RuntimeError: CUDA out of memory

# 解决方案
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MINERU_BATCH_SIZE=4
# 清理缓存
python -c "import torch; torch.cuda.empty_cache()"
```

#### 3. Table Transformer 依赖问题
```bash
# 错误信息
ImportError: timm not found

# 解决方案
pip install timm>=0.9.0
# 重启 Python 环境
```

#### 4. 性能问题诊断
```python
# 性能分析脚本
import time
import torch
from mineru.backend.pipeline.model_init import TableVLMProcessor

# 测试 VLM 加载时间
start = time.time()
processor = TableVLMProcessor(device="cuda:0")
load_time = time.time() - start
print(f"VLM load time: {load_time:.2f}s")

# 测试 GPU 利用率
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
print(f"GPU Utilization: {torch.cuda.utilization()}%")
```

### 日志分析

#### 启用详细日志
```bash
export MINERU_LOG_LEVEL=DEBUG
export MINERU_VLM_DEBUG=true

# 运行并查看日志
mineru -p document.pdf -o output/ --vlm 2>&1 | tee mineru_debug.log
```

#### 关键日志指标
```bash
# VLM 成功启用
grep "VLM enhancement enabled" mineru_debug.log

# 模型加载时间
grep "model init cost" mineru_debug.log

# 处理性能
grep "Processing pages" mineru_debug.log

# 错误诊断
grep -i "error\|warning\|failed" mineru_debug.log
```

## 📚 API 参考

### 命令行参数

```bash
mineru [OPTIONS] -p INPUT -o OUTPUT

Options:
  -p, --path PATH         输入文件或目录路径 [必需]
  -o, --output PATH       输出目录路径 [必需]
  --vlm, --enable-vlm     启用 VLM 增强 [默认: False]
  -d, --device TEXT       设备模式 (cpu/cuda/mps) [默认: auto]
  -l, --lang TEXT         OCR 语言 [默认: ch]
  -m, --method TEXT       解析方法 (auto/ocr/txt) [默认: auto]
  --vram INTEGER          GPU 显存限制 (GB)
  -b, --backend TEXT      后端类型 [默认: pipeline]
  -f, --formula           启用公式识别 [默认: True]
  -t, --table             启用表格识别 [默认: True]
  --source TEXT           模型源 (huggingface/modelscope) [默认: huggingface]
  -v, --version           显示版本信息
  --help                  显示帮助信息
```

### Python API

#### 核心函数
```python
from mineru.cli.common import do_parse

def do_parse(
    output_dir: str,
    pdf_file_names: List[str],
    pdf_bytes_list: List[bytes],
    p_lang_list: List[str],
    backend: str = "pipeline",
    parse_method: str = "auto", 
    formula_enable: bool = True,
    table_enable: bool = True,
    enable_vlm: bool = False,  # VLM 开关
    device_mode: str = None,
    **kwargs
) -> None
```

#### VLM 处理器类
```python
from mineru.backend.pipeline.model_init import TableVLMProcessor

class TableVLMProcessor:
    def __init__(
        self, 
        model_name: str = "microsoft/table-transformer-structure-recognition",
        device: str = None
    )
    
    def enhance_table_result(
        self, 
        image: PIL.Image,
        table_result: dict,
        context: str = ""
    ) -> dict
```

#### 模型配置类
```python
from mineru.backend.pipeline.model_init import MineruPipelineModel

class MineruPipelineModel:
    def __init__(
        self,
        device: str = "cpu",
        table_config: dict = None,
        formula_config: dict = None,
        lang: str = None,
        enable_vlm: bool = False,  # VLM 配置
        image_count: int = None,
        performance_priority: str = "balanced"
    )
```

### 环境变量

| 变量名 | 描述 | 默认值 | 示例 |
|--------|------|--------|------|
| `MINERU_DEVICE_MODE` | 设备模式 | auto | cuda:0 |
| `MINERU_VLM_ENABLED` | VLM 全局开关 | false | true |
| `MINERU_MIN_BATCH_INFERENCE_SIZE` | 批处理大小 | 384 | 512 |
| `MINERU_VIRTUAL_VRAM_SIZE` | 虚拟显存限制 | auto | 16 |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA 内存配置 | - | max_split_size_mb:512 |
| `TRANSFORMERS_CACHE` | 模型缓存目录 | ~/.cache | /tmp/transformers_cache |
| `MINERU_MODEL_SOURCE` | 模型源 | huggingface | modelscope |

## 📞 支持与贡献

### 获取帮助
- **GitHub Issues**: [提交问题](https://github.com/opendatalab/MinerU/issues)
- **文档**: [完整文档](https://github.com/opendatalab/MinerU/wiki)
- **社区讨论**: [Discussions](https://github.com/opendatalab/MinerU/discussions)

### 贡献指南
1. Fork 项目仓库
2. 创建功能分支: `git checkout -b feature-vlm-enhancement`
3. 提交更改: `git commit -am 'Add VLM enhancement'`
4. 推送分支: `git push origin feature-vlm-enhancement`
5. 创建 Pull Request

### 版本历史
- **v1.0.0**: VLM 集成基础版本
- **v1.1.0**: A100 生产环境优化
- **v1.2.0**: 多模型支持和性能优化

---

## 📜 许可证

本项目基于 Apache 2.0 许可证开源。详见 [LICENSE](LICENSE) 文件。

---

**🎉 感谢使用 MinerU VLM 增强功能！**

如果这个项目对您有帮助，请考虑给我们一个 ⭐ Star！