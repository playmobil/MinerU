#!/usr/bin/env python3
"""
MinerU VLM 环境设置脚本
自动检测硬件环境并设置最优配置
"""

import os
import sys
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def detect_system_info() -> Dict:
    """检测系统信息"""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": sys.version,
        "architecture": platform.machine(),
    }
    
    # 检测Linux发行版
    if info["os"] == "Linux":
        try:
            with open("/etc/os-release", "r") as f:
                os_release = f.read()
            for line in os_release.split("\n"):
                if line.startswith("ID="):
                    info["linux_distro"] = line.split("=")[1].strip('"')
                    break
        except:
            info["linux_distro"] = "unknown"
    
    return info

def detect_gpu_info() -> Dict:
    """检测GPU信息"""
    gpu_info = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_memory": [],
        "cuda_version": None
    }
    
    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        
        if gpu_info["cuda_available"]:
            gpu_info["gpu_count"] = torch.cuda.device_count()
            gpu_info["cuda_version"] = torch.version.cuda
            
            for i in range(gpu_info["gpu_count"]):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                gpu_info["gpu_names"].append(name)
                gpu_info["gpu_memory"].append(memory)
    except ImportError:
        print("⚠️  PyTorch not installed, cannot detect GPU info")
    
    # 检测MPS支持 (Apple Silicon)
    try:
        import torch
        gpu_info["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except:
        gpu_info["mps_available"] = False
    
    return gpu_info

def check_dependencies() -> Dict:
    """检查依赖包"""
    deps = {
        "torch": False,
        "transformers": False,
        "timm": False,
        "flash_attn": False,
        "xformers": False,
        "bitsandbytes": False
    }
    
    for pkg in deps:
        try:
            __import__(pkg)
            deps[pkg] = True
        except ImportError:
            deps[pkg] = False
    
    return deps

def recommend_config(system_info: Dict, gpu_info: Dict) -> Dict:
    """根据硬件推荐配置"""
    config = {
        "device": "cpu",
        "vlm_enabled": False,
        "batch_size": 4,
        "fp16": False,
        "performance_priority": "balanced",
        "table_model_type": "slanet_plus",
        "memory_optimization": True,
        "reasoning": []
    }
    
    # 设备选择
    if gpu_info["cuda_available"] and gpu_info["gpu_count"] > 0:
        config["device"] = "cuda:0"
        config["vlm_enabled"] = True
        config["fp16"] = True
        config["reasoning"].append("检测到CUDA GPU，启用GPU加速和VLM")
        
        # 根据GPU性能调整配置
        gpu_name = gpu_info["gpu_names"][0].lower() if gpu_info["gpu_names"] else ""
        gpu_memory = gpu_info["gpu_memory"][0] if gpu_info["gpu_memory"] else 0
        
        # 高端GPU配置
        if any(gpu in gpu_name for gpu in ["a100", "v100", "h100", "a6000"]):
            config.update({
                "batch_size": 32,
                "performance_priority": "accuracy",
                "table_model_type": "unitable",
                "memory_optimization": False,
                "min_batch_inference_size": 1024
            })
            config["reasoning"].append(f"检测到高端GPU ({gpu_name})，使用高性能配置")
            
        # 中端GPU配置
        elif any(gpu in gpu_name for gpu in ["rtx", "gtx", "quadro"]) and gpu_memory >= 8:
            config.update({
                "batch_size": 16,
                "performance_priority": "balanced",
                "table_model_type": "unitable",
                "min_batch_inference_size": 512
            })
            config["reasoning"].append(f"检测到中端GPU ({gpu_name}, {gpu_memory}GB)，使用均衡配置")
            
        # 低端GPU配置
        elif gpu_memory < 8:
            config.update({
                "batch_size": 8,
                "performance_priority": "speed", 
                "table_model_type": "slanet_plus",
                "memory_optimization": True,
                "min_batch_inference_size": 256
            })
            config["reasoning"].append(f"检测到低显存GPU ({gpu_memory}GB)，使用内存优化配置")
            
    elif gpu_info["mps_available"]:
        config.update({
            "device": "mps",
            "vlm_enabled": True,
            "batch_size": 8,
            "fp16": True,
            "performance_priority": "balanced",
            "table_model_type": "auto"
        })
        config["reasoning"].append("检测到Apple Silicon MPS，启用MPS加速")
        
    else:
        config.update({
            "device": "cpu",
            "vlm_enabled": False,
            "batch_size": 4,
            "fp16": False,
            "performance_priority": "speed",
            "table_model_type": "slanet_plus"
        })
        config["reasoning"].append("仅检测到CPU，使用CPU优化配置")
    
    return config

def generate_env_file(config: Dict, output_path: str = ".mineru_env"):
    """生成环境变量文件"""
    env_content = f"""# MinerU VLM Environment Configuration
# Auto-generated by setup_vlm_env.py

# Device Configuration
export MINERU_DEVICE_MODE={config['device']}
export MINERU_VLM_ENABLED={str(config['vlm_enabled']).lower()}

# Performance Configuration
export MINERU_MIN_BATCH_INFERENCE_SIZE={config.get('min_batch_inference_size', 384)}
export MINERU_PERFORMANCE_PRIORITY={config['performance_priority']}
export MINERU_TABLE_MODEL_TYPE={config['table_model_type']}

# Memory Configuration
export MINERU_MEMORY_OPTIMIZATION={str(config['memory_optimization']).lower()}
"""
    
    if config["device"].startswith("cuda"):
        env_content += f"""
# CUDA Optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDNN_BENCHMARK=1
"""
        
        if config.get("min_batch_inference_size", 0) >= 1024:
            env_content += """
# High-end GPU Optimization
export MINERU_FLASH_ATTENTION=true
export MINERU_USE_XFORMERS=true
"""

    env_content += """
# Cache Configuration
export TRANSFORMERS_CACHE=${HOME}/.cache/transformers
export HF_HOME=${HOME}/.cache/huggingface

# Logging
export MINERU_LOG_LEVEL=INFO
"""
    
    with open(output_path, "w") as f:
        f.write(env_content)
    
    return output_path

def install_missing_dependencies(deps: Dict, gpu_info: Dict):
    """安装缺失的依赖"""
    missing = [pkg for pkg, installed in deps.items() if not installed]
    
    if not missing:
        print("✅ 所有依赖已安装")
        return
    
    print(f"📦 准备安装缺失的依赖: {', '.join(missing)}")
    
    install_commands = []
    
    # PyTorch
    if "torch" in missing:
        if gpu_info["cuda_available"]:
            install_commands.append("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        else:
            install_commands.append("pip install torch torchvision torchaudio")
    
    # Transformers
    if "transformers" in missing:
        install_commands.append("pip install transformers>=4.35.0 accelerate>=0.20.0 sentencepiece protobuf")
    
    # TIMM
    if "timm" in missing:
        install_commands.append("pip install timm>=0.9.0")
    
    # 可选优化依赖
    if gpu_info["cuda_available"]:
        optional_deps = []
        if "flash_attn" in missing:
            optional_deps.append("flash-attn")
        if "xformers" in missing:
            optional_deps.append("xformers")
        if "bitsandbytes" in missing:
            optional_deps.append("bitsandbytes")
        
        if optional_deps:
            install_commands.append(f"pip install {' '.join(optional_deps)}")
    
    # 执行安装
    for cmd in install_commands:
        print(f"执行: {cmd}")
        try:
            subprocess.run(cmd.split(), check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  安装失败: {e}")

def create_test_script(config: Dict):
    """创建测试脚本"""
    test_content = f'''#!/usr/bin/env python3
"""
MinerU VLM 配置测试脚本
"""

import sys
import time
from pathlib import Path

def test_imports():
    """测试导入"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        import timm
        print(f"✅ TIMM {timm.__version__}")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    return True

def test_device():
    """测试设备"""
    import torch
    
    device = "{config['device']}"
    print(f"目标设备: {device}")
    
    if device.startswith("cuda"):
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"✅ 显存: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        else:
            print("❌ CUDA 不可用")
            return False
    elif device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✅ MPS 可用")
        else:
            print("❌ MPS 不可用") 
            return False
    else:
        print("✅ CPU 模式")
    
    return True

def test_vlm():
    """测试VLM"""
    if not {str(config['vlm_enabled']).lower()}:
        print("ℹ️  VLM 未启用")
        return True
        
    try:
        from mineru.backend.pipeline.model_init import VLM_AVAILABLE, TableVLMProcessor
        
        if not VLM_AVAILABLE:
            print("❌ VLM 依赖不可用")
            return False
            
        print("✅ VLM 依赖可用")
        
        # 测试VLM处理器创建
        start = time.time()
        processor = TableVLMProcessor(device="{config['device']}")
        load_time = time.time() - start
        print(f"✅ VLM 处理器创建成功 ({load_time:.1f}s)")
        
    except Exception as e:
        print(f"❌ VLM 测试失败: {e}")
        return False
        
    return True

def main():
    print("🔍 MinerU VLM 配置测试")
    print("=" * 40)
    
    # 测试导入
    if not test_imports():
        sys.exit(1)
    
    # 测试设备
    if not test_device():
        sys.exit(1)
    
    # 测试VLM
    if not test_vlm():
        sys.exit(1)
    
    print("=" * 40)
    print("🎉 所有测试通过！配置正确。")
    print()
    print("使用示例:")
    print("  source .mineru_env")
    print("  mineru -p document.pdf -o output/ {'--vlm' if config['vlm_enabled'] else ''}")

if __name__ == "__main__":
    main()
'''
    
    with open("test_vlm_config.py", "w") as f:
        f.write(test_content)
    
    os.chmod("test_vlm_config.py", 0o755)
    return "test_vlm_config.py"

def main():
    """主函数"""
    print("🚀 MinerU VLM 环境设置")
    print("=" * 50)
    
    # 检测系统信息
    print("📋 检测系统信息...")
    system_info = detect_system_info()
    print(f"  操作系统: {system_info['os']} ({system_info.get('linux_distro', '')})")
    print(f"  架构: {system_info['architecture']}")
    print(f"  Python: {sys.version.split()[0]}")
    
    # 检测GPU信息
    print("\n🔍 检测GPU信息...")
    gpu_info = detect_gpu_info()
    if gpu_info["cuda_available"]:
        for i, (name, memory) in enumerate(zip(gpu_info["gpu_names"], gpu_info["gpu_memory"])):
            print(f"  GPU {i}: {name} ({memory}GB)")
        print(f"  CUDA版本: {gpu_info['cuda_version']}")
    elif gpu_info["mps_available"]:
        print("  检测到Apple Silicon MPS支持")
    else:
        print("  未检测到GPU，将使用CPU模式")
    
    # 检查依赖
    print("\n📦 检查依赖...")
    deps = check_dependencies()
    for pkg, installed in deps.items():
        status = "✅" if installed else "❌"
        print(f"  {pkg}: {status}")
    
    # 推荐配置
    print("\n🎯 生成推荐配置...")
    config = recommend_config(system_info, gpu_info)
    
    print("\n推荐配置:")
    print(f"  设备: {config['device']}")
    print(f"  VLM启用: {config['vlm_enabled']}")
    print(f"  批处理大小: {config['batch_size']}")
    print(f"  性能优先级: {config['performance_priority']}")
    print(f"  表格模型: {config['table_model_type']}")
    
    print("\n配置理由:")
    for reason in config["reasoning"]:
        print(f"  • {reason}")
    
    # 询问是否安装依赖
    if not all(deps.values()):
        response = input("\n是否安装缺失的依赖？ (y/n): ")
        if response.lower() == 'y':
            install_missing_dependencies(deps, gpu_info)
    
    # 生成环境文件
    print("\n📝 生成配置文件...")
    env_file = generate_env_file(config)
    print(f"  环境文件: {env_file}")
    
    # 创建测试脚本
    test_script = create_test_script(config)
    print(f"  测试脚本: {test_script}")
    
    # 保存配置信息
    config_info = {
        "system_info": system_info,
        "gpu_info": gpu_info,
        "dependencies": deps,
        "recommended_config": config,
        "timestamp": time.time()
    }
    
    with open("mineru_config_info.json", "w") as f:
        json.dump(config_info, f, indent=2, default=str)
    
    print(f"  配置信息: mineru_config_info.json")
    
    print("\n🎉 环境设置完成！")
    print("\n下一步:")
    print("1. 加载环境变量: source .mineru_env") 
    print("2. 运行测试: python test_vlm_config.py")
    print("3. 开始使用: mineru -p document.pdf -o output/" + (" --vlm" if config['vlm_enabled'] else ""))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)