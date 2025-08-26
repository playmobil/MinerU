#!/usr/bin/env python3
"""
MinerU VLM 使用示例
展示各种使用场景和最佳实践
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import json

# 添加MinerU路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def example_1_basic_usage():
    """示例1: 基础VLM使用"""
    print("=" * 60)
    print("示例1: 基础VLM使用")
    print("=" * 60)
    
    from mineru.cli.common import do_parse
    
    # 基础配置
    pdf_path = "sample_document.pdf"  # 替换为实际文件路径
    output_dir = "output_basic"
    
    if not os.path.exists(pdf_path):
        print("⚠️  请将 sample_document.pdf 放置在当前目录")
        return
    
    # 读取PDF文件
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    print(f"📄 处理文件: {pdf_path}")
    print("🔧 启用VLM增强...")
    
    start_time = time.time()
    
    # 使用VLM增强处理
    do_parse(
        output_dir=output_dir,
        pdf_file_names=["sample_document"],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=["ch"],  # 中文
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        enable_vlm=True,  # 启用VLM
        device_mode="auto"
    )
    
    processing_time = time.time() - start_time
    print(f"✅ 处理完成，耗时: {processing_time:.2f}s")
    print(f"📁 输出目录: {output_dir}")

def example_2_custom_vlm_processor():
    """示例2: 自定义VLM处理器"""
    print("=" * 60)
    print("示例2: 自定义VLM处理器")
    print("=" * 60)
    
    from mineru.backend.pipeline.model_init import TableVLMProcessor
    
    # 创建自定义VLM处理器
    vlm_processor = TableVLMProcessor(
        model_name="microsoft/table-transformer-structure-recognition",
        device="auto"  # 自动选择设备
    )
    
    print("🔧 VLM处理器已创建")
    
    # 模拟处理表格图像
    if os.path.exists("sample_table.jpg"):
        image = Image.open("sample_table.jpg")
        
        # 模拟表格识别结果
        table_result = {
            "html": "<table><tr><td>示例</td><td>数据</td></tr></table>",
            "confidence": 0.95
        }
        
        print("🖼️  处理表格图像...")
        enhanced_result = vlm_processor.enhance_table_result(
            image=image,
            table_result=table_result,
            context="财务报表分析"
        )
        
        print("✅ VLM增强完成")
        print(f"增强结果: {enhanced_result}")
    else:
        print("⚠️  请提供 sample_table.jpg 文件进行测试")

def example_3_batch_processing():
    """示例3: 批量处理文档"""
    print("=" * 60)
    print("示例3: 批量处理文档") 
    print("=" * 60)
    
    from mineru.cli.common import do_parse
    
    # 批量处理配置
    input_dir = "batch_input"
    output_dir = "batch_output"
    
    if not os.path.exists(input_dir):
        print(f"⚠️  请创建 {input_dir} 目录并放入PDF文件")
        return
    
    # 获取所有PDF文件
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  未找到PDF文件")
        return
    
    print(f"📚 找到 {len(pdf_files)} 个PDF文件")
    
    # 批量处理
    pdf_names = []
    pdf_bytes_list = []
    lang_list = []
    
    for pdf_file in pdf_files:
        with open(pdf_file, 'rb') as f:
            pdf_bytes_list.append(f.read())
        pdf_names.append(pdf_file.stem)
        lang_list.append("ch")  # 假设都是中文
    
    print("🚀 开始批量处理...")
    start_time = time.time()
    
    do_parse(
        output_dir=output_dir,
        pdf_file_names=pdf_names,
        pdf_bytes_list=pdf_bytes_list,
        p_lang_list=lang_list,
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        enable_vlm=True,  # 启用VLM批量处理
        device_mode="auto"
    )
    
    total_time = time.time() - start_time
    avg_time = total_time / len(pdf_files)
    
    print(f"✅ 批量处理完成")
    print(f"📊 总耗时: {total_time:.2f}s")
    print(f"📊 平均耗时: {avg_time:.2f}s/文档")

def example_4_performance_comparison():
    """示例4: 性能对比测试"""
    print("=" * 60)
    print("示例4: 性能对比测试")
    print("=" * 60)
    
    from mineru.cli.common import do_parse
    
    test_file = "test_document.pdf"
    
    if not os.path.exists(test_file):
        print("⚠️  请提供 test_document.pdf 进行测试")
        return
    
    with open(test_file, 'rb') as f:
        pdf_bytes = f.read()
    
    # 测试配置
    test_configs = [
        {"name": "Pipeline Only", "enable_vlm": False},
        {"name": "Pipeline + VLM", "enable_vlm": True}
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n🧪 测试配置: {config['name']}")
        
        output_dir = f"test_output_{config['name'].lower().replace(' ', '_')}"
        
        start_time = time.time()
        
        do_parse(
            output_dir=output_dir,
            pdf_file_names=["test_document"],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            backend="pipeline",
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
            enable_vlm=config["enable_vlm"],
            device_mode="auto"
        )
        
        processing_time = time.time() - start_time
        results[config["name"]] = processing_time
        
        print(f"⏱️  处理时间: {processing_time:.2f}s")
    
    # 显示对比结果
    print(f"\n📊 性能对比结果:")
    print("-" * 30)
    for name, time_taken in results.items():
        print(f"{name:15}: {time_taken:6.2f}s")
    
    if len(results) == 2:
        pipeline_time = results["Pipeline Only"]
        vlm_time = results["Pipeline + VLM"] 
        overhead = ((vlm_time - pipeline_time) / pipeline_time) * 100
        print(f"\nVLM开销: +{overhead:.1f}%")

def example_5_configuration_examples():
    """示例5: 各种配置示例"""
    print("=" * 60)
    print("示例5: 各种配置示例")
    print("=" * 60)
    
    # 不同场景的配置
    configurations = {
        "高精度模式": {
            "enable_vlm": True,
            "performance_priority": "accuracy",
            "table_model_type": "unitable",
            "device_mode": "cuda:0",
            "batch_size": 8
        },
        
        "高速模式": {
            "enable_vlm": False,
            "performance_priority": "speed", 
            "table_model_type": "slanet_plus",
            "device_mode": "cuda:0",
            "batch_size": 32
        },
        
        "平衡模式": {
            "enable_vlm": True,
            "performance_priority": "balanced",
            "table_model_type": "auto",
            "device_mode": "auto",
            "batch_size": 16
        },
        
        "内存优化模式": {
            "enable_vlm": True,
            "performance_priority": "balanced",
            "table_model_type": "slanet_plus",
            "device_mode": "cuda:0",
            "batch_size": 4,
            "memory_optimization": True
        }
    }
    
    for name, config in configurations.items():
        print(f"\n🔧 {name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # 生成对应的命令行示例
        cmd_parts = ["mineru", "-p", "document.pdf", "-o", "output"]
        
        if config.get("enable_vlm"):
            cmd_parts.append("--vlm")
        
        if config.get("device_mode") and config["device_mode"] != "auto":
            cmd_parts.extend(["-d", config["device_mode"]])
        
        print(f"  命令: {' '.join(cmd_parts)}")

def example_6_error_handling():
    """示例6: 错误处理和降级"""
    print("=" * 60)
    print("示例6: 错误处理和降级")
    print("=" * 60)
    
    from mineru.backend.pipeline.model_init import VLM_AVAILABLE, TableVLMProcessor
    
    print(f"VLM可用性: {VLM_AVAILABLE}")
    
    if not VLM_AVAILABLE:
        print("⚠️  VLM不可用，系统会自动降级到Pipeline模式")
        print("这是正常的优雅降级行为")
    else:
        print("✅ VLM可用，可以使用增强功能")
        
        # 测试VLM处理器错误处理
        try:
            processor = TableVLMProcessor(device="cpu")  # 强制使用CPU测试
            print("✅ VLM处理器创建成功")
            
            # 测试处理空结果
            result = processor.enhance_table_result(
                image=Image.new("RGB", (100, 100), "white"),
                table_result={},
                context="测试"
            )
            print(f"处理结果: {type(result)}")
            
        except Exception as e:
            print(f"❌ VLM处理器测试失败: {e}")
            print("系统会自动降级处理")

def example_7_api_integration():
    """示例7: API集成示例"""
    print("=" * 60)
    print("示例7: API集成示例")
    print("=" * 60)
    
    # 模拟Web API使用场景
    def process_document_api(file_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """模拟API处理函数"""
        from mineru.cli.common import do_parse
        
        try:
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            
            output_dir = options.get("output_dir", "api_output")
            
            do_parse(
                output_dir=output_dir,
                pdf_file_names=[Path(file_path).stem],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[options.get("language", "ch")],
                backend="pipeline",
                parse_method="auto",
                formula_enable=options.get("formula_enable", True),
                table_enable=options.get("table_enable", True),
                enable_vlm=options.get("enable_vlm", False),
                device_mode=options.get("device", "auto")
            )
            
            return {
                "status": "success",
                "output_dir": output_dir,
                "vlm_enabled": options.get("enable_vlm", False)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    # API使用示例
    api_requests = [
        {
            "file": "document1.pdf",
            "options": {
                "enable_vlm": True,
                "language": "ch",
                "output_dir": "api_output_1"
            }
        },
        {
            "file": "document2.pdf", 
            "options": {
                "enable_vlm": False,
                "language": "en",
                "output_dir": "api_output_2"
            }
        }
    ]
    
    print("🌐 模拟API调用:")
    
    for i, request in enumerate(api_requests):
        print(f"\n请求 {i+1}:")
        print(f"  文件: {request['file']}")
        print(f"  选项: {request['options']}")
        
        if os.path.exists(request['file']):
            result = process_document_api(request['file'], request['options'])
            print(f"  结果: {result['status']}")
            if result['status'] == 'error':
                print(f"  错误: {result['error']}")
        else:
            print(f"  ⚠️  文件不存在: {request['file']}")

def main():
    """运行所有示例"""
    print("🚀 MinerU VLM 使用示例")
    print("=" * 60)
    
    examples = [
        ("基础VLM使用", example_1_basic_usage),
        ("自定义VLM处理器", example_2_custom_vlm_processor), 
        ("批量处理文档", example_3_batch_processing),
        ("性能对比测试", example_4_performance_comparison),
        ("各种配置示例", example_5_configuration_examples),
        ("错误处理和降级", example_6_error_handling),
        ("API集成示例", example_7_api_integration)
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n选择要运行的示例 (1-7, 0=全部, q=退出):")
    choice = input("请输入选择: ").strip()
    
    if choice.lower() == 'q':
        print("👋 退出")
        return
    
    try:
        if choice == '0':
            # 运行所有示例
            for name, func in examples:
                print(f"\n🔄 运行示例: {name}")
                try:
                    func()
                except Exception as e:
                    print(f"❌ 示例失败: {e}")
                print("\n" + "="*60)
        else:
            # 运行特定示例
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                name, func = examples[idx]
                print(f"\n🔄 运行示例: {name}")
                func()
            else:
                print("❌ 无效选择")
    except ValueError:
        print("❌ 请输入有效数字")
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()