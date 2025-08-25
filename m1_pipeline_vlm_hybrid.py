#!/usr/bin/env python3
"""
M1 Max MacBook 优化的 Pipeline + VLM-Transformers 混合表格处理系统
结合 MinerU Pipeline 的专业性和 VLM 的语义理解能力
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from PIL import Image
import logging

# 检查 M1 优化环境
def check_m1_environment():
    """检查 M1 Mac 运行环境"""
    info = {}
    
    # 检查 PyTorch MPS 支持
    info['pytorch_version'] = torch.__version__
    info['mps_available'] = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    info['mps_built'] = torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False
    
    # 检查设备
    if info['mps_available']:
        info['device'] = 'mps'
    elif torch.cuda.is_available():
        info['device'] = 'cuda'
    else:
        info['device'] = 'cpu'
    
    # 系统信息
    import platform
    info['platform'] = platform.platform()
    info['processor'] = platform.processor()
    info['is_m1_mac'] = 'arm64' in platform.platform().lower()
    
    return info

@dataclass 
class M1TableProcessingConfig:
    """M1 优化配置"""
    # Pipeline 配置
    pipeline_device: str = 'mps'
    pipeline_batch_size: int = 4
    pipeline_precision: str = 'fp16'  # M1 对 fp16 优化很好
    
    # VLM 配置
    vlm_model_name: str = 'microsoft/table-transformer-structure-recognition'  # 备选方案
    vlm_device: str = 'mps'
    vlm_batch_size: int = 1
    vlm_max_length: int = 512
    vlm_use_flash_attention: bool = False  # M1 暂不支持
    
    # 混合策略配置
    complexity_threshold: float = 0.4
    semantic_threshold: float = 0.3
    quality_threshold: float = 0.6
    
    # M1 性能优化
    use_mlx: bool = False  # 是否使用 MLX 框架
    memory_optimization: bool = True
    concurrent_processing: bool = True

class M1OptimizedVLMProcessor:
    """M1 优化的 VLM 处理器"""
    
    def __init__(self, config: M1TableProcessingConfig):
        self.config = config
        self.device = config.vlm_device
        self.logger = logging.getLogger(__name__)
        
        # 环境检查
        self.env_info = check_m1_environment()
        self.logger.info(f"M1 Environment: {self.env_info}")
        
        # 初始化 VLM 模型
        self._init_vlm_models()
        
    def _init_vlm_models(self):
        """初始化适合 M1 的 VLM 模型"""
        try:
            # 尝试使用 MLX 优化模型（如果可用）
            if self.config.use_mlx:
                self._init_mlx_models()
            else:
                self._init_huggingface_models()
                
        except Exception as e:
            self.logger.error(f"VLM model initialization failed: {e}")
            # 降级到基础模型
            self._init_fallback_models()
    
    def _init_huggingface_models(self):
        """初始化 Hugging Face VLM 模型"""
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        # 推荐的轻量 VLM 模型（适合 M1）
        model_options = [
            "microsoft/table-transformer-structure-recognition",  # 专门的表格模型
            "Salesforce/blip2-opt-2.7b",  # 轻量通用 VLM  
            "llava-hf/llava-1.5-7b-hf",   # LLaVA 模型
        ]
        
        model_name = self.config.vlm_model_name
        if model_name not in model_options:
            model_name = model_options[0]  # 默认使用表格专用模型
            
        self.logger.info(f"Loading VLM model: {model_name}")
        
        # 配置模型加载参数（M1 优化）
        model_kwargs = {
            'torch_dtype': torch.float16,  # M1 对 fp16 友好
            'device_map': 'auto' if self.device == 'mps' else None,
            'trust_remote_code': True
        }
        
        if self.config.memory_optimization:
            model_kwargs.update({
                'low_cpu_mem_usage': True,
                'load_in_8bit': False,  # M1 暂不完全支持量化
                'load_in_4bit': False
            })
        
        # 加载处理器和模型
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # 移动到设备
        if self.device == 'mps':
            self.model = self.model.to('mps')
            
        self.model.eval()
        self.logger.info(f"VLM model loaded successfully on {self.device}")
        
    def _init_mlx_models(self):
        """初始化 MLX 优化模型（Apple Silicon 专用）"""
        try:
            # 尝试导入 MLX VLM
            import mlx.core as mx
            from mlx_vlm import load, generate
            
            # MLX 优化的模型列表
            mlx_models = [
                "mlx-community/llava-1.5-7b-mlx",
                "mlx-community/blip2-mlx", 
            ]
            
            model_name = mlx_models[0]  # 默认使用 LLaVA MLX 版本
            self.logger.info(f"Loading MLX optimized model: {model_name}")
            
            self.mlx_model, self.mlx_processor = load(model_name)
            self.use_mlx = True
            
            self.logger.info("MLX VLM model loaded successfully")
            
        except ImportError:
            self.logger.warning("MLX not available, falling back to PyTorch")
            self._init_huggingface_models()
        except Exception as e:
            self.logger.error(f"MLX model loading failed: {e}")
            self._init_huggingface_models()
    
    def _init_fallback_models(self):
        """初始化备选基础模型"""
        self.logger.info("Initializing fallback table processing model")
        
        # 使用更轻量的替代方案
        self.use_simple_vlm = True
        self.logger.warning("Using simplified VLM processing")
        
    def process_table_with_vlm(
        self, 
        image: Image.Image, 
        context: Optional[str] = None,
        task_type: str = "structure_recognition"
    ) -> Dict[str, Any]:
        """使用 VLM 处理表格"""
        start_time = time.time()
        
        try:
            if hasattr(self, 'use_mlx') and self.use_mlx:
                return self._process_with_mlx(image, context, task_type)
            elif hasattr(self, 'use_simple_vlm') and self.use_simple_vlm:
                return self._process_with_simple_vlm(image, context)
            else:
                return self._process_with_huggingface(image, context, task_type)
                
        except Exception as e:
            self.logger.error(f"VLM processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _process_with_huggingface(self, image: Image.Image, context: str, task_type: str) -> Dict[str, Any]:
        """使用 Hugging Face 模型处理"""
        
        # 构建提示
        if task_type == "structure_recognition":
            prompt = "Analyze the table structure in this image and describe the layout, cells, and organization."
        elif task_type == "semantic_understanding":
            prompt = f"Understand this table's content and meaning. Context: {context or 'None'}"
        else:
            prompt = "Describe what you see in this table image."
        
        # 预处理
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # 移动到设备
        if self.device == 'mps':
            inputs = {k: v.to('mps') if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            # M1 优化推理参数
            generate_kwargs = {
                'max_length': self.config.vlm_max_length,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'pad_token_id': self.processor.tokenizer.eos_token_id
            }
            
            outputs = self.model.generate(**inputs, **generate_kwargs)
        
        # 解码结果
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'success': True,
            'response': response,
            'confidence': 0.85,  # 简化的置信度
            'task_type': task_type,
            'model_type': 'huggingface'
        }
    
    def _process_with_mlx(self, image: Image.Image, context: str, task_type: str) -> Dict[str, Any]:
        """使用 MLX 优化模型处理"""
        from mlx_vlm import generate
        
        # 构建提示
        prompt = f"Analyze this table image. Task: {task_type}. Context: {context or 'None'}"
        
        # MLX 推理
        response = generate(
            self.mlx_model,
            self.mlx_processor, 
            image,
            prompt,
            max_tokens=self.config.vlm_max_length,
            temp=0.7
        )
        
        return {
            'success': True,
            'response': response,
            'confidence': 0.90,  # MLX 优化通常更准确
            'task_type': task_type,
            'model_type': 'mlx'
        }
    
    def _process_with_simple_vlm(self, image: Image.Image, context: str) -> Dict[str, Any]:
        """简化的 VLM 处理（备选方案）"""
        
        # 基础图像分析
        width, height = image.size
        aspect_ratio = height / width if width > 0 else 1.0
        
        # 简化的表格分析
        analysis = {
            'structure': 'standard_table' if 0.5 < aspect_ratio < 2.0 else 'complex_layout',
            'estimated_cells': min(max(int((width * height) / 10000), 4), 100),
            'image_quality': 'good' if width > 800 and height > 600 else 'low'
        }
        
        response = f"Table analysis: {analysis['structure']}, estimated {analysis['estimated_cells']} cells, {analysis['image_quality']} quality"
        
        return {
            'success': True,
            'response': response,
            'confidence': 0.60,  # 较低置信度
            'task_type': 'simple_analysis',
            'model_type': 'fallback',
            'analysis': analysis
        }

class M1PipelineVLMHybrid:
    """M1 优化的 Pipeline + VLM 混合处理器"""
    
    def __init__(self, config: M1TableProcessingConfig = None):
        self.config = config or M1TableProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # 性能统计
        self.performance_stats = {
            'pipeline_only': {'count': 0, 'total_time': 0, 'accuracy': []},
            'vlm_only': {'count': 0, 'total_time': 0, 'accuracy': []},
            'hybrid': {'count': 0, 'total_time': 0, 'accuracy': []}
        }
        
        # 初始化组件
        self._init_pipeline()
        self._init_vlm_processor()
        
    def _init_pipeline(self):
        """初始化 MinerU Pipeline（M1 优化）"""
        try:
            # 导入 MinerU 组件
            sys.path.append('/Users/frank/mygit/github/MinerU')
            from mineru.backend.pipeline.model_init import MineruPipelineModel
            
            pipeline_config = {
                'table_config': {
                    'enable': True,
                    'model_type': 'auto',  # 自动选择适合 M1 的模型
                    'model_selection': 'auto'
                },
                'formula_config': {'enable': True},
                'device': self.config.pipeline_device,
                'performance_priority': 'balanced'
            }
            
            self.pipeline = MineruPipelineModel(**pipeline_config)
            self.logger.info("MinerU Pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {e}")
            self.pipeline = None
    
    def _init_vlm_processor(self):
        """初始化 VLM 处理器"""
        self.vlm_processor = M1OptimizedVLMProcessor(self.config)
        
    def analyze_table_complexity(self, image: Image.Image) -> Dict[str, float]:
        """分析表格复杂度（针对 M1 优化的轻量版本）"""
        import cv2
        
        # 转换为 numpy 数组
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        height, width = gray.shape
        
        # 快速复杂度分析（M1 优化）
        try:
            # 1. 边缘检测
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            # 2. 线条检测（简化版本）
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                  minLineLength=min(width, height)//4, maxLineGap=10)
            line_count = len(lines) if lines is not None else 0
            
            # 3. 文本密度估算
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            text_density = np.sum(binary == 0) / (height * width)
            
            # 4. 图像方差
            image_variance = np.var(gray) / (255 * 255)
            
            # 综合评分
            visual_complexity = min(edge_density * 2 + image_variance, 1.0)
            structural_complexity = min(line_count / 20 + text_density, 1.0)
            
            return {
                'visual_complexity': visual_complexity,
                'structural_complexity': structural_complexity,
                'overall_complexity': (visual_complexity + structural_complexity) / 2,
                'line_count': line_count,
                'edge_density': edge_density,
                'text_density': text_density
            }
            
        except Exception as e:
            self.logger.warning(f"Complexity analysis failed: {e}")
            return {
                'visual_complexity': 0.5,
                'structural_complexity': 0.5,
                'overall_complexity': 0.5,
                'error': str(e)
            }
    
    def select_processing_strategy(self, 
                                 image: Image.Image, 
                                 context: Optional[str] = None,
                                 user_priority: str = 'balanced') -> str:
        """选择处理策略"""
        
        # 分析复杂度
        complexity = self.analyze_table_complexity(image)
        overall_complexity = complexity['overall_complexity']
        
        # 分析语义需求
        semantic_score = 0.2  # 基础语义需求
        if context:
            semantic_keywords = ['financial', 'analysis', 'comparison', 'calculation']
            for keyword in semantic_keywords:
                if keyword in context.lower():
                    semantic_score = 0.8
                    break
        
        # 图像质量评估（简化）
        width, height = image.size
        image_quality = min((width * height) / (1920 * 1080), 1.0)
        
        # 决策逻辑
        if user_priority == 'speed':
            if overall_complexity < self.config.complexity_threshold:
                return 'pipeline_only'
            else:
                return 'pipeline_only'  # 速度优先始终用 Pipeline
                
        elif user_priority == 'accuracy':
            if semantic_score > self.config.semantic_threshold:
                return 'vlm_only'
            elif overall_complexity > self.config.complexity_threshold:
                return 'hybrid'
            else:
                return 'pipeline_only'
                
        else:  # balanced
            if overall_complexity < 0.3:
                return 'pipeline_only'
            elif semantic_score > 0.5 or overall_complexity > 0.7:
                return 'hybrid'
            else:
                return 'pipeline_only'
    
    def process_table(self, 
                     image: Image.Image,
                     context: Optional[str] = None,
                     user_priority: str = 'balanced') -> Dict[str, Any]:
        """混合表格处理主接口"""
        
        start_time = time.time()
        
        # 选择处理策略
        strategy = self.select_processing_strategy(image, context, user_priority)
        
        self.logger.info(f"Selected processing strategy: {strategy}")
        
        # 执行处理
        if strategy == 'pipeline_only':
            result = self._process_pipeline_only(image)
            self.performance_stats['pipeline_only']['count'] += 1
        elif strategy == 'vlm_only':
            result = self._process_vlm_only(image, context)
            self.performance_stats['vlm_only']['count'] += 1
        else:  # hybrid
            result = self._process_hybrid(image, context)
            self.performance_stats['hybrid']['count'] += 1
        
        # 记录性能
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        result['strategy'] = strategy
        
        # 更新统计
        stats = self.performance_stats[strategy.replace('_only', '')]
        stats['total_time'] += processing_time
        
        self.logger.info(f"Processing completed in {processing_time:.2f}s using {strategy}")
        
        return result
    
    def _process_pipeline_only(self, image: Image.Image) -> Dict[str, Any]:
        """纯 Pipeline 处理"""
        if not self.pipeline:
            return {'success': False, 'error': 'Pipeline not available'}
        
        try:
            # 使用 MinerU Pipeline 处理
            # 注意：这里需要根据实际 MinerU API 调整
            if hasattr(self.pipeline, 'table_model'):
                result = self.pipeline.table_model.predict(image)
                
                return {
                    'success': True,
                    'html': result[0] if result[0] else '',
                    'cell_bboxes': result[1] if len(result) > 1 else [],
                    'confidence': 0.92,
                    'method': 'pipeline_only'
                }
            else:
                return {'success': False, 'error': 'Table model not available'}
                
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_vlm_only(self, image: Image.Image, context: str) -> Dict[str, Any]:
        """纯 VLM 处理"""
        vlm_result = self.vlm_processor.process_table_with_vlm(
            image, context, "structure_recognition"
        )
        
        if vlm_result['success']:
            return {
                'success': True,
                'html': self._convert_vlm_to_html(vlm_result['response']),
                'semantic_analysis': vlm_result['response'],
                'confidence': vlm_result['confidence'],
                'method': 'vlm_only',
                'model_type': vlm_result.get('model_type', 'unknown')
            }
        else:
            return vlm_result
    
    def _process_hybrid(self, image: Image.Image, context: str) -> Dict[str, Any]:
        """混合处理"""
        
        if self.config.concurrent_processing:
            # 并行处理
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                pipeline_future = executor.submit(self._process_pipeline_only, image)
                vlm_future = executor.submit(self._process_vlm_only, image, context)
                
                pipeline_result = pipeline_future.result()
                vlm_result = vlm_future.result()
        else:
            # 串行处理
            pipeline_result = self._process_pipeline_only(image)
            if pipeline_result['success'] and pipeline_result.get('confidence', 0) > 0.9:
                # Pipeline 结果很好，无需 VLM 验证
                return {**pipeline_result, 'method': 'hybrid_pipeline_sufficient'}
            
            vlm_result = self._process_vlm_only(image, context)
        
        # 融合结果
        return self._fuse_results(pipeline_result, vlm_result)
    
    def _fuse_results(self, pipeline_result: Dict, vlm_result: Dict) -> Dict[str, Any]:
        """融合 Pipeline 和 VLM 结果"""
        
        if not pipeline_result['success'] and not vlm_result['success']:
            return {'success': False, 'error': 'Both methods failed'}
        
        if not pipeline_result['success']:
            return {**vlm_result, 'method': 'hybrid_vlm_fallback'}
        
        if not vlm_result['success']:
            return {**pipeline_result, 'method': 'hybrid_pipeline_fallback'}
        
        # 两个都成功，选择更可信的结果
        pipeline_conf = pipeline_result.get('confidence', 0.5)
        vlm_conf = vlm_result.get('confidence', 0.5)
        
        if vlm_conf > pipeline_conf + 0.1:
            # VLM 明显更好
            return {
                **vlm_result,
                'method': 'hybrid_vlm_primary',
                'pipeline_backup': pipeline_result
            }
        else:
            # Pipeline 为主，VLM 增强
            return {
                **pipeline_result,
                'method': 'hybrid_pipeline_primary',
                'semantic_enhancement': vlm_result.get('semantic_analysis'),
                'vlm_confidence': vlm_conf
            }
    
    def _convert_vlm_to_html(self, vlm_response: str) -> str:
        """将 VLM 响应转换为 HTML（简化实现）"""
        # 这是一个简化的转换，实际实现需要更复杂的 NLP 处理
        if 'table' in vlm_response.lower():
            return "<table><tr><td>VLM Generated Content</td></tr></table>"
        return ""
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能统计摘要"""
        summary = {}
        
        for strategy, stats in self.performance_stats.items():
            if stats['count'] > 0:
                avg_time = stats['total_time'] / stats['count']
                summary[strategy] = {
                    'count': stats['count'],
                    'average_time': avg_time,
                    'total_time': stats['total_time']
                }
            else:
                summary[strategy] = {'count': 0, 'average_time': 0, 'total_time': 0}
        
        return summary

# 使用示例和测试
def demo_m1_hybrid_processing():
    """演示 M1 混合处理"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 检查环境
    env_info = check_m1_environment()
    print("🖥️  M1 Mac 环境信息:")
    for key, value in env_info.items():
        print(f"   {key}: {value}")
    print()
    
    # 创建配置
    config = M1TableProcessingConfig(
        pipeline_device='mps' if env_info['mps_available'] else 'cpu',
        vlm_device='mps' if env_info['mps_available'] else 'cpu',
        use_mlx=False,  # 可以尝试设置为 True
        memory_optimization=True,
        concurrent_processing=True
    )
    
    # 创建混合处理器
    print("🚀 初始化 M1 Pipeline + VLM 混合处理器...")
    processor = M1PipelineVLMHybrid(config)
    
    # 模拟表格图像
    test_scenarios = [
        {
            'name': '简单财务表格',
            'image_size': (800, 600),
            'context': 'quarterly revenue report',
            'priority': 'speed'
        },
        {
            'name': '复杂科研数据表',
            'image_size': (1200, 900),
            'context': 'scientific research data with statistical analysis',
            'priority': 'accuracy'
        },
        {
            'name': '平衡处理场景',
            'image_size': (1000, 700),
            'context': 'business analysis report',
            'priority': 'balanced'
        }
    ]
    
    print("\\n📊 开始测试不同场景...")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\\n--- 场景 {i}: {scenario['name']} ---")
        
        # 创建模拟图像
        width, height = scenario['image_size']
        mock_image = Image.new('RGB', (width, height), color='white')
        
        # 处理表格
        result = processor.process_table(
            image=mock_image,
            context=scenario['context'],
            user_priority=scenario['priority']
        )
        
        print(f"✅ 处理策略: {result['strategy']}")
        print(f"⏱️  处理时间: {result['processing_time']:.3f}s")
        print(f"📈 处理成功: {result['success']}")
        
        if result['success']:
            print(f"🎯 置信度: {result.get('confidence', 'N/A')}")
            print(f"🔧 方法: {result.get('method', 'unknown')}")
    
    # 性能统计
    print("\\n📈 性能统计摘要:")
    summary = processor.get_performance_summary()
    for strategy, stats in summary.items():
        if stats['count'] > 0:
            print(f"   {strategy}: {stats['count']}次, 平均{stats['average_time']:.3f}s")
    
    print("\\n🎉 M1 混合处理演示完成！")

if __name__ == "__main__":
    demo_m1_hybrid_processing()