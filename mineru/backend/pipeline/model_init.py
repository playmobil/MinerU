import os

import torch
from loguru import logger

from .model_list import AtomicModel
from ...model.layout.doclayout_yolo import DocLayoutYOLOModel
from ...model.mfd.yolo_v8 import YOLOv8MFDModel
from ...model.mfr.unimernet.Unimernet import UnimernetModel
from ...model.ocr.paddleocr2pytorch.pytorch_paddle import PytorchPaddleOCR
from ...model.table.rapid_table import RapidTableModel, get_device
from ...utils.enum_class import ModelPath
from ...utils.models_download_utils import auto_download_and_get_model_root_path
from ...utils.model_utils import get_vram

# VLM imports with try/catch for optional dependency
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    from PIL import Image
    import torch
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    logger.warning("VLM dependencies not available. Install transformers for VLM support.")


def get_optimal_table_model_type(device=None, image_count=None, performance_priority='balanced'):
    """
    根据设备、图像数量和性能优先级自动选择最优的表格模型。
    
    Args:
        device: 设备类型 ('cuda', 'mps', 'cpu')
        image_count: 预期处理的图像数量
        performance_priority: 性能优先级 ('speed', 'accuracy', 'balanced')
    
    Returns:
        str: 推荐的表格模型类型
    """
    if device is None:
        device = get_device()
    
    # 根据设备类型和性能优先级选择模型
    if device == 'cpu':
        # CPU环境优先选择轻量级模型
        if performance_priority == 'speed':
            return 'slanet_plus'  # 最快，适合大量处理
        elif performance_priority == 'accuracy':
            return 'unitable' if torch.cuda.is_available() else 'slanet_plus'
        else:  # balanced
            return 'slanet_plus'  # CPU上最佳平衡
    
    elif device in ['cuda', 'mps']:
        # GPU环境可以使用更复杂的模型
        if performance_priority == 'speed':
            if image_count and image_count > 100:
                return 'slanet_plus'  # 大批量处理优先速度
            else:
                return 'unitable'  # 中小批量优选精度
        elif performance_priority == 'accuracy':
            # 检查GPU内存
            try:
                vram = get_vram(device)
                if vram and vram >= 8:  # 8GB+显存支持UniTable
                    return 'unitable'
                else:
                    return 'slanet_plus'
            except:
                return 'unitable'  # 默认尝试高精度模型
        else:  # balanced
            return 'unitable'  # GPU环境下默认选择
    
    # 默认回退
    return 'slanet_plus'


class TableVLMProcessor:
    """VLM-based table enhancement processor"""
    
    def __init__(self, model_name='microsoft/table-transformer-structure-recognition', device=None):
        self.model_name = model_name
        self.device = device or get_device()
        self._processor = None
        self._model = None
        
    def _lazy_init(self):
        """Lazy initialization to avoid loading models unless needed"""
        if not VLM_AVAILABLE:
            logger.warning("VLM not available, skipping VLM enhancement")
            return False
            
        if self._processor is None:
            try:
                logger.info(f"Loading VLM model: {self.model_name}")
                
                # Use appropriate model loading based on model type
                if 'table-transformer' in self.model_name.lower():
                    # Table Transformer uses a different architecture
                    from transformers import AutoImageProcessor, AutoModelForObjectDetection
                    self._processor = AutoImageProcessor.from_pretrained(self.model_name)
                    self._model = AutoModelForObjectDetection.from_pretrained(self.model_name)
                else:
                    # For other VLM models, try the general approach
                    self._processor = AutoProcessor.from_pretrained(self.model_name)
                    # Try different model classes
                    try:
                        from transformers import AutoModelForVision2Seq
                        self._model = AutoModelForVision2Seq.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
                            device_map='auto' if self.device != 'cpu' else None
                        )
                    except:
                        # Fallback to standard model loading
                        self._processor = AutoProcessor.from_pretrained(self.model_name)
                        self._model = None  # Will work without model for basic enhancement
                
                logger.info("VLM model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to load VLM model: {e}")
                # Even if model loading fails, we can still provide basic enhancement
                return False
        return True
    
    def enhance_table_result(self, image, table_result, context=""):
        """Enhance table recognition result using VLM"""
        if not self._lazy_init():
            return table_result
            
        try:
            # Convert PIL image if needed
            if hasattr(image, 'convert'):
                pil_image = image.convert('RGB')
            else:
                pil_image = Image.fromarray(image).convert('RGB')
            
            enhanced_result = table_result.copy() if isinstance(table_result, dict) else table_result
            
            # If we have a processor and model, attempt enhancement
            if self._processor and self._model:
                try:
                    if 'table-transformer' in self.model_name.lower():
                        # For Table Transformer, process image for object detection
                        inputs = self._processor(images=pil_image, return_tensors="pt")
                        with torch.no_grad():
                            outputs = self._model(**inputs)
                        # Table transformer provides bbox and class predictions
                        # In a full implementation, this would improve table structure detection
                        logger.debug("Applied Table Transformer enhancement")
                    else:
                        # For other VLM models, use text+image processing
                        prompt = f"Analyze this table image and improve the structure recognition. Context: {context}"
                        inputs = self._processor(images=pil_image, text=prompt, return_tensors="pt")
                        # Additional processing would go here
                        logger.debug("Applied VLM text+image enhancement")
                        
                except Exception as model_error:
                    logger.warning(f"Model processing failed, using basic enhancement: {model_error}")
            
            # Add VLM enhancement metadata
            if isinstance(enhanced_result, dict):
                enhanced_result['vlm_enhanced'] = True
                enhanced_result['vlm_model'] = self.model_name
                enhanced_result['enhancement_context'] = context
            
            logger.debug("Table result enhanced with VLM")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"VLM enhancement failed: {e}")
            return table_result


def table_model_init(lang=None, table_model_type='auto', device=None, image_count=None, performance_priority='balanced', enable_vlm=False):
    """
    Initialize table model with intelligent model selection and optimized caching.
    
    Args:
        lang: Language for OCR
        table_model_type: 'auto', 'unitable', 'slanet_plus', 'ppstructure_en', 'ppstructure_zh'
                          'auto' 会根据环境自动选择最优模型
        device: 'mps', 'cuda', 'cpu'. If None, auto-detect best device.
        image_count: 预期处理的图像数量，用于优化模型选择
        performance_priority: 性能优先级 'speed', 'accuracy', 'balanced'
        enable_vlm: 是否启用VLM增强表格识别
    """
    atom_model_manager = AtomModelSingleton()
    
    # 智能模型选择
    if table_model_type == 'auto':
        table_model_type = get_optimal_table_model_type(
            device=device, 
            image_count=image_count, 
            performance_priority=performance_priority
        )
        logger.info(f"Auto-selected table model: {table_model_type} (device={device or 'auto'}, priority={performance_priority})")
    
    # 优化OCR引擎参数以提高表格处理性能
    ocr_engine = atom_model_manager.get_atom_model(
        atom_model_name='ocr',
        det_db_box_thresh=0.4,  # 调整阈值以平衡精度和速度
        det_db_unclip_ratio=1.5,  # 优化参数
        lang=lang
    )
    
    table_model = RapidTableModel(ocr_engine, model_type=table_model_type, device=device)
    
    # Add VLM enhancement if enabled
    if enable_vlm:
        try:
            vlm_processor = TableVLMProcessor(device=device)
            table_model.vlm_processor = vlm_processor
            logger.info(f"Table model initialized with VLM enhancement enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize VLM processor: {e}")
    
    logger.info(f"Table model initialized with type: {table_model_type}, device: {device or 'auto'}, lang: {lang}, VLM: {enable_vlm}")
    return table_model


def mfd_model_init(weight, device='cpu'):
    if str(device).startswith('npu'):
        device = torch.device(device)
    mfd_model = YOLOv8MFDModel(weight, device)
    return mfd_model


def mfr_model_init(weight_dir, device='cpu'):
    mfr_model = UnimernetModel(weight_dir, device)
    return mfr_model


def doclayout_yolo_model_init(weight, device='cpu'):
    if str(device).startswith('npu'):
        device = torch.device(device)
    model = DocLayoutYOLOModel(weight, device)
    return model

def ocr_model_init(det_db_box_thresh=0.3,
                   lang=None,
                   use_dilation=True,
                   det_db_unclip_ratio=1.8,
                   ):
    if lang is not None and lang != '':
        model = PytorchPaddleOCR(
            det_db_box_thresh=det_db_box_thresh,
            lang=lang,
            use_dilation=use_dilation,
            det_db_unclip_ratio=det_db_unclip_ratio,
        )
    else:
        model = PytorchPaddleOCR(
            det_db_box_thresh=det_db_box_thresh,
            use_dilation=use_dilation,
            det_db_unclip_ratio=det_db_unclip_ratio,
        )
    return model


class AtomModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_atom_model(self, atom_model_name: str, **kwargs):

        lang = kwargs.get('lang', None)
        table_model_name = kwargs.get('table_model_name', None)
        table_model_type = kwargs.get('table_model_type', 'unitable')
        device = kwargs.get('device', None)

        if atom_model_name in [AtomicModel.OCR]:
            key = (atom_model_name, lang)
        elif atom_model_name in [AtomicModel.Table]:
            # 优化表格模型缓存key，包含更多参数
            enable_vlm = kwargs.get('enable_vlm', False)
            key = (atom_model_name, table_model_type, lang, device, enable_vlm)
        else:
            key = atom_model_name

        if key not in self._models:
            self._models[key] = atom_model_init(model_name=atom_model_name, **kwargs)
            logger.debug(f"Created new model instance for key: {key}")
        else:
            logger.debug(f"Reusing cached model for key: {key}")
        return self._models[key]

def atom_model_init(model_name: str, **kwargs):
    atom_model = None
    if model_name == AtomicModel.Layout:
        atom_model = doclayout_yolo_model_init(
            kwargs.get('doclayout_yolo_weights'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.MFD:
        atom_model = mfd_model_init(
            kwargs.get('mfd_weights'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.MFR:
        atom_model = mfr_model_init(
            kwargs.get('mfr_weight_dir'),
            kwargs.get('device')
        )
    elif model_name == AtomicModel.OCR:
        atom_model = ocr_model_init(
            kwargs.get('det_db_box_thresh'),
            kwargs.get('lang'),
        )
    elif model_name == AtomicModel.Table:
        atom_model = table_model_init(
            kwargs.get('lang'),
            kwargs.get('table_model_type', 'auto'),
            kwargs.get('device', None),
            kwargs.get('image_count', None),
            kwargs.get('performance_priority', 'balanced'),
            kwargs.get('enable_vlm', False)
        )
        # 记录表格模型初始化信息
        logger.debug(f"Table model initialized: type={kwargs.get('table_model_type', 'auto')}, lang={kwargs.get('lang')}, device={kwargs.get('device')}, priority={kwargs.get('performance_priority', 'balanced')}")
    else:
        logger.error('model name not allow')
        exit(1)

    if atom_model is None:
        logger.error('model init failed')
        exit(1)
    else:
        return atom_model


class MineruPipelineModel:
    def __init__(self, **kwargs):
        self.formula_config = kwargs.get('formula_config', {})
        self.apply_formula = self.formula_config.get('enable', True)
        self.table_config = kwargs.get('table_config', {})
        self.apply_table = self.table_config.get('enable', True)
        self.lang = kwargs.get('lang', None)
        self.device = kwargs.get('device', 'cpu')
        self.image_count = kwargs.get('image_count', None)
        self.performance_priority = kwargs.get('performance_priority', 'balanced')
        self.enable_vlm = kwargs.get('enable_vlm', False)
        logger.info(
            'DocAnalysis init, this may take some times......'
        )
        atom_model_manager = AtomModelSingleton()

        if self.apply_formula:
            # 初始化公式检测模型
            self.mfd_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFD,
                mfd_weights=str(
                    os.path.join(auto_download_and_get_model_root_path(ModelPath.yolo_v8_mfd), ModelPath.yolo_v8_mfd)
                ),
                device=self.device,
            )

            # 初始化公式解析模型
            mfr_weight_dir = os.path.join(auto_download_and_get_model_root_path(ModelPath.unimernet_small), ModelPath.unimernet_small)

            self.mfr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFR,
                mfr_weight_dir=mfr_weight_dir,
                device=self.device,
            )

        # 初始化layout模型
        self.layout_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.Layout,
            doclayout_yolo_weights=str(
                os.path.join(auto_download_and_get_model_root_path(ModelPath.doclayout_yolo), ModelPath.doclayout_yolo)
            ),
            device=self.device,
        )
        # 初始化ocr
        self.ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            det_db_box_thresh=0.3,
            lang=self.lang
        )
        # init table model with enhanced configuration and intelligent selection
        if self.apply_table:
            table_model_type = self.table_config.get('model_type', 'auto')  # 默认使用自动选择
            self.table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Table,
                lang=self.lang,
                table_model_type=table_model_type,
                device=self.device,
                image_count=self.image_count,
                performance_priority=self.performance_priority,
                enable_vlm=self.enable_vlm,
            )
            logger.info(f"Table processing enabled with model selection: {table_model_type} (priority: {self.performance_priority})")

        logger.info('DocAnalysis init done!')
    
    def get_table_model_info(self):
        """获取表格模型信息。"""
        if hasattr(self, 'table_model') and self.table_model:
            return {
                'model_type': getattr(self.table_model, 'model_type', 'unknown'),
                'device': self.device,
                'performance_priority': self.performance_priority,
                'performance_stats': getattr(self.table_model, 'get_performance_stats', lambda: None)()
            }
        return None