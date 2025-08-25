import os
import html
import cv2
import numpy as np
import torch
from loguru import logger
from rapid_table import RapidTable, RapidTableInput

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


def get_device():
    """
    Auto-detect the best available device.
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def escape_html(input_string):
    """Escape HTML Entities."""
    return html.escape(input_string)


class RapidTableModel(object):
    def __init__(self, ocr_engine, model_type='unitable', device=None):
        """
        Initialize RapidTableModel with configurable model type.
        
        Args:
            ocr_engine: OCR engine instance
            model_type: 'unitable', 'slanet_plus', 'ppstructure_en', 'ppstructure_zh'
            device: 'mps', 'cuda', 'cpu' (for unitable). If None, auto-detect best device.
        """
        # Auto-detect device if not specified
        if device is None:
            device = get_device()
        
        logger.info(f"Using device: {device} for table model")
        
        if model_type == 'unitable':
            # Unitable model - highest accuracy, PyTorch-based
            input_args = RapidTableInput(model_type='unitable', device=device)
        else:
            # Legacy SLANet+ model - ONNX-based
            slanet_plus_model_path = os.path.join(auto_download_and_get_model_root_path(ModelPath.slanet_plus), ModelPath.slanet_plus)
            input_args = RapidTableInput(model_type=model_type, model_path=slanet_plus_model_path)
        
        self.table_model = RapidTable(input_args)
        self.ocr_engine = ocr_engine
        self.model_type = model_type
        
        # 添加性能监控
        self._prediction_count = 0
        self._total_prediction_time = 0
        
        # 优化设置
        self._max_image_size = 2048  # 限制最大图像尺寸以控制内存使用


    def predict(self, image):
        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        # 增强的表格旋转检测算法
        img_height, img_width = bgr_image.shape[:2]
        img_aspect_ratio = img_height / img_width if img_width > 0 else 1.0
        img_is_portrait = img_aspect_ratio > 1.2

        rotation_angle = 0  # 记录旋转角度
        
        if img_is_portrait:
            # 使用更优化的OCR检测参数
            det_res = self.ocr_engine.ocr(bgr_image, rec=False)[0]
            
            is_rotated, rotation_angle = self._enhanced_rotation_detection(det_res, img_width, img_height)
            
            # 根据检测结果旋转图像
            if is_rotated:
                if rotation_angle == 90:
                    image = cv2.rotate(np.asarray(image), cv2.ROTATE_90_CLOCKWISE)
                elif rotation_angle == 270:
                    image = cv2.rotate(np.asarray(image), cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation_angle == 180:
                    image = cv2.rotate(np.asarray(image), cv2.ROTATE_180)
                
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                logger.debug(f"Table rotated by {rotation_angle} degrees")

        # Continue with OCR on potentially rotated image
        import time
        start_time = time.time()
        
        try:
            ocr_result = self.ocr_engine.ocr(bgr_image)[0]
            if ocr_result:
                ocr_result = [[item[0], escape_html(item[1][0]), item[1][1]] for item in ocr_result if
                          len(item) == 2 and isinstance(item[1], tuple)]
            else:
                ocr_result = None

            if ocr_result:
                # 优化的表格预测，添加预处理和后处理
                processed_image = self._preprocess_table_image(np.asarray(image))
                table_results = self.table_model(processed_image, ocr_result)
                
                html_code = table_results.pred_html
                table_cell_bboxes = table_results.cell_bboxes
                logic_points = table_results.logic_points
                elapse = table_results.elapse
                
                # 更新性能统计
                prediction_time = time.time() - start_time
                self._prediction_count += 1
                self._total_prediction_time += prediction_time
                
                if self._prediction_count % 10 == 0:  # 每10次预测输出一次统计
                    avg_time = self._total_prediction_time / self._prediction_count
                    logger.debug(f"Table prediction stats: avg_time={avg_time:.3f}s, count={self._prediction_count}")
                
                return html_code, table_cell_bboxes, logic_points, elapse
                
        except Exception as e:
            logger.exception(f"Table prediction error: {e}")

        return None, None, None, None
    
    def _enhanced_rotation_detection(self, det_res, img_width, img_height):
        """增强的表格旋转检测算法"""
        if not det_res or len(det_res) < 3:  # 需要足够的文本框来分析
            return False, 0
        
        vertical_count = 0
        horizontal_count = 0
        diagonal_count = 0
        total_boxes = len(det_res)
        
        # 分析文本框的方向和分布
        text_directions = []
        
        for box_ocr_res in det_res:
            try:
                p1, p2, p3, p4 = box_ocr_res
                
                # 计算文本框的主轴方向
                # 使用对角线向量来确定方向
                dx = p3[0] - p1[0]  # 主对角线的x差值
                dy = p3[1] - p1[1]  # 主对角线的y差值
                
                # 计算角度（弧度转角度）
                angle = np.arctan2(dy, dx) * 180 / np.pi
                if angle < 0:
                    angle += 180  # 角度正规化到0-180度
                
                text_directions.append(angle)
                
                # 统计不同方向的文本框
                if 75 <= angle <= 105:  # 垂直文本
                    vertical_count += 1
                elif angle <= 15 or angle >= 165:  # 水平文本
                    horizontal_count += 1
                else:  # 斜向文本
                    diagonal_count += 1
                    
            except (IndexError, ValueError) as e:
                logger.debug(f"Error processing text box: {e}")
                continue
        
        # 改进的旋转判断逻辑
        if total_boxes == 0:
            return False, 0
            
        vertical_ratio = vertical_count / total_boxes
        horizontal_ratio = horizontal_count / total_boxes
        
        # 优化的判断条件
        if vertical_ratio >= 0.4:  # 垂直文本占比较高，可能需要90度旋转
            # 进一步分析主要角度分布
            if text_directions:
                avg_angle = np.mean(text_directions)
                if 75 <= avg_angle <= 105:
                    return True, 90
        elif horizontal_ratio >= 0.6 and vertical_ratio < 0.2:
            # 水平文本占主导，但图像是竖向，可能不需要旋转
            return False, 0
        
        # 可以添加更多复杂的旋转检测逻辑，比如18 0度、270度等
        
        return False, 0
    
    def _preprocess_table_image(self, image):
        """预处理表格图像以提高识别精度"""
        try:
            # 基本的图像增强处理
            if len(image.shape) == 3:
                # 转换为灰度图进行对比度增强
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                # 自适应直方图均衡化提高对比度
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                # 转回 RGB格式
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                return enhanced_rgb
            else:
                return image
        except Exception as e:
            logger.debug(f"Image preprocessing failed, using original: {e}")
            return image
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if self._prediction_count > 0:
            avg_time = self._total_prediction_time / self._prediction_count
            return {
                'total_predictions': self._prediction_count,
                'total_time': self._total_prediction_time,
                'average_time': avg_time,
                'model_type': self.model_type
            }
        return None
