"""AI模型模块

包含历史气候分析、图像生成和区域预测模型。
"""

from .climate_analysis import ClimateAnalysisModel
from .regional_prediction import RegionalPredictionModel
from .base_model import BaseModel

__all__ = [
    "BaseModel",
    "ClimateAnalysisModel",
    "RegionalPredictionModel"
]