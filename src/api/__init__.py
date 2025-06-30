"""API模块

提供RESTful API接口，包括气候数据分析、图像生成和区域预测等功能。
"""

from .climate_api import ClimateAPI
from .visualization_api import VisualizationAPI
from .prediction_api import PredictionAPI
from .models import *
from .dependencies import *

__all__ = [
    "ClimateAPI",
    "VisualizationAPI", 
    "PredictionAPI"
]