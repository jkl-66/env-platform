"""API模块

提供RESTful API接口，包括气候数据分析、图像生成和区域预测等功能。
"""

from .visualization import router as visualization_router
from .prediction import router as prediction_router
from .models import *
from .dependencies import *
from .climate import router as climate_router

__all__ = [
    "climate_router",
    "visualization_router", 
    "prediction_router"
]