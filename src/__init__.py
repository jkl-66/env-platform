"""气候数据分析与生态警示系统

这是一个基于AI的气候数据分析与生态警示图像生成系统的主包。
"""

__version__ = "1.0.0"
__author__ = "Climate Research Team"
__email__ = "team@climate-research.org"

# 导入核心模块
from . import data_processing
from . import models
from . import api
from . import utils

__all__ = [
    "data_processing",
    "models", 
    "api",
    "utils"
]