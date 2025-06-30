"""日志管理模块

提供统一的日志配置和管理功能。
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
# 延迟导入以避免循环依赖


class InterceptHandler(logging.Handler):
    """拦截标准库日志并重定向到loguru"""
    
    def emit(self, record):
        # 获取对应的loguru级别
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # 查找调用者
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger(name: Optional[str] = None) -> logger:
    """设置日志配置
    
    Args:
        name: 日志器名称
        
    Returns:
        配置好的loguru logger实例
    """
    from .config import get_settings
    settings = get_settings()
    
    # 移除默认处理器
    logger.remove()
    
    # 控制台输出配置
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stdout,
        format=console_format,
        level=settings.LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 文件输出配置
    if settings.LOG_FILE:
        log_file_path = Path(settings.LOG_FILE)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        # 普通日志文件
        logger.add(
            log_file_path,
            format=file_format,
            level=settings.LOG_LEVEL,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
        
        # 错误日志文件
        error_log_path = log_file_path.parent / f"error_{log_file_path.name}"
        logger.add(
            error_log_path,
            format=file_format,
            level="ERROR",
            rotation="10 MB",
            retention="90 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    # 拦截标准库日志
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # 设置第三方库日志级别
    for logger_name in [
        "uvicorn",
        "uvicorn.error", 
        "uvicorn.access",
        "fastapi",
        "sqlalchemy",
        "asyncio"
    ]:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]
    
    if name:
        return logger.bind(name=name)
    
    return logger


def get_logger(name: str) -> logger:
    """获取指定名称的日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        绑定了名称的logger实例
    """
    return logger.bind(name=name)


# 预定义的日志器
api_logger = get_logger("api")
model_logger = get_logger("model")
data_logger = get_logger("data")
db_logger = get_logger("database")