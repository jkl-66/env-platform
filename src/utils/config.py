"""系统配置管理

使用Pydantic Settings管理应用配置，支持环境变量覆盖。
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from pathlib import Path
import os


class Settings(BaseSettings):
    """应用配置类"""
    
    # 基础配置
    APP_NAME: str = "气候数据分析与生态警示系统"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=True, description="调试模式")
    
    # 服务器配置
    HOST: str = Field(default="0.0.0.0", description="服务器主机")
    PORT: int = Field(default=8000, description="服务器端口")
    WORKERS: int = Field(default=1, description="工作进程数")
    
    # CORS配置
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="允许的跨域源"
    )
    
    # 数据库配置
    # MySQL (元数据)
    MYSQL_HOST: str = Field(default="localhost", description="MySQL主机")
    MYSQL_PORT: int = Field(default=3306, description="MySQL端口")
    MYSQL_DB: str = Field(default="climate_metadata", description="MySQL数据库名")
    MYSQL_USER: str = Field(default="jkl", description="MySQL用户名")
    MYSQL_PASSWORD: str = Field(default="922920", description="MySQL密码")
    
    # Kafka配置
    KAFKA_BOOTSTRAP_SERVERS: List[str] = Field(
        default=["localhost:9092"],
        description="Kafka服务器列表"
    )
    KAFKA_TOPIC_CLIMATE_DATA: str = Field(
        default="climate-data",
        description="气候数据主题"
    )
    
    # 文件存储配置
    DATA_ROOT_PATH: Path = Field(
        default=Path("e:/py codebase/env-platform/data"),
        description="数据根目录"
    )
    MODEL_ROOT_PATH: Path = Field(
        default=Path("models"),
        description="模型根目录"
    )
    
    @property
    def data_storage_path(self) -> str:
        """数据存储路径"""
        return str(self.DATA_ROOT_PATH)
    UPLOAD_MAX_SIZE: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        description="最大上传文件大小"
    )
    
    # AI模型配置
    # 设备配置
    DEVICE: str = Field(default="cpu", description="计算设备 (cpu/cuda)")
    
    # 图像生成模型
    DIFFUSION_MODEL_PATH: str = Field(
        default="runwayml/stable-diffusion-v1-5",
        description="扩散模型路径"
    )
    GAN_MODEL_PATH: Optional[str] = Field(
        default=None,
        description="GAN模型路径"
    )
    
    # 时间序列模型
    PROPHET_MODEL_PATH: Optional[str] = Field(
        default=None,
        description="Prophet模型路径"
    )
    LSTM_MODEL_PATH: Optional[str] = Field(
        default=None,
        description="LSTM模型路径"
    )
    
    # 预测模型
    XGBOOST_MODEL_PATH: Optional[str] = Field(
        default=None,
        description="XGBoost模型路径"
    )
    
    # API配置
    API_V1_PREFIX: str = Field(default="/api", description="API v1前缀")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="访问令牌过期时间(分钟)"
    )
    
    # 日志配置
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    LOG_FILE: Optional[str] = Field(default="logs/app.log", description="日志文件路径")
    
    # 外部API配置
    NOAA_API_KEY: Optional[str] = Field(default=None, description="NOAA API密钥")
    ECMWF_API_KEY: Optional[str] = Field(default=None, description="ECMWF API密钥")
    
    # 安全配置
    SECRET_KEY: str = Field(default="your_secret_key_here_change_in_production", description="应用密钥")
    
    # 开发环境配置
    ENABLE_DOCS: bool = Field(default=True, description="启用API文档")
    ENABLE_DEBUG_TOOLBAR: bool = Field(default=True, description="启用调试工具栏")
    
    # 性能配置
    MAX_WORKERS: int = Field(default=4, description="最大工作线程数")
    REQUEST_TIMEOUT: int = Field(default=300, description="请求超时时间(秒)")
    CACHE_TTL: int = Field(default=3600, description="缓存过期时间(秒)")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = 'ignore'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保数据目录存在
        self.DATA_ROOT_PATH.mkdir(parents=True, exist_ok=True)
        self.MODEL_ROOT_PATH.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.DATA_ROOT_PATH / "raw").mkdir(exist_ok=True)
        (self.DATA_ROOT_PATH / "processed").mkdir(exist_ok=True)
        (self.MODEL_ROOT_PATH / "trained").mkdir(exist_ok=True)
    
    @property
    def mysql_url(self) -> str:
        """MySQL连接URL"""
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"


# 全局设置实例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局设置实例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# 向后兼容的别名
get_config = get_settings


def load_config() -> Settings:
    """加载配置（向后兼容）"""
    return get_settings()


def reload_settings() -> Settings:
    """重新加载设置"""
    global _settings
    _settings = Settings()
    return _settings


# 导出全局settings实例
settings = get_settings()