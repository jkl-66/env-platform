"""基础模型类

为所有AI模型提供通用接口和功能。
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import pickle
import json
from datetime import datetime

from ..utils.config import get_settings
from ..utils.logger import get_logger

logger = get_logger("base_model")
settings = get_settings()


class BaseModel(ABC):
    """所有AI模型的基类
    
    提供模型的基本接口和通用功能。
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device or settings.DEVICE
        self.model = None
        self.is_trained = False
        self.training_history = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "model_name": model_name,
            "model_type": model_type,
            "device": self.device,
            "version": "1.0.0"
        }
        
        # 确保设备可用
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA不可用，回退到CPU")
            self.device = "cpu"
        
        logger.info(f"初始化模型: {model_name} ({model_type}) on {self.device}")
    
    @abstractmethod
    def build_model(self, **kwargs) -> None:
        """构建模型架构
        
        子类必须实现此方法来定义具体的模型结构。
        """
        pass
    
    @abstractmethod
    def train(
        self,
        train_data: Any,
        validation_data: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """训练模型
        
        Args:
            train_data: 训练数据
            validation_data: 验证数据
            **kwargs: 其他训练参数
            
        Returns:
            训练结果字典
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: Any, **kwargs) -> Any:
        """模型预测
        
        Args:
            input_data: 输入数据
            **kwargs: 其他预测参数
            
        Returns:
            预测结果
        """
        pass
    
    def save_model(self, save_path: Optional[Union[str, Path]] = None) -> Path:
        """保存模型
        
        Args:
            save_path: 保存路径，如果为None则使用默认路径
            
        Returns:
            实际保存路径
        """
        if save_path is None:
            save_path = settings.MODEL_ROOT_PATH / "trained" / f"{self.model_name}.pkl"
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 更新元数据
            self.metadata["saved_at"] = datetime.now().isoformat()
            self.metadata["is_trained"] = self.is_trained
            
            # 保存模型和元数据
            model_data = {
                "model": self.model,
                "metadata": self.metadata,
                "training_history": self.training_history,
                "model_type": self.model_type,
                "device": self.device
            }
            
            with open(save_path, "wb") as f:
                pickle.dump(model_data, f)
            
            # 保存元数据为JSON文件
            metadata_path = save_path.with_suffix(".json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"模型已保存到: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise
    
    def load_model(self, load_path: Union[str, Path]) -> None:
        """加载模型
        
        Args:
            load_path: 模型文件路径
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {load_path}")
        
        try:
            with open(load_path, "rb") as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.metadata = model_data.get("metadata", {})
            self.training_history = model_data.get("training_history", [])
            self.is_trained = model_data.get("metadata", {}).get("is_trained", False)
            
            # 设备兼容性检查
            saved_device = model_data.get("device", "cpu")
            if saved_device != self.device:
                logger.warning(f"模型设备 {saved_device} 与当前设备 {self.device} 不匹配")
                if hasattr(self.model, "to"):
                    self.model = self.model.to(self.device)
            
            logger.info(f"模型已从 {load_path} 加载")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        info = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "is_trained": self.is_trained,
            "metadata": self.metadata.copy()
        }
        
        # 添加模型参数数量（如果是PyTorch模型）
        if hasattr(self.model, "parameters"):
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            info["total_parameters"] = total_params
            info["trainable_parameters"] = trainable_params
        
        return info
    
    def validate_input(self, input_data: Any) -> bool:
        """验证输入数据
        
        Args:
            input_data: 输入数据
            
        Returns:
            验证是否通过
        """
        # 基础验证，子类可以重写
        if input_data is None:
            logger.error("输入数据不能为None")
            return False
        
        return True
    
    def preprocess_input(self, input_data: Any) -> Any:
        """预处理输入数据
        
        Args:
            input_data: 原始输入数据
            
        Returns:
            预处理后的数据
        """
        # 默认不做处理，子类可以重写
        return input_data
    
    def postprocess_output(self, output_data: Any) -> Any:
        """后处理输出数据
        
        Args:
            output_data: 模型原始输出
            
        Returns:
            后处理后的输出
        """
        # 默认不做处理，子类可以重写
        return output_data
    
    def add_training_record(self, epoch: int, metrics: Dict[str, float]) -> None:
        """添加训练记录
        
        Args:
            epoch: 训练轮次
            metrics: 指标字典
        """
        record = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.copy()
        }
        self.training_history.append(record)
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """获取训练历史
        
        Returns:
            训练历史记录列表
        """
        return self.training_history.copy()
    
    def reset_model(self) -> None:
        """重置模型状态"""
        self.model = None
        self.is_trained = False
        self.training_history = []
        logger.info(f"模型 {self.model_name} 已重置")
    
    def set_eval_mode(self) -> None:
        """设置模型为评估模式"""
        if hasattr(self.model, "eval"):
            self.model.eval()
    
    def set_train_mode(self) -> None:
        """设置模型为训练模式"""
        if hasattr(self.model, "train"):
            self.model.train()
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name}, type={self.model_type}, device={self.device})"
    
    def __repr__(self) -> str:
        return self.__str__()


class PyTorchBaseModel(BaseModel):
    """PyTorch模型基类
    
    为PyTorch模型提供额外的通用功能。
    """
    
    def __init__(self, model_name: str, model_type: str, device: Optional[str] = None):
        super().__init__(model_name, model_type, device)
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
    
    def setup_training(
        self,
        optimizer_class=torch.optim.Adam,
        optimizer_params: Optional[Dict] = None,
        criterion_class=nn.MSELoss,
        criterion_params: Optional[Dict] = None,
        scheduler_class=None,
        scheduler_params: Optional[Dict] = None
    ) -> None:
        """设置训练组件
        
        Args:
            optimizer_class: 优化器类
            optimizer_params: 优化器参数
            criterion_class: 损失函数类
            criterion_params: 损失函数参数
            scheduler_class: 学习率调度器类
            scheduler_params: 调度器参数
        """
        if self.model is None:
            raise RuntimeError("请先构建模型")
        
        # 设置优化器
        optimizer_params = optimizer_params or {"lr": 0.001}
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        
        # 设置损失函数
        criterion_params = criterion_params or {}
        self.criterion = criterion_class(**criterion_params)
        
        # 设置学习率调度器
        if scheduler_class:
            scheduler_params = scheduler_params or {}
            self.scheduler = scheduler_class(self.optimizer, **scheduler_params)
        
        logger.info("训练组件设置完成")
    
    def save_checkpoint(
        self,
        epoch: int,
        save_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """保存训练检查点
        
        Args:
            epoch: 当前训练轮次
            save_path: 保存路径
            
        Returns:
            检查点文件路径
        """
        if save_path is None:
            save_path = settings.MODEL_ROOT_PATH / "checkpoints" / f"{self.model_name}_epoch_{epoch}.pth"
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "training_history": self.training_history,
            "metadata": self.metadata
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"检查点已保存到: {save_path}")
        
        return save_path
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """加载训练检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            加载的训练轮次
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型状态
        if self.model:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # 加载优化器状态
        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # 加载调度器状态
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # 加载训练历史
        self.training_history = checkpoint.get("training_history", [])
        self.metadata.update(checkpoint.get("metadata", {}))
        
        epoch = checkpoint["epoch"]
        logger.info(f"检查点已从 {checkpoint_path} 加载，轮次: {epoch}")
        
        return epoch