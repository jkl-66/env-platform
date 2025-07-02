#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习模型管理器

负责气候预测模型的训练、评估、部署和管理。
"""

import asyncio
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple, Type
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier

# 深度学习库（可选）
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    warnings.warn("TensorFlow未安装，深度学习功能受限")

try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    warnings.warn("PyTorch未安装，深度学习功能受限")

from src.utils.logger import get_logger
from src.utils.config import get_config
from src.data_processing.data_storage import DataStorage

logger = get_logger(__name__)
config = get_config()


class ModelType(Enum):
    """模型类型"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIME_SERIES = "time_series"
    DEEP_LEARNING = "deep_learning"


class ModelStatus(Enum):
    """模型状态"""
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    model_type: str
    algorithm: str
    parameters: Dict[str, Any]
    features: List[str]
    target: str
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    random_state: int = 42
    

@dataclass
class ModelMetrics:
    """模型评估指标"""
    model_id: str
    model_type: str
    
    # 回归指标
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    
    # 分类指标
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    
    # 交叉验证指标
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # 其他信息
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ModelInfo:
    """模型信息"""
    id: str
    name: str
    model_type: str
    algorithm: str
    status: str
    config: ModelConfig
    metrics: Optional[ModelMetrics] = None
    file_path: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class ModelManager:
    """机器学习模型管理器"""
    
    def __init__(self, storage: Optional[DataStorage] = None):
        self.storage = storage or DataStorage()
        self.models_path = Path(getattr(config, 'MODEL_ROOT_PATH', Path('models')))
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # 模型注册表
        self.model_registry: Dict[str, ModelInfo] = {}
        
        # 支持的算法
        self.supported_algorithms = {
            ModelType.REGRESSION: {
                "linear_regression": LinearRegression,
                "random_forest": RandomForestRegressor,
                "gradient_boosting": GradientBoostingRegressor,
                "svr": SVR,
                "mlp": MLPRegressor
            },
            ModelType.CLASSIFICATION: {
                "logistic_regression": LogisticRegression,
                "random_forest": RandomForestClassifier,
                "gradient_boosting": GradientBoostingClassifier,
                "svc": SVC,
                "mlp": MLPClassifier
            }
        }
        
        # 加载已有模型
        self._load_model_registry()
    
    async def create_model(
        self,
        config: ModelConfig,
        data: pd.DataFrame
    ) -> str:
        """创建并训练模型"""
        try:
            logger.info(f"开始创建模型: {config.name}")
            
            # 生成模型ID
            model_id = self._generate_model_id(config.name)
            
            # 创建模型信息
            model_info = ModelInfo(
                id=model_id,
                name=config.name,
                model_type=config.model_type,
                algorithm=config.algorithm,
                status=ModelStatus.TRAINING.value,
                config=config
            )
            
            # 注册模型
            self.model_registry[model_id] = model_info
            
            # 准备数据
            X, y = self._prepare_training_data(data, config)
            
            # 训练模型
            model, metrics = await self._train_model(config, X, y)
            
            # 保存模型
            model_path = await self._save_model(model_id, model, config)
            
            # 更新模型信息
            model_info.status = ModelStatus.TRAINED.value
            model_info.metrics = metrics
            model_info.file_path = model_path
            model_info.updated_at = datetime.now()
            
            # 保存模型注册表
            await self._save_model_registry()
            
            logger.info(f"模型创建完成: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"创建模型失败: {e}")
            if model_id in self.model_registry:
                self.model_registry[model_id].status = ModelStatus.FAILED.value
            raise
    
    async def predict(
        self,
        model_id: str,
        data: pd.DataFrame
    ) -> np.ndarray:
        """使用模型进行预测"""
        try:
            logger.info(f"使用模型预测: {model_id}")
            
            # 获取模型信息
            model_info = self.model_registry.get(model_id)
            if not model_info:
                raise ValueError(f"模型不存在: {model_id}")
            
            if model_info.status != ModelStatus.TRAINED.value:
                raise ValueError(f"模型状态不正确: {model_info.status}")
            
            # 加载模型
            model = await self._load_model(model_id)
            
            # 准备预测数据
            X = self._prepare_prediction_data(data, model_info.config)
            
            # 执行预测
            start_time = datetime.now()
            predictions = model.predict(X)
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            # 更新预测时间统计
            if model_info.metrics:
                model_info.metrics.prediction_time = prediction_time
            
            logger.info(f"预测完成: {len(predictions)} 个样本")
            return predictions
            
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            raise
    
    async def evaluate_model(
        self,
        model_id: str,
        test_data: pd.DataFrame
    ) -> ModelMetrics:
        """评估模型性能"""
        try:
            logger.info(f"评估模型: {model_id}")
            
            # 获取模型信息
            model_info = self.model_registry.get(model_id)
            if not model_info:
                raise ValueError(f"模型不存在: {model_id}")
            
            # 更新状态
            model_info.status = ModelStatus.EVALUATING.value
            
            # 加载模型
            model = await self._load_model(model_id)
            
            # 准备测试数据
            X_test, y_test = self._prepare_training_data(test_data, model_info.config)
            
            # 执行预测
            y_pred = model.predict(X_test)
            
            # 计算评估指标
            metrics = self._calculate_metrics(
                model_id,
                model_info.config.model_type,
                y_test,
                y_pred,
                model
            )
            
            # 更新模型信息
            model_info.metrics = metrics
            model_info.status = ModelStatus.TRAINED.value
            model_info.updated_at = datetime.now()
            
            # 保存更新
            await self._save_model_registry()
            
            logger.info(f"模型评估完成: {model_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            if model_id in self.model_registry:
                self.model_registry[model_id].status = ModelStatus.FAILED.value
            raise
    
    async def deploy_model(self, model_id: str) -> bool:
        """部署模型"""
        try:
            logger.info(f"部署模型: {model_id}")
            
            # 获取模型信息
            model_info = self.model_registry.get(model_id)
            if not model_info:
                raise ValueError(f"模型不存在: {model_id}")
            
            if model_info.status != ModelStatus.TRAINED.value:
                raise ValueError(f"模型状态不正确，无法部署: {model_info.status}")
            
            # 验证模型文件存在
            if not model_info.file_path or not Path(model_info.file_path).exists():
                raise ValueError("模型文件不存在")
            
            # 更新状态
            model_info.status = ModelStatus.DEPLOYED.value
            model_info.updated_at = datetime.now()
            
            # 保存更新
            await self._save_model_registry()
            
            logger.info(f"模型部署完成: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"模型部署失败: {e}")
            return False
    
    async def delete_model(self, model_id: str) -> bool:
        """删除模型"""
        try:
            logger.info(f"删除模型: {model_id}")
            
            # 获取模型信息
            model_info = self.model_registry.get(model_id)
            if not model_info:
                return False
            
            # 删除模型文件
            if model_info.file_path:
                model_path = Path(model_info.file_path)
                if model_path.exists():
                    model_path.unlink()
            
            # 从注册表删除
            del self.model_registry[model_id]
            
            # 保存更新
            await self._save_model_registry()
            
            logger.info(f"模型删除完成: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除模型失败: {e}")
            return False
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[ModelInfo]:
        """列出模型"""
        models = list(self.model_registry.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if status:
            models = [m for m in models if m.status == status]
        
        # 按创建时间排序
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        return models
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self.model_registry.get(model_id)
    
    async def hyperparameter_tuning(
        self,
        config: ModelConfig,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        cv_folds: int = 5
    ) -> Tuple[str, Dict[str, Any]]:
        """超参数调优"""
        try:
            logger.info(f"开始超参数调优: {config.name}")
            
            # 准备数据
            X, y = self._prepare_training_data(data, config)
            
            # 获取基础模型
            model_class = self._get_model_class(config.model_type, config.algorithm)
            base_model = model_class(**config.parameters)
            
            # 网格搜索
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring='neg_mean_squared_error' if config.model_type == ModelType.REGRESSION.value else 'accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # 创建优化后的配置
            optimized_config = ModelConfig(
                name=f"{config.name}_optimized",
                model_type=config.model_type,
                algorithm=config.algorithm,
                parameters=grid_search.best_params_,
                features=config.features,
                target=config.target,
                validation_split=config.validation_split,
                cross_validation_folds=config.cross_validation_folds,
                random_state=config.random_state
            )
            
            # 创建优化后的模型
            model_id = await self.create_model(optimized_config, data)
            
            logger.info(f"超参数调优完成: {model_id}")
            logger.info(f"最佳参数: {grid_search.best_params_}")
            logger.info(f"最佳得分: {grid_search.best_score_}")
            
            return model_id, grid_search.best_params_
            
        except Exception as e:
            logger.error(f"超参数调优失败: {e}")
            raise
    
    def _generate_model_id(self, name: str) -> str:
        """生成模型ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{timestamp}"
    
    def _prepare_training_data(
        self,
        data: pd.DataFrame,
        config: ModelConfig
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
        # 检查特征列
        missing_features = [f for f in config.features if f not in data.columns]
        if missing_features:
            raise ValueError(f"缺少特征列: {missing_features}")
        
        # 检查目标列
        if config.target not in data.columns:
            raise ValueError(f"缺少目标列: {config.target}")
        
        # 提取特征和目标
        X = data[config.features].copy()
        y = data[config.target].copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        y = y.fillna(y.mean() if y.dtype in ['float64', 'int64'] else y.mode()[0])
        
        return X, y
    
    def _prepare_prediction_data(
        self,
        data: pd.DataFrame,
        config: ModelConfig
    ) -> pd.DataFrame:
        """准备预测数据"""
        # 检查特征列
        missing_features = [f for f in config.features if f not in data.columns]
        if missing_features:
            raise ValueError(f"缺少特征列: {missing_features}")
        
        # 提取特征
        X = data[config.features].copy()
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        return X
    
    async def _train_model(
        self,
        config: ModelConfig,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[Any, ModelMetrics]:
        """训练模型"""
        logger.info(f"训练模型: {config.algorithm}")
        
        start_time = datetime.now()
        
        # 获取模型类
        model_class = self._get_model_class(config.model_type, config.algorithm)
        
        # 创建模型实例
        model = model_class(**config.parameters)
        
        # 分割数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=config.validation_split,
            random_state=config.random_state
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 验证集预测
        y_pred = model.predict(X_val)
        
        # 交叉验证
        cv_scores = cross_val_score(
            model, X, y,
            cv=config.cross_validation_folds,
            scoring='neg_mean_squared_error' if config.model_type == ModelType.REGRESSION.value else 'accuracy'
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # 计算评估指标
        metrics = self._calculate_metrics(
            "",  # 临时ID
            config.model_type,
            y_val,
            y_pred,
            model,
            cv_scores=cv_scores,
            training_time=training_time
        )
        
        return model, metrics
    
    def _get_model_class(self, model_type: str, algorithm: str) -> Type:
        """获取模型类"""
        model_type_enum = ModelType(model_type)
        
        if model_type_enum not in self.supported_algorithms:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        algorithms = self.supported_algorithms[model_type_enum]
        
        if algorithm not in algorithms:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        return algorithms[algorithm]
    
    def _calculate_metrics(
        self,
        model_id: str,
        model_type: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Any,
        cv_scores: Optional[np.ndarray] = None,
        training_time: Optional[float] = None
    ) -> ModelMetrics:
        """计算评估指标"""
        metrics = ModelMetrics(
            model_id=model_id,
            model_type=model_type,
            training_time=training_time
        )
        
        if model_type == ModelType.REGRESSION.value:
            # 回归指标
            metrics.mse = mean_squared_error(y_true, y_pred)
            metrics.rmse = np.sqrt(metrics.mse)
            metrics.mae = mean_absolute_error(y_true, y_pred)
            metrics.r2 = r2_score(y_true, y_pred)
        
        elif model_type == ModelType.CLASSIFICATION.value:
            # 分类指标
            metrics.accuracy = accuracy_score(y_true, y_pred)
            metrics.precision = precision_score(y_true, y_pred, average='weighted')
            metrics.recall = recall_score(y_true, y_pred, average='weighted')
            metrics.f1 = f1_score(y_true, y_pred, average='weighted')
        
        # 交叉验证指标
        if cv_scores is not None:
            metrics.cv_scores = cv_scores.tolist()
            metrics.cv_mean = cv_scores.mean()
            metrics.cv_std = cv_scores.std()
        
        # 特征重要性
        if hasattr(model, 'feature_importances_'):
            feature_names = getattr(model, 'feature_names_in_', None)
            if feature_names is not None:
                metrics.feature_importance = dict(
                    zip(feature_names, model.feature_importances_)
                )
        
        return metrics
    
    async def _save_model(
        self,
        model_id: str,
        model: Any,
        config: ModelConfig
    ) -> str:
        """保存模型"""
        model_path = self.models_path / f"{model_id}.joblib"
        
        # 保存模型
        joblib.dump(model, model_path)
        
        # 保存配置
        config_path = self.models_path / f"{model_id}_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存: {model_path}")
        return str(model_path)
    
    async def _load_model(self, model_id: str) -> Any:
        """加载模型"""
        model_info = self.model_registry.get(model_id)
        if not model_info or not model_info.file_path:
            raise ValueError(f"模型文件路径不存在: {model_id}")
        
        model_path = Path(model_info.file_path)
        if not model_path.exists():
            raise ValueError(f"模型文件不存在: {model_path}")
        
        return joblib.load(model_path)
    
    def _load_model_registry(self):
        """加载模型注册表"""
        registry_path = self.models_path / "model_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                
                for model_id, model_data in registry_data.items():
                    # 重建ModelConfig
                    config_data = model_data['config']
                    config = ModelConfig(**config_data)
                    
                    # 重建ModelMetrics
                    metrics = None
                    if model_data.get('metrics'):
                        metrics_data = model_data['metrics']
                        if 'created_at' in metrics_data:
                            metrics_data['created_at'] = datetime.fromisoformat(
                                metrics_data['created_at']
                            )
                        metrics = ModelMetrics(**metrics_data)
                    
                    # 重建ModelInfo
                    model_info = ModelInfo(
                        id=model_data['id'],
                        name=model_data['name'],
                        model_type=model_data['model_type'],
                        algorithm=model_data['algorithm'],
                        status=model_data['status'],
                        config=config,
                        metrics=metrics,
                        file_path=model_data.get('file_path'),
                        created_at=datetime.fromisoformat(model_data['created_at']),
                        updated_at=datetime.fromisoformat(model_data['updated_at'])
                    )
                    
                    self.model_registry[model_id] = model_info
                
                logger.info(f"加载了 {len(self.model_registry)} 个模型")
                
            except Exception as e:
                logger.error(f"加载模型注册表失败: {e}")
    
    async def _save_model_registry(self):
        """保存模型注册表"""
        registry_path = self.models_path / "model_registry.json"
        
        try:
            registry_data = {}
            
            for model_id, model_info in self.model_registry.items():
                # 序列化ModelInfo
                model_data = {
                    'id': model_info.id,
                    'name': model_info.name,
                    'model_type': model_info.model_type,
                    'algorithm': model_info.algorithm,
                    'status': model_info.status,
                    'config': asdict(model_info.config),
                    'file_path': model_info.file_path,
                    'created_at': model_info.created_at.isoformat(),
                    'updated_at': model_info.updated_at.isoformat()
                }
                
                # 序列化ModelMetrics
                if model_info.metrics:
                    metrics_data = asdict(model_info.metrics)
                    if metrics_data.get('created_at'):
                        metrics_data['created_at'] = model_info.metrics.created_at.isoformat()
                    model_data['metrics'] = metrics_data
                
                registry_data[model_id] = model_data
            
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            
            logger.info("模型注册表已保存")
            
        except Exception as e:
            logger.error(f"保存模型注册表失败: {e}")
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        stats = {
            "total_models": len(self.model_registry),
            "by_type": {},
            "by_status": {},
            "by_algorithm": {},
            "performance_summary": {}
        }
        
        for model_info in self.model_registry.values():
            # 按类型统计
            model_type = model_info.model_type
            stats["by_type"][model_type] = stats["by_type"].get(model_type, 0) + 1
            
            # 按状态统计
            status = model_info.status
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            
            # 按算法统计
            algorithm = model_info.algorithm
            stats["by_algorithm"][algorithm] = stats["by_algorithm"].get(algorithm, 0) + 1
            
            # 性能统计
            if model_info.metrics:
                metrics = model_info.metrics
                if model_type not in stats["performance_summary"]:
                    stats["performance_summary"][model_type] = {
                        "count": 0,
                        "avg_training_time": 0,
                        "best_score": None
                    }
                
                perf = stats["performance_summary"][model_type]
                perf["count"] += 1
                
                if metrics.training_time:
                    perf["avg_training_time"] = (
                        (perf["avg_training_time"] * (perf["count"] - 1) + metrics.training_time) / 
                        perf["count"]
                    )
                
                # 最佳得分
                score = None
                if model_type == ModelType.REGRESSION.value and metrics.r2:
                    score = metrics.r2
                elif model_type == ModelType.CLASSIFICATION.value and metrics.accuracy:
                    score = metrics.accuracy
                
                if score and (perf["best_score"] is None or score > perf["best_score"]):
                    perf["best_score"] = score
        
        return stats


# 便捷函数
async def create_climate_model(
    name: str,
    model_type: str,
    algorithm: str,
    features: List[str],
    target: str,
    data: pd.DataFrame,
    parameters: Optional[Dict[str, Any]] = None,
    storage: Optional[DataStorage] = None
) -> str:
    """创建气候预测模型的便捷函数"""
    config = ModelConfig(
        name=name,
        model_type=model_type,
        algorithm=algorithm,
        parameters=parameters or {},
        features=features,
        target=target
    )
    
    manager = ModelManager(storage)
    return await manager.create_model(config, data)