#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气候预测引擎

负责执行气候预测任务，包括时间序列预测、空间预测和集成预测。
"""

import asyncio
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# 时间序列预测
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels未安装，时间序列预测功能受限")

# 深度学习时间序列
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# 科学计算
from scipy import interpolate, stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils.logger import get_logger
from src.utils.config import get_config
from src.data_processing.data_storage import DataStorage
from src.ml.model_manager import ModelManager, ModelInfo

logger = get_logger(__name__)
config = get_config()


class PredictionType(Enum):
    """预测类型"""
    TIME_SERIES = "time_series"
    SPATIAL = "spatial"
    ENSEMBLE = "ensemble"
    REAL_TIME = "real_time"


class PredictionStatus(Enum):
    """预测状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PredictionConfig:
    """预测配置"""
    prediction_type: str
    target_variable: str
    prediction_horizon: int  # 预测时间步数
    spatial_resolution: Optional[float] = None  # 空间分辨率（度）
    temporal_resolution: str = "daily"  # 时间分辨率
    confidence_interval: float = 0.95  # 置信区间
    ensemble_methods: Optional[List[str]] = None  # 集成方法
    preprocessing_steps: Optional[List[str]] = None  # 预处理步骤
    

@dataclass
class PredictionResult:
    """预测结果"""
    prediction_id: str
    config: PredictionConfig
    predictions: np.ndarray
    confidence_lower: Optional[np.ndarray] = None
    confidence_upper: Optional[np.ndarray] = None
    timestamps: Optional[List[datetime]] = None
    coordinates: Optional[Dict[str, np.ndarray]] = None  # lat, lon
    metadata: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class PredictionTask:
    """预测任务"""
    task_id: str
    config: PredictionConfig
    status: str
    model_ids: List[str]
    input_data_path: Optional[str] = None
    output_path: Optional[str] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    result: Optional[PredictionResult] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class PredictionEngine:
    """气候预测引擎"""
    
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        storage: Optional[DataStorage] = None
    ):
        self.model_manager = model_manager or ModelManager()
        self.storage = storage or DataStorage()
        self.predictions_path = Path(getattr(config, 'DATA_ROOT_PATH', Path('data')) / 'predictions')
        self.predictions_path.mkdir(parents=True, exist_ok=True)
        
        # 任务注册表
        self.tasks: Dict[str, PredictionTask] = {}
        
        # 预处理器
        self.scalers: Dict[str, Any] = {}
        
        # 加载已有任务
        self._load_tasks()
    
    async def create_prediction_task(
        self,
        config: PredictionConfig,
        model_ids: List[str],
        input_data: Union[pd.DataFrame, xr.Dataset, str]
    ) -> str:
        """创建预测任务"""
        try:
            logger.info(f"创建预测任务: {config.target_variable}")
            
            # 生成任务ID
            task_id = self._generate_task_id(config.target_variable)
            
            # 验证模型
            await self._validate_models(model_ids)
            
            # 保存输入数据
            input_data_path = await self._save_input_data(task_id, input_data)
            
            # 创建任务
            task = PredictionTask(
                task_id=task_id,
                config=config,
                status=PredictionStatus.PENDING.value,
                model_ids=model_ids,
                input_data_path=input_data_path
            )
            
            # 注册任务
            self.tasks[task_id] = task
            
            # 保存任务
            await self._save_tasks()
            
            logger.info(f"预测任务创建完成: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"创建预测任务失败: {e}")
            raise
    
    async def get_prediction_tasks(
        self,
        status: Optional[str] = None,
        prediction_type: Optional[str] = None,
        created_by: Optional[str] = None, # This is not used, but kept for API compatibility
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取预测任务列表"""
        tasks = list(self.tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]
        if prediction_type:
            tasks = [t for t in tasks if t.config.prediction_type == prediction_type]
        if start_date:
            tasks = [t for t in tasks if t.created_at >= start_date]
        if end_date:
            tasks = [t for t in tasks if t.created_at <= end_date]

        # Convert to dicts for API response
        task_dicts = []
        for task in tasks[:limit]:
            task_dicts.append({
                "task_id": task.task_id,
                "name": f"{task.config.target_variable} Prediction",
                "description": f"Prediction task for {task.config.target_variable}",
                "model_id": ",".join(task.model_ids),
                "model_name": "Ensemble Model" if len(task.model_ids) > 1 else task.model_ids[0],
                "prediction_type": task.config.prediction_type,
                "target_variables": [task.config.target_variable],
                "prediction_horizon": task.config.prediction_horizon,
                "status": task.status,
                "progress": int(task.progress * 100),
                "priority": "normal",
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "created_by": "system",
                "error_message": task.error_message,
                "result_summary": task.result.metrics if task.result else None
            })

        return task_dicts

    async def run_prediction_task(self, task_id: str) -> PredictionResult:
        """执行预测任务"""
        try:
            logger.info(f"开始执行预测任务: {task_id}")
            
            # 获取任务
            task = self.tasks.get(task_id)
            if not task:
                raise ValueError(f"任务不存在: {task_id}")
            
            # 更新状态
            task.status = PredictionStatus.RUNNING.value
            task.started_at = datetime.now()
            task.progress = 0.0
            
            # 加载输入数据
            input_data = await self._load_input_data(task.input_data_path)
            
            # 根据预测类型执行预测
            if task.config.prediction_type == PredictionType.TIME_SERIES.value:
                result = await self._run_time_series_prediction(task, input_data)
            elif task.config.prediction_type == PredictionType.SPATIAL.value:
                result = await self._run_spatial_prediction(task, input_data)
            elif task.config.prediction_type == PredictionType.ENSEMBLE.value:
                result = await self._run_ensemble_prediction(task, input_data)
            elif task.config.prediction_type == PredictionType.REAL_TIME.value:
                result = await self._run_real_time_prediction(task, input_data)
            else:
                raise ValueError(f"不支持的预测类型: {task.config.prediction_type}")
            
            # 保存结果
            output_path = await self._save_prediction_result(task_id, result)
            
            # 更新任务
            task.status = PredictionStatus.COMPLETED.value
            task.completed_at = datetime.now()
            task.progress = 100.0
            task.result = result
            task.output_path = output_path
            
            # 保存任务
            await self._save_tasks()
            
            logger.info(f"预测任务完成: {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"预测任务执行失败: {e}")
            
            # 更新错误状态
            if task_id in self.tasks:
                self.tasks[task_id].status = PredictionStatus.FAILED.value
                self.tasks[task_id].error_message = str(e)
                await self._save_tasks()
            
            raise
    
    async def get_prediction_result(self, task_id: str) -> Optional[PredictionResult]:
        """获取预测结果"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        if task.result:
            return task.result
        
        # 如果结果未加载，尝试从文件加载
        if task.output_path and Path(task.output_path).exists():
            return await self._load_prediction_result(task.output_path)
        
        return None
    
    async def cancel_prediction_task(self, task_id: str) -> bool:
        """取消预测任务"""
        try:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if task.status in [PredictionStatus.COMPLETED.value, PredictionStatus.FAILED.value]:
                return False
            
            task.status = PredictionStatus.CANCELLED.value
            await self._save_tasks()
            
            logger.info(f"预测任务已取消: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消预测任务失败: {e}")
            return False
    
    def list_prediction_tasks(
        self,
        status: Optional[str] = None,
        prediction_type: Optional[str] = None
    ) -> List[PredictionTask]:
        """列出预测任务"""
        tasks = list(self.tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        if prediction_type:
            tasks = [t for t in tasks if t.config.prediction_type == prediction_type]
        
        # 按创建时间排序
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        
        return tasks
    
    def get_task_info(self, task_id: str) -> Optional[PredictionTask]:
        """获取任务信息"""
        return self.tasks.get(task_id)
    
    async def _run_time_series_prediction(
        self,
        task: PredictionTask,
        input_data: Union[pd.DataFrame, xr.Dataset]
    ) -> PredictionResult:
        """执行时间序列预测"""
        logger.info("执行时间序列预测")
        
        config = task.config
        
        # 转换为DataFrame
        if isinstance(input_data, xr.Dataset):
            df = input_data.to_dataframe().reset_index()
        else:
            df = input_data.copy()
        
        # 确保时间列存在
        time_col = self._identify_time_column(df)
        if not time_col:
            raise ValueError("无法识别时间列")
        
        # 排序并设置时间索引
        df = df.sort_values(time_col)
        df.set_index(time_col, inplace=True)
        
        # 提取目标变量
        if config.target_variable not in df.columns:
            raise ValueError(f"目标变量不存在: {config.target_variable}")
        
        target_series = df[config.target_variable]
        
        # 数据预处理
        processed_series = await self._preprocess_time_series(target_series, config)
        
        # 更新进度
        task.progress = 20.0
        
        # 使用多个模型进行预测
        predictions_list = []
        
        for i, model_id in enumerate(task.model_ids):
            try:
                # 加载模型
                model_info = self.model_manager.get_model_info(model_id)
                if not model_info:
                    logger.warning(f"模型不存在: {model_id}")
                    continue
                
                # 执行预测
                if HAS_STATSMODELS and "arima" in model_info.algorithm.lower():
                    pred = await self._arima_predict(processed_series, config)
                elif HAS_TENSORFLOW and "lstm" in model_info.algorithm.lower():
                    pred = await self._lstm_predict(processed_series, config)
                else:
                    # 使用传统机器学习模型
                    pred = await self._ml_time_series_predict(model_id, processed_series, config)
                
                predictions_list.append(pred)
                
                # 更新进度
                task.progress = 20.0 + (i + 1) / len(task.model_ids) * 60.0
                
            except Exception as e:
                logger.error(f"模型 {model_id} 预测失败: {e}")
                continue
        
        if not predictions_list:
            raise ValueError("所有模型预测都失败了")
        
        # 集成预测结果
        final_predictions = np.mean(predictions_list, axis=0)
        
        # 计算置信区间
        confidence_lower, confidence_upper = self._calculate_confidence_interval(
            predictions_list, config.confidence_interval
        )
        
        # 生成时间戳
        last_time = df.index[-1]
        if config.temporal_resolution == "daily":
            freq = "D"
        elif config.temporal_resolution == "hourly":
            freq = "H"
        elif config.temporal_resolution == "monthly":
            freq = "M"
        else:
            freq = "D"
        
        timestamps = pd.date_range(
            start=last_time + pd.Timedelta(freq),
            periods=config.prediction_horizon,
            freq=freq
        ).tolist()
        
        # 计算评估指标（如果有验证数据）
        metrics = await self._calculate_prediction_metrics(
            processed_series, final_predictions, config
        )
        
        # 更新进度
        task.progress = 90.0
        
        # 创建结果
        result = PredictionResult(
            prediction_id=task.task_id,
            config=config,
            predictions=final_predictions,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            timestamps=timestamps,
            metadata={
                "models_used": task.model_ids,
                "data_points": len(processed_series),
                "prediction_method": "ensemble_average"
            },
            metrics=metrics
        )
        
        return result
    
    async def _run_spatial_prediction(
        self,
        task: PredictionTask,
        input_data: Union[pd.DataFrame, xr.Dataset]
    ) -> PredictionResult:
        """执行空间预测"""
        logger.info("执行空间预测")
        
        config = task.config
        
        # 确保是xarray数据集
        if isinstance(input_data, pd.DataFrame):
            # 尝试转换为xarray
            if 'lat' in input_data.columns and 'lon' in input_data.columns:
                ds = input_data.set_index(['lat', 'lon']).to_xarray()
            else:
                raise ValueError("DataFrame缺少空间坐标信息")
        else:
            ds = input_data
        
        # 检查目标变量
        if config.target_variable not in ds.data_vars:
            raise ValueError(f"目标变量不存在: {config.target_variable}")
        
        # 提取空间数据
        target_data = ds[config.target_variable]
        
        # 空间插值和预测
        predictions = await self._spatial_interpolation_predict(
            target_data, config, task.model_ids
        )
        
        # 更新进度
        task.progress = 80.0
        
        # 提取坐标
        coordinates = {
            'lat': ds.lat.values,
            'lon': ds.lon.values
        }
        
        # 创建结果
        result = PredictionResult(
            prediction_id=task.task_id,
            config=config,
            predictions=predictions,
            coordinates=coordinates,
            metadata={
                "models_used": task.model_ids,
                "spatial_resolution": config.spatial_resolution,
                "prediction_method": "spatial_interpolation"
            }
        )
        
        return result
    
    async def _run_ensemble_prediction(
        self,
        task: PredictionTask,
        input_data: Union[pd.DataFrame, xr.Dataset]
    ) -> PredictionResult:
        """执行集成预测"""
        logger.info("执行集成预测")
        
        config = task.config
        
        # 收集所有模型的预测结果
        all_predictions = []
        model_weights = []
        
        for i, model_id in enumerate(task.model_ids):
            try:
                # 获取模型信息
                model_info = self.model_manager.get_model_info(model_id)
                if not model_info:
                    continue
                
                # 准备模型输入数据
                model_input = await self._prepare_model_input(input_data, model_info)
                
                # 执行预测
                predictions = await self.model_manager.predict(model_id, model_input)
                all_predictions.append(predictions)
                
                # 计算模型权重（基于历史性能）
                weight = self._calculate_model_weight(model_info)
                model_weights.append(weight)
                
                # 更新进度
                task.progress = (i + 1) / len(task.model_ids) * 70.0
                
            except Exception as e:
                logger.error(f"模型 {model_id} 预测失败: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("所有模型预测都失败了")
        
        # 标准化权重
        model_weights = np.array(model_weights)
        model_weights = model_weights / model_weights.sum()
        
        # 加权集成
        ensemble_predictions = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, model_weights):
            ensemble_predictions += pred * weight
        
        # 计算预测不确定性
        prediction_std = np.std(all_predictions, axis=0)
        confidence_lower = ensemble_predictions - 1.96 * prediction_std
        confidence_upper = ensemble_predictions + 1.96 * prediction_std
        
        # 更新进度
        task.progress = 90.0
        
        # 创建结果
        result = PredictionResult(
            prediction_id=task.task_id,
            config=config,
            predictions=ensemble_predictions,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            metadata={
                "models_used": task.model_ids,
                "model_weights": model_weights.tolist(),
                "prediction_method": "weighted_ensemble",
                "ensemble_methods": config.ensemble_methods or ["weighted_average"]
            }
        )
        
        return result
    
    async def _run_real_time_prediction(
        self,
        task: PredictionTask,
        input_data: Union[pd.DataFrame, xr.Dataset]
    ) -> PredictionResult:
        """执行实时预测"""
        logger.info("执行实时预测")
        
        config = task.config
        
        # 实时预测通常使用最新的数据点
        if isinstance(input_data, xr.Dataset):
            df = input_data.to_dataframe().reset_index()
        else:
            df = input_data.copy()
        
        # 获取最新的数据
        latest_data = df.tail(1)
        
        # 使用最佳模型进行快速预测
        best_model_id = await self._select_best_model(task.model_ids)
        
        # 执行预测
        predictions = await self.model_manager.predict(best_model_id, latest_data)
        
        # 更新进度
        task.progress = 90.0
        
        # 创建结果
        result = PredictionResult(
            prediction_id=task.task_id,
            config=config,
            predictions=predictions,
            timestamps=[datetime.now() + timedelta(hours=i) for i in range(len(predictions))],
            metadata={
                "models_used": [best_model_id],
                "prediction_method": "real_time",
                "data_timestamp": datetime.now().isoformat()
            }
        )
        
        return result
    
    async def _arima_predict(
        self,
        series: pd.Series,
        config: PredictionConfig
    ) -> np.ndarray:
        """ARIMA时间序列预测"""
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库")
        
        # 检查平稳性
        adf_result = adfuller(series.dropna())
        if adf_result[1] > 0.05:
            # 差分使序列平稳
            series_diff = series.diff().dropna()
        else:
            series_diff = series
        
        # 拟合ARIMA模型
        model = ARIMA(series_diff, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # 预测
        forecast = fitted_model.forecast(steps=config.prediction_horizon)
        
        return forecast.values
    
    async def _lstm_predict(
        self,
        series: pd.Series,
        config: PredictionConfig
    ) -> np.ndarray:
        """LSTM时间序列预测"""
        if not HAS_TENSORFLOW:
            raise ImportError("需要安装TensorFlow库")
        
        # 数据预处理
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
        
        # 创建序列数据
        sequence_length = min(60, len(scaled_data) // 2)
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # 构建LSTM模型
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            layers.LSTM(50),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # 训练模型
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # 预测
        last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
        predictions = []
        
        for _ in range(config.prediction_horizon):
            pred = model.predict(last_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # 更新序列
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = pred[0, 0]
        
        # 反标准化
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return predictions.flatten()
    
    async def _ml_time_series_predict(
        self,
        model_id: str,
        series: pd.Series,
        config: PredictionConfig
    ) -> np.ndarray:
        """使用机器学习模型进行时间序列预测"""
        # 创建滞后特征
        lag_features = self._create_lag_features(series, lags=[1, 2, 3, 7, 30])
        
        # 移除缺失值
        lag_features = lag_features.dropna()
        
        if len(lag_features) == 0:
            raise ValueError("创建滞后特征后没有有效数据")
        
        # 使用模型预测
        predictions = await self.model_manager.predict(model_id, lag_features.tail(1))
        
        # 对于多步预测，需要迭代预测
        all_predictions = []
        current_features = lag_features.tail(1).copy()
        
        for _ in range(config.prediction_horizon):
            pred = await self.model_manager.predict(model_id, current_features)
            all_predictions.append(pred[0])
            
            # 更新特征（简化版本）
            # 在实际应用中，这里需要更复杂的特征更新逻辑
            current_features.iloc[0, 0] = pred[0]  # 更新第一个滞后特征
        
        return np.array(all_predictions)
    
    def _create_lag_features(
        self,
        series: pd.Series,
        lags: List[int]
    ) -> pd.DataFrame:
        """创建滞后特征"""
        df = pd.DataFrame()
        
        for lag in lags:
            df[f'lag_{lag}'] = series.shift(lag)
        
        # 添加移动平均特征
        df['ma_7'] = series.rolling(window=7).mean()
        df['ma_30'] = series.rolling(window=30).mean()
        
        # 添加趋势特征
        df['trend'] = range(len(series))
        
        # 添加季节性特征
        if hasattr(series.index, 'dayofyear'):
            df['day_of_year'] = series.index.dayofyear
            df['month'] = series.index.month
        
        return df
    
    async def _spatial_interpolation_predict(
        self,
        target_data: xr.DataArray,
        config: PredictionConfig,
        model_ids: List[str]
    ) -> np.ndarray:
        """空间插值预测"""
        # 提取有效数据点
        valid_data = target_data.where(~np.isnan(target_data), drop=True)
        
        if len(valid_data) == 0:
            raise ValueError("没有有效的空间数据点")
        
        # 获取坐标
        lats = valid_data.lat.values
        lons = valid_data.lon.values
        values = valid_data.values
        
        # 创建插值网格
        if config.spatial_resolution:
            lat_grid = np.arange(
                lats.min(), lats.max() + config.spatial_resolution,
                config.spatial_resolution
            )
            lon_grid = np.arange(
                lons.min(), lons.max() + config.spatial_resolution,
                config.spatial_resolution
            )
        else:
            lat_grid = np.linspace(lats.min(), lats.max(), 50)
            lon_grid = np.linspace(lons.min(), lons.max(), 50)
        
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # 执行插值
        interpolated = interpolate.griddata(
            (lons, lats), values,
            (lon_mesh, lat_mesh),
            method='cubic',
            fill_value=np.nan
        )
        
        return interpolated
    
    async def _prepare_model_input(
        self,
        input_data: Union[pd.DataFrame, xr.Dataset],
        model_info: ModelInfo
    ) -> pd.DataFrame:
        """为模型准备输入数据"""
        if isinstance(input_data, xr.Dataset):
            df = input_data.to_dataframe().reset_index()
        else:
            df = input_data.copy()
        
        # 确保包含模型需要的特征
        required_features = model_info.config.features
        available_features = [f for f in required_features if f in df.columns]
        
        if len(available_features) != len(required_features):
            missing = set(required_features) - set(available_features)
            logger.warning(f"缺少特征: {missing}")
        
        return df[available_features]
    
    def _calculate_model_weight(self, model_info: ModelInfo) -> float:
        """计算模型权重"""
        if not model_info.metrics:
            return 1.0
        
        # 基于模型性能计算权重
        if model_info.model_type == "regression":
            if model_info.metrics.r2:
                return max(0.1, model_info.metrics.r2)
        elif model_info.model_type == "classification":
            if model_info.metrics.accuracy:
                return max(0.1, model_info.metrics.accuracy)
        
        return 1.0
    
    async def _select_best_model(self, model_ids: List[str]) -> str:
        """选择最佳模型"""
        best_model_id = model_ids[0]
        best_score = 0.0
        
        for model_id in model_ids:
            model_info = self.model_manager.get_model_info(model_id)
            if not model_info or not model_info.metrics:
                continue
            
            # 计算综合得分
            score = self._calculate_model_weight(model_info)
            
            if score > best_score:
                best_score = score
                best_model_id = model_id
        
        return best_model_id
    
    def _identify_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """识别时间列"""
        time_candidates = ['time', 'date', 'datetime', 'timestamp']
        
        for col in df.columns:
            if col.lower() in time_candidates:
                return col
            
            # 检查数据类型
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        
        return None
    
    async def _preprocess_time_series(
        self,
        series: pd.Series,
        config: PredictionConfig
    ) -> pd.Series:
        """时间序列预处理"""
        processed = series.copy()
        
        if not config.preprocessing_steps:
            return processed
        
        for step in config.preprocessing_steps:
            if step == "fill_missing":
                processed = processed.interpolate(method='linear')
            elif step == "remove_outliers":
                Q1 = processed.quantile(0.25)
                Q3 = processed.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                processed = processed.clip(lower_bound, upper_bound)
            elif step == "detrend":
                # 简单去趋势
                trend = np.polyfit(range(len(processed)), processed.values, 1)
                trend_line = np.polyval(trend, range(len(processed)))
                processed = processed - trend_line
            elif step == "normalize":
                scaler_key = f"{config.target_variable}_scaler"
                if scaler_key not in self.scalers:
                    self.scalers[scaler_key] = StandardScaler()
                    processed = pd.Series(
                        self.scalers[scaler_key].fit_transform(processed.values.reshape(-1, 1)).flatten(),
                        index=processed.index
                    )
                else:
                    processed = pd.Series(
                        self.scalers[scaler_key].transform(processed.values.reshape(-1, 1)).flatten(),
                        index=processed.index
                    )
        
        return processed
    
    def _calculate_confidence_interval(
        self,
        predictions_list: List[np.ndarray],
        confidence_level: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算置信区间"""
        predictions_array = np.array(predictions_list)
        
        # 计算分位数
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        confidence_lower = np.quantile(predictions_array, lower_quantile, axis=0)
        confidence_upper = np.quantile(predictions_array, upper_quantile, axis=0)
        
        return confidence_lower, confidence_upper
    
    async def _calculate_prediction_metrics(
        self,
        actual: pd.Series,
        predicted: np.ndarray,
        config: PredictionConfig
    ) -> Dict[str, float]:
        """计算预测评估指标"""
        # 这里需要验证数据，实际应用中可能需要留出验证集
        # 暂时返回空字典
        return {}
    
    def _generate_task_id(self, target_variable: str) -> str:
        """生成任务ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"pred_{target_variable}_{timestamp}"
    
    async def _validate_models(self, model_ids: List[str]):
        """验证模型"""
        for model_id in model_ids:
            model_info = self.model_manager.get_model_info(model_id)
            if not model_info:
                raise ValueError(f"模型不存在: {model_id}")
            
            if model_info.status != "trained":
                raise ValueError(f"模型未训练完成: {model_id}")
    
    async def _save_input_data(
        self,
        task_id: str,
        input_data: Union[pd.DataFrame, xr.Dataset, str]
    ) -> str:
        """保存输入数据"""
        if isinstance(input_data, str):
            # 如果是文件路径，直接返回
            return input_data
        
        input_path = self.predictions_path / f"{task_id}_input.pkl"
        
        with open(input_path, 'wb') as f:
            pickle.dump(input_data, f)
        
        return str(input_path)
    
    async def _load_input_data(self, input_path: str) -> Union[pd.DataFrame, xr.Dataset]:
        """加载输入数据"""
        if input_path.endswith('.pkl'):
            with open(input_path, 'rb') as f:
                return pickle.load(f)
        elif input_path.endswith('.csv'):
            return pd.read_csv(input_path)
        elif input_path.endswith('.nc'):
            return xr.open_dataset(input_path)
        else:
            raise ValueError(f"不支持的文件格式: {input_path}")
    
    async def _save_prediction_result(
        self,
        task_id: str,
        result: PredictionResult
    ) -> str:
        """保存预测结果"""
        output_path = self.predictions_path / f"{task_id}_result.pkl"
        
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        
        return str(output_path)
    
    async def _load_prediction_result(self, output_path: str) -> PredictionResult:
        """加载预测结果"""
        with open(output_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_tasks(self):
        """加载任务"""
        tasks_file = self.predictions_path / "tasks.json"
        
        if tasks_file.exists():
            try:
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                
                for task_id, task_data in tasks_data.items():
                    # 重建PredictionConfig
                    config_data = task_data['config']
                    config = PredictionConfig(**config_data)
                    
                    # 重建PredictionTask
                    task = PredictionTask(
                        task_id=task_data['task_id'],
                        config=config,
                        status=task_data['status'],
                        model_ids=task_data['model_ids'],
                        input_data_path=task_data.get('input_data_path'),
                        output_path=task_data.get('output_path'),
                        progress=task_data.get('progress', 0.0),
                        error_message=task_data.get('error_message'),
                        created_at=datetime.fromisoformat(task_data['created_at']),
                        started_at=datetime.fromisoformat(task_data['started_at']) if task_data.get('started_at') else None,
                        completed_at=datetime.fromisoformat(task_data['completed_at']) if task_data.get('completed_at') else None
                    )
                    
                    self.tasks[task_id] = task
                
                logger.info(f"加载了 {len(self.tasks)} 个预测任务")
                
            except Exception as e:
                logger.error(f"加载预测任务失败: {e}")
    
    async def _save_tasks(self):
        """保存任务"""
        tasks_file = self.predictions_path / "tasks.json"
        
        try:
            tasks_data = {}
            
            for task_id, task in self.tasks.items():
                task_data = {
                    'task_id': task.task_id,
                    'config': asdict(task.config),
                    'status': task.status,
                    'model_ids': task.model_ids,
                    'input_data_path': task.input_data_path,
                    'output_path': task.output_path,
                    'progress': task.progress,
                    'error_message': task.error_message,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None
                }
                
                tasks_data[task_id] = task_data
            
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, indent=2, ensure_ascii=False)
            
            logger.info("预测任务已保存")
            
        except Exception as e:
            logger.error(f"保存预测任务失败: {e}")


# 便捷函数
async def create_time_series_prediction(
    target_variable: str,
    prediction_horizon: int,
    model_ids: List[str],
    input_data: Union[pd.DataFrame, xr.Dataset],
    temporal_resolution: str = "daily",
    engine: Optional[PredictionEngine] = None
) -> str:
    """创建时间序列预测任务的便捷函数"""
    config = PredictionConfig(
        prediction_type=PredictionType.TIME_SERIES.value,
        target_variable=target_variable,
        prediction_horizon=prediction_horizon,
        temporal_resolution=temporal_resolution
    )
    
    if engine is None:
        engine = PredictionEngine()
    
    return await engine.create_prediction_task(config, model_ids, input_data)


async def create_spatial_prediction(
    target_variable: str,
    model_ids: List[str],
    input_data: xr.Dataset,
    spatial_resolution: float = 0.1,
    engine: Optional[PredictionEngine] = None
) -> str:
    """创建空间预测任务的便捷函数"""
    config = PredictionConfig(
        prediction_type=PredictionType.SPATIAL.value,
        target_variable=target_variable,
        prediction_horizon=1,
        spatial_resolution=spatial_resolution
    )
    
    if engine is None:
        engine = PredictionEngine()
    
    return await engine.create_prediction_task(config, model_ids, input_data)