#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理器

负责气候数据的清洗、转换、聚合和质量控制。
"""

import asyncio
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import warnings
from dataclasses import dataclass
from enum import Enum

from scipy import stats, interpolate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

from src.utils.logger import get_logger
from src.utils.config import get_config
from .data_storage import DataStorage

logger = get_logger(__name__)
config = get_config()


class QualityFlag(Enum):
    """数据质量标志"""
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"
    MISSING = "missing"
    INTERPOLATED = "interpolated"
    OUTLIER = "outlier"


@dataclass
class ProcessingConfig:
    """数据处理配置"""
    # 质量控制参数
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0
    missing_threshold: float = 0.3  # 缺失值比例阈值
    
    # 插值参数
    interpolation_method: str = "linear"  # linear, cubic, nearest
    max_gap_hours: int = 6  # 最大插值间隔（小时）
    
    # 聚合参数
    temporal_resolution: str = "1H"  # 时间分辨率
    spatial_resolution: Optional[float] = None  # 空间分辨率（度）
    
    # 标准化参数
    normalization_method: str = "zscore"  # zscore, minmax, robust
    
    # 平滑参数
    smoothing_window: int = 5
    smoothing_method: str = "rolling_mean"  # rolling_mean, savgol, lowess


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, storage: Optional[DataStorage] = None):
        self.storage = storage or DataStorage()
        self.config = ProcessingConfig()
        
        # 处理统计
        self.processing_stats = {
            "records_processed": 0,
            "outliers_detected": 0,
            "missing_values_filled": 0,
            "quality_flags_assigned": 0
        }
    
    async def process_climate_data(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        config: Optional[ProcessingConfig] = None,
        save_result: bool = True
    ) -> Union[pd.DataFrame, xr.Dataset]:
        """处理气候数据"""
        try:
            logger.info("开始处理气候数据")
            
            if config:
                self.config = config
            
            # 根据数据类型选择处理方法
            if isinstance(data, pd.DataFrame):
                processed_data = await self._process_dataframe(data)
            elif isinstance(data, xr.Dataset):
                processed_data = await self._process_xarray(data)
            else:
                raise ValueError(f"不支持的数据类型: {type(data)}")
            
            # 保存处理结果
            if save_result and self.storage:
                await self._save_processed_data(processed_data)
            
            logger.info(f"数据处理完成，处理统计: {self.processing_stats}")
            return processed_data
            
        except Exception as e:
            logger.error(f"数据处理失败: {e}")
            raise
    
    async def _process_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理DataFrame数据"""
        logger.info(f"处理DataFrame数据，形状: {data.shape}")
        
        # 复制数据避免修改原始数据
        processed_data = data.copy()
        
        # 1. 数据类型转换和时间索引处理
        processed_data = self._prepare_dataframe(processed_data)
        
        # 2. 质量控制
        processed_data = await self._quality_control_dataframe(processed_data)
        
        # 3. 缺失值处理
        processed_data = await self._handle_missing_values_dataframe(processed_data)
        
        # 4. 异常值检测和处理
        processed_data = await self._detect_outliers_dataframe(processed_data)
        
        # 5. 数据平滑
        processed_data = await self._smooth_data_dataframe(processed_data)
        
        # 6. 时间聚合
        if self.config.temporal_resolution != "original":
            processed_data = await self._temporal_aggregation_dataframe(processed_data)
        
        # 7. 数据标准化
        processed_data = await self._normalize_data_dataframe(processed_data)
        
        return processed_data
    
    async def _process_xarray(self, data: xr.Dataset) -> xr.Dataset:
        """处理xarray数据"""
        logger.info(f"处理xarray数据，维度: {data.dims}")
        
        # 复制数据
        processed_data = data.copy()
        
        # 1. 质量控制
        processed_data = await self._quality_control_xarray(processed_data)
        
        # 2. 缺失值处理
        processed_data = await self._handle_missing_values_xarray(processed_data)
        
        # 3. 异常值检测
        processed_data = await self._detect_outliers_xarray(processed_data)
        
        # 4. 空间和时间聚合
        if self.config.spatial_resolution:
            processed_data = await self._spatial_aggregation_xarray(processed_data)
        
        if self.config.temporal_resolution != "original":
            processed_data = await self._temporal_aggregation_xarray(processed_data)
        
        return processed_data
    
    def _prepare_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备DataFrame数据"""
        # 尝试识别时间列
        time_columns = [col for col in data.columns 
                       if any(keyword in col.lower() 
                             for keyword in ['time', 'date', 'datetime'])]
        
        if time_columns and not isinstance(data.index, pd.DatetimeIndex):
            # 设置时间索引
            time_col = time_columns[0]
            try:
                data[time_col] = pd.to_datetime(data[time_col])
                data = data.set_index(time_col)
                logger.info(f"设置时间索引: {time_col}")
            except Exception as e:
                logger.warning(f"设置时间索引失败: {e}")
        
        # 数值列类型转换
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    async def _quality_control_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """DataFrame质量控制"""
        logger.info("执行质量控制")
        
        # 添加质量标志列
        for col in data.select_dtypes(include=[np.number]).columns:
            flag_col = f"{col}_quality_flag"
            data[flag_col] = QualityFlag.GOOD.value
        
        # 检查数据范围
        data = self._check_data_ranges(data)
        
        # 检查时间序列连续性
        if isinstance(data.index, pd.DatetimeIndex):
            data = self._check_temporal_continuity(data)
        
        return data
    
    def _check_data_ranges(self, data: pd.DataFrame) -> pd.DataFrame:
        """检查数据范围"""
        # 定义合理的气候变量范围
        variable_ranges = {
            'temperature': (-100, 60),  # 摄氏度
            'temp': (-100, 60),
            'humidity': (0, 100),  # 百分比
            'pressure': (800, 1100),  # hPa
            'wind_speed': (0, 200),  # m/s
            'precipitation': (0, 1000),  # mm
            'solar_radiation': (0, 1500)  # W/m²
        }
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if '_quality_flag' in col:
                continue
                
            # 查找匹配的变量类型
            var_type = None
            for var_name, (min_val, max_val) in variable_ranges.items():
                if var_name in col.lower():
                    var_type = var_name
                    break
            
            if var_type:
                min_val, max_val = variable_ranges[var_type]
                flag_col = f"{col}_quality_flag"
                
                # 标记超出范围的值
                out_of_range = (data[col] < min_val) | (data[col] > max_val)
                data.loc[out_of_range, flag_col] = QualityFlag.SUSPECT.value
                
                if out_of_range.sum() > 0:
                    logger.warning(f"{col}: {out_of_range.sum()} 个值超出合理范围 [{min_val}, {max_val}]")
        
        return data
    
    def _check_temporal_continuity(self, data: pd.DataFrame) -> pd.DataFrame:
        """检查时间序列连续性"""
        if len(data) < 2:
            return data
        
        # 计算时间间隔
        time_diffs = data.index.to_series().diff()
        median_interval = time_diffs.median()
        
        # 检测异常间隔
        large_gaps = time_diffs > median_interval * 3
        
        if large_gaps.sum() > 0:
            logger.warning(f"检测到 {large_gaps.sum()} 个时间间隔异常")
            
            # 在大间隔后的数据点标记为可疑
            for col in data.select_dtypes(include=[np.number]).columns:
                if '_quality_flag' not in col:
                    flag_col = f"{col}_quality_flag"
                    data.loc[large_gaps, flag_col] = QualityFlag.SUSPECT.value
        
        return data
    
    async def _handle_missing_values_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理DataFrame缺失值"""
        logger.info("处理缺失值")
        
        numeric_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if '_quality_flag' not in col]
        
        for col in numeric_columns:
            missing_ratio = data[col].isnull().sum() / len(data)
            
            if missing_ratio > self.config.missing_threshold:
                logger.warning(f"{col}: 缺失值比例 {missing_ratio:.2%} 超过阈值")
                continue
            
            if data[col].isnull().sum() > 0:
                # 根据配置选择插值方法
                filled_data = self._interpolate_missing_values(
                    data[col], 
                    method=self.config.interpolation_method
                )
                
                # 标记插值的值
                flag_col = f"{col}_quality_flag"
                interpolated_mask = data[col].isnull() & filled_data.notnull()
                data.loc[interpolated_mask, flag_col] = QualityFlag.INTERPOLATED.value
                
                # 更新数据
                data[col] = filled_data
                
                filled_count = interpolated_mask.sum()
                self.processing_stats["missing_values_filled"] += filled_count
                logger.info(f"{col}: 填充了 {filled_count} 个缺失值")
        
        return data
    
    def _interpolate_missing_values(
        self, 
        series: pd.Series, 
        method: str = "linear"
    ) -> pd.Series:
        """插值缺失值"""
        if method == "linear":
            return series.interpolate(method='linear', limit=self.config.max_gap_hours)
        elif method == "cubic":
            return series.interpolate(method='cubic', limit=self.config.max_gap_hours)
        elif method == "nearest":
            return series.interpolate(method='nearest', limit=self.config.max_gap_hours)
        elif method == "forward_fill":
            return series.fillna(method='ffill', limit=self.config.max_gap_hours)
        elif method == "backward_fill":
            return series.fillna(method='bfill', limit=self.config.max_gap_hours)
        else:
            return series.interpolate(method='linear', limit=self.config.max_gap_hours)
    
    async def _detect_outliers_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """检测DataFrame异常值"""
        logger.info("检测异常值")
        
        numeric_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if '_quality_flag' not in col]
        
        for col in numeric_columns:
            if data[col].isnull().all():
                continue
            
            outliers = self._detect_outliers_series(
                data[col], 
                method=self.config.outlier_method
            )
            
            if outliers.sum() > 0:
                flag_col = f"{col}_quality_flag"
                data.loc[outliers, flag_col] = QualityFlag.OUTLIER.value
                
                self.processing_stats["outliers_detected"] += outliers.sum()
                logger.info(f"{col}: 检测到 {outliers.sum()} 个异常值")
        
        return data
    
    def _detect_outliers_series(
        self, 
        series: pd.Series, 
        method: str = "iqr"
    ) -> pd.Series:
        """检测序列异常值"""
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(series.dropna()))
            outlier_indices = series.dropna().index[z_scores > self.config.outlier_threshold]
            return series.index.isin(outlier_indices)
        
        elif method == "isolation_forest":
            try:
                clf = IsolationForest(contamination=0.1, random_state=42)
                outliers = clf.fit_predict(series.dropna().values.reshape(-1, 1))
                outlier_indices = series.dropna().index[outliers == -1]
                return series.index.isin(outlier_indices)
            except Exception as e:
                logger.warning(f"Isolation Forest异常检测失败: {e}，回退到IQR方法")
                return self._detect_outliers_series(series, method="iqr")
        
        else:
            return pd.Series(False, index=series.index)
    
    async def _smooth_data_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据平滑"""
        if self.config.smoothing_window <= 1:
            return data
        
        logger.info(f"数据平滑，窗口大小: {self.config.smoothing_window}")
        
        numeric_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if '_quality_flag' not in col]
        
        for col in numeric_columns:
            if data[col].isnull().all():
                continue
            
            if self.config.smoothing_method == "rolling_mean":
                data[f"{col}_smoothed"] = data[col].rolling(
                    window=self.config.smoothing_window, 
                    center=True
                ).mean()
            
            elif self.config.smoothing_method == "savgol":
                try:
                    from scipy.signal import savgol_filter
                    valid_data = data[col].dropna()
                    if len(valid_data) > self.config.smoothing_window:
                        smoothed = savgol_filter(
                            valid_data.values, 
                            self.config.smoothing_window, 
                            3  # 多项式阶数
                        )
                        data.loc[valid_data.index, f"{col}_smoothed"] = smoothed
                except ImportError:
                    logger.warning("scipy未安装，使用滚动平均代替Savitzky-Golay滤波")
                    data[f"{col}_smoothed"] = data[col].rolling(
                        window=self.config.smoothing_window, 
                        center=True
                    ).mean()
        
        return data
    
    async def _temporal_aggregation_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """时间聚合"""
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("数据没有时间索引，跳过时间聚合")
            return data
        
        logger.info(f"时间聚合，目标分辨率: {self.config.temporal_resolution}")
        
        # 分离数值列和质量标志列
        numeric_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if '_quality_flag' not in col]
        flag_columns = [col for col in data.columns if '_quality_flag' in col]
        
        # 聚合数值列
        aggregated_numeric = data[numeric_columns].resample(
            self.config.temporal_resolution
        ).agg({
            col: ['mean', 'std', 'count'] for col in numeric_columns
        })
        
        # 展平多级列索引
        aggregated_numeric.columns = [
            f"{col}_{stat}" for col, stat in aggregated_numeric.columns
        ]
        
        # 聚合质量标志（取最差的质量等级）
        quality_priority = {
            QualityFlag.GOOD.value: 0,
            QualityFlag.INTERPOLATED.value: 1,
            QualityFlag.SUSPECT.value: 2,
            QualityFlag.OUTLIER.value: 3,
            QualityFlag.BAD.value: 4,
            QualityFlag.MISSING.value: 5
        }
        
        aggregated_flags = pd.DataFrame(index=aggregated_numeric.index)
        for flag_col in flag_columns:
            if flag_col in data.columns:
                # 将质量标志转换为数值，取最大值（最差质量），再转回标志
                flag_numeric = data[flag_col].map(quality_priority)
                worst_flag_numeric = flag_numeric.resample(
                    self.config.temporal_resolution
                ).max()
                
                # 转回质量标志
                reverse_priority = {v: k for k, v in quality_priority.items()}
                aggregated_flags[flag_col] = worst_flag_numeric.map(reverse_priority)
        
        # 合并结果
        result = pd.concat([aggregated_numeric, aggregated_flags], axis=1)
        
        return result
    
    async def _normalize_data_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据标准化"""
        logger.info(f"数据标准化，方法: {self.config.normalization_method}")
        
        numeric_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if '_quality_flag' not in col and '_smoothed' not in col]
        
        for col in numeric_columns:
            if data[col].isnull().all():
                continue
            
            if self.config.normalization_method == "zscore":
                data[f"{col}_normalized"] = (
                    data[col] - data[col].mean()
                ) / data[col].std()
            
            elif self.config.normalization_method == "minmax":
                data[f"{col}_normalized"] = (
                    data[col] - data[col].min()
                ) / (data[col].max() - data[col].min())
            
            elif self.config.normalization_method == "robust":
                median = data[col].median()
                mad = (data[col] - median).abs().median()
                data[f"{col}_normalized"] = (data[col] - median) / mad
        
        return data
    
    async def _quality_control_xarray(self, data: xr.Dataset) -> xr.Dataset:
        """xarray质量控制"""
        logger.info("执行xarray质量控制")
        
        # 为每个数据变量添加质量标志
        for var_name in data.data_vars:
            flag_name = f"{var_name}_quality_flag"
            data[flag_name] = xr.full_like(
                data[var_name], 
                QualityFlag.GOOD.value, 
                dtype='<U12'
            )
        
        return data
    
    async def _handle_missing_values_xarray(self, data: xr.Dataset) -> xr.Dataset:
        """处理xarray缺失值"""
        logger.info("处理xarray缺失值")
        
        for var_name in data.data_vars:
            if '_quality_flag' in var_name:
                continue
            
            var_data = data[var_name]
            
            # 时间维度插值
            if 'time' in var_data.dims:
                interpolated = var_data.interpolate_na(
                    dim='time', 
                    method=self.config.interpolation_method
                )
                
                # 标记插值的值
                interpolated_mask = var_data.isnull() & interpolated.notnull()
                if interpolated_mask.any():
                    flag_name = f"{var_name}_quality_flag"
                    data[flag_name] = data[flag_name].where(
                        ~interpolated_mask, 
                        QualityFlag.INTERPOLATED.value
                    )
                
                data[var_name] = interpolated
        
        return data
    
    async def _detect_outliers_xarray(self, data: xr.Dataset) -> xr.Dataset:
        """检测xarray异常值"""
        logger.info("检测xarray异常值")
        
        for var_name in data.data_vars:
            if '_quality_flag' in var_name:
                continue
            
            var_data = data[var_name]
            
            # 使用IQR方法检测异常值
            Q1 = var_data.quantile(0.25)
            Q3 = var_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (var_data < lower_bound) | (var_data > upper_bound)
            
            if outliers.any():
                flag_name = f"{var_name}_quality_flag"
                data[flag_name] = data[flag_name].where(
                    ~outliers, 
                    QualityFlag.OUTLIER.value
                )
                
                outlier_count = outliers.sum().item()
                self.processing_stats["outliers_detected"] += outlier_count
                logger.info(f"{var_name}: 检测到 {outlier_count} 个异常值")
        
        return data
    
    async def _spatial_aggregation_xarray(self, data: xr.Dataset) -> xr.Dataset:
        """空间聚合"""
        if not self.config.spatial_resolution:
            return data
        
        logger.info(f"空间聚合，目标分辨率: {self.config.spatial_resolution}度")
        
        # 简化实现：对经纬度网格进行粗化
        if 'lat' in data.dims and 'lon' in data.dims:
            # 计算聚合因子
            lat_res = float(data.lat[1] - data.lat[0])
            lon_res = float(data.lon[1] - data.lon[0])
            
            lat_factor = max(1, int(self.config.spatial_resolution / lat_res))
            lon_factor = max(1, int(self.config.spatial_resolution / lon_res))
            
            if lat_factor > 1 or lon_factor > 1:
                # 使用coarsen进行聚合
                data = data.coarsen(
                    lat=lat_factor, 
                    lon=lon_factor, 
                    boundary='trim'
                ).mean()
        
        return data
    
    async def _temporal_aggregation_xarray(self, data: xr.Dataset) -> xr.Dataset:
        """xarray时间聚合"""
        if 'time' not in data.dims:
            return data
        
        logger.info(f"xarray时间聚合，目标分辨率: {self.config.temporal_resolution}")
        
        # 使用resample进行时间聚合
        aggregated = data.resample(time=self.config.temporal_resolution).mean()
        
        return aggregated
    
    async def _save_processed_data(
        self, 
        data: Union[pd.DataFrame, xr.Dataset]
    ):
        """保存处理后的数据"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if isinstance(data, pd.DataFrame):
                filename = f"processed_dataframe_{timestamp}"
                file_path = await self.storage.save_dataframe(
                    data, 
                    filename, 
                    data_category="processed"
                )
            elif isinstance(data, xr.Dataset):
                filename = f"processed_dataset_{timestamp}"
                file_path = await self.storage.save_xarray(
                    data, 
                    filename, 
                    data_category="processed"
                )
            
            logger.info(f"处理后的数据已保存: {file_path}")
            
        except Exception as e:
            logger.error(f"保存处理后的数据失败: {e}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """获取处理摘要"""
        return {
            "processing_config": {
                "outlier_method": self.config.outlier_method,
                "outlier_threshold": self.config.outlier_threshold,
                "missing_threshold": self.config.missing_threshold,
                "interpolation_method": self.config.interpolation_method,
                "temporal_resolution": self.config.temporal_resolution,
                "normalization_method": self.config.normalization_method
            },
            "processing_stats": self.processing_stats.copy(),
            "timestamp": datetime.now().isoformat()
        }
    
    def reset_stats(self):
        """重置处理统计"""
        self.processing_stats = {
            "records_processed": 0,
            "outliers_detected": 0,
            "missing_values_filled": 0,
            "quality_flags_assigned": 0
        }


# 便捷函数
async def process_climate_dataframe(
    data: pd.DataFrame,
    config: Optional[ProcessingConfig] = None,
    storage: Optional[DataStorage] = None
) -> pd.DataFrame:
    """处理气候DataFrame的便捷函数"""
    processor = DataProcessor(storage)
    return await processor.process_climate_data(data, config)


async def process_climate_dataset(
    data: xr.Dataset,
    config: Optional[ProcessingConfig] = None,
    storage: Optional[DataStorage] = None
) -> xr.Dataset:
    """处理气候xarray数据集的便捷函数"""
    processor = DataProcessor(storage)
    return await processor.process_climate_data(data, config)