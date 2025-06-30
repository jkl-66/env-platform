#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史气候数据分析模型

提供历史气候数据的趋势分析、周期性检测、异常事件识别和模式识别功能。
支持多种时间序列分析方法和机器学习算法。
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import xarray as xr
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)
config = get_config()

warnings.filterwarnings('ignore')


class AnalysisType(Enum):
    """分析类型"""
    TREND = "trend"
    SEASONALITY = "seasonality"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    CORRELATION = "correlation"
    STATIONARITY = "stationarity"
    SPECTRAL = "spectral"
    EXTREME_EVENTS = "extreme_events"


class AnomalyMethod(Enum):
    """异常检测方法"""
    STATISTICAL = "statistical"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    DBSCAN = "dbscan"
    AUTOENCODER = "autoencoder"


class TrendMethod(Enum):
    """趋势分析方法"""
    LINEAR_REGRESSION = "linear_regression"
    POLYNOMIAL = "polynomial"
    MOVING_AVERAGE = "moving_average"
    LOWESS = "lowess"
    MANN_KENDALL = "mann_kendall"


@dataclass
class ClimateVariable:
    """气候变量"""
    name: str
    unit: str
    description: str
    normal_range: Tuple[float, float]
    extreme_thresholds: Tuple[float, float]  # (lower, upper)


@dataclass
class TrendResult:
    """趋势分析结果"""
    variable: str
    method: TrendMethod
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    confidence_interval: Tuple[float, float]
    trend_direction: str  # "increasing", "decreasing", "stable"
    significance: str     # "significant", "not_significant"
    annual_change: float
    decadal_change: float


@dataclass
class SeasonalityResult:
    """季节性分析结果"""
    variable: str
    has_seasonality: bool
    seasonal_strength: float
    dominant_periods: List[float]
    seasonal_component: np.ndarray
    trend_component: np.ndarray
    residual_component: np.ndarray
    seasonal_peaks: List[int]  # 月份
    seasonal_troughs: List[int]  # 月份


@dataclass
class AnomalyResult:
    """异常检测结果"""
    variable: str
    method: AnomalyMethod
    anomaly_indices: List[int]
    anomaly_scores: List[float]
    anomaly_dates: List[datetime]
    anomaly_values: List[float]
    threshold: float
    total_anomalies: int
    anomaly_rate: float


@dataclass
class PatternResult:
    """模式识别结果"""
    variable: str
    pattern_type: str
    clusters: List[int]
    cluster_centers: np.ndarray
    pattern_strength: float
    dominant_patterns: List[Dict[str, Any]]
    pattern_transitions: Dict[str, float]


@dataclass
class ExtremeEventResult:
    """极端事件分析结果"""
    variable: str
    event_type: str  # "heatwave", "coldwave", "drought", "flood"
    events: List[Dict[str, Any]]
    frequency: float  # 事件/年
    intensity_trend: TrendResult
    duration_trend: TrendResult
    return_periods: Dict[str, float]


@dataclass
class AnalysisReport:
    """分析报告"""
    dataset_name: str
    analysis_date: datetime
    time_period: Tuple[datetime, datetime]
    variables_analyzed: List[str]
    trend_results: Dict[str, TrendResult]
    seasonality_results: Dict[str, SeasonalityResult]
    anomaly_results: Dict[str, AnomalyResult]
    pattern_results: Dict[str, PatternResult]
    extreme_events: Dict[str, ExtremeEventResult]
    correlations: Dict[str, float]
    summary: Dict[str, Any]


class AutoEncoder(nn.Module):
    """用于异常检测的自编码器"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16]):
        super().__init__()
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # 移除最后的dropout
        
        # 解码器
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims[:-1])) + [input_dim]
        for hidden_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if hidden_dim != input_dim else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class HistoricalClimateAnalyzer:
    """历史气候数据分析器"""
    
    def __init__(self):
        self.climate_variables = self._define_climate_variables()
        self.scaler = StandardScaler()
        self.analysis_cache = {}
        
        # 模型存储
        self.autoencoder = None
        self.isolation_forest = None
        self.one_class_svm = None
        
        logger.info("历史气候数据分析器初始化完成")
    
    def _define_climate_variables(self) -> Dict[str, ClimateVariable]:
        """定义气候变量"""
        return {
            "temperature": ClimateVariable(
                name="温度",
                unit="°C",
                description="平均气温",
                normal_range=(-50, 50),
                extreme_thresholds=(-40, 45)
            ),
            "precipitation": ClimateVariable(
                name="降水量",
                unit="mm",
                description="月降水量",
                normal_range=(0, 500),
                extreme_thresholds=(0, 300)
            ),
            "humidity": ClimateVariable(
                name="湿度",
                unit="%",
                description="相对湿度",
                normal_range=(0, 100),
                extreme_thresholds=(10, 95)
            ),
            "pressure": ClimateVariable(
                name="气压",
                unit="hPa",
                description="海平面气压",
                normal_range=(950, 1050),
                extreme_thresholds=(960, 1040)
            ),
            "wind_speed": ClimateVariable(
                name="风速",
                unit="m/s",
                description="平均风速",
                normal_range=(0, 30),
                extreme_thresholds=(0, 25)
            )
        }
    
    def analyze_trends(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        variables: Optional[List[str]] = None,
        method: TrendMethod = TrendMethod.LINEAR_REGRESSION,
        confidence_level: float = 0.95
    ) -> Dict[str, TrendResult]:
        """分析长期趋势"""
        logger.info(f"开始趋势分析，方法: {method.value}")
        
        if isinstance(data, xr.Dataset):
            data = data.to_dataframe().reset_index()
        
        if variables is None:
            variables = [col for col in data.columns if col != 'time' and np.issubdtype(data[col].dtype, np.number)]
        
        results = {}
        
        for var in variables:
            if var not in data.columns:
                logger.warning(f"变量 {var} 不存在于数据中")
                continue
            
            try:
                result = self._analyze_single_trend(
                    data[var].dropna(), method, confidence_level
                )
                result.variable = var
                results[var] = result
                
                logger.info(f"{var} 趋势分析完成: {result.trend_direction}, 斜率={result.slope:.4f}")
            
            except Exception as e:
                logger.error(f"分析 {var} 趋势时出错: {e}")
        
        return results
    
    def _analyze_single_trend(
        self,
        series: pd.Series,
        method: TrendMethod,
        confidence_level: float
    ) -> TrendResult:
        """分析单个变量的趋势"""
        x = np.arange(len(series))
        y = series.values
        
        if method == TrendMethod.LINEAR_REGRESSION:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # 置信区间
            alpha = 1 - confidence_level
            t_val = stats.t.ppf(1 - alpha/2, len(x) - 2)
            margin_error = t_val * std_err
            confidence_interval = (slope - margin_error, slope + margin_error)
            
            r_squared = r_value ** 2
        
        elif method == TrendMethod.POLYNOMIAL:
            # 二次多项式拟合
            coeffs = np.polyfit(x, y, 2)
            slope = coeffs[1]  # 一次项系数
            intercept = coeffs[2]  # 常数项
            
            # 计算R²
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            p_value = 0.05  # 简化处理
            confidence_interval = (slope * 0.9, slope * 1.1)
        
        elif method == TrendMethod.MANN_KENDALL:
            # Mann-Kendall趋势检验
            n = len(y)
            s = 0
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if y[j] > y[i]:
                        s += 1
                    elif y[j] < y[i]:
                        s -= 1
            
            # 计算方差
            var_s = n * (n - 1) * (2 * n + 5) / 18
            
            # 标准化统计量
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
            
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            # Sen's slope estimator
            slopes = []
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if j != i:
                        slopes.append((y[j] - y[i]) / (j - i))
            
            slope = np.median(slopes) if slopes else 0
            intercept = np.median(y) - slope * np.median(x)
            r_squared = 0.5  # 简化处理
            confidence_interval = (slope * 0.9, slope * 1.1)
        
        else:
            # 默认线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
            confidence_interval = (slope - std_err, slope + std_err)
        
        # 判断趋势方向
        if abs(slope) < 1e-6:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        # 判断显著性
        significance = "significant" if p_value < 0.05 else "not_significant"
        
        # 计算年际和十年际变化
        annual_change = slope * 12 if len(series) > 12 else slope
        decadal_change = slope * 120 if len(series) > 120 else slope * 10
        
        return TrendResult(
            variable="",
            method=method,
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            p_value=p_value,
            confidence_interval=confidence_interval,
            trend_direction=trend_direction,
            significance=significance,
            annual_change=annual_change,
            decadal_change=decadal_change
        )
    
    def analyze_seasonality(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        variables: Optional[List[str]] = None,
        period: int = 12
    ) -> Dict[str, SeasonalityResult]:
        """分析季节性模式"""
        logger.info("开始季节性分析")
        
        if isinstance(data, xr.Dataset):
            data = data.to_dataframe().reset_index()
        
        if variables is None:
            variables = [col for col in data.columns if col != 'time' and np.issubdtype(data[col].dtype, np.number)]
        
        results = {}
        
        for var in variables:
            if var not in data.columns:
                continue
            
            try:
                series = data[var].dropna()
                if len(series) < 2 * period:
                    logger.warning(f"{var} 数据长度不足，跳过季节性分析")
                    continue
                
                result = self._analyze_single_seasonality(series, period)
                result.variable = var
                results[var] = result
                
                logger.info(f"{var} 季节性分析完成: 季节性强度={result.seasonal_strength:.3f}")
            
            except Exception as e:
                logger.error(f"分析 {var} 季节性时出错: {e}")
        
        return results
    
    def _analyze_single_seasonality(
        self,
        series: pd.Series,
        period: int
    ) -> SeasonalityResult:
        """分析单个变量的季节性"""
        # 时间序列分解
        if HAS_STATSMODELS and len(series) >= 2 * period:
            try:
                decomposition = seasonal_decompose(
                    series, model='additive', period=period, extrapolate_trend='freq'
                )
                
                seasonal_component = decomposition.seasonal.values
                trend_component = decomposition.trend.values
                residual_component = decomposition.resid.values
                
                # 计算季节性强度
                seasonal_var = np.var(seasonal_component[~np.isnan(seasonal_component)])
                residual_var = np.var(residual_component[~np.isnan(residual_component)])
                seasonal_strength = seasonal_var / (seasonal_var + residual_var) if (seasonal_var + residual_var) > 0 else 0
                
            except Exception:
                # 简单的季节性分析
                seasonal_component = self._simple_seasonal_decompose(series, period)
                trend_component = self._simple_trend(series)
                residual_component = series.values - seasonal_component - trend_component
                seasonal_strength = 0.5
        else:
            # 简单的季节性分析
            seasonal_component = self._simple_seasonal_decompose(series, period)
            trend_component = self._simple_trend(series)
            residual_component = series.values - seasonal_component - trend_component
            seasonal_strength = 0.5
        
        # 频谱分析寻找主导周期
        dominant_periods = self._find_dominant_periods(series)
        
        # 寻找季节性峰值和谷值
        seasonal_pattern = seasonal_component[:period] if len(seasonal_component) >= period else seasonal_component
        peaks = signal.find_peaks(seasonal_pattern)[0]
        troughs = signal.find_peaks(-seasonal_pattern)[0]
        
        # 转换为月份（假设period=12）
        seasonal_peaks = [(p % 12) + 1 for p in peaks] if period == 12 else list(peaks)
        seasonal_troughs = [(t % 12) + 1 for t in troughs] if period == 12 else list(troughs)
        
        # 判断是否有显著季节性
        has_seasonality = seasonal_strength > 0.1
        
        return SeasonalityResult(
            variable="",
            has_seasonality=has_seasonality,
            seasonal_strength=seasonal_strength,
            dominant_periods=dominant_periods,
            seasonal_component=seasonal_component,
            trend_component=trend_component,
            residual_component=residual_component,
            seasonal_peaks=seasonal_peaks,
            seasonal_troughs=seasonal_troughs
        )
    
    def _simple_seasonal_decompose(self, series: pd.Series, period: int) -> np.ndarray:
        """简单的季节性分解"""
        values = series.values
        seasonal = np.zeros_like(values)
        
        for i in range(period):
            indices = np.arange(i, len(values), period)
            if len(indices) > 1:
                seasonal[indices] = np.mean(values[indices])
        
        return seasonal
    
    def _simple_trend(self, series: pd.Series) -> np.ndarray:
        """简单的趋势计算"""
        x = np.arange(len(series))
        y = series.values
        
        # 线性趋势
        slope, intercept = np.polyfit(x, y, 1)
        trend = slope * x + intercept
        
        return trend
    
    def _find_dominant_periods(self, series: pd.Series) -> List[float]:
        """使用FFT寻找主导周期"""
        values = series.dropna().values
        if len(values) < 10:
            return []
        
        # 去除趋势
        detrended = signal.detrend(values)
        
        # FFT
        fft_values = fft(detrended)
        freqs = fftfreq(len(detrended))
        
        # 功率谱
        power = np.abs(fft_values) ** 2
        
        # 寻找峰值
        peaks, _ = signal.find_peaks(power[1:len(power)//2], height=np.max(power) * 0.1)
        
        # 转换为周期
        periods = []
        for peak in peaks:
            if freqs[peak + 1] != 0:
                period = 1 / abs(freqs[peak + 1])
                if 2 <= period <= len(values) / 2:
                    periods.append(period)
        
        return sorted(periods)[:5]  # 返回前5个主导周期
    
    def detect_anomalies(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        variables: Optional[List[str]] = None,
        method: AnomalyMethod = AnomalyMethod.ISOLATION_FOREST,
        contamination: float = 0.1
    ) -> Dict[str, AnomalyResult]:
        """检测异常值"""
        logger.info(f"开始异常检测，方法: {method.value}")
        
        if isinstance(data, xr.Dataset):
            data = data.to_dataframe().reset_index()
        
        if variables is None:
            variables = [col for col in data.columns if col != 'time' and np.issubdtype(data[col].dtype, np.number)]
        
        results = {}
        
        for var in variables:
            if var not in data.columns:
                continue
            
            try:
                series = data[var].dropna()
                if len(series) < 10:
                    logger.warning(f"{var} 数据点太少，跳过异常检测")
                    continue
                
                result = self._detect_single_anomaly(series, method, contamination)
                result.variable = var
                results[var] = result
                
                logger.info(f"{var} 异常检测完成: 发现 {result.total_anomalies} 个异常点")
            
            except Exception as e:
                logger.error(f"检测 {var} 异常时出错: {e}")
        
        return results
    
    def _detect_single_anomaly(
        self,
        series: pd.Series,
        method: AnomalyMethod,
        contamination: float
    ) -> AnomalyResult:
        """检测单个变量的异常"""
        values = series.values.reshape(-1, 1)
        
        if method == AnomalyMethod.STATISTICAL:
            # 基于统计的异常检测（3σ准则）
            mean_val = np.mean(values)
            std_val = np.std(values)
            threshold = 3 * std_val
            
            anomaly_scores = np.abs(values.flatten() - mean_val)
            anomaly_mask = anomaly_scores > threshold
            
        elif method == AnomalyMethod.ISOLATION_FOREST:
            # 孤立森林
            if self.isolation_forest is None:
                self.isolation_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42
                )
            
            anomaly_pred = self.isolation_forest.fit_predict(values)
            anomaly_scores = -self.isolation_forest.score_samples(values)
            anomaly_mask = anomaly_pred == -1
            threshold = np.percentile(anomaly_scores, (1 - contamination) * 100)
            
        elif method == AnomalyMethod.ONE_CLASS_SVM:
            # 单类SVM
            if self.one_class_svm is None:
                self.one_class_svm = OneClassSVM(
                    nu=contamination,
                    kernel='rbf',
                    gamma='scale'
                )
            
            anomaly_pred = self.one_class_svm.fit_predict(values)
            anomaly_scores = -self.one_class_svm.score_samples(values)
            anomaly_mask = anomaly_pred == -1
            threshold = 0
            
        elif method == AnomalyMethod.DBSCAN:
            # DBSCAN聚类
            dbscan = DBSCAN(eps=np.std(values) * 0.5, min_samples=max(2, len(values) // 20))
            clusters = dbscan.fit_predict(values)
            
            # 噪声点作为异常
            anomaly_mask = clusters == -1
            anomaly_scores = np.ones(len(values))
            anomaly_scores[anomaly_mask] = 2
            threshold = 1.5
            
        elif method == AnomalyMethod.AUTOENCODER and HAS_TORCH:
            # 自编码器异常检测
            if self.autoencoder is None:
                self.autoencoder = AutoEncoder(input_dim=1)
                self._train_autoencoder(values)
            
            with torch.no_grad():
                values_tensor = torch.FloatTensor(values)
                reconstructed = self.autoencoder(values_tensor)
                reconstruction_errors = torch.mean((values_tensor - reconstructed) ** 2, dim=1)
                anomaly_scores = reconstruction_errors.numpy()
            
            threshold = np.percentile(anomaly_scores, (1 - contamination) * 100)
            anomaly_mask = anomaly_scores > threshold
            
        else:
            # 默认使用统计方法
            mean_val = np.mean(values)
            std_val = np.std(values)
            threshold = 3 * std_val
            
            anomaly_scores = np.abs(values.flatten() - mean_val)
            anomaly_mask = anomaly_scores > threshold
        
        # 提取异常信息
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        anomaly_values = values[anomaly_mask].flatten().tolist()
        anomaly_scores_list = anomaly_scores[anomaly_mask].tolist() if hasattr(anomaly_scores, '__getitem__') else []
        
        # 生成异常日期（假设数据是按时间顺序的）
        base_date = datetime(2000, 1, 1)
        anomaly_dates = [base_date + timedelta(days=30 * i) for i in anomaly_indices]
        
        return AnomalyResult(
            variable="",
            method=method,
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores_list,
            anomaly_dates=anomaly_dates,
            anomaly_values=anomaly_values,
            threshold=float(threshold) if np.isscalar(threshold) else float(np.mean(threshold)),
            total_anomalies=len(anomaly_indices),
            anomaly_rate=len(anomaly_indices) / len(values)
        )
    
    def _train_autoencoder(self, data: np.ndarray, epochs: int = 100):
        """训练自编码器"""
        if not HAS_TORCH:
            return
        
        dataset = torch.FloatTensor(data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.autoencoder.train()
        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                reconstructed = self.autoencoder(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
    
    def identify_patterns(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        variables: Optional[List[str]] = None,
        n_clusters: int = 5
    ) -> Dict[str, PatternResult]:
        """识别气候模式"""
        logger.info("开始模式识别分析")
        
        if isinstance(data, xr.Dataset):
            data = data.to_dataframe().reset_index()
        
        if variables is None:
            variables = [col for col in data.columns if col != 'time' and np.issubdtype(data[col].dtype, np.number)]
        
        results = {}
        
        for var in variables:
            if var not in data.columns:
                continue
            
            try:
                series = data[var].dropna()
                if len(series) < n_clusters * 2:
                    logger.warning(f"{var} 数据点太少，跳过模式识别")
                    continue
                
                result = self._identify_single_pattern(series, n_clusters)
                result.variable = var
                results[var] = result
                
                logger.info(f"{var} 模式识别完成: 识别出 {len(result.dominant_patterns)} 个主要模式")
            
            except Exception as e:
                logger.error(f"识别 {var} 模式时出错: {e}")
        
        return results
    
    def _identify_single_pattern(self, series: pd.Series, n_clusters: int) -> PatternResult:
        """识别单个变量的模式"""
        # 特征工程：创建滑动窗口特征
        window_size = min(12, len(series) // 4)
        features = []
        
        for i in range(len(series) - window_size + 1):
            window = series.iloc[i:i + window_size].values
            # 统计特征
            feature_vector = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.median(window),
                stats.skew(window),
                stats.kurtosis(window)
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        # 计算模式强度（轮廓系数）
        from sklearn.metrics import silhouette_score
        try:
            pattern_strength = silhouette_score(features_scaled, clusters)
        except:
            pattern_strength = 0.5
        
        # 分析主要模式
        dominant_patterns = []
        for i in range(n_clusters):
            cluster_mask = clusters == i
            if np.sum(cluster_mask) > 0:
                cluster_features = features[cluster_mask]
                pattern_info = {
                    "cluster_id": i,
                    "size": np.sum(cluster_mask),
                    "percentage": np.sum(cluster_mask) / len(clusters) * 100,
                    "mean_value": np.mean(cluster_features[:, 0]),
                    "std_value": np.mean(cluster_features[:, 1]),
                    "characteristics": self._describe_pattern(cluster_features)
                }
                dominant_patterns.append(pattern_info)
        
        # 按大小排序
        dominant_patterns.sort(key=lambda x: x['size'], reverse=True)
        
        # 计算模式转换概率
        pattern_transitions = self._calculate_transitions(clusters)
        
        return PatternResult(
            variable="",
            pattern_type="temporal_clustering",
            clusters=clusters.tolist(),
            cluster_centers=kmeans.cluster_centers_,
            pattern_strength=pattern_strength,
            dominant_patterns=dominant_patterns,
            pattern_transitions=pattern_transitions
        )
    
    def _describe_pattern(self, cluster_features: np.ndarray) -> str:
        """描述模式特征"""
        mean_val = np.mean(cluster_features[:, 0])
        std_val = np.mean(cluster_features[:, 1])
        
        if std_val < 0.5:
            variability = "稳定"
        elif std_val < 1.0:
            variability = "中等变化"
        else:
            variability = "高变化"
        
        if mean_val > 0.5:
            level = "高值"
        elif mean_val > -0.5:
            level = "中等值"
        else:
            level = "低值"
        
        return f"{level}，{variability}"
    
    def _calculate_transitions(self, clusters: np.ndarray) -> Dict[str, float]:
        """计算模式转换概率"""
        transitions = {}
        n_clusters = len(np.unique(clusters))
        
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    # 计算从模式i到模式j的转换次数
                    transitions_count = 0
                    total_i = 0
                    
                    for k in range(len(clusters) - 1):
                        if clusters[k] == i:
                            total_i += 1
                            if clusters[k + 1] == j:
                                transitions_count += 1
                    
                    if total_i > 0:
                        transitions[f"{i}->{j}"] = transitions_count / total_i
                    else:
                        transitions[f"{i}->{j}"] = 0.0
        
        return transitions
    
    def analyze_extreme_events(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        variables: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None
    ) -> Dict[str, ExtremeEventResult]:
        """分析极端事件"""
        logger.info("开始极端事件分析")
        
        if isinstance(data, xr.Dataset):
            data = data.to_dataframe().reset_index()
        
        if variables is None:
            variables = [col for col in data.columns if col != 'time' and np.issubdtype(data[col].dtype, np.number)]
        
        if event_types is None:
            event_types = ["heatwave", "coldwave", "drought", "flood"]
        
        results = {}
        
        for var in variables:
            if var not in data.columns:
                continue
            
            try:
                series = data[var].dropna()
                if len(series) < 30:  # 至少需要30个数据点
                    continue
                
                for event_type in event_types:
                    if self._is_relevant_variable(var, event_type):
                        result = self._analyze_extreme_events_single(
                            series, var, event_type
                        )
                        results[f"{var}_{event_type}"] = result
                        
                        logger.info(f"{var} {event_type} 极端事件分析完成: 频率={result.frequency:.2f}/年")
            
            except Exception as e:
                logger.error(f"分析 {var} 极端事件时出错: {e}")
        
        return results
    
    def _is_relevant_variable(self, variable: str, event_type: str) -> bool:
        """判断变量是否与事件类型相关"""
        relevance_map = {
            "heatwave": ["temperature", "temp"],
            "coldwave": ["temperature", "temp"],
            "drought": ["precipitation", "rain", "humidity"],
            "flood": ["precipitation", "rain"]
        }
        
        relevant_vars = relevance_map.get(event_type, [])
        return any(rel_var in variable.lower() for rel_var in relevant_vars)
    
    def _analyze_extreme_events_single(
        self,
        series: pd.Series,
        variable: str,
        event_type: str
    ) -> ExtremeEventResult:
        """分析单个变量的极端事件"""
        # 定义阈值
        if event_type == "heatwave":
            threshold = np.percentile(series, 95)
            condition = lambda x: x > threshold
        elif event_type == "coldwave":
            threshold = np.percentile(series, 5)
            condition = lambda x: x < threshold
        elif event_type == "drought":
            threshold = np.percentile(series, 10)
            condition = lambda x: x < threshold
        elif event_type == "flood":
            threshold = np.percentile(series, 90)
            condition = lambda x: x > threshold
        else:
            threshold = np.percentile(series, 95)
            condition = lambda x: x > threshold
        
        # 识别事件
        extreme_mask = condition(series.values)
        events = []
        
        # 连续极端值作为一个事件
        in_event = False
        event_start = None
        event_values = []
        
        for i, (idx, value) in enumerate(series.items()):
            if extreme_mask[i]:
                if not in_event:
                    in_event = True
                    event_start = i
                    event_values = [value]
                else:
                    event_values.append(value)
            else:
                if in_event:
                    # 事件结束
                    events.append({
                        "start_index": event_start,
                        "end_index": i - 1,
                        "duration": i - event_start,
                        "intensity": np.max(event_values) if event_type in ["heatwave", "flood"] else np.min(event_values),
                        "mean_intensity": np.mean(event_values),
                        "values": event_values
                    })
                    in_event = False
        
        # 处理最后一个事件
        if in_event:
            events.append({
                "start_index": event_start,
                "end_index": len(series) - 1,
                "duration": len(series) - event_start,
                "intensity": np.max(event_values) if event_type in ["heatwave", "flood"] else np.min(event_values),
                "mean_intensity": np.mean(event_values),
                "values": event_values
            })
        
        # 计算频率（事件/年）
        total_years = len(series) / 12 if len(series) > 12 else 1
        frequency = len(events) / total_years
        
        # 分析强度和持续时间趋势
        if len(events) > 2:
            intensities = [event["intensity"] for event in events]
            durations = [event["duration"] for event in events]
            
            # 创建虚拟时间序列进行趋势分析
            intensity_series = pd.Series(intensities)
            duration_series = pd.Series(durations)
            
            intensity_trend = self._analyze_single_trend(
                intensity_series, TrendMethod.LINEAR_REGRESSION, 0.95
            )
            duration_trend = self._analyze_single_trend(
                duration_series, TrendMethod.LINEAR_REGRESSION, 0.95
            )
        else:
            # 创建空的趋势结果
            intensity_trend = TrendResult(
                variable=f"{variable}_{event_type}_intensity",
                method=TrendMethod.LINEAR_REGRESSION,
                slope=0, intercept=0, r_squared=0, p_value=1,
                confidence_interval=(0, 0),
                trend_direction="stable",
                significance="not_significant",
                annual_change=0, decadal_change=0
            )
            duration_trend = TrendResult(
                variable=f"{variable}_{event_type}_duration",
                method=TrendMethod.LINEAR_REGRESSION,
                slope=0, intercept=0, r_squared=0, p_value=1,
                confidence_interval=(0, 0),
                trend_direction="stable",
                significance="not_significant",
                annual_change=0, decadal_change=0
            )
        
        # 计算重现期
        return_periods = self._calculate_return_periods(events, event_type)
        
        return ExtremeEventResult(
            variable=variable,
            event_type=event_type,
            events=events,
            frequency=frequency,
            intensity_trend=intensity_trend,
            duration_trend=duration_trend,
            return_periods=return_periods
        )
    
    def _calculate_return_periods(self, events: List[Dict], event_type: str) -> Dict[str, float]:
        """计算重现期"""
        if not events:
            return {}
        
        intensities = [event["intensity"] for event in events]
        intensities.sort(reverse=True)
        
        return_periods = {}
        for i, intensity in enumerate(intensities[:5]):  # 前5个最强事件
            rank = i + 1
            return_period = len(events) / rank
            return_periods[f"intensity_{intensity:.2f}"] = return_period
        
        return return_periods
    
    def generate_comprehensive_report(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        dataset_name: str = "Climate Data",
        variables: Optional[List[str]] = None
    ) -> AnalysisReport:
        """生成综合分析报告"""
        logger.info(f"开始生成 {dataset_name} 的综合分析报告")
        
        if isinstance(data, xr.Dataset):
            data = data.to_dataframe().reset_index()
        
        if variables is None:
            variables = [col for col in data.columns if col != 'time' and np.issubdtype(data[col].dtype, np.number)]
        
        # 确定时间范围
        if 'time' in data.columns:
            time_period = (data['time'].min(), data['time'].max())
        else:
            time_period = (datetime(2000, 1, 1), datetime(2023, 12, 31))
        
        # 执行各种分析
        trend_results = self.analyze_trends(data, variables)
        seasonality_results = self.analyze_seasonality(data, variables)
        anomaly_results = self.detect_anomalies(data, variables)
        pattern_results = self.identify_patterns(data, variables)
        extreme_events = self.analyze_extreme_events(data, variables)
        
        # 计算变量间相关性
        correlations = {}
        numeric_data = data[variables].select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            for i, var1 in enumerate(corr_matrix.columns):
                for j, var2 in enumerate(corr_matrix.columns):
                    if i < j:
                        correlations[f"{var1}_vs_{var2}"] = corr_matrix.iloc[i, j]
        
        # 生成摘要
        summary = self._generate_summary(
            trend_results, seasonality_results, anomaly_results,
            pattern_results, extreme_events, correlations
        )
        
        report = AnalysisReport(
            dataset_name=dataset_name,
            analysis_date=datetime.now(),
            time_period=time_period,
            variables_analyzed=variables,
            trend_results=trend_results,
            seasonality_results=seasonality_results,
            anomaly_results=anomaly_results,
            pattern_results=pattern_results,
            extreme_events=extreme_events,
            correlations=correlations,
            summary=summary
        )
        
        logger.info(f"{dataset_name} 综合分析报告生成完成")
        return report
    
    def _generate_summary(
        self,
        trend_results: Dict[str, TrendResult],
        seasonality_results: Dict[str, SeasonalityResult],
        anomaly_results: Dict[str, AnomalyResult],
        pattern_results: Dict[str, PatternResult],
        extreme_events: Dict[str, ExtremeEventResult],
        correlations: Dict[str, float]
    ) -> Dict[str, Any]:
        """生成分析摘要"""
        summary = {
            "total_variables": len(trend_results),
            "significant_trends": sum(1 for r in trend_results.values() if r.significance == "significant"),
            "variables_with_seasonality": sum(1 for r in seasonality_results.values() if r.has_seasonality),
            "total_anomalies": sum(r.total_anomalies for r in anomaly_results.values()),
            "total_extreme_events": sum(len(r.events) for r in extreme_events.values()),
            "strong_correlations": sum(1 for corr in correlations.values() if abs(corr) > 0.7),
            "key_findings": []
        }
        
        # 关键发现
        findings = []
        
        # 趋势发现
        increasing_vars = [var for var, result in trend_results.items() 
                          if result.trend_direction == "increasing" and result.significance == "significant"]
        if increasing_vars:
            findings.append(f"显著上升趋势变量: {', '.join(increasing_vars)}")
        
        decreasing_vars = [var for var, result in trend_results.items() 
                          if result.trend_direction == "decreasing" and result.significance == "significant"]
        if decreasing_vars:
            findings.append(f"显著下降趋势变量: {', '.join(decreasing_vars)}")
        
        # 异常发现
        high_anomaly_vars = [var for var, result in anomaly_results.items() 
                            if result.anomaly_rate > 0.1]
        if high_anomaly_vars:
            findings.append(f"高异常率变量: {', '.join(high_anomaly_vars)}")
        
        # 极端事件发现
        frequent_events = [var for var, result in extreme_events.items() 
                          if result.frequency > 1.0]
        if frequent_events:
            findings.append(f"高频极端事件: {', '.join(frequent_events)}")
        
        summary["key_findings"] = findings
        
        return summary


# 便捷函数
def analyze_climate_data(
    data: Union[pd.DataFrame, xr.Dataset],
    dataset_name: str = "Climate Data",
    variables: Optional[List[str]] = None
) -> AnalysisReport:
    """分析气候数据的便捷函数"""
    analyzer = HistoricalClimateAnalyzer()
    return analyzer.generate_comprehensive_report(data, dataset_name, variables)


def detect_climate_anomalies(
    data: Union[pd.DataFrame, xr.Dataset],
    variables: Optional[List[str]] = None,
    method: AnomalyMethod = AnomalyMethod.ISOLATION_FOREST
) -> Dict[str, AnomalyResult]:
    """检测气候异常的便捷函数"""
    analyzer = HistoricalClimateAnalyzer()
    return analyzer.detect_anomalies(data, variables, method)


if __name__ == "__main__":
    # 测试代码
    # 创建示例数据
    dates = pd.date_range('2000-01-01', '2023-12-31', freq='M')
    np.random.seed(42)
    
    # 模拟温度数据（带趋势和季节性）
    trend = np.linspace(0, 2, len(dates))  # 2度升温趋势
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)  # 季节性
    noise = np.random.normal(0, 1, len(dates))
    temperature = 15 + trend + seasonal + noise
    
    # 添加一些异常值
    temperature[100] += 15  # 极端高温
    temperature[200] -= 10  # 极端低温
    
    # 模拟降水数据
    precipitation = np.random.gamma(2, 50, len(dates))
    precipitation[150:160] = 0  # 干旱期
    
    # 创建DataFrame
    climate_data = pd.DataFrame({
        'time': dates,
        'temperature': temperature,
        'precipitation': precipitation
    })
    
    # 执行分析
    analyzer = HistoricalClimateAnalyzer()
    
    print("=== 历史气候数据分析示例 ===")
    
    # 趋势分析
    trends = analyzer.analyze_trends(climate_data)
    for var, result in trends.items():
        print(f"\n{var} 趋势分析:")
        print(f"  趋势方向: {result.trend_direction}")
        print(f"  斜率: {result.slope:.4f}")
        print(f"  显著性: {result.significance}")
        print(f"  十年变化: {result.decadal_change:.2f}")
    
    # 异常检测
    anomalies = analyzer.detect_anomalies(climate_data)
    for var, result in anomalies.items():
        print(f"\n{var} 异常检测:")
        print(f"  异常点数量: {result.total_anomalies}")
        print(f"  异常率: {result.anomaly_rate:.2%}")
    
    # 生成综合报告
    report = analyzer.generate_comprehensive_report(climate_data, "示例气候数据")
    print(f"\n=== 综合分析报告 ===")
    print(f"数据集: {report.dataset_name}")
    print(f"分析变量数: {report.summary['total_variables']}")
    print(f"显著趋势数: {report.summary['significant_trends']}")
    print(f"总异常点数: {report.summary['total_anomalies']}")
    print(f"关键发现: {report.summary['key_findings']}")