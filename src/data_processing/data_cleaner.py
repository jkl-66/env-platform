"""数据清洗模块

负责数据质量控制、异常值过滤、格式统一和缺失值填充。
"""

import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.interpolate import interp1d
from dataclasses import dataclass
import warnings

from ..utils.logger import get_logger

logger = get_logger("data_cleaner")


@dataclass
class QualityControlConfig:
    """数据质量控制配置"""
    # 异常值检测
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0
    
    # 缺失值处理
    missing_threshold: float = 0.3  # 超过30%缺失则删除列
    interpolation_method: str = "linear"  # linear, cubic, nearest
    
    # 数据范围检查
    range_checks: Dict[str, Tuple[float, float]] = None
    
    # 时间序列特定
    time_resolution: str = "1H"  # 时间分辨率
    allow_duplicates: bool = False


class DataCleaner:
    """数据清洗器
    
    提供全面的数据清洗和质量控制功能。
    """
    
    def __init__(self, config: Optional[QualityControlConfig] = None):
        self.config = config or QualityControlConfig()
        self.scaler = StandardScaler()
        
        # 默认数据范围检查
        if self.config.range_checks is None:
            self.config.range_checks = {
                "temperature": (-100, 60),  # 摄氏度
                "precipitation": (0, 1000),  # mm
                "pressure": (800, 1100),  # hPa
                "humidity": (0, 100),  # %
                "wind_speed": (0, 200),  # m/s
            }
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        datetime_col: str = "date",
        value_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """清洗DataFrame数据
        
        Args:
            df: 输入DataFrame
            datetime_col: 时间列名
            value_cols: 数值列名列表
            
        Returns:
            清洗后的DataFrame
        """
        logger.info(f"开始清洗DataFrame数据，原始形状: {df.shape}")
        
        df_clean = df.copy()
        
        # 1. 处理时间列
        if datetime_col in df_clean.columns:
            df_clean = self._clean_datetime_column(df_clean, datetime_col)
        
        # 2. 识别数值列
        if value_cols is None:
            value_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # 3. 数据类型转换
        df_clean = self._convert_data_types(df_clean, value_cols)
        
        # 4. 移除重复行
        if not self.config.allow_duplicates:
            df_clean = self._remove_duplicates(df_clean, datetime_col)
        
        # 5. 范围检查
        df_clean = self._apply_range_checks(df_clean, value_cols)
        
        # 6. 异常值检测和处理
        df_clean = self._detect_and_handle_outliers(df_clean, value_cols)
        
        # 7. 缺失值处理
        df_clean = self._handle_missing_values(df_clean, value_cols)
        
        # 8. 时间序列规整
        if datetime_col in df_clean.columns:
            df_clean = self._regularize_time_series(df_clean, datetime_col)
        
        logger.info(f"DataFrame清洗完成，最终形状: {df_clean.shape}")
        return df_clean
    
    def clean_xarray(
        self,
        ds: xr.Dataset,
        time_dim: str = "time",
        spatial_dims: List[str] = None
    ) -> xr.Dataset:
        """清洗xarray Dataset数据
        
        Args:
            ds: 输入Dataset
            time_dim: 时间维度名
            spatial_dims: 空间维度名列表
            
        Returns:
            清洗后的Dataset
        """
        logger.info(f"开始清洗xarray数据，原始形状: {ds.sizes}")
        
        ds_clean = ds.copy()
        
        if spatial_dims is None:
            spatial_dims = ["lat", "lon"]
        
        # 1. 时间维度处理
        if time_dim in ds_clean.dims:
            ds_clean = self._clean_time_dimension(ds_clean, time_dim)
        
        # 2. 空间维度检查
        ds_clean = self._validate_spatial_dimensions(ds_clean, spatial_dims)
        
        # 3. 对每个数据变量进行清洗
        for var_name in ds_clean.data_vars:
            logger.info(f"清洗变量: {var_name}")
            ds_clean[var_name] = self._clean_data_variable(
                ds_clean[var_name], var_name
            )
        
        logger.info(f"xarray清洗完成，最终形状: {ds_clean.sizes}")
        return ds_clean
    
    def _clean_datetime_column(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """清洗时间列"""
        try:
            # 转换为datetime类型
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
            
            # 移除无效时间
            invalid_dates = df[datetime_col].isna()
            if invalid_dates.any():
                logger.warning(f"发现 {invalid_dates.sum()} 个无效时间值，已移除")
                df = df[~invalid_dates]
            
            # 排序
            df = df.sort_values(datetime_col).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"时间列清洗失败: {e}")
            return df
    
    def _convert_data_types(self, df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
        """转换数据类型"""
        for col in value_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception as e:
                    logger.warning(f"列 {col} 数据类型转换失败: {e}")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """移除重复行"""
        initial_count = len(df)
        
        if datetime_col in df.columns:
            # 基于时间和其他列去重
            df = df.drop_duplicates(subset=[datetime_col], keep="first")
        else:
            df = df.drop_duplicates()
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"移除了 {removed_count} 个重复行")
        
        return df
    
    def _apply_range_checks(self, df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
        """应用数据范围检查"""
        for col in value_cols:
            if col not in df.columns:
                continue
                
            # 查找匹配的范围检查规则
            range_check = None
            for pattern, (min_val, max_val) in self.config.range_checks.items():
                if pattern.lower() in col.lower():
                    range_check = (min_val, max_val)
                    break
            
            if range_check:
                min_val, max_val = range_check
                invalid_mask = (df[col] < min_val) | (df[col] > max_val)
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    logger.warning(
                        f"列 {col} 发现 {invalid_count} 个超出范围 [{min_val}, {max_val}] 的值"
                    )
                    df.loc[invalid_mask, col] = np.nan
        
        return df
    
    def _detect_and_handle_outliers(
        self,
        df: pd.DataFrame,
        value_cols: List[str]
    ) -> pd.DataFrame:
        """检测和处理异常值"""
        for col in value_cols:
            if col not in df.columns or df[col].isna().all():
                continue
            
            outlier_mask = self._detect_outliers(df[col])
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                logger.info(f"列 {col} 检测到 {outlier_count} 个异常值")
                
                # 将异常值设为NaN，后续通过插值处理
                df.loc[outlier_mask, col] = np.nan
        
        return df
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """检测异常值"""
        if self.config.outlier_method == "iqr":
            return self._detect_outliers_iqr(series)
        elif self.config.outlier_method == "zscore":
            return self._detect_outliers_zscore(series)
        elif self.config.outlier_method == "isolation_forest":
            return self._detect_outliers_isolation_forest(series)
        else:
            logger.warning(f"未知的异常值检测方法: {self.config.outlier_method}")
            return pd.Series(False, index=series.index)
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """使用IQR方法检测异常值"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """使用Z-score方法检测异常值"""
        z_scores = np.abs(stats.zscore(series.dropna()))
        outlier_indices = series.dropna().index[z_scores > self.config.outlier_threshold]
        
        outlier_mask = pd.Series(False, index=series.index)
        outlier_mask.loc[outlier_indices] = True
        
        return outlier_mask
    
    def _detect_outliers_isolation_forest(self, series: pd.Series) -> pd.Series:
        """使用孤立森林检测异常值"""
        try:
            from sklearn.ensemble import IsolationForest
            
            valid_data = series.dropna()
            if len(valid_data) < 10:
                return pd.Series(False, index=series.index)
            
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            outliers = iso_forest.fit_predict(valid_data.values.reshape(-1, 1))
            outlier_indices = valid_data.index[outliers == -1]
            
            outlier_mask = pd.Series(False, index=series.index)
            outlier_mask.loc[outlier_indices] = True
            
            return outlier_mask
            
        except ImportError:
            logger.warning("sklearn不可用，回退到IQR方法")
            return self._detect_outliers_iqr(series)
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        value_cols: List[str]
    ) -> pd.DataFrame:
        """处理缺失值"""
        # 1. 移除缺失值过多的列
        cols_to_drop = []
        for col in value_cols:
            if col in df.columns:
                missing_ratio = df[col].isna().sum() / len(df)
                if missing_ratio > self.config.missing_threshold:
                    cols_to_drop.append(col)
                    logger.warning(
                        f"列 {col} 缺失值比例 {missing_ratio:.2%} 超过阈值，已移除"
                    )
        
        df = df.drop(columns=cols_to_drop)
        value_cols = [col for col in value_cols if col not in cols_to_drop]
        
        # 2. 插值填充剩余缺失值
        for col in value_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = self._interpolate_missing_values(df[col])
        
        return df
    
    def _interpolate_missing_values(self, series: pd.Series) -> pd.Series:
        """插值填充缺失值"""
        if series.isna().all():
            return series
        
        try:
            if self.config.interpolation_method == "linear":
                return series.interpolate(method="linear")
            elif self.config.interpolation_method == "cubic":
                return series.interpolate(method="cubic")
            elif self.config.interpolation_method == "nearest":
                return series.interpolate(method="nearest")
            elif self.config.interpolation_method == "random_forest":
                return self._random_forest_imputation(series)
            else:
                return series.interpolate(method="linear")
                
        except Exception as e:
            logger.warning(f"插值失败，使用前向填充: {e}")
            return series.fillna(method="ffill").fillna(method="bfill")
    
    def _random_forest_imputation(self, series: pd.Series) -> pd.Series:
        """使用随机森林进行缺失值填充"""
        try:
            # 创建特征矩阵（使用索引作为特征）
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            
            # 分离有值和缺失的数据
            mask = ~np.isnan(y)
            X_train = X[mask]
            y_train = y[mask]
            X_missing = X[~mask]
            
            if len(X_train) < 5:  # 数据太少，回退到线性插值
                return series.interpolate(method="linear")
            
            # 训练随机森林
            rf = RandomForestRegressor(n_estimators=10, random_state=42)
            rf.fit(X_train, y_train)
            
            # 预测缺失值
            y_pred = rf.predict(X_missing)
            
            # 填充缺失值
            result = series.copy()
            result.iloc[~mask] = y_pred
            
            return result
            
        except Exception as e:
            logger.warning(f"随机森林插值失败，使用线性插值: {e}")
            return series.interpolate(method="linear")
    
    def _regularize_time_series(
        self,
        df: pd.DataFrame,
        datetime_col: str
    ) -> pd.DataFrame:
        """规整时间序列"""
        if datetime_col not in df.columns:
            return df
        
        try:
            # 设置时间索引
            df_ts = df.set_index(datetime_col)
            
            # 创建规整的时间索引
            start_time = df_ts.index.min()
            end_time = df_ts.index.max()
            regular_index = pd.date_range(
                start=start_time,
                end=end_time,
                freq=self.config.time_resolution
            )
            
            # 重新索引并插值
            df_regular = df_ts.reindex(regular_index)
            
            # 对数值列进行插值
            numeric_cols = df_regular.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_regular[col] = df_regular[col].interpolate(method="time")
            
            # 重置索引
            df_regular = df_regular.reset_index()
            df_regular = df_regular.rename(columns={"index": datetime_col})
            
            logger.info(f"时间序列已规整为 {self.config.time_resolution} 分辨率")
            return df_regular
            
        except Exception as e:
            logger.error(f"时间序列规整失败: {e}")
            return df
    
    def _clean_time_dimension(self, ds: xr.Dataset, time_dim: str) -> xr.Dataset:
        """清洗时间维度"""
        try:
            # 确保时间维度是datetime类型
            if not np.issubdtype(ds[time_dim].dtype, np.datetime64):
                ds[time_dim] = pd.to_datetime(ds[time_dim].values)
            
            # 移除重复时间点
            _, unique_indices = np.unique(ds[time_dim].values, return_index=True)
            if len(unique_indices) < len(ds[time_dim]):
                logger.warning(f"发现重复时间点，已移除")
                ds = ds.isel({time_dim: unique_indices})
            
            # 排序
            ds = ds.sortby(time_dim)
            
            return ds
            
        except Exception as e:
            logger.error(f"时间维度清洗失败: {e}")
            return ds
    
    def _validate_spatial_dimensions(
        self,
        ds: xr.Dataset,
        spatial_dims: List[str]
    ) -> xr.Dataset:
        """验证空间维度"""
        for dim in spatial_dims:
            if dim in ds.dims:
                # 检查坐标范围
                coord_values = ds[dim].values
                
                if dim == "lat":
                    if np.any((coord_values < -90) | (coord_values > 90)):
                        logger.warning(f"纬度坐标超出有效范围 [-90, 90]")
                elif dim == "lon":
                    if np.any((coord_values < -180) | (coord_values > 360)):
                        logger.warning(f"经度坐标超出有效范围 [-180, 360]")
        
        return ds
    
    def _clean_data_variable(self, data_array: xr.DataArray, var_name: str) -> xr.DataArray:
        """清洗单个数据变量"""
        # 应用范围检查
        for pattern, (min_val, max_val) in self.config.range_checks.items():
            if pattern.lower() in var_name.lower():
                invalid_mask = (data_array < min_val) | (data_array > max_val)
                invalid_count = invalid_mask.sum().item()
                
                if invalid_count > 0:
                    logger.warning(
                        f"变量 {var_name} 发现 {invalid_count} 个超出范围的值"
                    )
                    data_array = data_array.where(~invalid_mask)
                break
        
        return data_array
    
    def get_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """生成数据质量报告"""
        report = {
            "总行数": len(df),
            "总列数": len(df.columns),
            "数值列数": len(df.select_dtypes(include=[np.number]).columns),
            "缺失值统计": {},
            "数据类型": {},
            "基本统计": {}
        }
        
        # 缺失值统计
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_ratio = missing_count / len(df)
            report["缺失值统计"][col] = {
                "缺失数量": missing_count,
                "缺失比例": f"{missing_ratio:.2%}"
            }
        
        # 数据类型
        report["数据类型"] = df.dtypes.to_dict()
        
        # 数值列基本统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report["基本统计"] = df[numeric_cols].describe().to_dict()
        
        return report