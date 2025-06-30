#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图表生成模块

负责生成各种气候数据和预测结果的可视化图表。
"""

import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum
import base64
from io import BytesIO

# 可视化库
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns

# 地图可视化
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# 交互式可视化
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# 3D可视化
try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except ImportError:
    HAS_3D = False

from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)
config = get_config()

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class ChartType(Enum):
    """图表类型"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    MAP = "map"
    CONTOUR = "contour"
    SURFACE_3D = "surface_3d"
    TIME_SERIES = "time_series"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"


class OutputFormat(Enum):
    """输出格式"""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    BASE64 = "base64"


@dataclass
class ChartConfig:
    """图表配置"""
    title: str
    chart_type: str
    width: int = 800
    height: int = 600
    dpi: int = 100
    style: str = "default"
    color_palette: str = "husl"
    show_grid: bool = True
    show_legend: bool = True
    interactive: bool = False
    
    # 坐标轴配置
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_scale: str = "linear"  # linear, log
    y_scale: str = "linear"
    
    # 颜色和样式
    background_color: str = "white"
    grid_color: str = "lightgray"
    text_color: str = "black"
    
    # 特定图表配置
    extra_config: Optional[Dict[str, Any]] = None


@dataclass
class ChartData:
    """图表数据"""
    data: Union[pd.DataFrame, xr.Dataset, np.ndarray]
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChartGenerator:
    """图表生成器"""
    
    def __init__(self):
        self.output_path = Path(config.get("charts_path", "charts"))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 默认配置
        self.default_config = ChartConfig(
            title="气候数据图表",
            chart_type=ChartType.LINE.value
        )
        
        # 颜色主题
        self.color_themes = {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "climate": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592E83"],
            "temperature": ["#313695", "#4575b4", "#74add1", "#abd9e9", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"],
            "precipitation": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
        }
    
    def create_time_series_chart(
        self,
        data: pd.DataFrame,
        config: Optional[ChartConfig] = None,
        predictions: Optional[pd.DataFrame] = None,
        confidence_intervals: Optional[pd.DataFrame] = None
    ) -> str:
        """创建时间序列图表"""
        try:
            logger.info("创建时间序列图表")
            
            if config is None:
                config = self.default_config
                config.chart_type = ChartType.TIME_SERIES.value
            
            if config.interactive and HAS_PLOTLY:
                return self._create_interactive_time_series(data, config, predictions, confidence_intervals)
            else:
                return self._create_static_time_series(data, config, predictions, confidence_intervals)
                
        except Exception as e:
            logger.error(f"创建时间序列图表失败: {e}")
            raise
    
    def create_spatial_map(
        self,
        data: xr.Dataset,
        variable: str,
        config: Optional[ChartConfig] = None,
        time_slice: Optional[datetime] = None
    ) -> str:
        """创建空间地图"""
        try:
            logger.info(f"创建空间地图: {variable}")
            
            if config is None:
                config = self.default_config
                config.chart_type = ChartType.MAP.value
            
            if config.interactive and HAS_PLOTLY:
                return self._create_interactive_map(data, variable, config, time_slice)
            else:
                return self._create_static_map(data, variable, config, time_slice)
                
        except Exception as e:
            logger.error(f"创建空间地图失败: {e}")
            raise
    
    def create_correlation_matrix(
        self,
        data: pd.DataFrame,
        config: Optional[ChartConfig] = None,
        variables: Optional[List[str]] = None
    ) -> str:
        """创建相关性矩阵热图"""
        try:
            logger.info("创建相关性矩阵")
            
            if config is None:
                config = self.default_config
                config.chart_type = ChartType.CORRELATION.value
            
            # 选择数值列
            if variables:
                numeric_data = data[variables]
            else:
                numeric_data = data.select_dtypes(include=[np.number])
            
            # 计算相关性矩阵
            correlation_matrix = numeric_data.corr()
            
            if config.interactive and HAS_PLOTLY:
                return self._create_interactive_heatmap(correlation_matrix, config)
            else:
                return self._create_static_heatmap(correlation_matrix, config)
                
        except Exception as e:
            logger.error(f"创建相关性矩阵失败: {e}")
            raise
    
    def create_distribution_chart(
        self,
        data: pd.DataFrame,
        variable: str,
        config: Optional[ChartConfig] = None,
        group_by: Optional[str] = None
    ) -> str:
        """创建分布图"""
        try:
            logger.info(f"创建分布图: {variable}")
            
            if config is None:
                config = self.default_config
                config.chart_type = ChartType.DISTRIBUTION.value
            
            if config.interactive and HAS_PLOTLY:
                return self._create_interactive_distribution(data, variable, config, group_by)
            else:
                return self._create_static_distribution(data, variable, config, group_by)
                
        except Exception as e:
            logger.error(f"创建分布图失败: {e}")
            raise
    
    def create_comparison_chart(
        self,
        data: pd.DataFrame,
        config: Optional[ChartConfig] = None,
        chart_type: str = "bar"
    ) -> str:
        """创建对比图表"""
        try:
            logger.info(f"创建对比图表: {chart_type}")
            
            if config is None:
                config = self.default_config
                config.chart_type = chart_type
            
            if config.interactive and HAS_PLOTLY:
                return self._create_interactive_comparison(data, config)
            else:
                return self._create_static_comparison(data, config)
                
        except Exception as e:
            logger.error(f"创建对比图表失败: {e}")
            raise
    
    def create_3d_surface(
        self,
        data: xr.Dataset,
        variable: str,
        config: Optional[ChartConfig] = None
    ) -> str:
        """创建3D表面图"""
        try:
            logger.info(f"创建3D表面图: {variable}")
            
            if not HAS_3D:
                raise ImportError("需要安装3D可视化支持")
            
            if config is None:
                config = self.default_config
                config.chart_type = ChartType.SURFACE_3D.value
            
            return self._create_3d_surface(data, variable, config)
                
        except Exception as e:
            logger.error(f"创建3D表面图失败: {e}")
            raise
    
    def create_dashboard(
        self,
        charts_config: List[Dict[str, Any]],
        layout: str = "grid"
    ) -> str:
        """创建仪表板"""
        try:
            logger.info("创建仪表板")
            
            if not HAS_PLOTLY:
                raise ImportError("仪表板需要Plotly支持")
            
            return self._create_plotly_dashboard(charts_config, layout)
                
        except Exception as e:
            logger.error(f"创建仪表板失败: {e}")
            raise
    
    def _create_static_time_series(
        self,
        data: pd.DataFrame,
        config: ChartConfig,
        predictions: Optional[pd.DataFrame] = None,
        confidence_intervals: Optional[pd.DataFrame] = None
    ) -> str:
        """创建静态时间序列图"""
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        # 确保时间列是datetime类型
        time_col = self._identify_time_column(data)
        if time_col:
            data[time_col] = pd.to_datetime(data[time_col])
            data = data.set_index(time_col)
        
        # 绘制历史数据
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        colors = self.color_themes.get(config.color_palette, self.color_themes["default"])
        
        for i, col in enumerate(numeric_columns[:5]):  # 最多显示5个变量
            color = colors[i % len(colors)]
            ax.plot(data.index, data[col], label=col, color=color, linewidth=2)
        
        # 绘制预测数据
        if predictions is not None:
            pred_time_col = self._identify_time_column(predictions)
            if pred_time_col:
                predictions[pred_time_col] = pd.to_datetime(predictions[pred_time_col])
                predictions = predictions.set_index(pred_time_col)
            
            pred_numeric = predictions.select_dtypes(include=[np.number]).columns
            for i, col in enumerate(pred_numeric[:5]):
                color = colors[i % len(colors)]
                ax.plot(predictions.index, predictions[col], 
                       label=f'{col} (预测)', color=color, linestyle='--', linewidth=2)
        
        # 绘制置信区间
        if confidence_intervals is not None:
            ci_time_col = self._identify_time_column(confidence_intervals)
            if ci_time_col:
                confidence_intervals[ci_time_col] = pd.to_datetime(confidence_intervals[ci_time_col])
                confidence_intervals = confidence_intervals.set_index(ci_time_col)
            
            for i, col in enumerate(numeric_columns[:5]):
                if f'{col}_lower' in confidence_intervals.columns and f'{col}_upper' in confidence_intervals.columns:
                    color = colors[i % len(colors)]
                    ax.fill_between(
                        confidence_intervals.index,
                        confidence_intervals[f'{col}_lower'],
                        confidence_intervals[f'{col}_upper'],
                        alpha=0.2, color=color
                    )
        
        # 设置图表样式
        ax.set_title(config.title, fontsize=16, fontweight='bold')
        ax.set_xlabel(config.x_label or '时间', fontsize=12)
        ax.set_ylabel(config.y_label or '数值', fontsize=12)
        
        if config.show_grid:
            ax.grid(True, alpha=0.3)
        
        if config.show_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 格式化时间轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        filename = f"time_series_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_interactive_time_series(
        self,
        data: pd.DataFrame,
        config: ChartConfig,
        predictions: Optional[pd.DataFrame] = None,
        confidence_intervals: Optional[pd.DataFrame] = None
    ) -> str:
        """创建交互式时间序列图"""
        fig = go.Figure()
        
        # 确保时间列是datetime类型
        time_col = self._identify_time_column(data)
        if time_col:
            data[time_col] = pd.to_datetime(data[time_col])
        
        # 绘制历史数据
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        colors = self.color_themes.get(config.color_palette, self.color_themes["default"])
        
        for i, col in enumerate(numeric_columns[:5]):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=data[time_col] if time_col else data.index,
                y=data[col],
                mode='lines',
                name=col,
                line=dict(color=color, width=2)
            ))
        
        # 绘制预测数据
        if predictions is not None:
            pred_time_col = self._identify_time_column(predictions)
            if pred_time_col:
                predictions[pred_time_col] = pd.to_datetime(predictions[pred_time_col])
            
            pred_numeric = predictions.select_dtypes(include=[np.number]).columns
            for i, col in enumerate(pred_numeric[:5]):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=predictions[pred_time_col] if pred_time_col else predictions.index,
                    y=predictions[col],
                    mode='lines',
                    name=f'{col} (预测)',
                    line=dict(color=color, width=2, dash='dash')
                ))
        
        # 绘制置信区间
        if confidence_intervals is not None:
            ci_time_col = self._identify_time_column(confidence_intervals)
            if ci_time_col:
                confidence_intervals[ci_time_col] = pd.to_datetime(confidence_intervals[ci_time_col])
            
            for i, col in enumerate(numeric_columns[:5]):
                if f'{col}_lower' in confidence_intervals.columns and f'{col}_upper' in confidence_intervals.columns:
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=confidence_intervals[ci_time_col] if ci_time_col else confidence_intervals.index,
                        y=confidence_intervals[f'{col}_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    fig.add_trace(go.Scatter(
                        x=confidence_intervals[ci_time_col] if ci_time_col else confidence_intervals.index,
                        y=confidence_intervals[f'{col}_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f'rgba{tuple(list(plt.colors.to_rgba(color)[:3]) + [0.2])}',
                        name=f'{col} 置信区间',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # 设置布局
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label or '时间',
            yaxis_title=config.y_label or '数值',
            width=config.width,
            height=config.height,
            showlegend=config.show_legend,
            hovermode='x unified'
        )
        
        if config.show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # 保存图表
        filename = f"interactive_time_series_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_path / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def _create_static_map(
        self,
        data: xr.Dataset,
        variable: str,
        config: ChartConfig,
        time_slice: Optional[datetime] = None
    ) -> str:
        """创建静态地图"""
        if not HAS_CARTOPY:
            # 使用matplotlib创建简单地图
            return self._create_simple_map(data, variable, config, time_slice)
        
        fig = plt.figure(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 选择时间切片
        if time_slice and 'time' in data.dims:
            data_slice = data.sel(time=time_slice, method='nearest')
        else:
            data_slice = data.isel(time=0) if 'time' in data.dims else data
        
        # 提取变量数据
        var_data = data_slice[variable]
        
        # 创建等值线图
        if 'lat' in var_data.dims and 'lon' in var_data.dims:
            im = ax.contourf(
                var_data.lon, var_data.lat, var_data,
                levels=20, transform=ccrs.PlateCarree(),
                cmap='RdYlBu_r'
            )
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label(variable, fontsize=12)
        
        # 添加地图特征
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.OCEAN, color='lightblue')
        ax.add_feature(cfeature.LAND, color='lightgray')
        
        # 设置标题
        ax.set_title(config.title, fontsize=16, fontweight='bold')
        
        # 设置经纬度网格
        if config.show_grid:
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
        plt.tight_layout()
        
        # 保存图表
        filename = f"map_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_simple_map(
        self,
        data: xr.Dataset,
        variable: str,
        config: ChartConfig,
        time_slice: Optional[datetime] = None
    ) -> str:
        """创建简单地图（不使用cartopy）"""
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        # 选择时间切片
        if time_slice and 'time' in data.dims:
            data_slice = data.sel(time=time_slice, method='nearest')
        else:
            data_slice = data.isel(time=0) if 'time' in data.dims else data
        
        # 提取变量数据
        var_data = data_slice[variable]
        
        # 创建热图
        if 'lat' in var_data.dims and 'lon' in var_data.dims:
            im = ax.imshow(
                var_data.values,
                extent=[var_data.lon.min(), var_data.lon.max(), 
                       var_data.lat.min(), var_data.lat.max()],
                origin='lower',
                cmap='RdYlBu_r',
                aspect='auto'
            )
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(variable, fontsize=12)
        
        # 设置标签和标题
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        ax.set_title(config.title, fontsize=16, fontweight='bold')
        
        if config.show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        filename = f"simple_map_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_interactive_map(
        self,
        data: xr.Dataset,
        variable: str,
        config: ChartConfig,
        time_slice: Optional[datetime] = None
    ) -> str:
        """创建交互式地图"""
        # 选择时间切片
        if time_slice and 'time' in data.dims:
            data_slice = data.sel(time=time_slice, method='nearest')
        else:
            data_slice = data.isel(time=0) if 'time' in data.dims else data
        
        # 提取变量数据
        var_data = data_slice[variable]
        
        # 转换为DataFrame
        df = var_data.to_dataframe().reset_index()
        
        # 创建散点地图
        fig = px.scatter_mapbox(
            df,
            lat='lat',
            lon='lon',
            color=variable,
            size=variable,
            hover_data=[variable],
            color_continuous_scale='RdYlBu_r',
            title=config.title,
            mapbox_style='open-street-map',
            width=config.width,
            height=config.height
        )
        
        fig.update_layout(
            mapbox=dict(
                center=dict(
                    lat=df['lat'].mean(),
                    lon=df['lon'].mean()
                ),
                zoom=3
            )
        )
        
        # 保存图表
        filename = f"interactive_map_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_path / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def _create_static_heatmap(
        self,
        data: pd.DataFrame,
        config: ChartConfig
    ) -> str:
        """创建静态热图"""
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        # 创建热图
        sns.heatmap(
            data,
            annot=True,
            cmap='RdYlBu_r',
            center=0,
            square=True,
            ax=ax,
            cbar_kws={'shrink': 0.8}
        )
        
        ax.set_title(config.title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        filename = f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_interactive_heatmap(
        self,
        data: pd.DataFrame,
        config: ChartConfig
    ) -> str:
        """创建交互式热图"""
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='RdYlBu',
            reversescale=True,
            text=np.round(data.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height
        )
        
        # 保存图表
        filename = f"interactive_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_path / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def _create_static_distribution(
        self,
        data: pd.DataFrame,
        variable: str,
        config: ChartConfig,
        group_by: Optional[str] = None
    ) -> str:
        """创建静态分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(config.width/100, config.height/100), dpi=config.dpi)
        fig.suptitle(f'{variable} 分布分析', fontsize=16, fontweight='bold')
        
        # 直方图
        if group_by and group_by in data.columns:
            for group in data[group_by].unique():
                group_data = data[data[group_by] == group][variable]
                axes[0, 0].hist(group_data, alpha=0.7, label=str(group), bins=30)
            axes[0, 0].legend()
        else:
            axes[0, 0].hist(data[variable], bins=30, alpha=0.7)
        axes[0, 0].set_title('直方图')
        axes[0, 0].set_xlabel(variable)
        axes[0, 0].set_ylabel('频次')
        
        # 箱线图
        if group_by and group_by in data.columns:
            sns.boxplot(data=data, x=group_by, y=variable, ax=axes[0, 1])
        else:
            axes[0, 1].boxplot(data[variable].dropna())
        axes[0, 1].set_title('箱线图')
        
        # Q-Q图
        from scipy import stats
        stats.probplot(data[variable].dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q图')
        
        # 密度图
        if group_by and group_by in data.columns:
            for group in data[group_by].unique():
                group_data = data[data[group_by] == group][variable]
                axes[1, 1].hist(group_data, alpha=0.5, density=True, label=str(group), bins=30)
            axes[1, 1].legend()
        else:
            axes[1, 1].hist(data[variable], bins=30, alpha=0.7, density=True)
        axes[1, 1].set_title('密度图')
        axes[1, 1].set_xlabel(variable)
        axes[1, 1].set_ylabel('密度')
        
        plt.tight_layout()
        
        # 保存图表
        filename = f"distribution_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_interactive_distribution(
        self,
        data: pd.DataFrame,
        variable: str,
        config: ChartConfig,
        group_by: Optional[str] = None
    ) -> str:
        """创建交互式分布图"""
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('直方图', '箱线图', '小提琴图', '密度图'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 直方图
        if group_by and group_by in data.columns:
            for group in data[group_by].unique():
                group_data = data[data[group_by] == group][variable]
                fig.add_trace(
                    go.Histogram(x=group_data, name=str(group), opacity=0.7),
                    row=1, col=1
                )
        else:
            fig.add_trace(
                go.Histogram(x=data[variable], opacity=0.7),
                row=1, col=1
            )
        
        # 箱线图
        if group_by and group_by in data.columns:
            fig.add_trace(
                go.Box(x=data[group_by], y=data[variable]),
                row=1, col=2
            )
        else:
            fig.add_trace(
                go.Box(y=data[variable]),
                row=1, col=2
            )
        
        # 小提琴图
        if group_by and group_by in data.columns:
            fig.add_trace(
                go.Violin(x=data[group_by], y=data[variable]),
                row=2, col=1
            )
        else:
            fig.add_trace(
                go.Violin(y=data[variable]),
                row=2, col=1
            )
        
        # 密度图（使用直方图近似）
        if group_by and group_by in data.columns:
            for group in data[group_by].unique():
                group_data = data[data[group_by] == group][variable]
                fig.add_trace(
                    go.Histogram(x=group_data, histnorm='probability density', 
                               name=f'{group} 密度', opacity=0.7),
                    row=2, col=2
                )
        else:
            fig.add_trace(
                go.Histogram(x=data[variable], histnorm='probability density', opacity=0.7),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'{variable} 分布分析',
            width=config.width,
            height=config.height,
            showlegend=True
        )
        
        # 保存图表
        filename = f"interactive_distribution_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_path / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def _create_static_comparison(
        self,
        data: pd.DataFrame,
        config: ChartConfig
    ) -> str:
        """创建静态对比图"""
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        if config.chart_type == "bar":
            # 柱状图
            numeric_columns = data.select_dtypes(include=[np.number]).columns[:5]
            x_pos = np.arange(len(data))
            width = 0.8 / len(numeric_columns)
            
            for i, col in enumerate(numeric_columns):
                ax.bar(x_pos + i * width, data[col], width, label=col)
            
            ax.set_xticks(x_pos + width * (len(numeric_columns) - 1) / 2)
            ax.set_xticklabels(data.index)
            
        elif config.chart_type == "scatter":
            # 散点图
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) >= 2:
                ax.scatter(data[numeric_columns[0]], data[numeric_columns[1]], alpha=0.7)
                ax.set_xlabel(numeric_columns[0])
                ax.set_ylabel(numeric_columns[1])
        
        ax.set_title(config.title, fontsize=16, fontweight='bold')
        
        if config.show_legend and config.chart_type == "bar":
            ax.legend()
        
        if config.show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        filename = f"comparison_{config.chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_interactive_comparison(
        self,
        data: pd.DataFrame,
        config: ChartConfig
    ) -> str:
        """创建交互式对比图"""
        if config.chart_type == "bar":
            # 柱状图
            numeric_columns = data.select_dtypes(include=[np.number]).columns[:5]
            fig = go.Figure()
            
            for col in numeric_columns:
                fig.add_trace(go.Bar(
                    x=data.index,
                    y=data[col],
                    name=col
                ))
            
        elif config.chart_type == "scatter":
            # 散点图
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) >= 2:
                fig = px.scatter(
                    data,
                    x=numeric_columns[0],
                    y=numeric_columns[1],
                    title=config.title
                )
            else:
                fig = go.Figure()
        else:
            fig = go.Figure()
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend
        )
        
        # 保存图表
        filename = f"interactive_comparison_{config.chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_path / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def _create_3d_surface(
        self,
        data: xr.Dataset,
        variable: str,
        config: ChartConfig
    ) -> str:
        """创建3D表面图"""
        fig = plt.figure(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # 选择第一个时间切片
        if 'time' in data.dims:
            data_slice = data.isel(time=0)
        else:
            data_slice = data
        
        # 提取变量数据
        var_data = data_slice[variable]
        
        if 'lat' in var_data.dims and 'lon' in var_data.dims:
            # 创建网格
            lon_grid, lat_grid = np.meshgrid(var_data.lon, var_data.lat)
            
            # 创建3D表面
            surf = ax.plot_surface(
                lon_grid, lat_grid, var_data.values,
                cmap='RdYlBu_r',
                alpha=0.8
            )
            
            # 添加颜色条
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            ax.set_xlabel('经度')
            ax.set_ylabel('纬度')
            ax.set_zlabel(variable)
        
        ax.set_title(config.title, fontsize=16, fontweight='bold')
        
        # 保存图表
        filename = f"3d_surface_{variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = self.output_path / filename
        plt.savefig(filepath, dpi=config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_plotly_dashboard(
        self,
        charts_config: List[Dict[str, Any]],
        layout: str = "grid"
    ) -> str:
        """创建Plotly仪表板"""
        # 计算子图布局
        n_charts = len(charts_config)
        if layout == "grid":
            cols = int(np.ceil(np.sqrt(n_charts)))
            rows = int(np.ceil(n_charts / cols))
        else:
            rows = n_charts
            cols = 1
        
        # 创建子图
        subplot_titles = [chart.get('title', f'图表 {i+1}') for i, chart in enumerate(charts_config)]
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
        )
        
        # 添加图表
        for i, chart_config in enumerate(charts_config):
            row = i // cols + 1
            col = i % cols + 1
            
            chart_type = chart_config.get('type', 'line')
            data = chart_config.get('data')
            
            if chart_type == 'line' and isinstance(data, pd.DataFrame):
                for col_name in data.select_dtypes(include=[np.number]).columns[:3]:
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data[col_name], name=col_name, mode='lines'),
                        row=row, col=col
                    )
            
            elif chart_type == 'bar' and isinstance(data, pd.DataFrame):
                for col_name in data.select_dtypes(include=[np.number]).columns[:3]:
                    fig.add_trace(
                        go.Bar(x=data.index, y=data[col_name], name=col_name),
                        row=row, col=col
                    )
        
        fig.update_layout(
            title="气候数据仪表板",
            width=1200,
            height=800,
            showlegend=True
        )
        
        # 保存仪表板
        filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_path / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def _identify_time_column(self, data: pd.DataFrame) -> Optional[str]:
        """识别时间列"""
        time_candidates = ['time', 'date', 'datetime', 'timestamp']
        
        for col in data.columns:
            if col.lower() in time_candidates:
                return col
            
            # 检查数据类型
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                return col
        
        return None
    
    def export_chart(
        self,
        chart_path: str,
        output_format: str = "png",
        **kwargs
    ) -> str:
        """导出图表为不同格式"""
        try:
            input_path = Path(chart_path)
            if not input_path.exists():
                raise FileNotFoundError(f"图表文件不存在: {chart_path}")
            
            if output_format == OutputFormat.BASE64.value:
                # 转换为base64
                with open(input_path, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                return encoded
            
            elif output_format in [OutputFormat.PNG.value, OutputFormat.SVG.value, OutputFormat.PDF.value]:
                # 复制文件并更改扩展名
                output_path = input_path.with_suffix(f'.{output_format}')
                if input_path.suffix == f'.{output_format}':
                    return str(input_path)
                
                # 这里需要根据具体需求实现格式转换
                # 暂时返回原文件路径
                return str(input_path)
            
            else:
                raise ValueError(f"不支持的输出格式: {output_format}")
                
        except Exception as e:
            logger.error(f"导出图表失败: {e}")
            raise
    
    def get_chart_statistics(self) -> Dict[str, Any]:
        """获取图表统计信息"""
        chart_files = list(self.output_path.glob("*"))
        
        stats = {
            "total_charts": len(chart_files),
            "by_type": {},
            "by_format": {},
            "total_size_mb": 0
        }
        
        for file_path in chart_files:
            if file_path.is_file():
                # 按格式统计
                ext = file_path.suffix.lower()
                stats["by_format"][ext] = stats["by_format"].get(ext, 0) + 1
                
                # 按类型统计（从文件名推断）
                name = file_path.stem.lower()
                if "time_series" in name:
                    chart_type = "time_series"
                elif "map" in name:
                    chart_type = "map"
                elif "heatmap" in name:
                    chart_type = "heatmap"
                elif "distribution" in name:
                    chart_type = "distribution"
                elif "comparison" in name:
                    chart_type = "comparison"
                elif "3d" in name:
                    chart_type = "3d"
                elif "dashboard" in name:
                    chart_type = "dashboard"
                else:
                    chart_type = "other"
                
                stats["by_type"][chart_type] = stats["by_type"].get(chart_type, 0) + 1
                
                # 计算总大小
                stats["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        
        return stats


# 便捷函数
def create_climate_time_series(
    data: pd.DataFrame,
    title: str = "气候时间序列",
    interactive: bool = False,
    predictions: Optional[pd.DataFrame] = None,
    confidence_intervals: Optional[pd.DataFrame] = None
) -> str:
    """创建气候时间序列图表的便捷函数"""
    config = ChartConfig(
        title=title,
        chart_type=ChartType.TIME_SERIES.value,
        interactive=interactive,
        color_palette="climate"
    )
    
    generator = ChartGenerator()
    return generator.create_time_series_chart(data, config, predictions, confidence_intervals)


def create_climate_map(
    data: xr.Dataset,
    variable: str,
    title: Optional[str] = None,
    interactive: bool = False,
    time_slice: Optional[datetime] = None
) -> str:
    """创建气候地图的便捷函数"""
    config = ChartConfig(
        title=title or f"{variable} 空间分布",
        chart_type=ChartType.MAP.value,
        interactive=interactive,
        color_palette="climate"
    )
    
    generator = ChartGenerator()
    return generator.create_spatial_map(data, variable, config, time_slice)


def create_climate_dashboard(
    datasets: List[Dict[str, Any]],
    title: str = "气候数据仪表板"
) -> str:
    """创建气候数据仪表板的便捷函数"""
    generator = ChartGenerator()
    return generator.create_dashboard(datasets)