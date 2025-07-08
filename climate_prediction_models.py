#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于XGBoost和Transformer的气候预测模型
功能：从MySQL数据库读取气象数据，使用XGBoost和Transformer模型进行气候预测
作者：AI助手
日期：2024
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pickle

warnings.filterwarnings('ignore')

# 数据处理库
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost
import xgboost as xgb

# PyTorch和Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 数据库连接
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('climate_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeatherDataLoader:
    """
    气象数据加载器 - 从MySQL数据库加载数据
    """
    
    def __init__(self, mysql_config: Dict[str, str]):
        """
        初始化数据加载器
        
        Args:
            mysql_config: MySQL数据库连接配置
        """
        self.mysql_config = mysql_config
        self.engine = None
        self._init_database_connection()
    
    def _init_database_connection(self):
        """
        初始化数据库连接
        """
        try:
            connection_string = (
                f"mysql+pymysql://{self.mysql_config['user']}:"
                f"{self.mysql_config['password']}@{self.mysql_config['host']}:"
                f"{self.mysql_config['port']}/{self.mysql_config['database']}"
            )
            
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # 测试连接
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
            logger.info("✅ MySQL数据库连接成功")
            
        except Exception as e:
            logger.error(f"❌ MySQL数据库连接失败: {e}")
            raise
    
    def load_temperature_data(self, 
                            lat_range: Tuple[float, float] = (30.67, 31.88),  # 上海地区纬度范围
                            lon_range: Tuple[float, float] = (120.87, 122.2),  # 上海地区经度范围
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        从数据库加载指定区域的温度数据
        
        Args:
            lat_range: 纬度范围 (min_lat, max_lat)
            lon_range: 经度范围 (min_lon, max_lon)
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 温度数据
        """
        logger.info(f"🌡️ 正在加载温度数据...")
        logger.info(f"📍 纬度范围: {lat_range[0]}° - {lat_range[1]}°")
        logger.info(f"📍 经度范围: {lon_range[0]}° - {lon_range[1]}°")
        
        # 构建查询SQL
        conditions = [
            "variable = 't2m'",  # 2米气温
            f"latitude BETWEEN {lat_range[0]} AND {lat_range[1]}",
            f"longitude BETWEEN {lon_range[0]} AND {lon_range[1]}"
        ]
        
        if start_date:
            conditions.append(f"time >= '{start_date}'")
        if end_date:
            conditions.append(f"time <= '{end_date}'")
        
        where_clause = " AND ".join(conditions)
        
        sql = f"""
        SELECT time, latitude, longitude, value as temperature
        FROM grib_weather_data
        WHERE {where_clause}
        ORDER BY time, latitude, longitude
        """
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(sql, conn)
            
            if df.empty:
                logger.warning("⚠️ 未找到符合条件的温度数据，生成模拟数据")
                df = self._generate_mock_temperature_data(lat_range, lon_range)
            
            # 数据预处理
            df['time'] = pd.to_datetime(df['time'])
            df['temperature_celsius'] = df['temperature'] - 273.15  # 转换为摄氏度
            
            logger.info(f"✅ 加载完成，共 {len(df)} 条记录")
            logger.info(f"📊 时间范围: {df['time'].min()} - {df['time'].max()}")
            logger.info(f"📊 温度范围: {df['temperature_celsius'].min():.1f}°C - {df['temperature_celsius'].max():.1f}°C")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            logger.info("🔄 生成模拟数据作为替代")
            return self._generate_mock_temperature_data(lat_range, lon_range)
    
    def _generate_mock_temperature_data(self, 
                                      lat_range: Tuple[float, float],
                                      lon_range: Tuple[float, float]) -> pd.DataFrame:
        """
        生成模拟温度数据
        """
        logger.info("🔄 正在生成模拟温度数据...")
        
        # 时间范围：最近3年
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        # 空间网格
        lat_points = np.linspace(lat_range[0], lat_range[1], 10)
        lon_points = np.linspace(lon_range[0], lon_range[1], 10)
        
        data = []
        np.random.seed(42)
        
        for date in date_range:
            for lat in lat_points:
                for lon in lon_points:
                    # 基础温度（考虑季节性）
                    day_of_year = date.dayofyear
                    seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                    
                    # 添加随机噪声
                    noise = np.random.normal(0, 2)
                    temperature_k = seasonal_temp + 273.15 + noise
                    
                    data.append({
                        'time': date,
                        'latitude': lat,
                        'longitude': lon,
                        'temperature': temperature_k,
                        'temperature_celsius': temperature_k - 273.15
                    })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ 生成模拟数据完成，共 {len(df)} 条记录")
        
        return df

class WeatherTransformer(nn.Module):
    """
    基于Transformer的气候预测模型
    """
    
    def __init__(self, 
                 input_dim: int = 10,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_length: int = 365):
        """
        初始化Transformer模型
        
        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout率
            max_seq_length: 最大序列长度
        """
        super(WeatherTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.positional_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        创建位置编码
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 输出张量 [batch_size, seq_len, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        if seq_len <= self.max_seq_length:
            pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
            x = x + pos_encoding
        
        # Dropout
        x = self.dropout(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # 输出投影
        output = self.output_projection(x)  # [batch_size, seq_len, 1]
        
        return output

class ClimatePredictionSystem:
    """
    气候预测系统 - 结合XGBoost和Transformer模型
    """
    
    def __init__(self, mysql_config: Dict[str, str]):
        """
        初始化预测系统
        
        Args:
            mysql_config: MySQL数据库配置
        """
        self.mysql_config = mysql_config
        self.data_loader = WeatherDataLoader(mysql_config)
        
        # 模型
        self.xgb_model = None
        self.transformer_model = None
        
        # 数据预处理器
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 使用设备: {self.device}")
        
        # 数据
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.targets = None
    
    def load_and_prepare_data(self, 
                            lat_range: Tuple[float, float] = (30.67, 31.88),
                            lon_range: Tuple[float, float] = (120.87, 122.2),
                            sequence_length: int = 30) -> bool:
        """
        加载和准备训练数据
        
        Args:
            lat_range: 纬度范围
            lon_range: 经度范围
            sequence_length: 序列长度
            
        Returns:
            bool: 是否成功
        """
        try:
            logger.info("📊 正在加载和准备数据...")
            
            # 加载原始数据
            self.raw_data = self.data_loader.load_temperature_data(
                lat_range=lat_range,
                lon_range=lon_range
            )
            
            if self.raw_data.empty:
                logger.error("❌ 未能加载数据")
                return False
            
            # 数据聚合 - 按日期计算平均温度
            daily_data = self.raw_data.groupby('time').agg({
                'temperature_celsius': 'mean',
                'latitude': 'mean',
                'longitude': 'mean'
            }).reset_index()
            
            daily_data = daily_data.sort_values('time').reset_index(drop=True)
            
            # 创建特征
            self.processed_data = self._create_features(daily_data)
            
            # 创建序列数据
            self.features, self.targets = self._create_sequences(
                self.processed_data, sequence_length
            )
            
            logger.info(f"✅ 数据准备完成")
            logger.info(f"📊 特征形状: {self.features.shape}")
            logger.info(f"📊 目标形状: {self.targets.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据准备失败: {e}")
            return False
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建特征工程
        """
        logger.info("🔧 正在创建特征...")
        
        df = data.copy()
        
        # 时间特征
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day'] = df['time'].dt.day
        df['day_of_year'] = df['time'].dt.dayofyear
        df['week_of_year'] = df['time'].dt.isocalendar().week
        df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # 冬季
                                       3: 1, 4: 1, 5: 1,   # 春季
                                       6: 2, 7: 2, 8: 2,   # 夏季
                                       9: 3, 10: 3, 11: 3}) # 秋季
        
        # 周期性特征
        df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 滞后特征
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'temp_lag_{lag}'] = df['temperature_celsius'].shift(lag)
        
        # 滑动窗口统计特征
        for window in [3, 7, 14, 30]:
            df[f'temp_mean_{window}d'] = df['temperature_celsius'].rolling(window=window).mean()
            df[f'temp_std_{window}d'] = df['temperature_celsius'].rolling(window=window).std()
            df[f'temp_min_{window}d'] = df['temperature_celsius'].rolling(window=window).min()
            df[f'temp_max_{window}d'] = df['temperature_celsius'].rolling(window=window).max()
        
        # 温度变化特征
        df['temp_diff_1d'] = df['temperature_celsius'].diff(1)
        df['temp_diff_7d'] = df['temperature_celsius'].diff(7)
        
        # 删除包含NaN的行
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"✅ 特征创建完成，共 {len(df)} 条记录，{len(df.columns)} 个特征")
        
        return df
    
    def _create_sequences(self, data: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据
        """
        logger.info(f"🔧 正在创建序列数据 (序列长度: {sequence_length})...")
        
        # 选择特征列（排除时间和目标变量）
        feature_cols = [col for col in data.columns 
                       if col not in ['time', 'temperature_celsius']]
        
        # 准备数据
        feature_data = data[feature_cols].values
        target_data = data['temperature_celsius'].values
        
        # 标准化特征
        feature_data = self.feature_scaler.fit_transform(feature_data)
        
        # 创建序列
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(feature_data[i-sequence_length:i])
            y.append(target_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✅ 序列数据创建完成")
        logger.info(f"📊 输入序列形状: {X.shape}")
        logger.info(f"📊 目标序列形状: {y.shape}")
        
        return X, y
    
    def train_xgboost_model(self, test_size: float = 0.2) -> Dict[str, float]:
        """
        训练XGBoost模型
        
        Args:
            test_size: 测试集比例
            
        Returns:
            Dict[str, float]: 评估指标
        """
        logger.info("🚀 开始训练XGBoost模型...")
        
        if self.features is None or self.targets is None:
            raise ValueError("❌ 请先加载和准备数据")
        
        # 将序列数据展平用于XGBoost
        X_flat = self.features.reshape(self.features.shape[0], -1)
        y = self.targets
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # 标准化目标变量
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)).ravel()
        
        # 创建XGBoost模型
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        
        # 训练模型
        self.xgb_model.fit(
            X_train, y_train_scaled,
            eval_set=[(X_test, y_test_scaled)],
            verbose=100
        )
        
        # 预测
        y_pred_scaled = self.xgb_model.predict(X_test)
        y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # 计算评估指标
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"✅ XGBoost模型训练完成")
        logger.info(f"📊 MAE: {metrics['mae']:.4f}")
        logger.info(f"📊 RMSE: {metrics['rmse']:.4f}")
        logger.info(f"📊 R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def train_transformer_model(self, 
                              test_size: float = 0.2,
                              batch_size: int = 32,
                              epochs: int = 100,
                              learning_rate: float = 0.001) -> Dict[str, float]:
        """
        训练Transformer模型
        
        Args:
            test_size: 测试集比例
            batch_size: 批次大小
            epochs: 训练轮数
            learning_rate: 学习率
            
        Returns:
            Dict[str, float]: 评估指标
        """
        logger.info("🚀 开始训练Transformer模型...")
        
        if self.features is None or self.targets is None:
            raise ValueError("❌ 请先加载和准备数据")
        
        # 准备数据
        X = self.features
        y = self.targets
        
        # 分割数据集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 标准化目标变量
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)).ravel()
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test_scaled).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 创建模型
        input_dim = X.shape[2]
        self.transformer_model = WeatherTransformer(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_layers=6,
            dim_feedforward=512,
            dropout=0.1,
            max_seq_length=X.shape[1]
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = optim.AdamW(self.transformer_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        criterion = nn.MSELoss()
        
        # 训练循环
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # 训练阶段
            self.transformer_model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.transformer_model(batch_X)
                outputs = outputs[:, -1, 0]  # 取最后一个时间步的输出
                
                # 计算损失
                loss = criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer_model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            self.transformer_model.eval()
            with torch.no_grad():
                val_outputs = self.transformer_model(X_test_tensor)
                val_outputs = val_outputs[:, -1, 0]
                val_loss = criterion(val_outputs, y_test_tensor).item()
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.transformer_model.state_dict(), 'best_transformer_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")
            
            if patience_counter >= patience:
                logger.info(f"早停于第 {epoch} 轮")
                break
        
        # 加载最佳模型
        self.transformer_model.load_state_dict(torch.load('best_transformer_model.pth'))
        
        # 最终评估
        self.transformer_model.eval()
        with torch.no_grad():
            y_pred_scaled = self.transformer_model(X_test_tensor)[:, -1, 0].cpu().numpy()
            y_pred = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # 计算评估指标
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"✅ Transformer模型训练完成")
        logger.info(f"📊 MAE: {metrics['mae']:.4f}")
        logger.info(f"📊 RMSE: {metrics['rmse']:.4f}")
        logger.info(f"📊 R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def ensemble_predict(self, X: np.ndarray, weights: Tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
        """
        集成预测 - 结合XGBoost和Transformer的预测结果
        
        Args:
            X: 输入特征
            weights: 模型权重 (xgb_weight, transformer_weight)
            
        Returns:
            np.ndarray: 预测结果
        """
        if self.xgb_model is None or self.transformer_model is None:
            raise ValueError("❌ 请先训练模型")
        
        # XGBoost预测
        X_flat = X.reshape(X.shape[0], -1)
        xgb_pred_scaled = self.xgb_model.predict(X_flat)
        xgb_pred = self.scaler.inverse_transform(xgb_pred_scaled.reshape(-1, 1)).ravel()
        
        # Transformer预测
        self.transformer_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            transformer_pred_scaled = self.transformer_model(X_tensor)[:, -1, 0].cpu().numpy()
            transformer_pred = self.scaler.inverse_transform(transformer_pred_scaled.reshape(-1, 1)).ravel()
        
        # 加权平均
        ensemble_pred = weights[0] * xgb_pred + weights[1] * transformer_pred
        
        return ensemble_pred
    
    def predict_future(self, days: int = 30) -> pd.DataFrame:
        """
        预测未来气温
        
        Args:
            days: 预测天数
            
        Returns:
            pd.DataFrame: 预测结果
        """
        logger.info(f"🔮 正在预测未来 {days} 天的气温...")
        
        if self.processed_data is None:
            raise ValueError("❌ 请先加载和准备数据")
        
        # 获取最后的序列数据
        last_sequence = self.features[-1:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        # 获取最后的日期
        last_date = self.processed_data['time'].iloc[-1]
        
        for i in range(days):
            # 预测下一天
            pred = self.ensemble_predict(current_sequence)
            predictions.append(pred[0])
            
            # 更新序列（简化版本，实际应用中需要更复杂的特征更新）
            # 这里我们只是简单地滚动序列
            new_features = current_sequence[0, -1:].copy()
            current_sequence = np.concatenate([
                current_sequence[:, 1:],
                new_features.reshape(1, 1, -1)
            ], axis=1)
        
        # 创建预测结果DataFrame
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        results = pd.DataFrame({
            'date': future_dates,
            'predicted_temperature': predictions
        })
        
        logger.info(f"✅ 未来气温预测完成")
        
        return results
    
    def create_visualizations(self, predictions: pd.DataFrame = None):
        """
        创建可视化图表
        
        Args:
            predictions: 预测结果
        """
        logger.info("📊 正在创建可视化图表...")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('气候预测模型分析结果', fontsize=16, fontweight='bold')
        
        # 1. 历史温度趋势
        if self.processed_data is not None:
            axes[0, 0].plot(self.processed_data['time'], self.processed_data['temperature_celsius'], 
                           alpha=0.7, linewidth=1, label='历史气温')
            axes[0, 0].set_title('历史气温趋势')
            axes[0, 0].set_xlabel('时间')
            axes[0, 0].set_ylabel('温度 (°C)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 预测结果
        if predictions is not None:
            axes[0, 1].plot(predictions['date'], predictions['predicted_temperature'], 
                           'r-', linewidth=2, marker='o', markersize=4, label='预测气温')
            axes[0, 1].set_title('未来气温预测')
            axes[0, 1].set_xlabel('时间')
            axes[0, 1].set_ylabel('温度 (°C)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 旋转x轴标签
            for tick in axes[0, 1].get_xticklabels():
                tick.set_rotation(45)
        
        # 3. 温度分布
        if self.processed_data is not None:
            axes[1, 0].hist(self.processed_data['temperature_celsius'], bins=50, 
                           alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].set_title('历史气温分布')
            axes[1, 0].set_xlabel('温度 (°C)')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 季节性分析
        if self.processed_data is not None:
            monthly_temp = self.processed_data.groupby('month')['temperature_celsius'].mean()
            axes[1, 1].bar(monthly_temp.index, monthly_temp.values, 
                          color='lightcoral', alpha=0.8, edgecolor='black')
            axes[1, 1].set_title('月平均气温')
            axes[1, 1].set_xlabel('月份')
            axes[1, 1].set_ylabel('平均温度 (°C)')
            axes[1, 1].set_xticks(range(1, 13))
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = f"outputs/climate_prediction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        os.makedirs('outputs', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 图表已保存至: {output_path}")
        
        plt.show()
    
    def save_models(self, model_dir: str = 'models/climate_prediction'):
        """
        保存训练好的模型
        
        Args:
            model_dir: 模型保存目录
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存XGBoost模型
        if self.xgb_model is not None:
            xgb_path = os.path.join(model_dir, 'xgboost_model.pkl')
            with open(xgb_path, 'wb') as f:
                pickle.dump(self.xgb_model, f)
            logger.info(f"✅ XGBoost模型已保存至: {xgb_path}")
        
        # 保存Transformer模型
        if self.transformer_model is not None:
            transformer_path = os.path.join(model_dir, 'transformer_model.pth')
            torch.save(self.transformer_model.state_dict(), transformer_path)
            logger.info(f"✅ Transformer模型已保存至: {transformer_path}")
        
        # 保存预处理器
        scaler_path = os.path.join(model_dir, 'scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'target_scaler': self.scaler,
                'feature_scaler': self.feature_scaler
            }, f)
        logger.info(f"✅ 预处理器已保存至: {scaler_path}")
    
    def load_models(self, model_dir: str = 'models/climate_prediction'):
        """
        加载训练好的模型
        
        Args:
            model_dir: 模型保存目录
        """
        # 加载XGBoost模型
        xgb_path = os.path.join(model_dir, 'xgboost_model.pkl')
        if os.path.exists(xgb_path):
            with open(xgb_path, 'rb') as f:
                self.xgb_model = pickle.load(f)
            logger.info(f"✅ XGBoost模型已加载")
        
        # 加载预处理器
        scaler_path = os.path.join(model_dir, 'scalers.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.scaler = scalers['target_scaler']
                self.feature_scaler = scalers['feature_scaler']
            logger.info(f"✅ 预处理器已加载")
        
        logger.info(f"📁 模型加载完成")


def main():
    """
    主函数 - 演示完整的气候预测流程
    """
    logger.info("🌟 开始气候预测系统演示")
    
    try:
        # 1. 初始化预测系统
        prediction_system = ClimatePredictionSystem()
        
        # 2. 加载数据
        logger.info("📊 正在加载气象数据...")
        prediction_system.load_data_from_mysql(
            lat_range=(30.67, 31.88),  # 北纬30°40′至31°53′
            lon_range=(120.87, 122.20),  # 东经120°52′至122°12′
            start_date='2020-01-01',
            end_date='2024-12-31'
        )
        
        # 3. 准备训练数据
        logger.info("🔧 正在准备训练数据...")
        prediction_system.prepare_features()
        X, y = prediction_system.create_sequences(sequence_length=30, target_days=1)
        
        # 4. 训练XGBoost模型
        logger.info("🚀 开始训练XGBoost模型...")
        xgb_metrics = prediction_system.train_xgboost_model(test_size=0.2)
        
        # 5. 训练Transformer模型
        logger.info("🚀 开始训练Transformer模型...")
        transformer_metrics = prediction_system.train_transformer_model(
            test_size=0.2,
            batch_size=32,
            epochs=50,
            learning_rate=0.001
        )
        
        # 6. 模型性能对比
        logger.info("📊 模型性能对比:")
        logger.info(f"XGBoost - MAE: {xgb_metrics['mae']:.4f}, RMSE: {xgb_metrics['rmse']:.4f}, R²: {xgb_metrics['r2']:.4f}")
        logger.info(f"Transformer - MAE: {transformer_metrics['mae']:.4f}, RMSE: {transformer_metrics['rmse']:.4f}, R²: {transformer_metrics['r2']:.4f}")
        
        # 7. 预测未来气温
        logger.info("🔮 正在预测未来30天气温...")
        future_predictions = prediction_system.predict_future(days=30)
        
        # 8. 创建可视化
        logger.info("📊 正在创建可视化图表...")
        prediction_system.create_visualizations(future_predictions)
        
        # 9. 保存模型
        logger.info("💾 正在保存模型...")
        prediction_system.save_models()
        
        # 10. 保存预测结果
        output_path = f"outputs/future_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs('outputs', exist_ok=True)
        future_predictions.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"📄 预测结果已保存至: {output_path}")
        
        # 11. 显示预测结果摘要
        logger.info("📋 预测结果摘要:")
        logger.info(f"预测期间: {future_predictions['date'].min()} 至 {future_predictions['date'].max()}")
        logger.info(f"平均预测温度: {future_predictions['predicted_temperature'].mean():.2f}°C")
        logger.info(f"最高预测温度: {future_predictions['predicted_temperature'].max():.2f}°C")
        logger.info(f"最低预测温度: {future_predictions['predicted_temperature'].min():.2f}°C")
        
        logger.info("✅ 气候预测系统演示完成!")
        
    except Exception as e:
        logger.error(f"❌ 系统运行出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()