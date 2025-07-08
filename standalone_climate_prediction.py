#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立气候预测系统
基于XGBoost和Transformer模型的气候预测演示
不依赖MySQL数据库，使用模拟数据

作者: AI Assistant
日期: 2024-12-19
"""

import os
import sys
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class WeatherTransformer(nn.Module):
    """
    用于气象预测的Transformer模型
    """
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 512, 
                 dropout: float = 0.1, max_seq_length: int = 100):
        super(WeatherTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        创建位置编码
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        if seq_len <= self.max_seq_length:
            pos_encoding = self.pos_encoding[:, :seq_len, :].to(x.device)
            x = x + pos_encoding
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 输出投影
        x = self.dropout(x)
        x = self.output_projection(x)
        
        return x


class StandaloneClimatePrediction:
    """
    独立气候预测系统
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 数据相关
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.targets = None
        
        # 预处理器
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # 模型
        self.xgb_model = None
        self.transformer_model = None
    
    def generate_mock_weather_data(self, 
                                 start_date: str = '2020-01-01',
                                 end_date: str = '2024-12-31',
                                 lat_range: Tuple[float, float] = (30.67, 31.88),
                                 lon_range: Tuple[float, float] = (120.87, 122.20)) -> pd.DataFrame:
        """
        生成模拟气象数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            lat_range: 纬度范围
            lon_range: 经度范围
            
        Returns:
            pd.DataFrame: 模拟气象数据
        """
        logger.info(f"生成模拟气象数据: {start_date} 至 {end_date}")
        logger.info(f"区域范围: 纬度 {lat_range}, 经度 {lon_range}")
        
        # 创建日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # 设置随机种子以确保可重复性
        np.random.seed(42)
        
        # 生成基础气象数据
        # 温度：季节性变化 + 随机噪声
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        base_temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # 季节性变化
        temperature = base_temp + np.random.normal(0, 3, n_days)  # 添加噪声
        
        # 气压：相对稳定 + 小幅波动
        pressure = 1013 + 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 8, n_days)
        
        # 湿度：季节性变化
        humidity = 65 + 15 * np.sin(2 * np.pi * (day_of_year - 120) / 365) + np.random.normal(0, 10, n_days)
        humidity = np.clip(humidity, 20, 95)  # 限制在合理范围内
        
        # 风速：随机变化
        wind_speed = np.abs(np.random.normal(8, 4, n_days))
        wind_speed = np.clip(wind_speed, 0, 25)
        
        # 降水量：随机生成，夏季较多
        summer_factor = 1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 150) / 365)
        precipitation = np.random.exponential(2 * summer_factor, n_days)
        precipitation = np.clip(precipitation, 0, 50)
        
        # 云量：与湿度相关
        cloud_cover = 30 + 0.5 * humidity + np.random.normal(0, 15, n_days)
        cloud_cover = np.clip(cloud_cover, 0, 100)
        
        # 创建DataFrame
        data = pd.DataFrame({
            'time': dates,
            'latitude': np.random.uniform(lat_range[0], lat_range[1], n_days),
            'longitude': np.random.uniform(lon_range[0], lon_range[1], n_days),
            'temperature_celsius': temperature,
            'pressure_hpa': pressure,
            'humidity_percent': humidity,
            'wind_speed_ms': wind_speed,
            'precipitation_mm': precipitation,
            'cloud_cover_percent': cloud_cover
        })
        
        logger.info(f"生成了 {len(data)} 条气象记录")
        logger.info(f"温度范围: {data['temperature_celsius'].min():.2f}°C 至 {data['temperature_celsius'].max():.2f}°C")
        
        self.raw_data = data
        return data
    
    def prepare_features(self) -> pd.DataFrame:
        """
        准备特征数据
        
        Returns:
            pd.DataFrame: 处理后的特征数据
        """
        logger.info("开始特征工程...")
        
        if self.raw_data is None:
            raise ValueError("请先生成或加载数据")
        
        data = self.raw_data.copy()
        
        # 时间特征
        data['year'] = data['time'].dt.year
        data['month'] = data['time'].dt.month
        data['day'] = data['time'].dt.day
        data['day_of_year'] = data['time'].dt.dayofyear
        data['week_of_year'] = data['time'].dt.isocalendar().week
        data['season'] = ((data['month'] - 1) // 3) + 1
        
        # 周期性特征（正弦余弦编码）
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)
        
        # 滞后特征（前几天的温度）
        for lag in [1, 2, 3, 7, 14]:
            data[f'temp_lag_{lag}'] = data['temperature_celsius'].shift(lag)
        
        # 滚动统计特征
        for window in [3, 7, 14, 30]:
            data[f'temp_rolling_mean_{window}'] = data['temperature_celsius'].rolling(window=window).mean()
            data[f'temp_rolling_std_{window}'] = data['temperature_celsius'].rolling(window=window).std()
            data[f'pressure_rolling_mean_{window}'] = data['pressure_hpa'].rolling(window=window).mean()
            data[f'humidity_rolling_mean_{window}'] = data['humidity_percent'].rolling(window=window).mean()
        
        # 气象指数
        data['heat_index'] = data['temperature_celsius'] + 0.5 * data['humidity_percent'] / 100 * (data['temperature_celsius'] - 14)
        data['comfort_index'] = data['temperature_celsius'] - 0.4 * (data['temperature_celsius'] - 10) * (1 - data['humidity_percent'] / 100)
        
        # 删除包含NaN的行
        data = data.dropna()
        
        logger.info(f"特征工程完成，数据形状: {data.shape}")
        logger.info(f"特征列数: {data.shape[1]}")
        
        self.processed_data = data
        return data
    
    def create_sequences(self, sequence_length: int = 30, target_days: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列数据
        
        Args:
            sequence_length: 输入序列长度
            target_days: 预测天数
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 特征序列和目标序列
        """
        logger.info(f"创建时间序列，序列长度: {sequence_length}, 预测天数: {target_days}")
        
        if self.processed_data is None:
            raise ValueError("请先进行特征工程")
        
        data = self.processed_data.copy()
        
        # 选择特征列（排除时间和目标变量）
        feature_cols = [col for col in data.columns if col not in ['time', 'temperature_celsius']]
        
        # 标准化特征
        features_scaled = self.feature_scaler.fit_transform(data[feature_cols])
        targets = data['temperature_celsius'].values
        
        # 创建序列
        X, y = [], []
        
        for i in range(sequence_length, len(data) - target_days + 1):
            # 输入序列
            X.append(features_scaled[i-sequence_length:i])
            # 目标值（未来target_days天的平均温度）
            y.append(np.mean(targets[i:i+target_days]))
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"输入序列形状: {X.shape}")
        logger.info(f"目标序列形状: {y.shape}")
        
        self.features = X
        self.targets = y
        
        return X, y
    
    def train_xgboost_model(self, test_size: float = 0.2) -> Dict[str, float]:
        """
        训练XGBoost模型
        
        Args:
            test_size: 测试集比例
            
        Returns:
            Dict[str, float]: 评估指标
        """
        logger.info("开始训练XGBoost模型...")
        
        if self.features is None or self.targets is None:
            raise ValueError("请先创建序列数据")
        
        # 将序列数据展平用于XGBoost
        X_flat = self.features.reshape(self.features.shape[0], -1)
        y = self.targets
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # 标准化目标变量
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).ravel()
        
        # 创建XGBoost模型
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # 训练模型
        self.xgb_model.fit(X_train, y_train_scaled)
        
        # 预测
        y_pred_scaled = self.xgb_model.predict(X_test)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # 计算评估指标
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"XGBoost模型训练完成")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def train_transformer_model(self, 
                              test_size: float = 0.2,
                              batch_size: int = 32,
                              epochs: int = 50,
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
        logger.info("开始训练Transformer模型...")
        
        if self.features is None or self.targets is None:
            raise ValueError("请先创建序列数据")
        
        # 准备数据
        X = self.features
        y = self.targets
        
        # 分割数据集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 标准化目标变量
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).ravel()
        
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
            num_layers=4,
            dim_feedforward=256,
            dropout=0.1,
            max_seq_length=X.shape[1]
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = optim.AdamW(self.transformer_model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()
        
        # 训练循环
        best_loss = float('inf')
        patience_counter = 0
        patience = 10
        
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
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # 计算评估指标
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Transformer模型训练完成")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def ensemble_predict(self, X: np.ndarray, weights: Tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
        """
        集成预测
        
        Args:
            X: 输入特征
            weights: 模型权重 (xgb_weight, transformer_weight)
            
        Returns:
            np.ndarray: 预测结果
        """
        if self.xgb_model is None or self.transformer_model is None:
            raise ValueError("请先训练模型")
        
        # XGBoost预测
        X_flat = X.reshape(X.shape[0], -1)
        xgb_pred_scaled = self.xgb_model.predict(X_flat)
        xgb_pred = self.target_scaler.inverse_transform(xgb_pred_scaled.reshape(-1, 1)).ravel()
        
        # Transformer预测
        self.transformer_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            transformer_pred_scaled = self.transformer_model(X_tensor)[:, -1, 0].cpu().numpy()
            transformer_pred = self.target_scaler.inverse_transform(transformer_pred_scaled.reshape(-1, 1)).ravel()
        
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
        logger.info(f"正在预测未来 {days} 天的气温...")
        
        if self.processed_data is None:
            raise ValueError("请先准备数据")
        
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
            
            # 更新序列（简化版本）
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
        
        logger.info(f"未来气温预测完成")
        
        return results
    
    def create_visualizations(self, predictions: pd.DataFrame = None):
        """
        创建可视化图表
        
        Args:
            predictions: 预测结果
        """
        logger.info("正在创建可视化图表...")
        
        # 设置图表样式
        plt.style.use('default')
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
        logger.info(f"图表已保存至: {output_path}")
        
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
            logger.info(f"XGBoost模型已保存至: {xgb_path}")
        
        # 保存Transformer模型
        if self.transformer_model is not None:
            transformer_path = os.path.join(model_dir, 'transformer_model.pth')
            torch.save(self.transformer_model.state_dict(), transformer_path)
            logger.info(f"Transformer模型已保存至: {transformer_path}")
        
        # 保存预处理器
        scaler_path = os.path.join(model_dir, 'scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler
            }, f)
        logger.info(f"预处理器已保存至: {scaler_path}")


def main():
    """
    主函数 - 演示完整的气候预测流程
    """
    logger.info("开始独立气候预测系统演示")
    logger.info("目标区域: 东经120°52′至122°12′，北纬30°40′至31°53′")
    logger.info("使用模型: XGBoost + Transformer")
    
    try:
        # 1. 初始化预测系统
        prediction_system = StandaloneClimatePrediction()
        
        # 2. 生成模拟数据
        logger.info("\n" + "="*50)
        logger.info("步骤1: 生成模拟气象数据")
        logger.info("="*50)
        
        prediction_system.generate_mock_weather_data(
            start_date='2020-01-01',
            end_date='2024-12-31',
            lat_range=(30.67, 31.88),  # 北纬30°40′至31°53′
            lon_range=(120.87, 122.20)  # 东经120°52′至122°12′
        )
        
        # 3. 特征工程
        logger.info("\n" + "="*50)
        logger.info("步骤2: 特征工程")
        logger.info("="*50)
        
        prediction_system.prepare_features()
        X, y = prediction_system.create_sequences(sequence_length=30, target_days=1)
        
        # 4. 训练XGBoost模型
        logger.info("\n" + "="*50)
        logger.info("步骤3: 训练XGBoost模型")
        logger.info("="*50)
        
        xgb_metrics = prediction_system.train_xgboost_model(test_size=0.2)
        
        # 5. 训练Transformer模型
        logger.info("\n" + "="*50)
        logger.info("步骤4: 训练Transformer模型")
        logger.info("="*50)
        
        transformer_metrics = prediction_system.train_transformer_model(
            test_size=0.2,
            batch_size=32,
            epochs=30,
            learning_rate=0.001
        )
        
        # 6. 模型性能对比
        logger.info("\n" + "="*50)
        logger.info("步骤5: 模型性能对比")
        logger.info("="*50)
        
        logger.info(f"XGBoost模型 - MAE: {xgb_metrics['mae']:.4f}, RMSE: {xgb_metrics['rmse']:.4f}, R²: {xgb_metrics['r2']:.4f}")
        logger.info(f"Transformer模型 - MAE: {transformer_metrics['mae']:.4f}, RMSE: {transformer_metrics['rmse']:.4f}, R²: {transformer_metrics['r2']:.4f}")
        
        # 7. 预测未来气温
        logger.info("\n" + "="*50)
        logger.info("步骤6: 预测未来气温")
        logger.info("="*50)
        
        future_predictions = prediction_system.predict_future(days=30)
        
        # 8. 创建可视化
        logger.info("\n" + "="*50)
        logger.info("步骤7: 创建可视化图表")
        logger.info("="*50)
        
        prediction_system.create_visualizations(future_predictions)
        
        # 9. 保存模型和结果
        logger.info("\n" + "="*50)
        logger.info("步骤8: 保存模型和结果")
        logger.info("="*50)
        
        prediction_system.save_models()
        
        # 保存预测结果
        output_path = f"outputs/future_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs('outputs', exist_ok=True)
        future_predictions.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"预测结果已保存至: {output_path}")
        
        # 保存原始数据
        raw_data_path = f"outputs/mock_weather_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        prediction_system.raw_data.to_csv(raw_data_path, index=False, encoding='utf-8-sig')
        logger.info(f"原始数据已保存至: {raw_data_path}")
        
        # 10. 显示预测结果摘要
        logger.info("\n" + "="*50)
        logger.info("预测结果摘要")
        logger.info("="*50)
        
        logger.info(f"预测期间: {future_predictions['date'].min()} 至 {future_predictions['date'].max()}")
        logger.info(f"平均预测温度: {future_predictions['predicted_temperature'].mean():.2f}°C")
        logger.info(f"最高预测温度: {future_predictions['predicted_temperature'].max():.2f}°C")
        logger.info(f"最低预测温度: {future_predictions['predicted_temperature'].min():.2f}°C")
        
        # 显示前10天的详细预测
        logger.info("\n未来10天详细预测:")
        for i, row in future_predictions.head(10).iterrows():
            logger.info(f"{row['date'].strftime('%Y-%m-%d')}: {row['predicted_temperature']:.2f}°C")
        
        logger.info("\n" + "="*50)
        logger.info("气候预测系统演示完成!")
        logger.info("="*50)
        
        # 显示输出文件位置
        logger.info("\n输出文件位置:")
        logger.info("- 模型文件: models/climate_prediction/")
        logger.info("- 预测结果: outputs/")
        logger.info("- 可视化图表: outputs/")
        logger.info("- 原始数据: outputs/")
        
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()