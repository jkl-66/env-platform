#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能夏季最高气温预测系统 - 深度学习版本
使用深度神经网络预测上海地区（东经120°52′至122°12′，北纬30°40′至31°53′）夏季最高气温
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 深度学习相关库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn.functional as F
    import xgboost as xgb
    from sqlalchemy import create_engine
    import json

    # GPU配置
    print("🔧 配置GPU加速...")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        print(f"✅ 检测到 {gpu_count} 个GPU设备，已启用GPU加速")
        print(f"📱 GPU设备: {gpu_names}")
        print(f"🎯 使用设备: {device}")
    else:
        device = torch.device('cpu')
        print("⚠️ 未检测到GPU设备，将使用CPU训练")

    print("✅ PyTorch 已加载")
except ImportError:
    print("⚠️ 正在安装 PyTorch...")
    import subprocess
    subprocess.check_call(["pip", "install", "torch", "torchvision", "torchaudio"])
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    
    # GPU配置
    print("🔧 配置GPU加速...")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        print(f"✅ 检测到 {gpu_count} 个GPU设备，已启用GPU加速")
        print(f"📱 GPU设备: {gpu_names}")
        print(f"🎯 使用设备: {device}")
    else:
        device = torch.device('cpu')
        print("⚠️ 未检测到GPU设备，将使用CPU训练")

# 科学计算库
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.signal import savgol_filter
import math

# 数据处理库
try:
    import xarray as xr
    print("✅ xarray 已加载")
except ImportError:
    print("⚠️ 正在安装 xarray...")
    import subprocess
    subprocess.check_call(["pip", "install", "xarray"])
    import xarray as xr

try:
    import cfgrib
    print("✅ cfgrib 已加载")
except ImportError:
    print("⚠️ 正在安装 cfgrib...")
    import subprocess
    subprocess.check_call(["pip", "install", "cfgrib"])
    import cfgrib

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class IntelligentTemperaturePrediction:
    """
    智能夏季最高气温预测系统
    使用深度学习模型预测上海地区夏季最高气温
    """
    
    def __init__(self, grib_file_path=None, db_config_path='mysql_config.json'):
        self.grib_file_path = grib_file_path
        self.data = None
        self.processed_data = None
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 上海地区地理范围
        self.lon_range = (120.87, 122.2)  # 东经120°52′至122°12′
        self.lat_range = (30.67, 31.88)   # 北纬30°40′至31°53′
        
        # 上海地区夏季气温特征（基于历史数据）
        self.shanghai_climate = {
            'summer_avg_high': 32.0,  # 夏季平均最高气温
            'summer_max_record': 40.9,  # 历史最高气温记录
            'summer_min_high': 28.0,   # 夏季最低的最高气温
            'peak_months': [7, 8],     # 最热月份
            'heat_wave_threshold': 35.0  # 高温预警阈值
        }
        
        print("🌡️ 智能夏季最高气温预测系统已初始化")
        print(f"📍 目标区域: 上海地区 (东经{self.lon_range[0]:.2f}°-{self.lon_range[1]:.2f}°, 北纬{self.lat_range[0]:.2f}°-{self.lat_range[1]:.2f}°)")

    def load_data_from_db(self):
        """
        从MySQL数据库加载气候数据
        """
        try:
            with open(self.db_config_path, 'r') as f:
                config = json.load(f)['mysql']
            
            engine = create_engine(f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/jkl")
            
            query = """
            SELECT * FROM climate_metadata 
            WHERE latitude BETWEEN 30.67 AND 31.88 
            AND longitude BETWEEN 120.87 AND 122.2
            """
            
            self.data = pd.read_sql(query, engine)
            
            if self.data.empty:
                print("数据库中未找到指定范围内的数据，将生成模拟数据。")
                return self._generate_shanghai_simulated_data()

            self.data['time'] = pd.to_datetime(self.data['time'])
            self.data = self.data.set_index('time')
            self.data = self.data.rename(columns={'temperature': 't2m'})
            self.processed_data = self.data.copy()

            print(f"✅ 从数据库成功加载 {len(self.data)} 条数据")
            return True
        except Exception as e:
            print(f"❌ 从数据库加载数据失败: {e}")
            print("🔄 将生成上海地区模拟数据...")
            return self._generate_shanghai_simulated_data()

    def load_grib_data(self):
        """
        加载GRIB数据并提取上海地区数据
        """
        if not self.grib_file_path:
            print("⚠️ 未提供GRIB文件路径，将生成上海地区模拟数据")
            return self._generate_shanghai_simulated_data()
        
        try:
            print(f"📂 正在加载GRIB数据: {self.grib_file_path}")
            
            # 检查文件是否存在
            import os
            if not os.path.exists(self.grib_file_path):
                print(f"❌ 文件不存在: {self.grib_file_path}")
                print("🔄 将生成上海地区模拟数据...")
                return self._generate_shanghai_simulated_data()
            
            # 使用xarray和cfgrib读取GRIB数据
            self.data = xr.open_dataset(self.grib_file_path, engine='cfgrib')
            
            # 查找温度相关变量
            temp_vars = [var for var in self.data.data_vars if any(keyword in var.lower() 
                        for keyword in ['temp', 't2m', 'temperature', '2t'])]
            
            if not temp_vars:
                print("⚠️ 未找到温度变量，查看所有变量:")
                print(list(self.data.data_vars.keys()))
                if self.data.data_vars:
                    temp_var = list(self.data.data_vars.keys())[0]
                    print(f"📊 使用变量: {temp_var}")
                else:
                    raise ValueError("GRIB文件中未找到任何数据变量")
            else:
                temp_var = temp_vars[0]
                print(f"🌡️ 找到温度变量: {temp_var}")
            
            # 重命名温度变量为标准名称
            if temp_var != 't2m':
                self.data = self.data.rename({temp_var: 't2m'})
            
            # 转换为摄氏度（如果需要）
            if self.data['t2m'].attrs.get('units') == 'K':
                self.data['t2m'] = self.data['t2m'] - 273.15
                self.data['t2m'].attrs['units'] = 'C'
            
            # 提取上海地区数据
            self.data = self.data.sel(
                longitude=slice(self.lon_range[0], self.lon_range[1]),
                latitude=slice(self.lat_range[0], self.lat_range[1])
            )
            
            print(f"✅ 数据加载成功!")
            print(f"📊 数据维度: {self.data.dims}")
            print(f"🗓️ 时间范围: {self.data.time.min().values} 到 {self.data.time.max().values}")
            
            return True
            
        except Exception as e:
            print(f"❌ 读取GRIB文件失败: {e}")
            print("🔄 将生成上海地区模拟数据...")
            return self._generate_shanghai_simulated_data()
    
    def _generate_shanghai_simulated_data(self):
        """
        生成符合上海地区气候特征的模拟数据
        """
        print("🔄 正在生成上海地区夏季气温模拟数据...")
        
        # 生成1980-2024年的数据
        start_year = 1980
        end_year = 2024
        years = list(range(start_year, end_year + 1))
        
        # 创建时间序列
        dates = []
        temperatures = []
        
        for year in years:
            # 夏季月份：6, 7, 8月
            for month in [6, 7, 8]:
                days_in_month = 30 if month in [6, 8] else 31
                for day in range(1, days_in_month + 1):
                    try:
                        date = datetime(year, month, day)
                        dates.append(date)
                        
                        # 生成符合上海气候的温度数据
                        temp = self._generate_realistic_shanghai_temperature(year, month, day)
                        temperatures.append(temp)
                    except ValueError:
                        continue
        
        # 创建DataFrame
        self.processed_data = pd.DataFrame({
            'date': dates,
            'temperature': temperatures,
            'year': [d.year for d in dates],
            'month': [d.month for d in dates],
            'day': [d.day for d in dates],
            'day_of_year': [d.timetuple().tm_yday for d in dates]
        })
        
        print(f"✅ 生成了 {len(self.processed_data)} 个数据点")
        print(f"🌡️ 温度范围: {self.processed_data['temperature'].min():.1f}°C - {self.processed_data['temperature'].max():.1f}°C")
        print(f"📊 平均温度: {self.processed_data['temperature'].mean():.1f}°C")
        
        return True
    
    def _generate_realistic_shanghai_temperature(self, year, month, day):
        """
        生成符合上海地区实际气候的温度数据
        """
        # 基础温度（根据月份）
        base_temps = {6: 29.0, 7: 33.0, 8: 32.5}
        base_temp = base_temps[month]
        
        # 年际变化趋势（全球变暖影响）
        warming_trend = (year - 1980) * 0.02  # 每年升温0.02°C
        
        # 月内变化（中旬最热）
        if day <= 10:
            month_factor = 0.8 + (day / 10) * 0.2
        elif day <= 20:
            month_factor = 1.0
        else:
            month_factor = 1.0 - ((day - 20) / 10) * 0.15
        
        # 随机波动
        daily_variation = np.random.normal(0, 2.5)
        
        # 极端天气事件（热浪）
        heat_wave_prob = 0.05 if month == 7 else 0.02
        if np.random.random() < heat_wave_prob:
            heat_wave_boost = np.random.uniform(3, 8)
        else:
            heat_wave_boost = 0
        
        # 计算最终温度
        temperature = base_temp + warming_trend + (base_temp * (month_factor - 1)) + daily_variation + heat_wave_boost
        
        # 确保温度在合理范围内
        temperature = np.clip(temperature, 22.0, 42.0)
        
        return temperature
    
    def create_features(self):
        """
        创建模型的特征
        """
        print("🔧 正在创建深度学习特征...")
        
        df = self.processed_data.copy()
        
        # 时间特征
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 滞后特征
        for lag in [1, 2, 3, 5, 7, 14]:
            df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
        
        # 滑动窗口统计特征
        for window in [3, 7, 14, 30]:
            df[f'temp_mean_{window}'] = df['temperature'].rolling(window=window).mean()
            df[f'temp_std_{window}'] = df['temperature'].rolling(window=window).std()
            df[f'temp_max_{window}'] = df['temperature'].rolling(window=window).max()
            df[f'temp_min_{window}'] = df['temperature'].rolling(window=window).min()
        
        # 年际趋势
        df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        
        # 热浪指标
        df['is_heat_wave'] = (df['temperature'] > self.shanghai_climate['heat_wave_threshold']).astype(int)
        df['heat_wave_duration'] = df.groupby((df['is_heat_wave'] != df['is_heat_wave'].shift()).cumsum())['is_heat_wave'].cumsum()
        
        # 删除包含NaN的行
        df = df.dropna()
        
        self.processed_data = df
        print(f"✅ 特征创建完成，共 {len(df)} 个样本，{len(df.columns)} 个特征")
        
        return df
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    def build_models(self, input_dim, model_type='xgboost'):
        """
        构建模型
        """
        if model_type == 'xgboost':
            model = xgb.XGBRegressor(objective='reg:squarederror',
                                     n_estimators=1000,
                                     learning_rate=0.05,
                                     max_depth=5,
                                     min_child_weight=1,
                                     gamma=0.1,
                                     subsample=0.8,
                                     colsample_bytree=0.8,
                                     reg_alpha=0.005,
                                     random_state=42,
                                     n_jobs=-1)
        elif model_type == 'transformer':
            class TransformerModel(nn.Module):
                def __init__(self, input_dim, model_dim=128, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
                    super(TransformerModel, self).__init__()
                    self.model_type = 'Transformer'
                    self.src_mask = None
                    self.pos_encoder = PositionalEncoding(model_dim, dropout)
                    self.transformer = nn.Transformer(model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
                    self.encoder = nn.Linear(input_dim, model_dim)
                    self.decoder = nn.Linear(model_dim, 1)

                def forward(self, src):
                    src = self.encoder(src)
                    src = self.pos_encoder(src)
                    output = self.transformer(src, src)
                    output = self.decoder(output)
                    return output
            model = TransformerModel(input_dim)
            model = model.to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
        
        return model
    
    def train_models(self):
        """
        训练多个深度学习模型
        """
        print("🚀 开始训练深度学习模型...")
        
        # 准备特征和目标变量
        feature_cols = [col for col in self.processed_data.columns 
                       if col not in ['date', 'temperature']]
        
        X = self.processed_data[feature_cols].values
        y = self.processed_data['temperature'].values
        
        # 数据标准化
        self.scalers['X'] = StandardScaler()
        self.scalers['y'] = StandardScaler()
        
        X_scaled = self.scalers['X'].fit_transform(X)
        y_scaled = self.scalers['y'].fit_transform(y.reshape(-1, 1)).ravel()
        
        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=5)
        
        model_configs = {
            'xgboost': {'type': 'xgboost'},
            'transformer': {'type': 'transformer', 'epochs': 100}
        }
        
        for model_name, config in model_configs.items():
            print(f"\n🔄 训练 {model_name} 模型...")
            
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                print(f"  📊 训练折 {fold + 1}/5...")
                
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
                
                input_dim = X_train.shape[1]
                
                # 构建模型
                model = self.build_models(input_dim, config['type'])
                
                # 转换为PyTorch张量
                X_train_tensor = torch.FloatTensor(X_train).to(self.device)
                y_train_tensor = torch.FloatTensor(y_train).to(self.device)
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val).to(self.device)
                
                # 创建数据加载器
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                
                # 定义损失函数和优化器
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-6)
                
                # 训练模型
                best_val_loss = float('inf')
                patience_counter = 0
                patience = 20
                
                for epoch in range(config['epochs']):
                    model.train()
                    train_loss = 0.0
                    
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                    
                    # 验证
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor).squeeze()
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    scheduler.step(val_loss)
                    
                    # 早停
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break
                
                # 恢复最佳模型
                model.load_state_dict(best_model_state)
                
                # 评估
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_tensor).squeeze().cpu().numpy()
                    val_score = mean_absolute_error(y_val, val_pred)
                    fold_scores.append(val_score)
            
            avg_score = np.mean(fold_scores)
            print(f"  ✅ {model_name} 平均MAE: {avg_score:.4f}")
            
            X_final = X_scaled
            input_dim = X_final.shape[1]
            
            final_model = self.build_models(input_dim, config['type'])
            
            # 转换为PyTorch张量
            X_final_tensor = torch.FloatTensor(X_final).to(self.device)
            y_final_tensor = torch.FloatTensor(y_scaled).to(self.device)
            
            # 创建数据加载器
            final_dataset = TensorDataset(X_final_tensor, y_final_tensor)
            final_loader = DataLoader(final_dataset, batch_size=32, shuffle=True)
            
            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = optim.Adam(final_model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-6)
            
            # 训练最终模型
            best_loss = float('inf')
            patience_counter = 0
            patience = 30
            
            for epoch in range(config['epochs']):
                final_model.train()
                epoch_loss = 0.0
                
                for batch_X, batch_y in final_loader:
                    optimizer.zero_grad()
                    outputs = final_model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_epoch_loss = epoch_loss / len(final_loader)
                scheduler.step(avg_epoch_loss)
                
                # 早停
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                    best_final_state = final_model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            # 恢复最佳模型
            final_model.load_state_dict(best_final_state)
            
            self.models[model_name] = {
                'model': final_model,
                'score': avg_score,
                'type': config['type']
            }
        
        print("\n✅ 所有模型训练完成!")
    
    def predict_future_temperatures(self, years_ahead=5):
        """
        预测未来几年的夏季最高气温
        """
        print(f"🔮 预测未来 {years_ahead} 年的夏季最高气温...")
        
        current_year = self.processed_data['year'].max()
        future_years = list(range(current_year + 1, current_year + years_ahead + 1))
        
        predictions = {}
        
        for model_name, model_info in self.models.items():
            print(f"  📊 使用 {model_name} 模型预测...")
            
            model = model_info['model']
            model_type = model_info['type']
            
            year_predictions = []
            
            for year in future_years:
                # 为每年生成夏季日期
                summer_dates = []
                for month in [6, 7, 8]:
                    days_in_month = 30 if month in [6, 8] else 31
                    for day in range(1, days_in_month + 1):
                        try:
                            date = datetime(year, month, day)
                            summer_dates.append(date)
                        except ValueError:
                            continue
                
                # 创建特征
                future_features = []
                for date in summer_dates:
                    features = self._create_future_features(date, year)
                    future_features.append(features)
                
                future_features = np.array(future_features)
                
                # 标准化特征
                future_features_scaled = self.scalers['X'].transform(future_features)
                
                # 根据模型类型调整数据形状
                if model_type in ['lstm', 'gru']:
                    future_features_scaled = future_features_scaled.reshape(
                        future_features_scaled.shape[0], future_features_scaled.shape[1], 1
                    )
                
                # 预测
                model.eval()
                with torch.no_grad():
                    future_tensor = torch.FloatTensor(future_features_scaled).to(self.device)
                    pred_scaled = model(future_tensor).squeeze().cpu().numpy()
                pred_temp = self.scalers['y'].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
                
                # 找到最高温度和对应日期
                max_temp_idx = np.argmax(pred_temp)
                max_temp = pred_temp[max_temp_idx]
                max_temp_date = summer_dates[max_temp_idx]
                
                year_predictions.append({
                    'year': year,
                    'max_temperature': max_temp,
                    'max_temp_date': max_temp_date,
                    'avg_summer_temp': np.mean(pred_temp)
                })
            
            predictions[model_name] = year_predictions
        
        self.results['future_predictions'] = predictions
        return predictions
    
    def _create_future_features(self, date, year):
        """
        为未来日期创建特征
        """
        month = date.month
        day = date.day
        day_of_year = date.timetuple().tm_yday
        
        # 基础特征
        features = {
            'year': year,
            'month': month,
            'day': day,
            'day_of_year': day_of_year,
            'sin_day': np.sin(2 * np.pi * day_of_year / 365),
            'cos_day': np.cos(2 * np.pi * day_of_year / 365),
            'sin_month': np.sin(2 * np.pi * month / 12),
            'cos_month': np.cos(2 * np.pi * month / 12),
            'year_normalized': (year - self.processed_data['year'].min()) / (self.processed_data['year'].max() - self.processed_data['year'].min())
        }
        
        # 滞后特征（使用历史平均值）
        historical_avg = self.processed_data[
            (self.processed_data['month'] == month) & 
            (self.processed_data['day'] == day)
        ]['temperature'].mean()
        
        for lag in [1, 2, 3, 5, 7, 14]:
            features[f'temp_lag_{lag}'] = historical_avg
        
        # 滑动窗口特征（使用历史平均值）
        for window in [3, 7, 14, 30]:
            features[f'temp_mean_{window}'] = historical_avg
            features[f'temp_std_{window}'] = 2.0  # 假设标准差
            features[f'temp_max_{window}'] = historical_avg + 3
            features[f'temp_min_{window}'] = historical_avg - 3
        
        # 热浪特征
        features['is_heat_wave'] = 0
        features['heat_wave_duration'] = 0
        
        # 确保特征顺序与训练时一致
        feature_cols = [col for col in self.processed_data.columns 
                       if col not in ['date', 'temperature']]
        
        return [features.get(col, 0) for col in feature_cols]
    
    def create_visualizations(self):
        """
        创建可视化图表
        """
        print("📊 正在创建可视化图表...")
        
        # 创建输出目录
        import os
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 历史数据分析图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('上海地区夏季最高气温智能分析', fontsize=16, fontweight='bold')
        
        # 年际变化趋势
        yearly_max = self.processed_data.groupby('year')['temperature'].max()
        axes[0, 0].plot(yearly_max.index, yearly_max.values, 'b-', linewidth=2, alpha=0.7)
        axes[0, 0].plot(yearly_max.index, savgol_filter(yearly_max.values, 5, 2), 'r-', linewidth=3)
        axes[0, 0].set_title('年度最高气温变化趋势', fontweight='bold')
        axes[0, 0].set_xlabel('年份')
        axes[0, 0].set_ylabel('温度 (°C)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 月度分布
        monthly_stats = self.processed_data.groupby('month')['temperature'].agg(['mean', 'max', 'min'])
        axes[0, 1].plot(monthly_stats.index, monthly_stats['mean'], 'g-', linewidth=3, label='平均')
        axes[0, 1].plot(monthly_stats.index, monthly_stats['max'], 'r-', linewidth=2, label='最高')
        axes[0, 1].plot(monthly_stats.index, monthly_stats['min'], 'b-', linewidth=2, label='最低')
        axes[0, 1].set_title('夏季各月温度分布', fontweight='bold')
        axes[0, 1].set_xlabel('月份')
        axes[0, 1].set_ylabel('温度 (°C)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 温度分布直方图
        axes[1, 0].hist(self.processed_data['temperature'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(self.shanghai_climate['heat_wave_threshold'], color='red', linestyle='--', linewidth=2, label='高温预警线')
        axes[1, 0].set_title('温度分布直方图', fontweight='bold')
        axes[1, 0].set_xlabel('温度 (°C)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 模型性能对比
        model_names = list(self.models.keys())
        model_scores = [self.models[name]['score'] for name in model_names]
        bars = axes[1, 1].bar(model_names, model_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 1].set_title('深度学习模型性能对比 (MAE)', fontweight='bold')
        axes[1, 1].set_ylabel('平均绝对误差')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars, model_scores):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/intelligent_temperature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 未来预测图
        if 'future_predictions' in self.results:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('上海地区夏季最高气温智能预测', fontsize=16, fontweight='bold')
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for i, (model_name, predictions) in enumerate(self.results['future_predictions'].items()):
                years = [p['year'] for p in predictions]
                max_temps = [p['max_temperature'] for p in predictions]
                avg_temps = [p['avg_summer_temp'] for p in predictions]
                
                # 最高温度预测
                axes[0, 0].plot(years, max_temps, 'o-', color=colors[i], linewidth=2, 
                              markersize=8, label=f'{model_name}')
                
                # 平均温度预测
                axes[0, 1].plot(years, avg_temps, 's-', color=colors[i], linewidth=2, 
                              markersize=8, label=f'{model_name}')
            
            axes[0, 0].set_title('未来年度最高气温预测', fontweight='bold')
            axes[0, 0].set_xlabel('年份')
            axes[0, 0].set_ylabel('最高温度 (°C)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(self.shanghai_climate['heat_wave_threshold'], color='red', 
                              linestyle='--', alpha=0.7, label='高温预警线')
            
            axes[0, 1].set_title('未来夏季平均气温预测', fontweight='bold')
            axes[0, 1].set_xlabel('年份')
            axes[0, 1].set_ylabel('平均温度 (°C)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 预测日期分布
            best_model = min(self.models.keys(), key=lambda x: self.models[x]['score'])
            best_predictions = self.results['future_predictions'][best_model]
            
            months = [p['max_temp_date'].month for p in best_predictions]
            days = [p['max_temp_date'].day for p in best_predictions]
            
            axes[1, 0].scatter(months, days, s=100, alpha=0.7, c=colors[0])
            axes[1, 0].set_title(f'最高气温出现日期预测 ({best_model})', fontweight='bold')
            axes[1, 0].set_xlabel('月份')
            axes[1, 0].set_ylabel('日期')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 温度趋势对比
            historical_yearly_max = self.processed_data.groupby('year')['temperature'].max()
            future_max_temps = [p['max_temperature'] for p in best_predictions]
            future_years = [p['year'] for p in best_predictions]
            
            axes[1, 1].plot(historical_yearly_max.index, historical_yearly_max.values, 
                          'b-', linewidth=2, alpha=0.7, label='历史数据')
            axes[1, 1].plot(future_years, future_max_temps, 
                          'r-', linewidth=3, marker='o', markersize=8, label='预测数据')
            axes[1, 1].set_title('历史与预测温度趋势对比', fontweight='bold')
            axes[1, 1].set_xlabel('年份')
            axes[1, 1].set_ylabel('最高温度 (°C)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/intelligent_temperature_predictions.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ 可视化图表创建完成!")
    
    def generate_report(self):
        """
        生成分析报告
        """
        print("📝 正在生成分析报告...")
        
        import os
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = f"{output_dir}/intelligent_temperature_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 上海地区夏季最高气温智能预测分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析区域: 上海地区 (东经{self.lon_range[0]:.2f}°-{self.lon_range[1]:.2f}°, 北纬{self.lat_range[0]:.2f}°-{self.lat_range[1]:.2f}°)\n\n")
            
            # 数据概况
            f.write("## 数据概况\n")
            f.write(f"- 分析年份: {self.processed_data['year'].min()} - {self.processed_data['year'].max()}\n")
            f.write(f"- 数据点数: {len(self.processed_data)} 个\n")
            f.write(f"- 平均气温: {self.processed_data['temperature'].mean():.1f}°C\n")
            f.write(f"- 最高气温: {self.processed_data['temperature'].max():.1f}°C\n")
            f.write(f"- 最低气温: {self.processed_data['temperature'].min():.1f}°C\n")
            f.write(f"- 高温天数(>{self.shanghai_climate['heat_wave_threshold']}°C): {(self.processed_data['temperature'] > self.shanghai_climate['heat_wave_threshold']).sum()} 天\n\n")
            
            # 深度学习模型性能
            f.write("## 深度学习模型性能\n")
            for model_name, model_info in self.models.items():
                f.write(f"- {model_name}: MAE = {model_info['score']:.4f}\n")
            
            best_model = min(self.models.keys(), key=lambda x: self.models[x]['score'])
            f.write(f"\n最佳模型: {best_model}\n\n")
            
            # 未来预测
            if 'future_predictions' in self.results:
                f.write("## 未来预测结果\n")
                best_predictions = self.results['future_predictions'][best_model]
                
                for pred in best_predictions:
                    date_str = pred['max_temp_date'].strftime('%m月%d日')
                    f.write(f"- {pred['year']}年: 最高气温 {pred['max_temperature']:.1f}°C, 出现日期 {date_str}\n")
                    f.write(f"  夏季平均气温: {pred['avg_summer_temp']:.1f}°C\n")
                
                # 趋势分析
                temps = [p['max_temperature'] for p in best_predictions]
                if len(temps) > 1:
                    trend = (temps[-1] - temps[0]) / (len(temps) - 1)
                    f.write(f"\n预测趋势: 每年变化 {trend:+.2f}°C\n")
            
            # 气候风险评估
            f.write("\n## 气候风险评估\n")
            if 'future_predictions' in self.results:
                future_max_temps = [p['max_temperature'] for p in best_predictions]
                extreme_heat_years = [p['year'] for p in best_predictions 
                                    if p['max_temperature'] > 38.0]
                
                f.write(f"- 预测期内最高温度: {max(future_max_temps):.1f}°C\n")
                f.write(f"- 极端高温年份(>38°C): {extreme_heat_years}\n")
                f.write(f"- 高温风险等级: {'高' if max(future_max_temps) > 39 else '中' if max(future_max_temps) > 37 else '低'}\n")
            
            f.write("\n## 建议措施\n")
            f.write("- 加强夏季高温预警系统\n")
            f.write("- 完善城市热岛效应缓解措施\n")
            f.write("- 提升公共场所降温设施\n")
            f.write("- 制定极端高温应急预案\n")
        
        print(f"✅ 分析报告已保存至: {report_path}")
    
    def run_analysis(self):
        """
        运行完整的智能分析流程
        """
        print("🚀 开始智能夏季最高气温预测分析...")
        print("=" * 60)
        
        try:
            # 1. 加载数据
            if self.grib_file_path:
                self.load_grib_data()
            else:
                self.load_data_from_db()
            
            # 2. 创建特征
            self.create_features()
            
            # 3. 训练深度学习模型
            self.train_models()
            
            # 4. 预测未来温度
            self.predict_future_temperatures()
            
            # 5. 创建可视化
            self.create_visualizations()
            
            # 6. 生成报告
            self.generate_report()
            
            print("\n" + "=" * 60)
            print("🎉 智能分析完成! 所有结果已保存至 outputs/ 目录")
            print("\n📊 生成的文件:")
            print("  - intelligent_temperature_analysis.png (历史数据分析)")
            print("  - intelligent_temperature_predictions.png (未来预测图表)")
            print("  - intelligent_temperature_report.txt (详细分析报告)")
            
            # 显示最佳模型的预测结果
            if 'future_predictions' in self.results:
                best_model = min(self.models.keys(), key=lambda x: self.models[x]['score'])
                best_predictions = self.results['future_predictions'][best_model]
                
                print(f"\n🏆 最佳模型: {best_model}")
                print("🔮 未来预测摘要:")
                for pred in best_predictions:
                    date_str = pred['max_temp_date'].strftime('%m月%d日')
                    print(f"  {pred['year']}年: 最高气温 {pred['max_temperature']:.1f}°C, 出现日期 {date_str}")
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

# 主程序
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Intelligent Summer Temperature Prediction System')
    parser.add_argument('--grib_file', type=str, help='Path to the GRIB file for analysis.')
    args = parser.parse_args()

    # 创建预测系统实例
    predictor = IntelligentTemperaturePrediction(args.grib_file)
    
    # 运行分析
    predictor.run_analysis()