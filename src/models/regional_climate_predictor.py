#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区域气候问题预测模型

基于全球气候数据预测特定区域未来可能出现的气候相关问题。
支持多种机器学习和深度学习模型。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xarray as xr
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class ClimateRisk(Enum):
    """气候风险类型"""
    DROUGHT = "drought"
    FLOOD = "flood"
    HEATWAVE = "heatwave"
    COLDWAVE = "coldwave"
    STORM = "storm"
    SEA_LEVEL_RISE = "sea_level_rise"
    WILDFIRE = "wildfire"
    CROP_FAILURE = "crop_failure"
    WATER_SCARCITY = "water_scarcity"
    ECOSYSTEM_DISRUPTION = "ecosystem_disruption"


class RiskLevel(Enum):
    """风险等级"""
    VERY_LOW = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    VERY_HIGH = 4
    EXTREME = 5


@dataclass
class GlobalClimateData:
    """全球气候数据"""
    global_temp_anomaly: float  # 全球温度异常 (°C)
    co2_concentration: float    # CO2浓度 (ppm)
    sea_level_change: float     # 海平面变化 (mm)
    arctic_ice_extent: float    # 北极海冰范围 (million km²)
    ocean_ph: float            # 海洋pH值
    precipitation_anomaly: float # 全球降水异常 (%)
    solar_radiation: float      # 太阳辐射 (W/m²)
    volcanic_activity: float    # 火山活动指数
    el_nino_index: float       # 厄尔尼诺指数
    atlantic_oscillation: float # 大西洋振荡指数


@dataclass
class RegionalFeatures:
    """区域特征"""
    latitude: float
    longitude: float
    elevation: float           # 海拔 (m)
    distance_to_coast: float   # 距海岸距离 (km)
    population_density: float  # 人口密度 (人/km²)
    gdp_per_capita: float     # 人均GDP (USD)
    forest_coverage: float     # 森林覆盖率 (%)
    agricultural_area: float   # 农业用地比例 (%)
    urban_area: float         # 城市化率 (%)
    water_resources: float     # 水资源丰富度指数


@dataclass
class ClimateScenario:
    """气候情景"""
    name: str
    description: str
    global_warming: float      # 全球升温 (°C)
    co2_increase: float        # CO2增加 (ppm)
    precipitation_change: float # 降水变化 (%)
    extreme_events_frequency: float # 极端事件频率倍数


@dataclass
class PredictionResult:
    """预测结果"""
    region_name: str
    scenario: ClimateScenario
    risk_predictions: Dict[ClimateRisk, float]  # 风险概率 (0-1)
    risk_levels: Dict[ClimateRisk, RiskLevel]
    confidence_scores: Dict[ClimateRisk, float]
    time_horizon: int          # 预测时间范围 (年)
    prediction_date: datetime
    recommendations: List[str]


class ClimateDataset(Dataset):
    """气候数据集"""
    
    def __init__(
        self,
        global_data: List[GlobalClimateData],
        regional_data: List[RegionalFeatures],
        targets: List[Dict[ClimateRisk, float]],
        sequence_length: int = 12
    ):
        self.global_data = global_data
        self.regional_data = regional_data
        self.targets = targets
        self.sequence_length = sequence_length
        
        # 标准化数据
        self.scaler_global = StandardScaler()
        self.scaler_regional = StandardScaler()
        
        self._prepare_data()
    
    def _prepare_data(self):
        """准备和标准化数据"""
        # 转换为numpy数组
        global_features = np.array([
            [d.global_temp_anomaly, d.co2_concentration, d.sea_level_change,
             d.arctic_ice_extent, d.ocean_ph, d.precipitation_anomaly,
             d.solar_radiation, d.volcanic_activity, d.el_nino_index,
             d.atlantic_oscillation] for d in self.global_data
        ])
        
        regional_features = np.array([
            [r.latitude, r.longitude, r.elevation, r.distance_to_coast,
             r.population_density, r.gdp_per_capita, r.forest_coverage,
             r.agricultural_area, r.urban_area, r.water_resources] for r in self.regional_data
        ])
        
        # 标准化
        self.global_features_scaled = self.scaler_global.fit_transform(global_features)
        self.regional_features_scaled = self.scaler_regional.fit_transform(regional_features)
        
        # 准备目标数据
        self.target_data = []
        for target_dict in self.targets:
            target_vector = [target_dict.get(risk, 0.0) for risk in ClimateRisk]
            self.target_data.append(target_vector)
        
        self.target_data = np.array(self.target_data)
    
    def __len__(self):
        return len(self.global_data) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # 时间序列全球数据
        global_seq = self.global_features_scaled[idx:idx + self.sequence_length]
        
        # 区域特征（静态）
        regional_features = self.regional_features_scaled[idx + self.sequence_length - 1]
        
        # 目标
        target = self.target_data[idx + self.sequence_length - 1]
        
        return {
            'global_sequence': torch.FloatTensor(global_seq),
            'regional_features': torch.FloatTensor(regional_features),
            'target': torch.FloatTensor(target)
        }


class LSTMClimatePredictor(nn.Module):
    """基于LSTM的气候预测模型"""
    
    def __init__(
        self,
        global_feature_dim: int = 10,
        regional_feature_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 10,  # 对应ClimateRisk的数量
        dropout: float = 0.2
    ):
        super().__init__()
        
        # LSTM处理全球时间序列数据
        self.lstm = nn.LSTM(
            input_size=global_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # 区域特征处理
        self.regional_fc = nn.Sequential(
            nn.Linear(regional_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.output_fc = nn.Linear(64, output_dim)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout
        )
    
    def forward(self, global_sequence, regional_features):
        batch_size = global_sequence.size(0)
        
        # LSTM处理全球时间序列
        lstm_out, (hidden, cell) = self.lstm(global_sequence)
        
        # 应用注意力机制
        lstm_out_transposed = lstm_out.transpose(0, 1)  # (seq_len, batch, hidden)
        attended_out, _ = self.attention(
            lstm_out_transposed, lstm_out_transposed, lstm_out_transposed
        )
        attended_out = attended_out.transpose(0, 1)  # (batch, seq_len, hidden)
        
        # 取最后一个时间步的输出
        global_features = attended_out[:, -1, :]  # (batch, hidden)
        
        # 处理区域特征
        regional_processed = self.regional_fc(regional_features)
        
        # 特征融合
        combined_features = torch.cat([global_features, regional_processed], dim=1)
        fused_features = self.fusion_fc(combined_features)
        
        # 输出预测
        output = self.output_fc(fused_features)
        output = torch.sigmoid(output)  # 输出概率
        
        return output


class CNNClimatePredictor(nn.Module):
    """基于CNN的气候预测模型"""
    
    def __init__(
        self,
        global_feature_dim: int = 10,
        regional_feature_dim: int = 10,
        sequence_length: int = 12,
        output_dim: int = 10
    ):
        super().__init__()
        
        # 1D CNN处理时间序列
        self.conv_layers = nn.Sequential(
            nn.Conv1d(global_feature_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 区域特征处理
        self.regional_fc = nn.Sequential(
            nn.Linear(regional_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 输出层
        self.output_fc = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, global_sequence, regional_features):
        # CNN处理时间序列 (batch, seq_len, features) -> (batch, features, seq_len)
        global_sequence = global_sequence.transpose(1, 2)
        conv_out = self.conv_layers(global_sequence)
        conv_out = conv_out.squeeze(-1)  # (batch, 256)
        
        # 处理区域特征
        regional_out = self.regional_fc(regional_features)
        
        # 特征融合和输出
        combined = torch.cat([conv_out, regional_out], dim=1)
        output = self.output_fc(combined)
        
        return output


class RegionalClimatePredictor:
    """区域气候预测器"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型
        self.lstm_model = None
        self.cnn_model = None
        self.rf_model = None
        self.gbm_model = None
        self.svm_model = None
        
        # 数据预处理器
        self.scaler_global = StandardScaler()
        self.scaler_regional = StandardScaler()
        
        # 预定义气候情景
        self.climate_scenarios = self._load_climate_scenarios()
        
        # 区域数据库
        self.region_database = self._load_region_database()
        
        # 风险阈值
        self.risk_thresholds = self._define_risk_thresholds()
    
    def _load_climate_scenarios(self) -> Dict[str, ClimateScenario]:
        """加载预定义气候情景"""
        return {
            "RCP2.6": ClimateScenario(
                name="RCP2.6",
                description="低排放情景，全球升温控制在2°C以内",
                global_warming=1.5,
                co2_increase=50,
                precipitation_change=2,
                extreme_events_frequency=1.2
            ),
            "RCP4.5": ClimateScenario(
                name="RCP4.5",
                description="中等排放情景，温和的气候变化",
                global_warming=2.5,
                co2_increase=150,
                precipitation_change=5,
                extreme_events_frequency=1.5
            ),
            "RCP6.0": ClimateScenario(
                name="RCP6.0",
                description="高排放情景，显著的气候变化",
                global_warming=3.5,
                co2_increase=250,
                precipitation_change=8,
                extreme_events_frequency=2.0
            ),
            "RCP8.5": ClimateScenario(
                name="RCP8.5",
                description="极高排放情景，灾难性气候变化",
                global_warming=5.0,
                co2_increase=400,
                precipitation_change=12,
                extreme_events_frequency=3.0
            )
        }
    
    def _load_region_database(self) -> Dict[str, RegionalFeatures]:
        """加载区域数据库"""
        return {
            "北京": RegionalFeatures(
                latitude=39.9, longitude=116.4, elevation=44,
                distance_to_coast=150, population_density=1300,
                gdp_per_capita=24000, forest_coverage=35,
                agricultural_area=15, urban_area=85, water_resources=3
            ),
            "上海": RegionalFeatures(
                latitude=31.2, longitude=121.5, elevation=4,
                distance_to_coast=0, population_density=3800,
                gdp_per_capita=25000, forest_coverage=15,
                agricultural_area=10, urban_area=90, water_resources=4
            ),
            "广州": RegionalFeatures(
                latitude=23.1, longitude=113.3, elevation=21,
                distance_to_coast=120, population_density=1800,
                gdp_per_capita=22000, forest_coverage=45,
                agricultural_area=20, urban_area=80, water_resources=6
            ),
            "成都": RegionalFeatures(
                latitude=30.7, longitude=104.1, elevation=500,
                distance_to_coast=1500, population_density=1200,
                gdp_per_capita=18000, forest_coverage=40,
                agricultural_area=35, urban_area=65, water_resources=5
            ),
            "新疆乌鲁木齐": RegionalFeatures(
                latitude=43.8, longitude=87.6, elevation=800,
                distance_to_coast=3000, population_density=300,
                gdp_per_capita=15000, forest_coverage=5,
                agricultural_area=25, urban_area=75, water_resources=2
            )
        }
    
    def _define_risk_thresholds(self) -> Dict[ClimateRisk, List[float]]:
        """定义风险等级阈值"""
        return {
            ClimateRisk.DROUGHT: [0.1, 0.2, 0.4, 0.6, 0.8],
            ClimateRisk.FLOOD: [0.1, 0.2, 0.4, 0.6, 0.8],
            ClimateRisk.HEATWAVE: [0.15, 0.3, 0.5, 0.7, 0.85],
            ClimateRisk.COLDWAVE: [0.1, 0.2, 0.4, 0.6, 0.8],
            ClimateRisk.STORM: [0.1, 0.25, 0.45, 0.65, 0.8],
            ClimateRisk.SEA_LEVEL_RISE: [0.05, 0.15, 0.3, 0.5, 0.7],
            ClimateRisk.WILDFIRE: [0.1, 0.2, 0.4, 0.6, 0.8],
            ClimateRisk.CROP_FAILURE: [0.1, 0.2, 0.4, 0.6, 0.8],
            ClimateRisk.WATER_SCARCITY: [0.1, 0.25, 0.45, 0.65, 0.8],
            ClimateRisk.ECOSYSTEM_DISRUPTION: [0.1, 0.2, 0.4, 0.6, 0.8]
        }
    
    def build_models(self):
        """构建所有预测模型"""
        logger.info("构建区域气候预测模型...")
        
        # 深度学习模型
        self.lstm_model = LSTMClimatePredictor().to(self.device)
        self.cnn_model = CNNClimatePredictor().to(self.device)
        
        # 传统机器学习模型
        self.rf_model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.gbm_model = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
        self.svm_model = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        logger.info("区域气候预测模型构建完成")
    
    def train_models(
        self,
        train_data: List[Tuple[GlobalClimateData, RegionalFeatures, Dict[ClimateRisk, float]]],
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """训练所有模型"""
        logger.info("开始训练区域气候预测模型...")
        
        # 准备数据
        global_data = [item[0] for item in train_data]
        regional_data = [item[1] for item in train_data]
        targets = [item[2] for item in train_data]
        
        # 创建数据集
        dataset = ClimateDataset(global_data, regional_data, targets)
        
        # 分割训练和验证数据
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        results = {}
        
        # 训练LSTM模型
        results['lstm'] = self._train_deep_model(
            self.lstm_model, train_loader, val_loader, epochs, learning_rate, 'LSTM'
        )
        
        # 训练CNN模型
        results['cnn'] = self._train_deep_model(
            self.cnn_model, train_loader, val_loader, epochs, learning_rate, 'CNN'
        )
        
        # 训练传统机器学习模型
        results['traditional'] = self._train_traditional_models(dataset)
        
        logger.info("区域气候预测模型训练完成")
        return results
    
    def _train_deep_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        learning_rate: float,
        model_name: str
    ) -> Dict[str, Any]:
        """训练深度学习模型"""
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                global_seq = batch['global_sequence'].to(self.device)
                regional_feat = batch['regional_features'].to(self.device)
                target = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                output = model(global_seq, regional_feat)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    global_seq = batch['global_sequence'].to(self.device)
                    regional_feat = batch['regional_features'].to(self.device)
                    target = batch['target'].to(self.device)
                    
                    output = model(global_seq, regional_feat)
                    loss = criterion(output, target)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"{model_name} Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
    
    def _train_traditional_models(self, dataset: ClimateDataset) -> Dict[str, Any]:
        """训练传统机器学习模型"""
        # 准备特征和目标
        X = []
        y = []
        
        for i in range(len(dataset)):
            data = dataset[i]
            global_seq = data['global_sequence'].numpy()
            regional_feat = data['regional_features'].numpy()
            target = data['target'].numpy()
            
            # 将时间序列展平并与区域特征连接
            features = np.concatenate([global_seq.flatten(), regional_feat])
            X.append(features)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        # 训练每个风险类型的模型
        for i, risk in enumerate(ClimateRisk):
            y_train_risk = y_train[:, i]
            y_test_risk = y_test[:, i]
            
            # 随机森林
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train_risk)
            rf_pred = rf_model.predict(X_test)
            
            results[f'{risk.value}_rf'] = {
                'mse': mean_squared_error(y_test_risk, rf_pred),
                'mae': mean_absolute_error(y_test_risk, rf_pred),
                'r2': r2_score(y_test_risk, rf_pred)
            }
        
        return results
    
    def predict_regional_risks(
        self,
        region_name: str,
        scenario_name: str,
        global_data: List[GlobalClimateData],
        time_horizon: int = 30,
        model_ensemble: bool = True
    ) -> PredictionResult:
        """预测区域气候风险"""
        if region_name not in self.region_database:
            raise ValueError(f"未知区域: {region_name}")
        
        if scenario_name not in self.climate_scenarios:
            raise ValueError(f"未知气候情景: {scenario_name}")
        
        regional_features = self.region_database[region_name]
        scenario = self.climate_scenarios[scenario_name]
        
        # 调整全球数据以反映气候情景
        adjusted_global_data = self._adjust_global_data_for_scenario(
            global_data, scenario
        )
        
        # 准备输入数据
        dataset = ClimateDataset(
            adjusted_global_data,
            [regional_features] * len(adjusted_global_data),
            [{}] * len(adjusted_global_data)  # 空目标
        )
        
        # 获取最后一个样本用于预测
        if len(dataset) > 0:
            sample = dataset[-1]
            global_seq = sample['global_sequence'].unsqueeze(0).to(self.device)
            regional_feat = sample['regional_features'].unsqueeze(0).to(self.device)
        else:
            raise ValueError("数据不足以进行预测")
        
        # 模型预测
        predictions = {}
        confidence_scores = {}
        
        if model_ensemble and self.lstm_model and self.cnn_model:
            # 集成预测
            self.lstm_model.eval()
            self.cnn_model.eval()
            
            with torch.no_grad():
                lstm_pred = self.lstm_model(global_seq, regional_feat)
                cnn_pred = self.cnn_model(global_seq, regional_feat)
                
                # 平均预测结果
                ensemble_pred = (lstm_pred + cnn_pred) / 2
                ensemble_pred = ensemble_pred.cpu().numpy()[0]
            
            for i, risk in enumerate(ClimateRisk):
                predictions[risk] = float(ensemble_pred[i])
                confidence_scores[risk] = 0.8  # 集成模型置信度较高
        
        elif self.lstm_model:
            # 仅使用LSTM
            self.lstm_model.eval()
            with torch.no_grad():
                pred = self.lstm_model(global_seq, regional_feat)
                pred = pred.cpu().numpy()[0]
            
            for i, risk in enumerate(ClimateRisk):
                predictions[risk] = float(pred[i])
                confidence_scores[risk] = 0.7
        
        else:
            # 使用简单的启发式方法
            predictions = self._heuristic_prediction(regional_features, scenario)
            confidence_scores = {risk: 0.5 for risk in ClimateRisk}
        
        # 转换为风险等级
        risk_levels = {}
        for risk, prob in predictions.items():
            thresholds = self.risk_thresholds[risk]
            level = RiskLevel.VERY_LOW
            for i, threshold in enumerate(thresholds):
                if prob >= threshold:
                    level = RiskLevel(i + 1)
            risk_levels[risk] = level
        
        # 生成建议
        recommendations = self._generate_recommendations(
            region_name, scenario, risk_levels
        )
        
        return PredictionResult(
            region_name=region_name,
            scenario=scenario,
            risk_predictions=predictions,
            risk_levels=risk_levels,
            confidence_scores=confidence_scores,
            time_horizon=time_horizon,
            prediction_date=datetime.now(),
            recommendations=recommendations
        )
    
    def _adjust_global_data_for_scenario(
        self,
        global_data: List[GlobalClimateData],
        scenario: ClimateScenario
    ) -> List[GlobalClimateData]:
        """根据气候情景调整全球数据"""
        adjusted_data = []
        
        for data in global_data:
            adjusted = GlobalClimateData(
                global_temp_anomaly=data.global_temp_anomaly + scenario.global_warming,
                co2_concentration=data.co2_concentration + scenario.co2_increase,
                sea_level_change=data.sea_level_change * (1 + scenario.global_warming / 5),
                arctic_ice_extent=data.arctic_ice_extent * (1 - scenario.global_warming / 10),
                ocean_ph=data.ocean_ph - scenario.co2_increase / 1000,
                precipitation_anomaly=data.precipitation_anomaly + scenario.precipitation_change,
                solar_radiation=data.solar_radiation,
                volcanic_activity=data.volcanic_activity,
                el_nino_index=data.el_nino_index * scenario.extreme_events_frequency,
                atlantic_oscillation=data.atlantic_oscillation
            )
            adjusted_data.append(adjusted)
        
        return adjusted_data
    
    def _heuristic_prediction(
        self,
        regional_features: RegionalFeatures,
        scenario: ClimateScenario
    ) -> Dict[ClimateRisk, float]:
        """启发式预测方法（当模型不可用时）"""
        predictions = {}
        
        # 基于地理位置和气候情景的简单规则
        base_risk = 0.1
        scenario_multiplier = {
            "RCP2.6": 1.0,
            "RCP4.5": 1.5,
            "RCP6.0": 2.0,
            "RCP8.5": 3.0
        }.get(scenario.name, 1.0)
        
        # 干旱风险
        drought_risk = base_risk
        if regional_features.water_resources < 5:
            drought_risk += 0.2
        if regional_features.latitude > 30 or regional_features.latitude < -30:
            drought_risk += 0.1
        predictions[ClimateRisk.DROUGHT] = min(drought_risk * scenario_multiplier, 1.0)
        
        # 洪水风险
        flood_risk = base_risk
        if regional_features.distance_to_coast < 100:
            flood_risk += 0.3
        if regional_features.elevation < 100:
            flood_risk += 0.2
        predictions[ClimateRisk.FLOOD] = min(flood_risk * scenario_multiplier, 1.0)
        
        # 热浪风险
        heatwave_risk = base_risk
        if abs(regional_features.latitude) < 40:
            heatwave_risk += 0.2
        if regional_features.urban_area > 70:
            heatwave_risk += 0.1
        predictions[ClimateRisk.HEATWAVE] = min(heatwave_risk * scenario_multiplier, 1.0)
        
        # 海平面上升风险
        sea_level_risk = 0.0
        if regional_features.distance_to_coast < 50:
            sea_level_risk = 0.3
        elif regional_features.distance_to_coast < 200:
            sea_level_risk = 0.1
        predictions[ClimateRisk.SEA_LEVEL_RISE] = min(sea_level_risk * scenario_multiplier, 1.0)
        
        # 其他风险的默认值
        for risk in ClimateRisk:
            if risk not in predictions:
                predictions[risk] = min(base_risk * scenario_multiplier, 1.0)
        
        return predictions
    
    def _generate_recommendations(
        self,
        region_name: str,
        scenario: ClimateScenario,
        risk_levels: Dict[ClimateRisk, RiskLevel]
    ) -> List[str]:
        """生成适应性建议"""
        recommendations = []
        
        # 基于风险等级生成建议
        for risk, level in risk_levels.items():
            if level.value >= 3:  # 高风险或以上
                if risk == ClimateRisk.DROUGHT:
                    recommendations.append("建议加强水资源管理，发展节水技术和雨水收集系统")
                elif risk == ClimateRisk.FLOOD:
                    recommendations.append("建议完善防洪基础设施，建立早期预警系统")
                elif risk == ClimateRisk.HEATWAVE:
                    recommendations.append("建议增加城市绿化，改善建筑隔热，建立降温中心")
                elif risk == ClimateRisk.SEA_LEVEL_RISE:
                    recommendations.append("建议加强海岸防护，考虑迁移低洼地区居民")
                elif risk == ClimateRisk.WILDFIRE:
                    recommendations.append("建议加强森林管理，建立防火带，完善消防设施")
        
        # 通用建议
        if scenario.global_warming > 2.0:
            recommendations.append("建议制定长期气候适应计划，提高基础设施韧性")
        
        if not recommendations:
            recommendations.append("当前风险水平较低，建议继续监测气候变化趋势")
        
        return recommendations
    
    def get_risk_summary(self, prediction_result: PredictionResult) -> Dict[str, Any]:
        """获取风险摘要"""
        high_risks = [
            risk.value for risk, level in prediction_result.risk_levels.items()
            if level.value >= 3
        ]
        
        overall_risk_score = np.mean(list(prediction_result.risk_predictions.values()))
        
        return {
            "region": prediction_result.region_name,
            "scenario": prediction_result.scenario.name,
            "overall_risk_score": overall_risk_score,
            "risk_level": "高" if overall_risk_score > 0.6 else "中" if overall_risk_score > 0.3 else "低",
            "high_risk_categories": high_risks,
            "top_recommendations": prediction_result.recommendations[:3],
            "confidence": np.mean(list(prediction_result.confidence_scores.values()))
        }


# 便捷函数
def create_global_climate_data(
    temp_anomaly: float = 1.0,
    co2_concentration: float = 420.0,
    **kwargs
) -> GlobalClimateData:
    """创建全球气候数据的便捷函数"""
    return GlobalClimateData(
        global_temp_anomaly=temp_anomaly,
        co2_concentration=co2_concentration,
        sea_level_change=kwargs.get('sea_level_change', 3.0),
        arctic_ice_extent=kwargs.get('arctic_ice_extent', 12.0),
        ocean_ph=kwargs.get('ocean_ph', 8.1),
        precipitation_anomaly=kwargs.get('precipitation_anomaly', 0.0),
        solar_radiation=kwargs.get('solar_radiation', 1361.0),
        volcanic_activity=kwargs.get('volcanic_activity', 0.0),
        el_nino_index=kwargs.get('el_nino_index', 0.0),
        atlantic_oscillation=kwargs.get('atlantic_oscillation', 0.0)
    )


def predict_regional_climate_risk(
    region_name: str,
    scenario_name: str = "RCP4.5",
    global_temp_increase: float = 2.0,
    co2_increase: float = 100.0
) -> PredictionResult:
    """预测区域气候风险的便捷函数"""
    predictor = RegionalClimatePredictor()
    predictor.build_models()
    
    # 创建示例全球数据
    global_data = [
        create_global_climate_data(
            temp_anomaly=global_temp_increase,
            co2_concentration=420 + co2_increase
        )
    ]
    
    return predictor.predict_regional_risks(
        region_name=region_name,
        scenario_name=scenario_name,
        global_data=global_data
    )


if __name__ == "__main__":
    # 测试代码
    predictor = RegionalClimatePredictor()
    predictor.build_models()
    
    # 预测北京在RCP4.5情景下的气候风险
    result = predict_regional_climate_risk(
        region_name="北京",
        scenario_name="RCP4.5",
        global_temp_increase=2.5
    )
    
    # 打印结果摘要
    summary = predictor.get_risk_summary(result)
    print(f"区域: {summary['region']}")
    print(f"情景: {summary['scenario']}")
    print(f"总体风险等级: {summary['risk_level']}")
    print(f"高风险类别: {summary['high_risk_categories']}")
    print(f"主要建议: {summary['top_recommendations']}")