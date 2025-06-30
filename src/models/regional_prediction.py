"""区域气候预测模型

实现全球-区域降尺度气候预测，结合机器学习和深度学习方法。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings

try:
    import xarray as xr
except ImportError:
    warnings.warn("xarray未安装，NetCDF数据处理功能受限")
    xr = None

from .base_model import PyTorchBaseModel
from ..utils.logger import get_logger

logger = get_logger("regional_prediction")


class SpatialTransformer(nn.Module):
    """时空Transformer模型"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 365
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # Transformer编码
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 输出预测
        output = self.output_projection(encoded)
        
        return output


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class ClimateGNN(nn.Module):
    """气候影响网络图神经网络"""
    
    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # 图卷积层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 输入层
        if use_attention:
            self.convs.append(GATConv(node_features, hidden_dim, heads=4, concat=False))
        else:
            self.convs.append(GCNConv(node_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            if use_attention:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 输出层
        if use_attention:
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))
        else:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index, batch=None):
        # 图卷积
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # 图级别池化
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # 预测
        output = self.predictor(x)
        
        return output


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    
    def __init__(self, input_channels: int, output_dim: int = 256):
        super().__init__()
        
        # 不同尺度的卷积分支
        self.scale1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        # 多尺度特征提取
        feat1 = self.scale1(x)
        feat2 = self.scale2(x)
        feat3 = self.scale3(x)
        
        # 特征连接
        combined = torch.cat([feat1, feat2, feat3], dim=1)
        
        # 特征融合
        output = self.fusion(combined)
        
        return output


class RegionalPredictionModel(PyTorchBaseModel):
    """区域气候预测模型
    
    集成多种预测方法：时空Transformer、GNN、传统机器学习。
    """
    
    def __init__(self, device: Optional[str] = None):
        super().__init__("regional_prediction", "prediction", device)
        
        # 深度学习模型
        self.transformer_model = None
        self.gnn_model = None
        self.feature_extractor = None
        
        # 传统机器学习模型
        self.xgb_model = None
        self.gbr_model = None
        
        # 数据预处理器
        self.global_scaler = StandardScaler()
        self.regional_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # 模型权重（集成学习）
        self.ensemble_weights = {
            "transformer": 0.3,
            "gnn": 0.3,
            "xgb": 0.2,
            "gbr": 0.2
        }
        
        # 预测目标
        self.prediction_targets = [
            "drought_index",
            "heat_days",
            "precipitation_anomaly",
            "temperature_anomaly",
            "extreme_weather_risk"
        ]
    
    def build_model(
        self,
        global_features: int = 20,
        regional_features: int = 15,
        sequence_length: int = 365,
        hidden_dim: int = 256,
        num_targets: int = 5
    ) -> None:
        """构建模型架构
        
        Args:
            global_features: 全球特征维度
            regional_features: 区域特征维度
            sequence_length: 序列长度
            hidden_dim: 隐藏层维度
            num_targets: 预测目标数量
        """
        logger.info("构建区域气候预测模型...")
        
        total_features = global_features + regional_features
        
        # 时空Transformer模型
        self.transformer_model = SpatialTransformer(
            input_dim=total_features,
            d_model=hidden_dim,
            nhead=8,
            num_layers=6,
            max_seq_len=sequence_length
        ).to(self.device)
        
        # 图神经网络模型
        self.gnn_model = ClimateGNN(
            node_features=total_features,
            hidden_dim=hidden_dim,
            num_layers=3,
            use_attention=True
        ).to(self.device)
        
        # 多尺度特征提取器
        self.feature_extractor = MultiScaleFeatureExtractor(
            input_channels=1,  # 假设输入是单通道网格数据
            output_dim=hidden_dim
        ).to(self.device)
        
        # 传统机器学习模型
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.gbr_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # 设置主模型为Transformer
        self.model = self.transformer_model
        
        logger.info("区域气候预测模型构建完成")
    
    def train(
        self,
        train_data: Dict[str, Any],
        validation_data: Optional[Dict[str, Any]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """训练模型
        
        Args:
            train_data: 训练数据字典
            validation_data: 验证数据字典
            epochs: 训练轮次
            batch_size: 批次大小
            learning_rate: 学习率
            
        Returns:
            训练结果
        """
        logger.info("开始训练区域气候预测模型...")
        
        results = {}
        
        # 训练Transformer模型
        if "transformer_data" in train_data:
            logger.info("训练Transformer模型...")
            transformer_results = self._train_transformer(
                train_data["transformer_data"],
                validation_data.get("transformer_data") if validation_data else None,
                epochs, batch_size, learning_rate
            )
            results["transformer"] = transformer_results
        
        # 训练GNN模型
        if "gnn_data" in train_data:
            logger.info("训练GNN模型...")
            gnn_results = self._train_gnn(
                train_data["gnn_data"],
                validation_data.get("gnn_data") if validation_data else None,
                epochs, batch_size, learning_rate
            )
            results["gnn"] = gnn_results
        
        # 训练传统ML模型
        if "ml_data" in train_data:
            logger.info("训练传统机器学习模型...")
            ml_results = self._train_traditional_ml(train_data["ml_data"])
            results["traditional_ml"] = ml_results
        
        self.is_trained = True
        logger.info("区域气候预测模型训练完成")
        
        return results
    
    def predict(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """模型预测
        
        Args:
            input_data: 输入数据字典
            **kwargs: 其他参数
            
        Returns:
            预测结果字典
        """
        if not self.is_trained:
            logger.warning("模型未训练，使用默认参数")
        
        results = {}
        predictions = {}
        
        # Transformer预测
        if "sequence_data" in input_data and self.transformer_model is not None:
            transformer_pred = self._predict_transformer(input_data["sequence_data"])
            predictions["transformer"] = transformer_pred
        
        # GNN预测
        if "graph_data" in input_data and self.gnn_model is not None:
            gnn_pred = self._predict_gnn(input_data["graph_data"])
            predictions["gnn"] = gnn_pred
        
        # 传统ML预测
        if "tabular_data" in input_data:
            if self.xgb_model is not None:
                xgb_pred = self._predict_xgb(input_data["tabular_data"])
                predictions["xgb"] = xgb_pred
            
            if self.gbr_model is not None:
                gbr_pred = self._predict_gbr(input_data["tabular_data"])
                predictions["gbr"] = gbr_pred
        
        # 集成预测
        if predictions:
            ensemble_pred = self._ensemble_predictions(predictions)
            results["ensemble_prediction"] = ensemble_pred
            results["individual_predictions"] = predictions
        
        # 风险等级评估
        if "ensemble_prediction" in results:
            risk_assessment = self._assess_risk_levels(results["ensemble_prediction"])
            results["risk_assessment"] = risk_assessment
        
        return results
    
    def _train_transformer(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]],
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> Dict[str, Any]:
        """训练Transformer模型"""
        # 设置训练组件
        self.setup_training(
            optimizer_params={"lr": learning_rate},
            criterion_class=nn.MSELoss
        )
        
        # 准备数据
        X_train = torch.FloatTensor(train_data["X"]).to(self.device)
        y_train = torch.FloatTensor(train_data["y"]).to(self.device)
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = None
        if val_data is not None:
            X_val = torch.FloatTensor(val_data["X"]).to(self.device)
            y_val = torch.FloatTensor(val_data["y"]).to(self.device)
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        
        train_losses = []
        val_losses = []
        
        self.transformer_model.train()
        
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                outputs = self.transformer_model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证
            if val_loader:
                val_loss = self._validate_transformer(val_loader)
                val_losses.append(val_loss)
            
            # 记录训练历史
            metrics = {"train_loss": avg_train_loss}
            if val_loader:
                metrics["val_loss"] = val_loss
            
            self.add_training_record(epoch, metrics)
            
            if epoch % 10 == 0:
                logger.info(f"Transformer Epoch {epoch}: train_loss={avg_train_loss:.4f}")
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1] if val_losses else None
        }
    
    def _train_gnn(
        self,
        train_data: Dict[str, Any],
        val_data: Optional[Dict[str, Any]],
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> Dict[str, Any]:
        """训练GNN模型"""
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 准备图数据
        train_graphs = train_data["graphs"]
        train_targets = torch.FloatTensor(train_data["targets"]).to(self.device)
        
        train_losses = []
        
        self.gnn_model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # 批处理图数据
            for i in range(0, len(train_graphs), batch_size):
                batch_graphs = train_graphs[i:i+batch_size]
                batch_targets = train_targets[i:i+batch_size]
                
                # 创建批次
                batch = Batch.from_data_list(batch_graphs).to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.gnn_model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(outputs.squeeze(), batch_targets)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(train_graphs) // batch_size + 1)
            train_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"GNN Epoch {epoch}: loss={avg_loss:.4f}")
        
        return {
            "train_losses": train_losses,
            "final_loss": train_losses[-1]
        }
    
    def _train_traditional_ml(self, train_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """训练传统机器学习模型"""
        X_train = train_data["X"]
        y_train = train_data["y"]
        
        # 数据预处理
        X_scaled = self.global_scaler.fit_transform(X_train)
        y_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # 分割训练和验证数据
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        results = {}
        
        # 训练XGBoost
        try:
            self.xgb_model.fit(X_train_split, y_train_split)
            xgb_pred = self.xgb_model.predict(X_val_split)
            xgb_mse = mean_squared_error(y_val_split, xgb_pred)
            xgb_r2 = r2_score(y_val_split, xgb_pred)
            
            results["xgb"] = {
                "mse": xgb_mse,
                "r2": xgb_r2,
                "feature_importance": self.xgb_model.feature_importances_.tolist()
            }
            logger.info(f"XGBoost训练完成: MSE={xgb_mse:.4f}, R2={xgb_r2:.4f}")
        except Exception as e:
            logger.error(f"XGBoost训练失败: {e}")
            results["xgb"] = {"error": str(e)}
        
        # 训练GradientBoosting
        try:
            self.gbr_model.fit(X_train_split, y_train_split)
            gbr_pred = self.gbr_model.predict(X_val_split)
            gbr_mse = mean_squared_error(y_val_split, gbr_pred)
            gbr_r2 = r2_score(y_val_split, gbr_pred)
            
            results["gbr"] = {
                "mse": gbr_mse,
                "r2": gbr_r2,
                "feature_importance": self.gbr_model.feature_importances_.tolist()
            }
            logger.info(f"GBR训练完成: MSE={gbr_mse:.4f}, R2={gbr_r2:.4f}")
        except Exception as e:
            logger.error(f"GBR训练失败: {e}")
            results["gbr"] = {"error": str(e)}
        
        return results
    
    def _predict_transformer(self, data: np.ndarray) -> np.ndarray:
        """Transformer预测"""
        self.transformer_model.eval()
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(data).to(self.device)
            if input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)
            
            prediction = self.transformer_model(input_tensor)
            return prediction.cpu().numpy()
    
    def _predict_gnn(self, graph_data: Data) -> np.ndarray:
        """GNN预测"""
        self.gnn_model.eval()
        
        with torch.no_grad():
            graph_data = graph_data.to(self.device)
            prediction = self.gnn_model(graph_data.x, graph_data.edge_index)
            return prediction.cpu().numpy()
    
    def _predict_xgb(self, data: np.ndarray) -> np.ndarray:
        """XGBoost预测"""
        if self.xgb_model is None:
            return np.array([])
        
        scaled_data = self.global_scaler.transform(data)
        prediction = self.xgb_model.predict(scaled_data)
        return self.target_scaler.inverse_transform(prediction.reshape(-1, 1)).ravel()
    
    def _predict_gbr(self, data: np.ndarray) -> np.ndarray:
        """GradientBoosting预测"""
        if self.gbr_model is None:
            return np.array([])
        
        scaled_data = self.global_scaler.transform(data)
        prediction = self.gbr_model.predict(scaled_data)
        return self.target_scaler.inverse_transform(prediction.reshape(-1, 1)).ravel()
    
    def _ensemble_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """集成多个模型的预测结果"""
        ensemble_pred = None
        total_weight = 0.0
        
        for model_name, pred in predictions.items():
            if model_name in self.ensemble_weights and len(pred) > 0:
                weight = self.ensemble_weights[model_name]
                
                if ensemble_pred is None:
                    ensemble_pred = weight * pred
                else:
                    ensemble_pred += weight * pred
                
                total_weight += weight
        
        if ensemble_pred is not None and total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def _assess_risk_levels(self, predictions: np.ndarray) -> Dict[str, Any]:
        """评估风险等级"""
        if len(predictions) == 0:
            return {"error": "无预测结果"}
        
        # 假设预测值范围为0-1，转换为风险等级
        risk_levels = []
        risk_descriptions = []
        
        for pred in predictions.flatten():
            if pred < 0.2:
                level = "低风险"
                desc = "气候条件相对稳定"
            elif pred < 0.4:
                level = "中低风险"
                desc = "轻微气候异常"
            elif pred < 0.6:
                level = "中等风险"
                desc = "明显气候变化"
            elif pred < 0.8:
                level = "高风险"
                desc = "严重气候异常"
            else:
                level = "极高风险"
                desc = "极端气候事件"
            
            risk_levels.append(level)
            risk_descriptions.append(desc)
        
        return {
            "risk_levels": risk_levels,
            "risk_descriptions": risk_descriptions,
            "average_risk": float(np.mean(predictions)),
            "max_risk": float(np.max(predictions)),
            "min_risk": float(np.min(predictions))
        }
    
    def _validate_transformer(self, val_loader) -> float:
        """验证Transformer模型"""
        self.transformer_model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = self.transformer_model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y)
                total_loss += loss.item()
        
        self.transformer_model.train()
        return total_loss / len(val_loader)
    
    def predict_regional_climate(
        self,
        global_scenario: str,
        region_coords: Tuple[float, float],
        time_horizon: int = 30,
        variables: List[str] = None
    ) -> Dict[str, Any]:
        """区域气候预测
        
        Args:
            global_scenario: 全球情景（如RCP8.5）
            region_coords: 区域坐标（经度，纬度）
            time_horizon: 预测时间范围（年）
            variables: 预测变量列表
            
        Returns:
            预测结果
        """
        if variables is None:
            variables = self.prediction_targets
        
        # 构建输入数据（这里需要根据实际数据格式调整）
        input_data = {
            "scenario": global_scenario,
            "longitude": region_coords[0],
            "latitude": region_coords[1],
            "time_horizon": time_horizon
        }
        
        # 模拟输入数据（实际应用中需要从数据源获取）
        sequence_data = np.random.randn(1, 365, 35)  # 1年的日数据，35个特征
        tabular_data = np.random.randn(1, 35)  # 汇总特征
        
        prediction_input = {
            "sequence_data": sequence_data,
            "tabular_data": tabular_data
        }
        
        # 执行预测
        results = self.predict(prediction_input)
        
        # 格式化输出
        formatted_results = {
            "region": {
                "longitude": region_coords[0],
                "latitude": region_coords[1]
            },
            "scenario": global_scenario,
            "time_horizon": time_horizon,
            "predictions": results,
            "variables": variables
        }
        
        return formatted_results
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """获取特征重要性"""
        importance = {}
        
        if self.xgb_model is not None:
            importance["xgb"] = self.xgb_model.feature_importances_.tolist()
        
        if self.gbr_model is not None:
            importance["gbr"] = self.gbr_model.feature_importances_.tolist()
        
        return importance
    
    def update_ensemble_weights(self, new_weights: Dict[str, float]) -> None:
        """更新集成模型权重
        
        Args:
            new_weights: 新的权重字典
        """
        for model_name, weight in new_weights.items():
            if model_name in self.ensemble_weights:
                self.ensemble_weights[model_name] = weight
        
        # 归一化权重
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            for model_name in self.ensemble_weights:
                self.ensemble_weights[model_name] /= total_weight
        
        logger.info(f"集成权重已更新: {self.ensemble_weights}")