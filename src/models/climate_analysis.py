"""气候分析模型

实现历史气候数据分析，包括时间序列分析、异常检测和模式识别。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import signal
from scipy.stats import zscore
import warnings

try:
    from prophet import Prophet
except ImportError:
    Prophet = None
    warnings.warn("Prophet未安装，时间序列分析功能受限")

from .base_model import PyTorchBaseModel
from ..utils.logger import get_logger

logger = get_logger("climate_analysis")


class LSTMModel(nn.Module):
    """LSTM时间序列模型"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 应用dropout和全连接层
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output


class AutoEncoder(nn.Module):
    """自编码器用于异常检测"""
    
    def __init__(self, input_size: int, encoding_dim: int = 32):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CNNLSTMModel(nn.Module):
    """CNN-LSTM混合模型用于气候模态识别"""
    
    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        num_classes: int,
        cnn_features: int = 64,
        lstm_hidden: int = 128
    ):
        super().__init__()
        
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_features),
            nn.Conv1d(cnn_features, cnn_features * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_features * 2),
            nn.AdaptiveAvgPool1d(sequence_length // 2)
        )
        
        # LSTM时序建模
        self.lstm = nn.LSTM(
            input_size=cnn_features * 2,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden // 2, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        cnn_out = self.cnn(x)
        
        # 转换为LSTM输入格式
        cnn_out = cnn_out.transpose(1, 2)  # (batch_size, seq_len, features)
        
        lstm_out, _ = self.lstm(cnn_out)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 分类
        output = self.classifier(last_output)
        
        return output


class ClimateAnalysisModel(PyTorchBaseModel):
    """气候分析模型
    
    集成多种分析方法：时间序列分析、异常检测、模式识别。
    """
    
    def __init__(self, device: Optional[str] = None):
        super().__init__("climate_analysis", "analysis", device)
        
        # 子模型
        self.lstm_model = None
        self.autoencoder = None
        self.cnn_lstm_model = None
        self.prophet_model = None
        
        # 传统ML模型
        self.isolation_forest = None
        self.dbscan = None
        self.scaler = StandardScaler()
        
        # 分析结果缓存
        self.analysis_cache = {}

    def analyze_trends(self, data: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """分析时间序列趋势"""
        if Prophet is None:
            raise ImportError("Prophet is not installed, cannot perform trend analysis.")

        df = data[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
        
        model = Prophet()
        model.fit(df)
        
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        
        return {
            "forecast": forecast.to_dict(orient="records"),
            "model": model
        }

    def detect_anomalies(self, data: pd.DataFrame, value_cols: List[str], contamination: float = 0.05) -> pd.DataFrame:
        """使用孤立森林检测异常"""
        X = data[value_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = self.isolation_forest.fit_predict(X_scaled)
        
        result_df = data.copy()
        result_df["anomaly"] = predictions
        
        return result_df[result_df["anomaly"] == -1]
    
    def build_model(
        self,
        input_size: int = 1,
        sequence_length: int = 30,
        lstm_hidden: int = 64,
        autoencoder_dim: int = 32,
        num_climate_modes: int = 5
    ) -> None:
        """构建模型架构
        
        Args:
            input_size: 输入特征维度
            sequence_length: 序列长度
            lstm_hidden: LSTM隐藏层大小
            autoencoder_dim: 自编码器编码维度
            num_climate_modes: 气候模态数量
        """
        logger.info("构建气候分析模型...")
        self.lstm_model = LSTMModel(input_size=input_size, hidden_size=lstm_hidden)
        self.autoencoder = AutoEncoder(input_size=input_size, encoding_dim=autoencoder_dim)
        self.cnn_lstm_model = CNNLSTMModel(
            input_channels=input_size,
            sequence_length=sequence_length,
            num_classes=num_climate_modes
        )
        
        # LSTM时间序列模型
        self.lstm_model = LSTMModel(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            output_size=1
        ).to(self.device)
        
        # 自编码器异常检测模型
        self.autoencoder = AutoEncoder(
            input_size=input_size,
            encoding_dim=autoencoder_dim
        ).to(self.device)
        
        # CNN-LSTM气候模态识别模型
        self.cnn_lstm_model = CNNLSTMModel(
            input_channels=input_size,
            sequence_length=sequence_length,
            num_classes=num_climate_modes
        ).to(self.device)
        
        # 传统ML模型
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        self.dbscan = DBSCAN(
            eps=0.5,
            min_samples=5
        )
        
        # 设置主模型为LSTM（用于保存/加载）
        self.model = self.lstm_model
        
        logger.info("气候分析模型构建完成")
    
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
        logger.info("开始训练气候分析模型...")
        
        results = {}
        
        # 训练LSTM模型
        if "time_series" in train_data:
            logger.info("训练LSTM时间序列模型...")
            lstm_results = self._train_lstm(
                train_data["time_series"],
                validation_data.get("time_series") if validation_data else None,
                epochs, batch_size, learning_rate
            )
            results["lstm"] = lstm_results
        
        # 训练自编码器
        if "anomaly_detection" in train_data:
            logger.info("训练自编码器异常检测模型...")
            ae_results = self._train_autoencoder(
                train_data["anomaly_detection"],
                validation_data.get("anomaly_detection") if validation_data else None,
                epochs, batch_size, learning_rate
            )
            results["autoencoder"] = ae_results
        
        # 训练CNN-LSTM模型
        if "pattern_recognition" in train_data:
            logger.info("训练CNN-LSTM模式识别模型...")
            cnn_lstm_results = self._train_cnn_lstm(
                train_data["pattern_recognition"],
                validation_data.get("pattern_recognition") if validation_data else None,
                epochs, batch_size, learning_rate
            )
            results["cnn_lstm"] = cnn_lstm_results
        
        # 训练传统ML模型
        if "traditional_ml" in train_data:
            logger.info("训练传统机器学习模型...")
            ml_results = self._train_traditional_ml(train_data["traditional_ml"])
            results["traditional_ml"] = ml_results
        
        self.is_trained = True
        logger.info("气候分析模型训练完成")
        
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
        
        # 时间序列预测
        if "time_series" in input_data:
            results["time_series_forecast"] = self._predict_time_series(
                input_data["time_series"]
            )
        
        # 异常检测
        if "anomaly_data" in input_data:
            results["anomalies"] = self._detect_anomalies(
                input_data["anomaly_data"]
            )
        
        # 模式识别
        if "pattern_data" in input_data:
            results["climate_patterns"] = self._recognize_patterns(
                input_data["pattern_data"]
            )
        
        # 趋势分析
        if "trend_data" in input_data:
            results["trend_analysis"] = self._analyze_trends(
                input_data["trend_data"]
            )
        
        return results
    
    def _train_lstm(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray],
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> Dict[str, Any]:
        """训练LSTM模型"""
        # 设置训练组件
        self.setup_training(
            optimizer_params={"lr": learning_rate},
            criterion_class=nn.MSELoss
        )
        
        # 准备数据
        train_loader = self._prepare_sequence_data(train_data, batch_size)
        val_loader = self._prepare_sequence_data(val_data, batch_size) if val_data is not None else None
        
        train_losses = []
        val_losses = []
        
        self.lstm_model.train()
        
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.lstm_model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证
            if val_loader:
                val_loss = self._validate_lstm(val_loader)
                val_losses.append(val_loss)
            
            # 记录训练历史
            metrics = {"train_loss": avg_train_loss}
            if val_loader:
                metrics["val_loss"] = val_loss
            
            self.add_training_record(epoch, metrics)
            
            if epoch % 10 == 0:
                logger.info(f"LSTM Epoch {epoch}: train_loss={avg_train_loss:.4f}")
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1] if val_losses else None
        }
    
    def _train_autoencoder(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray],
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> Dict[str, Any]:
        """训练自编码器"""
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 准备数据
        train_tensor = torch.FloatTensor(train_data).to(self.device)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_tensor, train_tensor),
            batch_size=batch_size,
            shuffle=True
        )
        
        train_losses = []
        
        self.autoencoder.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                reconstructed = self.autoencoder(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"AutoEncoder Epoch {epoch}: loss={avg_loss:.4f}")
        
        return {
            "train_losses": train_losses,
            "final_loss": train_losses[-1]
        }
    
    def _train_cnn_lstm(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]],
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> Dict[str, Any]:
        """训练CNN-LSTM模型"""
        optimizer = torch.optim.Adam(self.cnn_lstm_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 准备数据
        X_train = torch.FloatTensor(train_data["X"]).to(self.device)
        y_train = torch.LongTensor(train_data["y"]).to(self.device)
        
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )
        
        train_losses = []
        train_accuracies = []
        
        self.cnn_lstm_model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.cnn_lstm_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100 * correct / total
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            if epoch % 10 == 0:
                logger.info(f"CNN-LSTM Epoch {epoch}: loss={avg_loss:.4f}, acc={accuracy:.2f}%")
        
        return {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "final_loss": train_losses[-1],
            "final_accuracy": train_accuracies[-1]
        }
    
    def _train_traditional_ml(self, train_data: np.ndarray) -> Dict[str, Any]:
        """训练传统机器学习模型"""
        # 标准化数据
        scaled_data = self.scaler.fit_transform(train_data)
        
        # 训练孤立森林
        self.isolation_forest.fit(scaled_data)
        
        # 训练DBSCAN（无监督，只需要fit）
        cluster_labels = self.dbscan.fit_predict(scaled_data)
        
        return {
            "isolation_forest_trained": True,
            "dbscan_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            "noise_points": list(cluster_labels).count(-1)
        }
    
    def _predict_time_series(self, data: np.ndarray) -> Dict[str, Any]:
        """时间序列预测"""
        if self.lstm_model is None:
            return {"error": "LSTM模型未初始化"}
        
        self.lstm_model.eval()
        
        with torch.no_grad():
            # 准备输入数据
            input_tensor = torch.FloatTensor(data).unsqueeze(0).to(self.device)
            prediction = self.lstm_model(input_tensor)
            
            return {
                "prediction": prediction.cpu().numpy().tolist(),
                "confidence": 0.85  # 简化的置信度
            }
    
    def _detect_anomalies(self, data: np.ndarray) -> Dict[str, Any]:
        """异常检测"""
        results = {}
        
        # 使用自编码器检测异常
        if self.autoencoder is not None:
            self.autoencoder.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(data).to(self.device)
                reconstructed = self.autoencoder(input_tensor)
                reconstruction_error = torch.mean((input_tensor - reconstructed) ** 2, dim=1)
                
                # 设置阈值（可以根据训练数据调整）
                threshold = torch.quantile(reconstruction_error, 0.95)
                anomalies = reconstruction_error > threshold
                
                results["autoencoder"] = {
                    "anomaly_indices": torch.where(anomalies)[0].cpu().numpy().tolist(),
                    "reconstruction_errors": reconstruction_error.cpu().numpy().tolist(),
                    "threshold": threshold.item()
                }
        
        # 使用孤立森林检测异常
        if self.isolation_forest is not None:
            scaled_data = self.scaler.transform(data)
            anomaly_scores = self.isolation_forest.decision_function(scaled_data)
            anomalies = self.isolation_forest.predict(scaled_data)
            
            results["isolation_forest"] = {
                "anomaly_indices": np.where(anomalies == -1)[0].tolist(),
                "anomaly_scores": anomaly_scores.tolist()
            }
        
        return results
    
    def _recognize_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """气候模式识别"""
        if self.cnn_lstm_model is None:
            return {"error": "CNN-LSTM模型未初始化"}
        
        self.cnn_lstm_model.eval()
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(data).to(self.device)
            predictions = self.cnn_lstm_model(input_tensor)
            probabilities = torch.softmax(predictions, dim=1)
            
            return {
                "predicted_patterns": torch.argmax(predictions, dim=1).cpu().numpy().tolist(),
                "pattern_probabilities": probabilities.cpu().numpy().tolist()
            }
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """趋势分析"""
        results = {}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            series = data[column].dropna()
            
            if len(series) < 10:
                continue
            
            # 线性趋势
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series.values, 1)
            trend_slope = coeffs[0]
            
            # 季节性分解（简化版）
            if len(series) >= 24:  # 至少需要2个周期
                try:
                    # 使用移动平均进行简单的趋势提取
                    window = min(12, len(series) // 4)
                    trend = series.rolling(window=window, center=True).mean()
                    seasonal = series - trend
                    residual = series - trend - seasonal
                    
                    results[column] = {
                        "linear_trend_slope": trend_slope,
                        "trend_direction": "increasing" if trend_slope > 0 else "decreasing",
                        "seasonal_variance": seasonal.var(),
                        "residual_variance": residual.var(),
                        "trend_significance": abs(trend_slope) > series.std() / len(series)
                    }
                except Exception as e:
                    logger.warning(f"趋势分析失败 {column}: {e}")
                    results[column] = {"error": str(e)}
        
        return results
    
    def _prepare_sequence_data(self, data: np.ndarray, batch_size: int, sequence_length: int = 30):
        """准备序列数据"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        sequences = torch.FloatTensor(sequences)
        targets = torch.FloatTensor(targets)
        
        dataset = torch.utils.data.TensorDataset(sequences, targets)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def _validate_lstm(self, val_loader) -> float:
        """验证LSTM模型"""
        self.lstm_model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.lstm_model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        self.lstm_model.train()
        return total_loss / len(val_loader)
    
    def analyze_climate_data(
        self,
        data: pd.DataFrame,
        analysis_types: List[str] = None
    ) -> Dict[str, Any]:
        """综合气候数据分析
        
        Args:
            data: 气候数据DataFrame
            analysis_types: 分析类型列表
            
        Returns:
            分析结果字典
        """
        if analysis_types is None:
            analysis_types = ["trend", "anomaly", "seasonality"]
        
        results = {}
        
        # 趋势分析
        if "trend" in analysis_types:
            results["trend_analysis"] = self._analyze_trends(data)
        
        # 异常检测
        if "anomaly" in analysis_types and len(data) > 0:
            numeric_data = data.select_dtypes(include=[np.number]).values
            if numeric_data.size > 0:
                results["anomaly_detection"] = self._detect_anomalies(numeric_data)
        
        # 季节性分析
        if "seasonality" in analysis_types:
            results["seasonality_analysis"] = self._analyze_seasonality(data)
        
        return results
    
    def _analyze_seasonality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """季节性分析"""
        results = {}
        
        # 假设数据有时间索引或时间列
        if hasattr(data.index, 'month'):
            time_index = data.index
        elif 'date' in data.columns:
            time_index = pd.to_datetime(data['date'])
        else:
            return {"error": "未找到时间信息"}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            series = data[column].dropna()
            
            if len(series) < 24:  # 至少需要2年数据
                continue
            
            try:
                # 按月份分组分析
                monthly_data = series.groupby(time_index.month)
                monthly_means = monthly_data.mean()
                monthly_stds = monthly_data.std()
                
                # 计算季节性强度
                overall_mean = series.mean()
                seasonal_strength = (monthly_means.max() - monthly_means.min()) / overall_mean
                
                results[column] = {
                    "monthly_means": monthly_means.to_dict(),
                    "monthly_stds": monthly_stds.to_dict(),
                    "seasonal_strength": seasonal_strength,
                    "peak_month": monthly_means.idxmax(),
                    "trough_month": monthly_means.idxmin()
                }
                
            except Exception as e:
                logger.warning(f"季节性分析失败 {column}: {e}")
                results[column] = {"error": str(e)}
        
        return results