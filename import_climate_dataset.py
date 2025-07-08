#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全球极端气候事件数据集导入与分析脚本

该脚本用于导入全球0.5°逐年复合极端气候事件数据集（1951-2022年），
并使用气象研究人员的传统方法和AI方法分析召回率和准确率，最后生成对比图表。

使用方法:
    python import_climate_dataset.py
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 尝试导入项目模块，如果失败则使用简化版本
try:
    from src.data_processing.data_storage import DataStorage
    from src.ml.model_manager import ModelManager
    from src.utils.logger import get_logger
    from src.utils.config import get_settings
    USE_PROJECT_MODULES = True
except ImportError as e:
    print(f"警告: 无法导入项目模块 ({e})，将使用简化版本")
    USE_PROJECT_MODULES = False
    
    # 简化的日志记录器
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    # 简化的设置
    class SimpleSettings:
        def __init__(self):
            self.database_url = "sqlite:///climate_data.db"
    
    def get_settings():
        return SimpleSettings()

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

logger = get_logger(__name__)
settings = get_settings()

class ClimateDatasetAnalyzer:
    """全球极端气候事件数据集分析器"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.data = None
        self.results = {}
        
        if USE_PROJECT_MODULES:
            self.data_storage = DataStorage()
            self.model_manager = ModelManager()
        else:
            self.data_storage = None
            self.model_manager = None
            # 创建输出目录
            self.output_dir = Path('outputs') / 'climate_analysis'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """初始化数据存储"""
        if USE_PROJECT_MODULES and self.data_storage:
            await self.data_storage.initialize()
            logger.info("数据存储初始化完成")
        else:
            logger.info("使用简化模式，跳过数据存储初始化")
    
    def load_dataset(self) -> pd.DataFrame:
        """加载数据集"""
        logger.info(f"正在加载数据集: {self.dataset_path}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")
        
        # 根据文件扩展名选择加载方式
        if self.dataset_path.suffix.lower() == '.csv':
            self.data = pd.read_csv(self.dataset_path, encoding='utf-8')
        elif self.dataset_path.suffix.lower() in ['.xlsx', '.xls']:
            self.data = pd.read_excel(self.dataset_path)
        elif self.dataset_path.suffix.lower() == '.nc':
            import xarray as xr
            ds = xr.open_dataset(self.dataset_path)
            self.data = ds.to_dataframe().reset_index()
        else:
            raise ValueError(f"不支持的文件格式: {self.dataset_path.suffix}")
        
        logger.info(f"数据集加载完成，形状: {self.data.shape}")
        logger.info(f"列名: {list(self.data.columns)}")
        
        return self.data
    
    async def import_to_database(self) -> str:
        """将数据集导入数据库"""
        logger.info("正在将数据集导入数据库...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if USE_PROJECT_MODULES and self.data_storage:
            # 使用完整的数据存储系统
            filename = f"climate_extreme_events_{timestamp}"
            
            # 保存为parquet格式以提高性能
            file_path = self.data_storage.save_dataframe(
                self.data, 
                filename, 
                data_category="processed", 
                format="parquet"
            )
            
            # 创建数据记录
            record_id = await self.data_storage.save_data_record(
                source="Global Climate Dataset",
                data_type="extreme_events",
                location="Global",
                start_time=datetime(1951, 1, 1),
                end_time=datetime(2022, 12, 31),
                file_path=file_path,
                file_format="parquet",
                file_size=os.path.getsize(file_path),
                variables=list(self.data.columns),
                data_metadata={
                    "description": "全球0.5°逐年复合极端气候事件数据集（1951-2022年）",
                    "resolution": "0.5度",
                    "temporal_coverage": "1951-2022",
                    "data_type": "extreme_climate_events"
                }
            )
            
            logger.info(f"数据集已导入数据库，记录ID: {record_id}")
            return record_id
        else:
            # 简化模式：直接保存到本地文件
            filename = f"climate_extreme_events_{timestamp}.parquet"
            file_path = self.output_dir / filename
            
            self.data.to_parquet(file_path)
            
            # 保存元数据
            metadata = {
                "source": "Global Climate Dataset",
                "data_type": "extreme_events",
                "location": "Global",
                "start_time": "1951-01-01",
                "end_time": "2022-12-31",
                "file_path": str(file_path),
                "file_format": "parquet",
                "file_size": os.path.getsize(file_path),
                "variables": list(self.data.columns),
                "description": "全球0.5°逐年复合极端气候事件数据集（1951-2022年）",
                "resolution": "0.5度",
                "temporal_coverage": "1951-2022"
            }
            
            metadata_path = self.output_dir / f"metadata_{timestamp}.json"
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"数据集已保存到本地文件: {file_path}")
            logger.info(f"元数据已保存: {metadata_path}")
            return f"local_file_{timestamp}"
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """数据预处理"""
        logger.info("正在进行数据预处理...")
        
        # 假设数据集包含极端事件标识列
        # 这里需要根据实际数据集结构调整
        if 'extreme_event' in self.data.columns:
            target_col = 'extreme_event'
        elif 'is_extreme' in self.data.columns:
            target_col = 'is_extreme'
        else:
            # 如果没有明确的极端事件标识，基于阈值创建
            # 这里以温度为例，可以根据实际情况调整
            temp_cols = [col for col in self.data.columns if 'temp' in col.lower() or 'temperature' in col.lower()]
            if temp_cols:
                temp_col = temp_cols[0]
                threshold = self.data[temp_col].quantile(0.95)  # 95分位数作为极端事件阈值
                self.data['extreme_event'] = (self.data[temp_col] > threshold).astype(int)
                target_col = 'extreme_event'
            else:
                raise ValueError("无法识别极端事件标识列，请检查数据集结构")
        
        # 分离特征和目标变量
        feature_cols = [col for col in self.data.columns if col != target_col]
        X = self.data[feature_cols].select_dtypes(include=[np.number])
        y = self.data[target_col]
        
        # 处理缺失值
        X = X.fillna(X.mean())
        
        logger.info(f"预处理完成，特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
        logger.info(f"极端事件比例: {y.mean():.3f}")
        
        return X, y
    
    def meteorological_method_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """气象研究人员的传统方法分析"""
        logger.info("正在使用气象研究人员的传统方法进行分析...")
        
        # 传统气象方法：基于阈值的简单分类
        results = {}
        
        # 方法1：基于温度阈值
        temp_cols = [col for col in X.columns if 'temp' in col.lower()]
        if temp_cols:
            temp_col = temp_cols[0]
            temp_threshold = X[temp_col].quantile(0.9)
            temp_pred = (X[temp_col] > temp_threshold).astype(int)
            
            results['temp_threshold'] = {
                'accuracy': accuracy_score(y, temp_pred),
                'precision': precision_score(y, temp_pred, zero_division=0),
                'recall': recall_score(y, temp_pred, zero_division=0),
                'f1': f1_score(y, temp_pred, zero_division=0)
            }
        
        # 方法2：基于降水阈值
        precip_cols = [col for col in X.columns if 'precip' in col.lower() or 'rain' in col.lower()]
        if precip_cols:
            precip_col = precip_cols[0]
            precip_threshold = X[precip_col].quantile(0.95)
            precip_pred = (X[precip_col] > precip_threshold).astype(int)
            
            results['precip_threshold'] = {
                'accuracy': accuracy_score(y, precip_pred),
                'precision': precision_score(y, precip_pred, zero_division=0),
                'recall': recall_score(y, precip_pred, zero_division=0),
                'f1': f1_score(y, precip_pred, zero_division=0)
            }
        
        # 方法3：综合指数方法
        # 标准化特征并计算综合指数
        X_normalized = (X - X.mean()) / X.std()
        composite_index = X_normalized.mean(axis=1)
        composite_threshold = composite_index.quantile(0.9)
        composite_pred = (composite_index > composite_threshold).astype(int)
        
        results['composite_index'] = {
            'accuracy': accuracy_score(y, composite_pred),
            'precision': precision_score(y, composite_pred, zero_division=0),
            'recall': recall_score(y, composite_pred, zero_division=0),
            'f1': f1_score(y, composite_pred, zero_division=0)
        }
        
        logger.info("传统气象方法分析完成")
        return results
    
    def ai_method_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """AI方法分析"""
        logger.info("正在使用AI方法进行分析...")
        
        # 分割训练和测试数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        results = {}
        
        # 方法1：随机森林
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, zero_division=0),
            'recall': recall_score(y_test, rf_pred, zero_division=0),
            'f1': f1_score(y_test, rf_pred, zero_division=0)
        }
        
        # 方法2：逻辑回归
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        results['logistic_regression'] = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred, zero_division=0),
            'recall': recall_score(y_test, lr_pred, zero_division=0),
            'f1': f1_score(y_test, lr_pred, zero_division=0)
        }
        
        # 保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if USE_PROJECT_MODULES and self.model_manager:
            # 保存到模型管理器
            rf_model_path = self.model_manager.models_path / f"climate_rf_{timestamp}.joblib"
        else:
            # 简化模式：保存到本地目录
            models_dir = self.output_dir / 'models'
            models_dir.mkdir(exist_ok=True)
            rf_model_path = models_dir / f"climate_rf_{timestamp}.joblib"
        
        import joblib
        joblib.dump(rf_model, rf_model_path)
        logger.info(f"随机森林模型已保存: {rf_model_path}")
        
        logger.info("AI方法分析完成")
        return results
    
    def create_comparison_chart(self, meteorological_results: Dict, ai_results: Dict):
        """创建对比图表"""
        logger.info("正在创建对比图表...")
        
        # 准备数据
        methods = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        method_types = []
        
        # 气象方法结果
        for method, metrics in meteorological_results.items():
            methods.append(f"气象-{method}")
            accuracies.append(metrics['accuracy'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1'])
            method_types.append('传统气象方法')
        
        # AI方法结果
        for method, metrics in ai_results.items():
            methods.append(f"AI-{method}")
            accuracies.append(metrics['accuracy'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1'])
            method_types.append('AI方法')
        
        # 创建DataFrame
        df_results = pd.DataFrame({
            '方法': methods,
            '准确率': accuracies,
            '精确率': precisions,
            '召回率': recalls,
            'F1分数': f1_scores,
            '方法类型': method_types
        })
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('气象研究方法 vs AI方法 - 极端气候事件检测性能对比', fontsize=16, fontweight='bold')
        
        # 准确率对比
        sns.barplot(data=df_results, x='方法', y='准确率', hue='方法类型', ax=axes[0, 0])
        axes[0, 0].set_title('准确率对比')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 精确率对比
        sns.barplot(data=df_results, x='方法', y='精确率', hue='方法类型', ax=axes[0, 1])
        axes[0, 1].set_title('精确率对比')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 召回率对比
        sns.barplot(data=df_results, x='方法', y='召回率', hue='方法类型', ax=axes[1, 0])
        axes[1, 0].set_title('召回率对比')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # F1分数对比
        sns.barplot(data=df_results, x='方法', y='F1分数', hue='方法类型', ax=axes[1, 1])
        axes[1, 1].set_title('F1分数对比')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = Path('outputs') / 'climate_analysis' / f'method_comparison_{timestamp}.png'
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        
        logger.info(f"对比图表已保存: {chart_path}")
        
        # 显示图表
        plt.show()
        
        # 保存结果数据
        results_path = chart_path.parent / f'method_comparison_data_{timestamp}.csv'
        df_results.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"结果数据已保存: {results_path}")
        
        return chart_path, results_path
    
    def print_summary(self, meteorological_results: Dict, ai_results: Dict):
        """打印分析摘要"""
        print("\n" + "="*80)
        print("全球极端气候事件数据集分析结果摘要")
        print("="*80)
        
        print("\n📊 传统气象研究方法结果:")
        for method, metrics in meteorological_results.items():
            print(f"\n  {method}:")
            print(f"    准确率: {metrics['accuracy']:.3f}")
            print(f"    精确率: {metrics['precision']:.3f}")
            print(f"    召回率: {metrics['recall']:.3f}")
            print(f"    F1分数: {metrics['f1']:.3f}")
        
        print("\n🤖 AI方法结果:")
        for method, metrics in ai_results.items():
            print(f"\n  {method}:")
            print(f"    准确率: {metrics['accuracy']:.3f}")
            print(f"    精确率: {metrics['precision']:.3f}")
            print(f"    召回率: {metrics['recall']:.3f}")
            print(f"    F1分数: {metrics['f1']:.3f}")
        
        # 找出最佳方法
        all_results = {**meteorological_results, **ai_results}
        best_method = max(all_results.keys(), key=lambda x: all_results[x]['f1'])
        best_f1 = all_results[best_method]['f1']
        
        print(f"\n🏆 最佳方法: {best_method} (F1分数: {best_f1:.3f})")
        print("="*80)
    
    async def run_analysis(self):
        """运行完整分析流程"""
        try:
            # 初始化
            await self.initialize()
            
            # 加载数据集
            self.load_dataset()
            
            # 导入数据库
            record_id = await self.import_to_database()
            
            # 数据预处理
            X, y = self.preprocess_data()
            
            # 传统气象方法分析
            meteorological_results = self.meteorological_method_analysis(X, y)
            
            # AI方法分析
            ai_results = self.ai_method_analysis(X, y)
            
            # 创建对比图表
            chart_path, results_path = self.create_comparison_chart(meteorological_results, ai_results)
            
            # 打印摘要
            self.print_summary(meteorological_results, ai_results)
            
            return {
                'record_id': record_id,
                'meteorological_results': meteorological_results,
                'ai_results': ai_results,
                'chart_path': str(chart_path),
                'results_path': str(results_path)
            }
            
        except Exception as e:
            logger.error(f"分析过程中发生错误: {e}", exc_info=True)
            raise
        finally:
            if USE_PROJECT_MODULES and self.data_storage:
                await self.data_storage.close()

def create_sample_data():
    """创建示例极端气候事件数据集用于演示"""
    print("📊 正在生成示例极端气候事件数据集...")
    
    # 创建data目录
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成时间序列（1951-2022年）
    years = np.random.randint(1951, 2023, n_samples)
    months = np.random.randint(1, 13, n_samples)
    
    # 生成地理坐标（全球0.5度网格）
    latitudes = np.random.uniform(-90, 90, n_samples)
    longitudes = np.random.uniform(-180, 180, n_samples)
    
    # 生成气象变量
    temperatures = np.random.normal(15, 10, n_samples)  # 温度 (°C)
    precipitation = np.random.exponential(50, n_samples)  # 降水量 (mm)
    wind_speed = np.random.gamma(2, 5, n_samples)  # 风速 (m/s)
    humidity = np.random.uniform(30, 100, n_samples)  # 湿度 (%)
    pressure = np.random.normal(1013, 20, n_samples)  # 气压 (hPa)
    
    # 生成极端事件标识（基于多个条件）
    extreme_temp = (temperatures > np.percentile(temperatures, 95)) | (temperatures < np.percentile(temperatures, 5))
    extreme_precip = precipitation > np.percentile(precipitation, 95)
    extreme_wind = wind_speed > np.percentile(wind_speed, 90)
    
    # 综合极端事件判断
    extreme_event = (extreme_temp | extreme_precip | extreme_wind).astype(int)
    
    # 创建DataFrame
    sample_data = pd.DataFrame({
        'year': years,
        'month': months,
        'latitude': latitudes,
        'longitude': longitudes,
        'temperature': temperatures,
        'precipitation': precipitation,
        'wind_speed': wind_speed,
        'humidity': humidity,
        'pressure': pressure,
        'extreme_event': extreme_event
    })
    
    # 保存示例数据
    sample_file = data_dir / 'sample_climate_extreme_events.csv'
    sample_data.to_csv(sample_file, index=False, encoding='utf-8')
    
    print(f"✅ 示例数据已创建: {sample_file}")
    print(f"📈 数据形状: {sample_data.shape}")
    print(f"🔥 极端事件比例: {extreme_event.mean():.3f}")
    
    return str(sample_file)

async def main():
    """主函数"""
    # 数据集路径
    dataset_path = r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体"
    
    # 检查路径是否存在
    if not Path(dataset_path).exists():
        print(f"❌ 错误: 数据集路径不存在: {dataset_path}")
        print("请检查路径是否正确，或将数据集文件放置在指定位置。")
        print("\n🔧 正在创建示例数据用于演示...")
        dataset_path = create_sample_data()
    else:
        # 如果是目录，查找数据文件
        if Path(dataset_path).is_dir():
            # 支持更多文件格式
            extensions = ['*.csv', '*.xlsx', '*.xls', '*.nc', '*.txt', '*.dat', '*.json']
            data_files = []
            
            for ext in extensions:
                data_files.extend(list(Path(dataset_path).glob(ext)))
            
            if not data_files:
                print(f"❌ 错误: 在目录 {dataset_path} 中未找到支持的数据文件")
                print("支持的文件格式: .csv, .xlsx, .xls, .nc, .txt, .dat, .json")
                print("\n💡 建议:")
                print("1. 将数据文件复制到项目目录 'data' 文件夹中")
                print("2. 或者直接指定数据文件的完整路径")
                
                # 创建示例数据用于演示
                print("\n🔧 正在创建示例数据用于演示...")
                dataset_path = create_sample_data()
            else:
                # 使用第一个找到的数据文件
                dataset_path = str(data_files[0])
                print(f"📁 找到数据文件: {dataset_path}")
    
    print("\n🚀 开始全球极端气候事件数据集分析...")
    print(f"📂 数据集路径: {dataset_path}")
    
    # 创建分析器并运行分析
    analyzer = ClimateDatasetAnalyzer(dataset_path)
    
    try:
        results = await analyzer.run_analysis()
        
        print("\n✅ 分析完成!")
        print(f"📊 对比图表: {results['chart_path']}")
        print(f"📋 结果数据: {results['results_path']}")
        print(f"🗄️ 数据库记录ID: {results['record_id']}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        logger.error(f"分析失败: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())