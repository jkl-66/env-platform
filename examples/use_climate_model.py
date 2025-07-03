#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气候模型使用示例

演示如何使用已训练的气候模型进行温度预测
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ml.model_manager import ModelManager
from src.ml.prediction_engine import PredictionEngine, PredictionConfig, PredictionType
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_time_features(base_time):
    """
    为给定的时间创建时间特征
    """
    dt = pd.to_datetime(base_time)
    
    features = {
        'year': dt.year,
        'month': dt.month,
        'day': dt.day,
        'hour': dt.hour,
        'day_of_year': dt.dayofyear,
        'season': {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}[dt.month],
        'month_sin': np.sin(2 * np.pi * dt.month / 12),
        'month_cos': np.cos(2 * np.pi * dt.month / 12),
        'day_sin': np.sin(2 * np.pi * dt.dayofyear / 365.25),
        'day_cos': np.cos(2 * np.pi * dt.dayofyear / 365.25),
        'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dt.hour / 24)
    }
    
    return features


async def example_1_simple_prediction():
    """
    示例1: 简单的单点温度预测
    """
    print("\n=== 示例1: 简单的单点温度预测 ===")
    
    # 创建模型管理器
    model_manager = ModelManager()
    
    # 获取可用模型
    models = model_manager.list_models()
    if not models:
        print("没有找到可用的模型，请先训练模型")
        return
    
    # 使用第一个可用的模型
    model_id = models[0].id
    print(f"使用模型: {model_id}")
    print(f"模型算法: {models[0].algorithm}")
    print(f"目标变量: {models[0].config.target}")
    
    # 设置预测时间（当前时间）
    prediction_time = datetime.now()
    time_features = create_time_features(prediction_time)
    
    # 准备预测数据（北京的气象数据）
    predict_data = pd.DataFrame({
        'latitude': [39.9042],      # 北京纬度
        'longitude': [116.4074],    # 北京经度
        'number': [0],              # GRIB数据编号
        'step': [0],                # 时间步长
        'surface': [1],             # 地表层标识
        'msl': [101325.0],          # 海平面压力（Pa）
        'sst': [15.5],              # 海表温度（°C）
        'sp': [101000.0],           # 地面压力（Pa）
        'quality_score': [1.0],     # 数据质量分数
        **{k: [v] for k, v in time_features.items()}  # 添加时间特征
    })
    
    print(f"预测时间: {prediction_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"预测月份: {time_features['month']}月")
    print(f"预测季节: {['', '冬季', '春季', '夏季', '秋季'][time_features['season']]}")
    
    print(f"\n输入数据:")
    print(predict_data)
    
    try:
        # 执行预测
        predictions = await model_manager.predict(model_id, predict_data)
        
        print(f"\n预测结果:")
        print(f"北京预测温度: {predictions[0]:.2f}°C")
        
    except Exception as e:
        print(f"预测失败: {e}")


async def example_2_batch_prediction():
    """
    示例2: 批量预测多个城市的温度
    """
    print("\n=== 示例2: 批量预测多个城市的温度 ===")
    
    model_manager = ModelManager()
    models = model_manager.list_models()
    
    if not models:
        print("没有找到可用的模型")
        return
    
    model_id = models[0].id
    
    # 设置预测时间（当前时间）
    prediction_time = datetime.now()
    time_features = create_time_features(prediction_time)
    
    # 准备多个城市的数据
    cities_data = pd.DataFrame({
        'city': ['北京', '上海', '广州', '深圳', '成都'],
        'latitude': [39.9042, 31.2304, 23.1291, 22.5431, 30.5728],
        'longitude': [116.4074, 121.4737, 113.2644, 114.0579, 104.0668],
        'number': [0, 0, 0, 0, 0],              # GRIB数据编号
        'step': [0, 0, 0, 0, 0],                # 时间步长
        'surface': [1, 1, 1, 1, 1],             # 地表层标识
        'msl': [101325.0, 101325.0, 101325.0, 101325.0, 101325.0],
        'sst': [15.5, 18.2, 22.8, 23.5, 16.8],  # 不同地区的海表温度
        'sp': [101000.0, 101000.0, 101000.0, 101000.0, 101000.0],
        'quality_score': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    
    # 为每个城市添加时间特征
    for key, value in time_features.items():
        cities_data[key] = value
    
    print(f"预测时间: {prediction_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"预测月份: {time_features['month']}月")
    print(f"预测季节: {['', '冬季', '春季', '夏季', '秋季'][time_features['season']]}")
    
    print(f"\n输入数据:")
    print(cities_data[['city', 'latitude', 'longitude', 'sst']])
    
    try:
        # 执行批量预测
        prediction_data = cities_data.drop('city', axis=1)  # 移除非数值列
        predictions = await model_manager.predict(model_id, prediction_data)
        
        print(f"\n预测结果:")
        for i, city in enumerate(cities_data['city']):
            print(f"{city}: {predictions[i]:.2f}°C")
            
        # 保存结果
        cities_data['predicted_temperature'] = predictions
        output_path = project_root / "data" / "predictions" / "cities_temperature_predictions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cities_data.to_csv(output_path, index=False)
        print(f"\n结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"批量预测失败: {e}")


async def example_3_region_analysis():
    """
    示例3: 区域温度分析
    """
    print("\n=== 示例3: 区域温度分析 ===")
    
    model_manager = ModelManager()
    models = model_manager.list_models()
    
    if not models:
        print("没有找到可用的模型")
        return
    
    model_id = models[0].id
    
    # 设置预测时间（当前时间）
    prediction_time = datetime.now()
    time_features = create_time_features(prediction_time)
    
    # 创建网格数据（华北地区）
    lats = np.arange(35, 45, 1.0)  # 35°N到45°N，每1度一个点
    lons = np.arange(110, 120, 1.0)  # 110°E到120°E，每1度一个点
    
    # 创建网格
    lat_grid, lon_grid = np.meshgrid(lats, lons)
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    
    # 创建区域数据
    region_data = pd.DataFrame({
        'latitude': lat_flat,
        'longitude': lon_flat,
        'number': [0] * len(lat_flat),              # GRIB数据编号
        'step': [0] * len(lat_flat),                # 时间步长
        'surface': [1] * len(lat_flat),             # 地表层标识
        'msl': [101325.0] * len(lat_flat),
        'sst': np.random.uniform(12, 20, len(lat_flat)),  # 随机海表温度
        'sp': [101000.0] * len(lat_flat),
        'quality_score': [1.0] * len(lat_flat)
    })
    
    # 为每个网格点添加时间特征
    for key, value in time_features.items():
        region_data[key] = value
    
    print(f"预测时间: {prediction_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"预测月份: {time_features['month']}月")
    print(f"预测季节: {['', '冬季', '春季', '夏季', '秋季'][time_features['season']]}")
    
    print(f"\n分析区域: {len(region_data)} 个网格点")
    print(f"纬度范围: {lats.min()}°N - {lats.max()}°N")
    print(f"经度范围: {lons.min()}°E - {lons.max()}°E")
    
    try:
        # 执行区域预测
        predictions = await model_manager.predict(model_id, region_data)
        
        # 分析结果
        region_data['predicted_temperature'] = predictions
        
        print(f"\n区域温度分析结果:")
        print(f"最低温度: {predictions.min():.2f}°C")
        print(f"最高温度: {predictions.max():.2f}°C")
        print(f"平均温度: {predictions.mean():.2f}°C")
        print(f"温度标准差: {predictions.std():.2f}°C")
        
        # 找出最热和最冷的点
        max_idx = np.argmax(predictions)
        min_idx = np.argmin(predictions)
        
        print(f"\n最热点: ({lat_flat[max_idx]:.1f}°N, {lon_flat[max_idx]:.1f}°E) - {predictions[max_idx]:.2f}°C")
        print(f"最冷点: ({lat_flat[min_idx]:.1f}°N, {lon_flat[min_idx]:.1f}°E) - {predictions[min_idx]:.2f}°C")
        
        # 保存结果
        output_path = project_root / "data" / "predictions" / "region_temperature_analysis.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        region_data.to_csv(output_path, index=False)
        print(f"\n区域分析结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"区域分析失败: {e}")


async def example_4_prediction_engine():
    """
    示例4: 使用预测引擎进行高级预测
    """
    print("\n=== 示例4: 使用预测引擎进行高级预测 ===")
    
    model_manager = ModelManager()
    prediction_engine = PredictionEngine(model_manager=model_manager)
    
    models = model_manager.list_models()
    if not models:
        print("没有找到可用的模型")
        return
    
    model_ids = [models[0].id]
    
    # 设置预测时间（当前时间）
    prediction_time = datetime.now()
    time_features = create_time_features(prediction_time)
    
    # 配置预测任务
    config = PredictionConfig(
        prediction_type=PredictionType.REAL_TIME.value,
        target_variable="t2m",
        prediction_horizon=1,
        temporal_resolution="daily",
        confidence_interval=0.95
    )
    
    # 准备输入数据
    input_data = pd.DataFrame({
        'latitude': [40.0, 41.0, 42.0],
        'longitude': [116.0, 117.0, 118.0],
        'number': [0, 0, 0],              # GRIB数据编号
        'step': [0, 0, 0],                # 时间步长
        'surface': [1, 1, 1],             # 地表层标识
        'msl': [101325.0, 101320.0, 101330.0],
        'sst': [15.2, 15.4, 15.6],
        'sp': [101000.0, 100995.0, 101005.0],
        'quality_score': [1.0, 1.0, 1.0]
    })
    
    # 为每行数据添加时间特征
    for key, value in time_features.items():
        input_data[key] = value
    
    print(f"预测时间: {prediction_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"预测月份: {time_features['month']}月")
    print(f"预测季节: {['', '冬季', '春季', '夏季', '秋季'][time_features['season']]}")
    
    print(f"\n输入数据:")
    print(input_data)
    
    try:
        # 创建预测任务
        task_id = await prediction_engine.create_prediction_task(
            config=config,
            model_ids=model_ids,
            input_data=input_data
        )
        
        print(f"\n创建预测任务: {task_id}")
        
        # 执行预测任务
        result = await prediction_engine.run_prediction_task(task_id)
        
        print(f"\n预测任务完成!")
        print(f"预测结果: {result.predictions}")
        
        if result.confidence_lower is not None:
            print(f"置信区间下界: {result.confidence_lower}")
            print(f"置信区间上界: {result.confidence_upper}")
        
        if result.metadata:
            print(f"元数据: {result.metadata}")
            
    except Exception as e:
        print(f"预测引擎任务失败: {e}")


def show_model_info():
    """
    显示可用模型信息
    """
    print("\n=== 可用模型信息 ===")
    
    model_manager = ModelManager()
    models = model_manager.list_models()
    
    if not models:
        print("没有找到可用的模型")
        print("请先运行 'python scripts/train_climate_model.py' 训练模型")
        return
    
    for model in models:
        print(f"\n模型ID: {model.id}")
        print(f"模型名称: {model.name}")
        print(f"模型类型: {model.model_type}")
        print(f"算法: {model.algorithm}")
        print(f"状态: {model.status}")
        print(f"目标变量: {model.config.target}")
        print(f"特征变量: {model.config.features}")
        
        if model.metrics:
            print(f"性能指标:")
            if model.metrics.r2:
                print(f"  R²得分: {model.metrics.r2:.4f}")
            if model.metrics.rmse:
                print(f"  RMSE: {model.metrics.rmse:.4f}")
            if model.metrics.mae:
                print(f"  MAE: {model.metrics.mae:.4f}")
        
        print(f"创建时间: {model.created_at}")


async def main():
    """
    主函数：运行所有示例
    """
    print("气候模型使用示例")
    print("=" * 50)
    
    # 显示模型信息
    show_model_info()
    
    # 运行示例
    await example_1_simple_prediction()
    await example_2_batch_prediction()
    await example_3_region_analysis()
    await example_4_prediction_engine()
    
    print("\n=== 所有示例运行完成 ===")
    print("\n更多使用方法请参考: docs/MODEL_USAGE_GUIDE.md")


if __name__ == "__main__":
    asyncio.run(main())