#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气候预测系统测试脚本
"""

import os
import sys
import traceback
from datetime import datetime

print("开始测试气候预测系统...")
print(f"Python版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")

try:
    print("\n1. 测试导入模块...")
    
    # 测试基础模块导入
    import numpy as np
    print("✓ numpy导入成功")
    
    import pandas as pd
    print("✓ pandas导入成功")
    
    import torch
    print(f"✓ PyTorch导入成功，版本: {torch.__version__}")
    print(f"✓ CUDA可用: {torch.cuda.is_available()}")
    
    import xgboost as xgb
    print(f"✓ XGBoost导入成功，版本: {xgb.__version__}")
    
    import sklearn
    print(f"✓ scikit-learn导入成功，版本: {sklearn.__version__}")
    
    import matplotlib.pyplot as plt
    print("✓ matplotlib导入成功")
    
    print("\n2. 测试自定义模块导入...")
    
    # 添加当前目录到Python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from grib_to_mysql import GRIBToMySQLProcessor
    print("✓ GRIBToMySQLProcessor导入成功")
    
    from climate_prediction_models import ClimatePredictionSystem, WeatherDataLoader, WeatherTransformer
    print("✓ 气候预测模块导入成功")
    
    print("\n3. 测试数据库连接...")
    
    try:
        # 使用默认配置文件
        processor = GRIBToMySQLProcessor('mysql_config.json')
        print("✓ GRIB处理器初始化成功")
        
        # 测试数据库连接
        stats = processor.get_statistics()
        print(f"✓ 数据库连接成功，当前记录数: {stats.get('total_records', 0)}")
        
        processor.close()
        
    except Exception as db_error:
        print(f"✗ 数据库连接失败: {str(db_error)}")
    
    print("\n4. 测试气候预测系统初始化...")
    
    try:
        prediction_system = ClimatePredictionSystem('mysql_config.json')
        print("✓ 气候预测系统初始化成功")
        print(f"✓ 使用设备: {prediction_system.device}")
        
    except Exception as init_error:
        print(f"✗ 气候预测系统初始化失败: {str(init_error)}")
        prediction_system = None
    
    print("\n5. 测试模拟数据生成...")
    
    try:
        # 创建简单的模拟数据
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n_points = len(dates)
        
        # 生成模拟气象数据
        np.random.seed(42)
        temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n_points) / 365) + np.random.normal(0, 2, n_points)
        pressure = 1013 + np.random.normal(0, 10, n_points)
        humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(n_points) / 365 + np.pi/4) + np.random.normal(0, 5, n_points)
        
        # 创建DataFrame
        test_data = pd.DataFrame({
            'time': dates,
            'latitude': [31.0] * n_points,
            'longitude': [121.0] * n_points,
            'temperature_celsius': temperature,
            'pressure_hpa': pressure,
            'humidity_percent': humidity
        })
        
        print(f"✓ 模拟数据生成成功，数据形状: {test_data.shape}")
        print(f"✓ 温度范围: {test_data['temperature_celsius'].min():.2f}°C 至 {test_data['temperature_celsius'].max():.2f}°C")
        
        # 保存测试数据
        os.makedirs('outputs', exist_ok=True)
        test_data.to_csv('outputs/test_climate_data.csv', index=False)
        print("✓ 测试数据已保存至 outputs/test_climate_data.csv")
        
    except Exception as data_error:
        print(f"✗ 模拟数据生成失败: {str(data_error)}")
    
    print("\n6. 测试特征工程...")
    
    X, y = None, None
    try:
        if prediction_system is not None:
            # 使用气候预测系统处理数据
            prediction_system.processed_data = test_data.copy()
            prediction_system.prepare_features()
            
            print(f"✓ 特征工程完成，特征数量: {prediction_system.processed_data.shape[1]}")
            
            # 创建序列数据
            X, y = prediction_system.create_sequences(sequence_length=7, target_days=1)
            print(f"✓ 序列数据创建成功")
            print(f"✓ 输入序列形状: {X.shape}")
            print(f"✓ 目标序列形状: {y.shape}")
        else:
            print("✗ 预测系统未初始化，跳过特征工程")
        
    except Exception as feature_error:
        print(f"✗ 特征工程失败: {str(feature_error)}")
    
    print("\n7. 测试小规模模型训练...")
    
    try:
        if prediction_system is not None and X is not None and X.shape[0] > 10:  # 确保有足够的数据
            # 训练XGBoost模型（小规模）
            xgb_metrics = prediction_system.train_xgboost_model(test_size=0.3)
            print(f"✓ XGBoost训练成功 - MAE: {xgb_metrics['mae']:.4f}")
            
            # 训练Transformer模型（小规模）
            transformer_metrics = prediction_system.train_transformer_model(
                test_size=0.3,
                batch_size=8,
                epochs=5,
                learning_rate=0.01
            )
            print(f"✓ Transformer训练成功 - MAE: {transformer_metrics['mae']:.4f}")
            
            # 测试预测
            future_predictions = prediction_system.predict_future(days=7)
            print(f"✓ 未来预测成功，预测了 {len(future_predictions)} 天")
            
            # 保存预测结果
            future_predictions.to_csv('outputs/test_predictions.csv', index=False)
            print("✓ 预测结果已保存至 outputs/test_predictions.csv")
            
        else:
            print("✗ 系统未初始化或数据量不足，跳过模型训练")
            
    except Exception as train_error:
        print(f"✗ 模型训练失败: {str(train_error)}")
        print(f"错误详情: {traceback.format_exc()}")
    
    print("\n" + "="*50)
    print("测试完成!")
    print("="*50)
    
except Exception as e:
    print(f"\n❌ 测试过程中发生错误: {str(e)}")
    print(f"错误详情: {traceback.format_exc()}")

print("\n测试脚本执行完毕。")