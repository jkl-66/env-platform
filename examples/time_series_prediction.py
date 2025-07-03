#!/usr/bin/env python3
"""
时间序列气候预测示例

本脚本演示如何使用气候模型进行时间序列预测，包括：
1. 预测未来24小时的温度变化
2. 预测一周内的温度趋势
3. 季节性温度对比分析
4. 不同时间点的温度预测可视化
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.model_manager import ModelManager

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

async def predict_24h_temperature():
    """
    示例1: 预测未来24小时的温度变化
    """
    print("\n=== 示例1: 预测未来24小时的温度变化 ===")
    
    model_manager = ModelManager()
    
    # 设置基准时间（当前时间）
    base_time = datetime.now()
    
    # 生成未来24小时的时间点（每小时一个预测）
    time_points = [base_time + timedelta(hours=h) for h in range(24)]
    
    # 准备预测数据
    predictions_data = []
    for time_point in time_points:
        time_features = create_time_features(time_point)
        
        data_point = {
            'latitude': 39.9042,      # 北京纬度
            'longitude': 116.4074,    # 北京经度
            'number': 0,
            'step': 0,
            'surface': 1013.25,
            'msl': 1013.25,
            'sst': 285.15,
            'sp': 101325.0,
            'quality_score': 0.95,
            **time_features
        }
        predictions_data.append(data_point)
    
    prediction_df = pd.DataFrame(predictions_data)
    
    try:
        # 进行预测（使用最新的包含时间特征的模型）
        result = await model_manager.predict('temperature_prediction_rf_20250703_112511', prediction_df)
        
        # 处理结果
        results_df = pd.DataFrame({
            'time': time_points,
            'hour': [t.hour for t in time_points],
            'temp_k': result,
            'temp_c': result - 273.15,
            'confidence': [0.85] * len(result)  # 模拟置信度
        })
        
        print(f"\n北京未来24小时温度预测（从 {base_time.strftime('%Y-%m-%d %H:%M')} 开始）:")
        print("时间\t\t温度(°C)\t置信度")
        print("-" * 40)
        
        for _, row in results_df.iterrows():
            print(f"{row['time'].strftime('%m-%d %H:%M')}\t{row['temp_c']:.1f}°C\t\t{row['confidence']:.3f}")
        
        # 分析温度变化
        max_temp = results_df['temp_c'].max()
        min_temp = results_df['temp_c'].min()
        max_idx = results_df['temp_c'].idxmax()
        min_idx = results_df['temp_c'].idxmin()
        max_time = results_df.iloc[max_idx]['time']
        min_time = results_df.iloc[min_idx]['time']
        
        print(f"\n温度变化分析:")
        print(f"最高温度: {max_temp:.1f}°C (时间: {max_time.strftime('%H:%M')})")
        print(f"最低温度: {min_temp:.1f}°C (时间: {min_time.strftime('%H:%M')})")
        print(f"温差: {max_temp - min_temp:.1f}°C")
        
        # 保存结果
        output_file = project_root / "examples" / "24h_temperature_forecast.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n详细结果已保存到: {output_file}")
        
        return results_df
        
    except Exception as e:
        print(f"未来24小时预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None

async def predict_weekly_trend():
    """
    示例2: 预测一周内的温度趋势
    """
    print("\n=== 示例2: 预测一周内的温度趋势 ===")
    
    model_manager = ModelManager()
    
    # 设置基准时间
    base_time = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)  # 每天中午12点
    
    # 生成一周的时间点（每天一个预测）
    time_points = [base_time + timedelta(days=d) for d in range(7)]
    
    # 准备预测数据
    predictions_data = []
    for time_point in time_points:
        time_features = create_time_features(time_point)
        
        data_point = {
            'latitude': 39.9042,      # 北京纬度
            'longitude': 116.4074,    # 北京经度
            'number': 0,
            'step': 0,
            'surface': 1013.25,
            'msl': 1013.25,
            'sst': 285.15,
            'sp': 101325.0,
            'quality_score': 0.95,
            **time_features
        }
        predictions_data.append(data_point)
    
    prediction_df = pd.DataFrame(predictions_data)
    
    try:
        # 进行预测（使用最新的包含时间特征的模型）
        result = await model_manager.predict('temperature_prediction_rf_20250703_112511', prediction_df)
        
        # 处理结果
        results_df = pd.DataFrame({
            'date': [t.date() for t in time_points],
            'weekday': [t.strftime('%A') for t in time_points],
            'temp_k': result,
            'temp_c': result - 273.15,
            'confidence': [0.85] * len(result)  # 模拟置信度
        })
        
        print(f"\n北京未来一周温度趋势（每天中午12点）:")
        print("日期\t\t星期\t\t温度(°C)\t置信度")
        print("-" * 50)
        
        for _, row in results_df.iterrows():
            print(f"{row['date']}\t{row['weekday'][:3]}\t\t{row['temp_c']:.1f}°C\t\t{row['confidence']:.3f}")
        
        # 分析温度趋势
        temp_trend = np.polyfit(range(len(results_df)), results_df['temp_c'], 1)[0]
        avg_temp = results_df['temp_c'].mean()
        
        print(f"\n一周温度趋势分析:")
        print(f"平均温度: {avg_temp:.1f}°C")
        if temp_trend > 0.1:
            print(f"温度趋势: 上升 (+{temp_trend:.2f}°C/天)")
        elif temp_trend < -0.1:
            print(f"温度趋势: 下降 ({temp_trend:.2f}°C/天)")
        else:
            print(f"温度趋势: 稳定 ({temp_trend:.2f}°C/天)")
        
        # 保存结果
        output_file = project_root / "examples" / "weekly_temperature_trend.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n详细结果已保存到: {output_file}")
        
        return results_df
        
    except Exception as e:
        print(f"一周趋势预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None

async def seasonal_comparison():
    """
    示例3: 季节性温度对比分析
    """
    print("\n=== 示例3: 季节性温度对比分析 ===")
    
    model_manager = ModelManager()
    
    # 设置四个季节的代表性日期（每个季节的中间月份15日中午12点）
    current_year = datetime.now().year
    seasonal_dates = {
        '春季': datetime(current_year, 4, 15, 12, 0),  # 4月15日
        '夏季': datetime(current_year, 7, 15, 12, 0),  # 7月15日
        '秋季': datetime(current_year, 10, 15, 12, 0), # 10月15日
        '冬季': datetime(current_year, 1, 15, 12, 0)   # 1月15日
    }
    
    # 准备预测数据
    predictions_data = []
    season_names = []
    
    for season, date in seasonal_dates.items():
        time_features = create_time_features(date)
        
        data_point = {
            'latitude': 39.9042,      # 北京纬度
            'longitude': 116.4074,    # 北京经度
            'number': 0,
            'step': 0,
            'surface': 1013.25,
            'msl': 1013.25,
            'sst': 285.15,
            'sp': 101325.0,
            'quality_score': 0.95,
            **time_features
        }
        predictions_data.append(data_point)
        season_names.append(season)
    
    prediction_df = pd.DataFrame(predictions_data)
    
    try:
        # 进行预测（使用最新的包含时间特征的模型）
        result = await model_manager.predict('temperature_prediction_rf_20250703_112511', prediction_df)
        
        # 处理结果
        results_df = pd.DataFrame({
            'season': season_names,
            'date': list(seasonal_dates.values()),
            'temp_k': result,
            'temp_c': result - 273.15,
            'confidence': [0.85] * len(result)  # 模拟置信度
        })
        
        print(f"\n北京四季温度对比分析（{current_year}年）:")
        print("季节\t\t日期\t\t温度(°C)\t置信度")
        print("-" * 50)
        
        for _, row in results_df.iterrows():
            print(f"{row['season']}\t\t{row['date'].strftime('%m-%d')}\t\t{row['temp_c']:.1f}°C\t\t{row['confidence']:.3f}")
        
        # 分析季节性差异
        max_idx = results_df['temp_c'].idxmax()
        min_idx = results_df['temp_c'].idxmin()
        max_season = results_df.iloc[max_idx]['season']
        min_season = results_df.iloc[min_idx]['season']
        max_temp = results_df['temp_c'].max()
        min_temp = results_df['temp_c'].min()
        
        print(f"\n季节性分析:")
        print(f"最热季节: {max_season} ({max_temp:.1f}°C)")
        print(f"最冷季节: {min_season} ({min_temp:.1f}°C)")
        print(f"季节温差: {max_temp - min_temp:.1f}°C")
        
        # 保存结果
        output_file = project_root / "examples" / "seasonal_temperature_comparison.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n详细结果已保存到: {output_file}")
        
        return results_df
        
    except Exception as e:
        print(f"季节性对比分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

async def multi_city_time_comparison():
    """
    示例4: 多城市不同时间点温度对比
    """
    print("\n=== 示例4: 多城市不同时间点温度对比 ===")
    
    model_manager = ModelManager()
    
    # 定义城市信息
    cities = {
        '北京': {'lat': 39.9042, 'lon': 116.4074},
        '上海': {'lat': 31.2304, 'lon': 121.4737},
        '广州': {'lat': 23.1291, 'lon': 113.2644},
        '哈尔滨': {'lat': 45.8038, 'lon': 126.5349}
    }
    
    # 定义时间点（今天的不同时刻）
    base_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    time_points = {
        '凌晨': base_date.replace(hour=3),
        '上午': base_date.replace(hour=9),
        '中午': base_date.replace(hour=12),
        '下午': base_date.replace(hour=15),
        '傍晚': base_date.replace(hour=18),
        '夜晚': base_date.replace(hour=21)
    }
    
    # 准备预测数据
    predictions_data = []
    city_names = []
    time_names = []
    
    for city_name, city_info in cities.items():
        for time_name, time_point in time_points.items():
            time_features = create_time_features(time_point)
            
            data_point = {
                'latitude': city_info['lat'],
                'longitude': city_info['lon'],
                'number': 0,
                'step': 0,
                'surface': 1013.25,
                'msl': 1013.25,
                'sst': 285.15,
                'sp': 101325.0,
                'quality_score': 0.95,
                **time_features
            }
            predictions_data.append(data_point)
            city_names.append(city_name)
            time_names.append(time_name)
    
    prediction_df = pd.DataFrame(predictions_data)
    
    try:
        # 进行预测（使用最新的包含时间特征的模型）
        result = await model_manager.predict('temperature_prediction_rf_20250703_112511', prediction_df)
        
        # 处理结果
        results_df = pd.DataFrame({
            'city': city_names,
            'time_period': time_names,
            'temp_k': result,
            'temp_c': result - 273.15,
            'confidence': [0.85] * len(result)  # 模拟置信度
        })
        
        print(f"\n多城市不同时间点温度对比（{base_date.strftime('%Y-%m-%d')}）:")
        
        # 按城市分组显示
        for city in cities.keys():
            city_data = results_df[results_df['city'] == city]
            print(f"\n{city}:")
            print("时间\t\t温度(°C)\t置信度")
            print("-" * 30)
            for _, row in city_data.iterrows():
                print(f"{row['time_period']}\t\t{row['temp_c']:.1f}°C\t\t{row['confidence']:.3f}")
        
        # 分析各城市的日温差
        print(f"\n各城市日温差分析:")
        for city in cities.keys():
            city_data = results_df[results_df['city'] == city]
            max_temp = city_data['temp_c'].max()
            min_temp = city_data['temp_c'].min()
            temp_range = max_temp - min_temp
            print(f"{city}: {temp_range:.1f}°C (最高: {max_temp:.1f}°C, 最低: {min_temp:.1f}°C)")
        
        # 保存结果
        output_file = project_root / "examples" / "multi_city_time_comparison.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n详细结果已保存到: {output_file}")
        
        return results_df
        
    except Exception as e:
        print(f"多城市时间对比失败: {e}")
        import traceback
        traceback.print_exc()
        return None

async def show_model_info():
    """
    显示可用模型信息
    """
    print("=== 可用的气候预测模型 ===")
    
    model_manager = ModelManager()
    models = model_manager.list_models()  # 移除await，因为这是同步方法
    
    if not models:
        print("没有找到可用的模型。请先运行 scripts/train_climate_model.py 训练模型。")
        return
    
    for model in models:
        print(f"模型ID: {model.id}")
        print(f"模型类型: {model.model_type}")
        print(f"目标变量: {model.config.target}")
        print(f"创建时间: {model.created_at}")
        print("-" * 50)

async def main():
    """
    运行所有时间序列预测示例
    """
    print("气候模型时间序列预测示例")
    print("=" * 50)
    
    # 显示模型信息
    await show_model_info()
    
    try:
        # 运行所有示例
        await predict_24h_temperature()
        await predict_weekly_trend()
        await seasonal_comparison()
        await multi_city_time_comparison()
        
        print("\n=== 所有时间序列预测示例运行完成 ===")
        print("\n生成的文件:")
        print("- examples/24h_temperature_forecast.csv")
        print("- examples/weekly_temperature_trend.csv")
        print("- examples/seasonal_temperature_comparison.csv")
        print("- examples/multi_city_time_comparison.csv")
        print("\n更多使用方法请参考: docs/MODEL_USAGE_GUIDE.md")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())