#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版历史气候模式识别演示

本脚本创建更真实的模拟气候数据，展示AI技术在识别历史气候模式、趋势和异常事件方面的能力。
包含：
1. 明显的全球变暖趋势
2. 季节性变化模式
3. 极端天气事件（热浪、寒潮、干旱、洪涝）
4. 气候振荡模式（如ENSO）
5. 突变点检测
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.historical_climate_analyzer import (
    HistoricalClimateAnalyzer,
    AnomalyMethod,
    TrendMethod
)
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

def create_realistic_climate_data():
    """
    创建更真实的历史气候数据，包含多种气候模式和事件
    """
    logger.info("创建真实的历史气候数据...")
    
    # 创建50年的月度数据（1970-2023）
    start_date = datetime(1970, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='M')
    
    np.random.seed(42)
    n_points = len(dates)
    time_index = np.arange(n_points)
    
    # === 温度数据 ===
    # 1. 全球变暖趋势（1970-2023年升温约1.5度）
    warming_trend = 1.5 * time_index / n_points
    
    # 2. 季节性变化（北半球）
    seasonal_temp = 15 * np.sin(2 * np.pi * time_index / 12 - np.pi/2)
    
    # 3. 年际变化（ENSO等）
    enso_cycle = 2 * np.sin(2 * np.pi * time_index / (3.5 * 12))  # 3.5年周期
    
    # 4. 十年际变化（PDO等）
    decadal_cycle = 1.5 * np.sin(2 * np.pi * time_index / (20 * 12))  # 20年周期
    
    # 5. 随机噪声
    temp_noise = np.random.normal(0, 1.5, n_points)
    
    # 基础温度
    base_temp = 14.0
    temperature = base_temp + warming_trend + seasonal_temp + enso_cycle + decadal_cycle + temp_noise
    
    # 添加极端事件
    # 热浪事件（夏季更频繁）
    summer_months = np.where((time_index % 12 >= 5) & (time_index % 12 <= 7))[0]
    heatwave_indices = np.random.choice(summer_months, min(30, len(summer_months)), replace=False)
    temperature[heatwave_indices] += np.random.uniform(5, 12, len(heatwave_indices))
    
    # 寒潮事件（冬季更频繁）
    winter_months = np.where((time_index % 12 <= 1) | (time_index % 12 >= 11))[0]
    coldwave_indices = np.random.choice(winter_months, min(20, len(winter_months)), replace=False)
    temperature[coldwave_indices] -= np.random.uniform(8, 15, len(coldwave_indices))
    
    # 气候突变点（1980年代和2000年代）
    shift_1980s = np.where((time_index >= 10*12) & (time_index <= 15*12))[0]
    temperature[shift_1980s] += 0.8  # 1980年代升温
    
    shift_2000s = np.where(time_index >= 30*12)[0]
    temperature[shift_2000s] += 0.5  # 2000年代后进一步升温
    
    # === 降水数据 ===
    # 1. 季节性降水（夏季多雨）
    seasonal_precip = 50 + 40 * np.sin(2 * np.pi * time_index / 12)
    
    # 2. ENSO对降水的影响
    enso_precip_effect = 20 * np.sin(2 * np.pi * time_index / (3.5 * 12) + np.pi/4)
    
    # 3. 长期趋势（某些地区降水减少）
    precip_trend = -10 * time_index / n_points
    
    # 4. 随机变化
    precip_noise = np.random.gamma(2, 15, n_points)
    
    precipitation = seasonal_precip + enso_precip_effect + precip_trend + precip_noise
    precipitation = np.maximum(precipitation, 0)  # 降水不能为负
    
    # 添加极端降水事件
    # 干旱事件（连续低降水）
    drought_periods = [
        (15*12, 15*12+18),  # 1985年干旱
        (25*12, 25*12+24),  # 1995年干旱
        (40*12, 40*12+15)   # 2010年干旱
    ]
    
    for start, end in drought_periods:
        if end < n_points:
            precipitation[start:end] *= 0.3
    
    # 洪涝事件（极端降水）
    flood_indices = np.random.choice(n_points, 25, replace=False)
    precipitation[flood_indices] += np.random.uniform(150, 400, 25)
    
    # === 其他气象变量 ===
    
    # 湿度（与温度和降水相关）
    humidity = 65 + 15 * np.sin(2 * np.pi * time_index / 12) - 0.3 * warming_trend + \
               0.1 * (precipitation - np.mean(precipitation)) + np.random.normal(0, 3, n_points)
    humidity = np.clip(humidity, 20, 95)
    
    # 风速（季节性变化 + 极端事件）
    wind_speed = 12 + 4 * np.sin(2 * np.pi * time_index / 12 + np.pi) + np.random.exponential(2, n_points)
    
    # 添加风暴事件
    storm_indices = np.random.choice(n_points, 15, replace=False)
    wind_speed[storm_indices] += np.random.uniform(20, 40, 15)
    
    # 气压（与温度反相关）
    pressure = 1013 - 0.5 * warming_trend + 8 * np.sin(2 * np.pi * time_index / 12 + np.pi) + \
               np.random.normal(0, 5, n_points)
    
    # 海表温度（与陆地温度相关但变化较小）
    sst = temperature * 0.8 + 2 + np.random.normal(0, 0.8, n_points)
    
    # 创建DataFrame
    climate_data = pd.DataFrame({
        'time': dates,
        'temperature': temperature,
        'precipitation': precipitation,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure,
        'sea_surface_temp': sst
    })
    
    logger.info(f"创建了包含 {len(climate_data)} 条记录的真实气候数据")
    logger.info(f"时间范围: {dates[0]} 至 {dates[-1]}")
    logger.info(f"变量: {list(climate_data.columns[1:])}")
    
    return climate_data

def analyze_climate_patterns(data):
    """
    执行全面的气候模式分析
    """
    logger.info("=== 开始全面气候模式分析 ===")
    
    analyzer = HistoricalClimateAnalyzer()
    
    print("\n🌍 历史气候模式识别与分析")
    print("=" * 60)
    print(f"📊 数据概况: {len(data)} 个月的气候数据 ({data['time'].min().year}-{data['time'].max().year})")
    print(f"🌡️ 分析变量: {', '.join(data.columns[1:])}")
    print()
    
    # 1. 趋势分析
    print("📈 1. 长期趋势分析")
    print("-" * 30)
    
    trend_results = analyzer.analyze_trends(data, method=TrendMethod.LINEAR_REGRESSION)
    
    for var, result in trend_results.items():
        if result.significance == 'significant':
            direction_emoji = {
                'increasing': '🔺',
                'decreasing': '🔻',
                'stable': '➡️'
            }.get(result.trend_direction, '❓')
            
            print(f"  {direction_emoji} {var}:")
            print(f"    • 趋势: {result.trend_direction} (p={result.p_value:.4f})")
            print(f"    • 年际变化: {result.annual_change:.4f}")
            print(f"    • 50年总变化: {result.annual_change * 50:.2f}")
            print(f"    • R²: {result.r_squared:.4f}")
            print()
    
    # 2. 季节性分析
    print("🔄 2. 季节性模式分析")
    print("-" * 30)
    
    seasonality_results = analyzer.analyze_seasonality(data)
    
    for var, result in seasonality_results.items():
        if result.has_seasonality:
            print(f"  🔄 {var}:")
            print(f"    • 季节性强度: {result.seasonal_strength:.4f}")
            if result.seasonal_peaks:
                months = ['1月', '2月', '3月', '4月', '5月', '6月', 
                         '7月', '8月', '9月', '10月', '11月', '12月']
                peak_months = [months[int(peak)-1] for peak in result.seasonal_peaks if 1 <= int(peak) <= 12]
                print(f"    • 峰值月份: {', '.join(peak_months)}")
            print()
    
    # 3. 异常事件检测
    print("🚨 3. 异常事件检测")
    print("-" * 30)
    
    # 使用多种方法检测异常
    methods = [AnomalyMethod.ISOLATION_FOREST, AnomalyMethod.STATISTICAL]
    
    for method in methods:
        print(f"\n  方法: {method.value}")
        anomaly_results = analyzer.detect_anomalies(data, method=method, contamination=0.05)
        
        for var, result in anomaly_results.items():
            if result.total_anomalies > 0:
                print(f"    🚨 {var}: {result.total_anomalies} 个异常点 ({result.anomaly_rate:.1%})")
                
                # 显示最极端的异常值
                if result.anomaly_values:
                    extreme_values = sorted(result.anomaly_values, key=abs, reverse=True)[:3]
                    print(f"      最极端值: {[f'{v:.2f}' for v in extreme_values]}")
    
    # 4. 极端事件分析
    print("\n⚡ 4. 极端事件分析")
    print("-" * 30)
    
    extreme_results = analyzer.analyze_extreme_events(data)
    
    event_emojis = {
        'heatwave': '🔥',
        'coldwave': '🧊',
        'drought': '🏜️',
        'flood': '🌊'
    }
    
    for event_key, result in extreme_results.items():
        if len(result.events) > 0:
            emoji = event_emojis.get(result.event_type, '⚡')
            print(f"  {emoji} {result.variable} - {result.event_type}:")
            print(f"    • 事件总数: {len(result.events)}")
            print(f"    • 年均频率: {result.frequency:.2f} 次/年")
            
            if result.events:
                intensities = [event['intensity'] for event in result.events]
                durations = [event['duration'] for event in result.events]
                
                print(f"    • 平均强度: {np.mean(intensities):.2f}")
                print(f"    • 最大强度: {np.max(intensities):.2f}")
                print(f"    • 平均持续: {np.mean(durations):.1f} 个月")
                
                # 检查趋势
                if result.intensity_trend.significance == 'significant':
                    trend_dir = '增强' if result.intensity_trend.trend_direction == 'increasing' else '减弱'
                    print(f"    • 强度趋势: {trend_dir} ⚠️")
            print()
    
    # 5. 模式识别
    print("🔍 5. 气候模式识别")
    print("-" * 30)
    
    pattern_results = analyzer.identify_patterns(data, n_clusters=4)
    
    for var, result in pattern_results.items():
        if result.pattern_strength > 0.3:  # 只显示较强的模式
            print(f"  🔍 {var}:")
            print(f"    • 模式强度: {result.pattern_strength:.4f}")
            print(f"    • 识别模式数: {len(result.dominant_patterns)}")
            
            for i, pattern in enumerate(result.dominant_patterns[:2]):  # 显示前2个主要模式
                print(f"      模式 {i+1}: {pattern['characteristics']} ({pattern['percentage']:.1f}%)")
            print()
    
    return {
        'trends': trend_results,
        'seasonality': seasonality_results,
        'anomalies': anomaly_results,
        'extremes': extreme_results,
        'patterns': pattern_results
    }

def create_comprehensive_visualizations(data, analysis_results):
    """
    创建综合的可视化分析图表
    """
    logger.info("创建综合可视化分析...")
    
    # 设置中文字体和图表样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    # 创建大型图表
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. 温度长期趋势 (大图)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(data['time'], data['temperature'], alpha=0.6, linewidth=0.8, color='steelblue', label='月度温度')
    
    # 添加趋势线
    if 'temperature' in analysis_results['trends']:
        trend = analysis_results['trends']['temperature']
        x_numeric = np.arange(len(data))
        trend_line = trend.slope * x_numeric + trend.intercept
        # 确保数字格式化正确显示
        trend_str = f"{trend.annual_change:.4f}".replace('.', '.')
        ax1.plot(data['time'], trend_line, 'r-', linewidth=3, 
                label=f'趋势线 ({trend_str}°C/年)')
    
    # 添加5年移动平均
    temp_5yr = data['temperature'].rolling(window=60, center=True).mean()
    ax1.plot(data['time'], temp_5yr, 'orange', linewidth=2, label='5年移动平均')
    
    ax1.set_title('全球温度长期变化趋势 (1970-2023)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('温度 (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 降水异常检测
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(data['time'], data['precipitation'], alpha=0.7, linewidth=1, color='blue')
    
    # 标记异常点
    if 'precipitation' in analysis_results['anomalies']:
        anomaly_result = analysis_results['anomalies']['precipitation']
        if anomaly_result.anomaly_indices:
            anomaly_times = data['time'].iloc[anomaly_result.anomaly_indices]
            anomaly_values = [data['precipitation'].iloc[i] for i in anomaly_result.anomaly_indices]
            # 确保数字格式化正确显示
            anomaly_count = len(anomaly_result.anomaly_indices)
            ax2.scatter(anomaly_times, anomaly_values, color='red', s=30, 
                       label=f'异常点 ({anomaly_count}个)', zorder=5)
    
    ax2.set_title('降水异常检测')
    ax2.set_ylabel('降水量 (mm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 季节性模式
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 计算月度平均值
    monthly_temp = data.groupby(data['time'].dt.month)['temperature'].mean()
    monthly_precip = data.groupby(data['time'].dt.month)['precipitation'].mean()
    
    months = ['1月', '2月', '3月', '4月', '5月', '6月', 
             '7月', '8月', '9月', '10月', '11月', '12月']
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(range(1, 13), monthly_temp, 'ro-', linewidth=2, label='温度')
    line2 = ax3_twin.plot(range(1, 13), monthly_precip, 'bs-', linewidth=2, label='降水')
    
    ax3.set_xlabel('月份')
    ax3.set_ylabel('温度 (°C)', color='red')
    ax3_twin.set_ylabel('降水量 (mm)', color='blue')
    ax3.set_title('季节性变化模式')
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels([m[:2] for m in months])
    ax3.grid(True, alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # 4. 变量相关性热图
    ax4 = fig.add_subplot(gs[1, 2])
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    
    im = ax4.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels([col[:8] for col in corr_matrix.columns], rotation=45, ha='right')
    ax4.set_yticklabels([col[:8] for col in corr_matrix.columns])
    ax4.set_title('变量相关性矩阵')
    
    # 添加数值标签
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax4, shrink=0.8)
    
    # 5. 极端事件时间序列
    ax5 = fig.add_subplot(gs[2, :])
    
    # 绘制温度和极端事件
    ax5.plot(data['time'], data['temperature'], alpha=0.5, color='gray', linewidth=0.8)
    
    # 标记极端事件
    colors = {'heatwave': 'red', 'coldwave': 'blue', 'drought': 'brown', 'flood': 'cyan'}
    
    for event_key, result in analysis_results['extremes'].items():
        if len(result.events) > 0 and 'temperature' in result.variable:
            color = colors.get(result.event_type, 'black')
            for event in result.events:
                start_idx = event['start_index']
                end_idx = event['end_index']
                if start_idx < len(data) and end_idx < len(data):
                    event_time = data['time'].iloc[start_idx:end_idx+1]
                    event_temp = data['temperature'].iloc[start_idx:end_idx+1]
                    ax5.plot(event_time, event_temp, color=color, linewidth=3, alpha=0.8)
    
    ax5.set_title('极端温度事件识别')
    ax5.set_ylabel('温度 (°C)')
    ax5.grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [plt.Line2D([0], [0], color=color, lw=3, label=event_type) 
                      for event_type, color in colors.items()]
    ax5.legend(handles=legend_elements, loc='upper left')
    
    # 6. 年代际变化对比
    ax6 = fig.add_subplot(gs[3, 0])
    
    # 按年代分组
    data['decade'] = (data['time'].dt.year // 10) * 10
    decade_temp = data.groupby('decade')['temperature'].mean()
    
    bars = ax6.bar(decade_temp.index, decade_temp.values, 
                   color=['lightblue', 'skyblue', 'steelblue', 'darkblue', 'navy', 'midnightblue'])
    ax6.set_title('年代际温度变化')
    ax6.set_xlabel('年代')
    ax6.set_ylabel('平均温度 (°C)')
    
    # 添加数值标签
    for bar, temp in zip(bars, decade_temp.values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{temp:.1f}°C', ha='center', va='bottom')
    
    # 7. 异常事件频率统计
    ax7 = fig.add_subplot(gs[3, 1])
    
    anomaly_counts = {}
    for var, result in analysis_results['anomalies'].items():
        if result.total_anomalies > 0:
            anomaly_counts[var[:8]] = result.total_anomalies
    
    if anomaly_counts:
        vars_list = list(anomaly_counts.keys())
        counts = list(anomaly_counts.values())
        
        bars = ax7.bar(vars_list, counts, color='orange', alpha=0.7)
        ax7.set_title('异常事件统计')
        ax7.set_ylabel('异常点数量')
        ax7.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
    
    # 8. 模式强度对比
    ax8 = fig.add_subplot(gs[3, 2])
    
    pattern_strengths = {}
    for var, result in analysis_results['patterns'].items():
        pattern_strengths[var[:8]] = result.pattern_strength
    
    if pattern_strengths:
        vars_list = list(pattern_strengths.keys())
        strengths = list(pattern_strengths.values())
        
        bars = ax8.bar(vars_list, strengths, color='green', alpha=0.7)
        ax8.set_title('气候模式强度')
        ax8.set_ylabel('模式强度')
        ax8.tick_params(axis='x', rotation=45)
        ax8.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, strength in zip(bars, strengths):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{strength:.3f}', ha='center', va='bottom')
    
    plt.suptitle('历史气候模式识别与分析综合报告', fontsize=18, fontweight='bold', y=0.98)
    
    # 保存图表
    output_dir = "outputs/climate_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(f"{output_dir}/enhanced_climate_analysis.png", dpi=300, bbox_inches='tight')
    logger.info(f"综合分析图表已保存到 {output_dir}/enhanced_climate_analysis.png")
    
    plt.show()

def generate_analysis_summary(data, analysis_results):
    """
    生成分析摘要报告
    """
    print("\n📋 分析摘要报告")
    print("=" * 60)
    
    # 数据概况
    print(f"📊 数据时间跨度: {data['time'].min().year}-{data['time'].max().year} ({len(data)} 个月)")
    print(f"🌡️ 温度范围: {data['temperature'].min():.1f}°C 至 {data['temperature'].max():.1f}°C")
    print(f"🌧️ 降水范围: {data['precipitation'].min():.1f}mm 至 {data['precipitation'].max():.1f}mm")
    print()
    
    # 趋势发现
    significant_trends = {var: result for var, result in analysis_results['trends'].items() 
                         if result.significance == 'significant'}
    
    print(f"📈 显著趋势发现 ({len(significant_trends)} 个):")
    for var, result in significant_trends.items():
        direction = '上升' if result.trend_direction == 'increasing' else '下降'
        print(f"  • {var}: {direction}趋势，年变化率 {result.annual_change:.4f}")
    print()
    
    # 季节性发现
    seasonal_vars = {var: result for var, result in analysis_results['seasonality'].items() 
                    if result.has_seasonality}
    
    print(f"🔄 季节性模式 ({len(seasonal_vars)} 个变量):")
    for var, result in seasonal_vars.items():
        print(f"  • {var}: 季节性强度 {result.seasonal_strength:.3f}")
    print()
    
    # 异常事件统计
    total_anomalies = sum(result.total_anomalies for result in analysis_results['anomalies'].values())
    print(f"🚨 异常事件: 共检测到 {total_anomalies} 个异常点")
    
    high_anomaly_vars = {var: result for var, result in analysis_results['anomalies'].items() 
                        if result.anomaly_rate > 0.05}
    
    for var, result in high_anomaly_vars.items():
        print(f"  • {var}: {result.total_anomalies} 个异常点 ({result.anomaly_rate:.1%})")
    print()
    
    # 极端事件统计
    total_extreme_events = sum(len(result.events) for result in analysis_results['extremes'].values())
    print(f"⚡ 极端事件: 共识别 {total_extreme_events} 个极端事件")
    
    for event_key, result in analysis_results['extremes'].items():
        if len(result.events) > 0:
            print(f"  • {result.variable} {result.event_type}: {len(result.events)} 次 ({result.frequency:.2f}/年)")
    print()
    
    # 模式识别结果
    strong_patterns = {var: result for var, result in analysis_results['patterns'].items() 
                      if result.pattern_strength > 0.3}
    
    print(f"🔍 气候模式: 识别出 {len(strong_patterns)} 个强模式")
    for var, result in strong_patterns.items():
        print(f"  • {var}: 模式强度 {result.pattern_strength:.3f}")
    print()
    
    # 关键发现
    print("🎯 关键发现:")
    
    # 温度趋势
    if 'temperature' in significant_trends:
        temp_trend = significant_trends['temperature']
        total_change = temp_trend.annual_change * (data['time'].max().year - data['time'].min().year)
        print(f"  🔥 温度显著上升: {total_change:.2f}°C ({data['time'].min().year}-{data['time'].max().year})")
    
    # 极端事件趋势
    extreme_trends = []
    for event_key, result in analysis_results['extremes'].items():
        if (result.intensity_trend.significance == 'significant' or 
            result.duration_trend.significance == 'significant'):
            extreme_trends.append(f"{result.event_type}事件强度或持续时间有显著变化")
    
    if extreme_trends:
        print(f"  ⚠️ 极端事件变化: {'; '.join(extreme_trends)}")
    
    # 异常率高的变量
    high_anomaly_list = [var for var, result in analysis_results['anomalies'].items() 
                        if result.anomaly_rate > 0.08]
    if high_anomaly_list:
        print(f"  🚨 高异常率变量: {', '.join(high_anomaly_list)}")
    
    print()
    print("✅ 分析完成！AI技术成功识别了历史气候数据中的多种模式、趋势和异常事件。")

def main():
    """
    主函数：执行增强版历史气候模式分析
    """
    print("🌍 增强版历史气候模式识别与分析系统")
    print("=" * 70)
    print("利用AI技术深度分析历史气候数据，识别模式、趋势和异常事件")
    print()
    
    try:
        # 1. 创建真实的气候数据
        data = create_realistic_climate_data()
        
        # 2. 执行全面分析
        analysis_results = analyze_climate_patterns(data)
        
        # 3. 创建可视化
        create_comprehensive_visualizations(data, analysis_results)
        
        # 4. 生成摘要报告
        generate_analysis_summary(data, analysis_results)
        
        # 5. 保存详细结果
        output_dir = "outputs/climate_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据
        data.to_csv(f"{output_dir}/simulated_climate_data.csv", index=False, encoding='utf-8-sig')
        
        # 保存分析结果摘要
        with open(f"{output_dir}/enhanced_analysis_summary.txt", 'w', encoding='utf-8') as f:
            f.write("增强版历史气候模式分析摘要\n")
            f.write("=" * 50 + "\n\n")
            
            # 写入关键统计信息
            f.write(f"数据时间跨度: {data['time'].min().year}-{data['time'].max().year}\n")
            f.write(f"数据点数量: {len(data)} 个月\n")
            f.write(f"分析变量: {', '.join(data.columns[1:])}\n\n")
            
            # 趋势分析结果
            f.write("显著趋势:\n")
            for var, result in analysis_results['trends'].items():
                if result.significance == 'significant':
                    f.write(f"  {var}: {result.trend_direction}, 年变化率 {result.annual_change:.6f}\n")
            
            # 异常检测结果
            f.write("\n异常检测:\n")
            for var, result in analysis_results['anomalies'].items():
                if result.total_anomalies > 0:
                    f.write(f"  {var}: {result.total_anomalies} 个异常点 ({result.anomaly_rate:.2%})\n")
            
            # 极端事件
            f.write("\n极端事件:\n")
            for event_key, result in analysis_results['extremes'].items():
                if len(result.events) > 0:
                    f.write(f"  {result.variable} {result.event_type}: {len(result.events)} 次\n")
        
        logger.info(f"详细结果已保存到 {output_dir} 目录")
        
        print("\n🎉 增强版历史气候模式分析完成！")
        print("📁 所有结果文件已保存到 outputs/climate_analysis/ 目录")
        print("📊 综合可视化图表已生成并显示")
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()