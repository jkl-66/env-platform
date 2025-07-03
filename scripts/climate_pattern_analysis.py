#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史气候模式识别与分析示例

本脚本演示如何使用AI技术辅助研究人员识别历史气候模式、趋势和异常事件。
实现了以下功能：
1. 趋势分析与周期性检测
2. 异常事件检测
3. 模式识别
4. 极端事件分析
5. 综合报告生成
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
    TrendMethod,
    analyze_climate_data,
    detect_climate_anomalies
)
from src.data_processing.grib_processor import GRIBProcessor
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_climate_data():
    """
    加载历史气候数据
    """
    logger.info("正在加载历史气候数据...")
    
    # 尝试从GRIB文件加载数据
    grib_file = "data/raw/6cd7cc57755a5204a65bc7db615cd36b.grib"
    
    if os.path.exists(grib_file):
        try:
            processor = GRIBProcessor()
            df = processor.process_grib_to_dataframe(grib_file, sample_size=1000)
            
            # 确保有时间列
            if 'time' not in df.columns and 'valid_time' in df.columns:
                df['time'] = df['valid_time']
            elif 'time' not in df.columns:
                # 创建虚拟时间序列
                start_date = datetime(2020, 1, 1)
                df['time'] = [start_date + timedelta(hours=6*i) for i in range(len(df))]
            
            logger.info(f"从GRIB文件加载了 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.warning(f"无法加载GRIB文件: {e}，将使用模拟数据")
    
    # 创建模拟的历史气候数据
    logger.info("创建模拟历史气候数据...")
    
    # 创建30年的月度数据
    start_date = datetime(1990, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='M')
    
    np.random.seed(42)
    n_points = len(dates)
    
    # 模拟温度数据（全球变暖趋势 + 季节性 + 噪声）
    years = np.arange(n_points) / 12
    warming_trend = 0.02 * years  # 每年0.02度的升温
    seasonal_temp = 10 * np.sin(2 * np.pi * np.arange(n_points) / 12)  # 季节性变化
    temp_noise = np.random.normal(0, 2, n_points)
    temperature = 15 + warming_trend + seasonal_temp + temp_noise
    
    # 添加极端事件
    # 热浪事件
    heatwave_indices = np.random.choice(n_points, 20, replace=False)
    temperature[heatwave_indices] += np.random.uniform(8, 15, 20)
    
    # 寒潮事件
    coldwave_indices = np.random.choice(n_points, 15, replace=False)
    temperature[coldwave_indices] -= np.random.uniform(8, 12, 15)
    
    # 模拟降水数据（带干旱和洪涝）
    base_precip = 80 + 30 * np.sin(2 * np.pi * np.arange(n_points) / 12)  # 季节性降水
    precip_noise = np.random.gamma(2, 20, n_points)
    precipitation = base_precip + precip_noise
    
    # 干旱事件（连续低降水）
    drought_start = np.random.choice(n_points - 6, 5)
    for start in drought_start:
        precipitation[start:start+6] *= 0.2
    
    # 洪涝事件（极端降水）
    flood_indices = np.random.choice(n_points, 10, replace=False)
    precipitation[flood_indices] += np.random.uniform(200, 400, 10)
    
    # 模拟湿度数据
    humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(n_points) / 12) + np.random.normal(0, 5, n_points)
    humidity = np.clip(humidity, 0, 100)
    
    # 模拟风速数据
    wind_speed = 8 + 3 * np.sin(2 * np.pi * np.arange(n_points) / 6) + np.random.exponential(2, n_points)
    
    # 模拟气压数据
    pressure = 1013 + 10 * np.sin(2 * np.pi * np.arange(n_points) / 12) + np.random.normal(0, 8, n_points)
    
    # 创建DataFrame
    climate_data = pd.DataFrame({
        'time': dates,
        'temperature': temperature,
        'precipitation': precipitation,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure
    })
    
    logger.info(f"创建了包含 {len(climate_data)} 条记录的模拟气候数据")
    return climate_data

def analyze_trends_and_seasonality(analyzer, data):
    """
    分析趋势和季节性
    """
    logger.info("=== 开始趋势分析与周期性检测 ===")
    
    # 趋势分析
    trend_results = analyzer.analyze_trends(data)
    
    print("\n📈 趋势分析结果:")
    for var, result in trend_results.items():
        direction_emoji = {
            'increasing': '📈',
            'decreasing': '📉',
            'stable': '➡️'
        }.get(result.trend_direction, '❓')
        
        significance_emoji = '✅' if result.significance == 'significant' else '❌'
        
        print(f"  {direction_emoji} {var}:")
        print(f"    趋势方向: {result.trend_direction}")
        print(f"    年际变化: {result.annual_change:.4f}")
        print(f"    十年变化: {result.decadal_change:.4f}")
        print(f"    显著性: {result.significance} {significance_emoji}")
        print(f"    R²: {result.r_squared:.4f}")
        print()
    
    # 季节性分析
    seasonality_results = analyzer.analyze_seasonality(data)
    
    print("\n🔄 季节性分析结果:")
    for var, result in seasonality_results.items():
        seasonality_emoji = '✅' if result.has_seasonality else '❌'
        
        print(f"  🔄 {var}:")
        print(f"    季节性: {'是' if result.has_seasonality else '否'} {seasonality_emoji}")
        print(f"    季节性强度: {result.seasonal_strength:.4f}")
        if result.dominant_periods:
            print(f"    主导周期: {result.dominant_periods}")
        print()
    
    return trend_results, seasonality_results

def detect_anomalies_and_patterns(analyzer, data):
    """
    检测异常和识别模式
    """
    logger.info("=== 开始异常事件检测 ===")
    
    # 异常检测 - 使用多种方法
    methods = [AnomalyMethod.ISOLATION_FOREST, AnomalyMethod.STATISTICAL, AnomalyMethod.ONE_CLASS_SVM]
    
    all_anomaly_results = {}
    for method in methods:
        try:
            anomaly_results = analyzer.detect_anomalies(data, method=method)
            all_anomaly_results[method.value] = anomaly_results
        except Exception as e:
            logger.warning(f"异常检测方法 {method.value} 失败: {e}")
    
    # 显示异常检测结果
    print("\n🚨 异常事件检测结果:")
    for method_name, results in all_anomaly_results.items():
        print(f"\n  方法: {method_name}")
        for var, result in results.items():
            anomaly_emoji = '🚨' if result.total_anomalies > 0 else '✅'
            print(f"    {anomaly_emoji} {var}:")
            print(f"      异常点数量: {result.total_anomalies}")
            print(f"      异常率: {result.anomaly_rate:.2%}")
            if result.total_anomalies > 0:
                print(f"      异常值范围: {min(result.anomaly_values):.2f} ~ {max(result.anomaly_values):.2f}")
    
    # 模式识别
    logger.info("=== 开始模式识别 ===")
    
    pattern_results = analyzer.identify_patterns(data)
    
    print("\n🔍 模式识别结果:")
    for var, result in pattern_results.items():
        print(f"  🔍 {var}:")
        print(f"    模式类型: {result.pattern_type}")
        print(f"    模式强度: {result.pattern_strength:.4f}")
        print(f"    主要模式数量: {len(result.dominant_patterns)}")
        
        for i, pattern in enumerate(result.dominant_patterns[:3]):  # 显示前3个主要模式
            print(f"      模式 {i+1}: {pattern['characteristics']} ({pattern['percentage']:.1f}%)")
        print()
    
    return all_anomaly_results, pattern_results

def analyze_extreme_events(analyzer, data):
    """
    分析极端事件
    """
    logger.info("=== 开始极端事件分析 ===")
    
    extreme_results = analyzer.analyze_extreme_events(data)
    
    print("\n⚡ 极端事件分析结果:")
    for event_key, result in extreme_results.items():
        event_emoji = {
            'heatwave': '🔥',
            'coldwave': '🧊',
            'drought': '🏜️',
            'flood': '🌊'
        }
        
        event_type = result.event_type
        emoji = event_emoji.get(event_type, '⚡')
        
        print(f"  {emoji} {result.variable} - {event_type}:")
        print(f"    事件数量: {len(result.events)}")
        print(f"    年均频率: {result.frequency:.2f} 次/年")
        
        if result.events:
            intensities = [event['intensity'] for event in result.events]
            durations = [event['duration'] for event in result.events]
            
            print(f"    平均强度: {np.mean(intensities):.2f}")
            print(f"    最大强度: {np.max(intensities):.2f}")
            print(f"    平均持续时间: {np.mean(durations):.1f} 个时间步")
            print(f"    最长持续时间: {np.max(durations)} 个时间步")
            
            # 显示强度和持续时间趋势
            if result.intensity_trend.significance == 'significant':
                trend_emoji = '📈' if result.intensity_trend.trend_direction == 'increasing' else '📉'
                print(f"    强度趋势: {result.intensity_trend.trend_direction} {trend_emoji}")
            
            if result.duration_trend.significance == 'significant':
                trend_emoji = '📈' if result.duration_trend.trend_direction == 'increasing' else '📉'
                print(f"    持续时间趋势: {result.duration_trend.trend_direction} {trend_emoji}")
        
        print()
    
    return extreme_results

def generate_comprehensive_report(analyzer, data):
    """
    生成综合分析报告
    """
    logger.info("=== 生成综合分析报告 ===")
    
    report = analyzer.generate_comprehensive_report(data, "历史气候数据分析")
    
    print("\n📊 综合分析报告:")
    print(f"  📅 数据集: {report.dataset_name}")
    print(f"  📅 分析时间: {report.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  📅 数据时间范围: {report.time_period[0]} 至 {report.time_period[1]}")
    print(f"  📊 分析变量数: {report.summary['total_variables']}")
    print(f"  📈 显著趋势数: {report.summary['significant_trends']}")
    print(f"  🔄 季节性变量数: {report.summary['variables_with_seasonality']}")
    print(f"  🚨 总异常点数: {report.summary['total_anomalies']}")
    print(f"  ⚡ 总极端事件数: {report.summary['total_extreme_events']}")
    print(f"  🔗 强相关性数: {report.summary['strong_correlations']}")
    
    print("\n🔍 关键发现:")
    for finding in report.summary['key_findings']:
        print(f"    • {finding}")
    
    # 显示变量间相关性
    if report.correlations:
        print("\n🔗 变量间相关性 (|r| > 0.5):")
        strong_corrs = {k: v for k, v in report.correlations.items() if abs(v) > 0.5}
        for pair, corr in sorted(strong_corrs.items(), key=lambda x: abs(x[1]), reverse=True):
            corr_emoji = '🔴' if abs(corr) > 0.8 else '🟡' if abs(corr) > 0.6 else '🟢'
            print(f"    {corr_emoji} {pair}: {corr:.3f}")
    
    return report

def create_visualizations(data, trend_results, anomaly_results, extreme_results):
    """
    创建可视化图表
    """
    logger.info("创建可视化图表...")
    
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
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('历史气候模式分析可视化', fontsize=16, fontweight='bold')
    
    # 1. 温度趋势图
    ax1 = axes[0, 0]
    if 'temperature' in data.columns:
        ax1.plot(data['time'], data['temperature'], alpha=0.7, linewidth=1)
        
        # 添加趋势线
        if 'temperature' in trend_results:
            trend = trend_results['temperature']
            x_numeric = np.arange(len(data))
            trend_line = trend.slope * x_numeric + trend.intercept
            ax1.plot(data['time'], trend_line, 'r--', linewidth=2, 
                    # 确保数字格式化正确显示
                slope_str = f"{trend.slope:.4f}".replace('.', '.')
                label=f'趋势线 (斜率: {slope_str})')
        
        ax1.set_title('温度变化趋势')
        ax1.set_ylabel('温度 (°C)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 降水异常检测
    ax2 = axes[0, 1]
    if 'precipitation' in data.columns:
        ax2.plot(data['time'], data['precipitation'], alpha=0.7, linewidth=1, label='降水量')
        
        # 标记异常点
        if 'isolation_forest' in anomaly_results and 'precipitation' in anomaly_results['isolation_forest']:
            anomaly_result = anomaly_results['isolation_forest']['precipitation']
            if anomaly_result.anomaly_indices:
                anomaly_times = data['time'].iloc[anomaly_result.anomaly_indices]
                anomaly_values = [data['precipitation'].iloc[i] for i in anomaly_result.anomaly_indices]
                ax2.scatter(anomaly_times, anomaly_values, color='red', s=50, 
                           # 确保数字格式化正确显示
                anomaly_count = len(anomaly_result.anomaly_indices)
                label=f'异常点 ({anomaly_count}个)', zorder=5)
        
        ax2.set_title('降水量异常检测')
        ax2.set_ylabel('降水量 (mm)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 多变量相关性热图
    ax3 = axes[1, 0]
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        im = ax3.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(corr_matrix.columns)))
        ax3.set_yticks(range(len(corr_matrix.columns)))
        ax3.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax3.set_yticklabels(corr_matrix.columns)
        ax3.set_title('变量相关性矩阵')
        
        # 添加数值标签
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3, shrink=0.8)
    
    # 4. 极端事件统计
    ax4 = axes[1, 1]
    if extreme_results:
        event_counts = {}
        for key, result in extreme_results.items():
            event_type = result.event_type
            if event_type not in event_counts:
                event_counts[event_type] = 0
            event_counts[event_type] += len(result.events)
        
        if event_counts:
            events = list(event_counts.keys())
            counts = list(event_counts.values())
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
            
            bars = ax4.bar(events, counts, color=colors[:len(events)])
            ax4.set_title('极端事件统计')
            ax4.set_ylabel('事件数量')
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = "outputs/climate_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(f"{output_dir}/climate_pattern_analysis.png", dpi=300, bbox_inches='tight')
    logger.info(f"可视化图表已保存到 {output_dir}/climate_pattern_analysis.png")
    
    plt.show()

def save_results_to_files(report, output_dir="outputs/climate_analysis"):
    """
    保存分析结果到文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存综合报告
    report_file = f"{output_dir}/comprehensive_climate_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"历史气候模式分析综合报告\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"数据集: {report.dataset_name}\n")
        f.write(f"分析时间: {report.analysis_date}\n")
        f.write(f"数据时间范围: {report.time_period[0]} 至 {report.time_period[1]}\n\n")
        
        f.write(f"分析摘要:\n")
        f.write(f"  分析变量数: {report.summary['total_variables']}\n")
        f.write(f"  显著趋势数: {report.summary['significant_trends']}\n")
        f.write(f"  季节性变量数: {report.summary['variables_with_seasonality']}\n")
        f.write(f"  总异常点数: {report.summary['total_anomalies']}\n")
        f.write(f"  总极端事件数: {report.summary['total_extreme_events']}\n")
        f.write(f"  强相关性数: {report.summary['strong_correlations']}\n\n")
        
        f.write(f"关键发现:\n")
        for finding in report.summary['key_findings']:
            f.write(f"  • {finding}\n")
    
    # 保存详细的趋势分析结果
    trend_file = f"{output_dir}/trend_analysis_results.csv"
    trend_data = []
    for var, result in report.trend_results.items():
        trend_data.append({
            'variable': var,
            'trend_direction': result.trend_direction,
            'slope': result.slope,
            'r_squared': result.r_squared,
            'p_value': result.p_value,
            'significance': result.significance,
            'annual_change': result.annual_change,
            'decadal_change': result.decadal_change
        })
    
    pd.DataFrame(trend_data).to_csv(trend_file, index=False, encoding='utf-8-sig')
    
    # 保存异常检测结果
    anomaly_file = f"{output_dir}/anomaly_detection_results.csv"
    anomaly_data = []
    for var, result in report.anomaly_results.items():
        anomaly_data.append({
            'variable': var,
            'method': result.method.value,
            'total_anomalies': result.total_anomalies,
            'anomaly_rate': result.anomaly_rate,
            'threshold': result.threshold
        })
    
    pd.DataFrame(anomaly_data).to_csv(anomaly_file, index=False, encoding='utf-8-sig')
    
    logger.info(f"分析结果已保存到 {output_dir} 目录")

def main():
    """
    主函数：执行完整的历史气候模式分析
    """
    print("🌍 历史气候模式识别与分析系统")
    print("=" * 50)
    print("利用AI技术辅助研究人员识别历史气候模式、趋势和异常事件")
    print()
    
    try:
        # 1. 加载数据
        data = load_climate_data()
        print(f"✅ 成功加载 {len(data)} 条气候数据记录")
        print(f"📊 数据变量: {list(data.columns)}")
        print(f"📅 时间范围: {data['time'].min()} 至 {data['time'].max()}")
        print()
        
        # 2. 初始化分析器
        analyzer = HistoricalClimateAnalyzer()
        
        # 3. 趋势分析与周期性检测
        trend_results, seasonality_results = analyze_trends_and_seasonality(analyzer, data)
        
        # 4. 异常事件检测和模式识别
        anomaly_results, pattern_results = detect_anomalies_and_patterns(analyzer, data)
        
        # 5. 极端事件分析
        extreme_results = analyze_extreme_events(analyzer, data)
        
        # 6. 生成综合报告
        report = generate_comprehensive_report(analyzer, data)
        
        # 7. 创建可视化
        create_visualizations(data, trend_results, anomaly_results, extreme_results)
        
        # 8. 保存结果
        save_results_to_files(report)
        
        print("\n🎉 历史气候模式分析完成！")
        print("📁 结果文件已保存到 outputs/climate_analysis/ 目录")
        print("📊 可视化图表已生成并显示")
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()