#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文字体显示测试脚本

测试修复后的可视化代码是否能正确显示中文字符。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.utils.font_config import configure_chinese_fonts, test_chinese_font_display, print_font_info
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_test_visualization():
    """
    创建测试可视化图表，验证中文字符显示
    """
    print("🎨 创建中文字体测试图表...")
    
    # 配置中文字体
    configure_chinese_fonts()
    
    # 创建测试数据
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    temperature = 15 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12) + np.random.normal(0, 2, len(dates))
    precipitation = 50 + 30 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12 + np.pi/2) + np.random.normal(0, 10, len(dates))
    humidity = 60 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12 + np.pi/4) + np.random.normal(0, 5, len(dates))
    
    # 创建综合测试图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('中文字体显示测试 - 气候数据可视化', fontsize=16, fontweight='bold')
    
    # 1. 温度趋势图
    ax1 = axes[0, 0]
    ax1.plot(dates, temperature, 'r-', linewidth=2, label='月平均温度')
    ax1.set_title('温度变化趋势分析')
    ax1.set_xlabel('时间')
    ax1.set_ylabel('温度 (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加中文注释
    ax1.annotate('夏季高温期', xy=(dates[6], temperature[6]), xytext=(dates[8], temperature[6]+5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    # 2. 降水量柱状图
    ax2 = axes[0, 1]
    bars = ax2.bar(dates[::3], precipitation[::3], width=60, alpha=0.7, color='blue', label='季度降水量')
    ax2.set_title('降水量分布统计')
    ax2.set_xlabel('时间')
    ax2.set_ylabel('降水量 (mm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars[::4]):  # 每年标注一次
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.0f}mm', ha='center', va='bottom')
    
    # 3. 湿度散点图
    ax3 = axes[1, 0]
    colors = ['red' if h > 70 else 'blue' if h < 50 else 'green' for h in humidity]
    scatter = ax3.scatter(dates, humidity, c=colors, alpha=0.6, s=30)
    ax3.set_title('湿度变化散点图')
    ax3.set_xlabel('时间')
    ax3.set_ylabel('相对湿度 (%)')
    ax3.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='高湿度 (>70%)'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='适中湿度 (50-70%)'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='低湿度 (<50%)')]
    ax3.legend(handles=legend_elements)
    
    # 4. 综合对比图
    ax4 = axes[1, 1]
    
    # 标准化数据用于对比
    temp_norm = (temperature - temperature.mean()) / temperature.std()
    precip_norm = (precipitation - precipitation.mean()) / precipitation.std()
    humid_norm = (humidity - humidity.mean()) / humidity.std()
    
    ax4.plot(dates, temp_norm, 'r-', linewidth=2, label='温度（标准化）')
    ax4.plot(dates, precip_norm, 'b-', linewidth=2, label='降水量（标准化）')
    ax4.plot(dates, humid_norm, 'g-', linewidth=2, label='湿度（标准化）')
    
    ax4.set_title('多变量标准化对比')
    ax4.set_xlabel('时间')
    ax4.set_ylabel('标准化数值')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 添加文本框说明
    textstr = '\n'.join([
        '数据说明：',
        '• 温度：月平均气温',
        '• 降水：月累计降水量', 
        '• 湿度：月平均相对湿度',
        '• 时间跨度：2020-2023年'
    ])
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('outputs/font_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'chinese_font_test.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"✅ 测试图表已保存到: {output_file}")
    
    # 显示图表（如果在交互环境中）
    try:
        plt.show()
    except:
        pass
    
    plt.close()
    
    return str(output_file)

def create_climate_analysis_test():
    """
    创建气候分析测试图表，模拟实际使用场景
    """
    print("🌡️ 创建气候分析测试图表...")
    
    # 配置中文字体
    configure_chinese_fonts()
    
    # 创建模拟的气候分析数据
    np.random.seed(42)
    time_points = pd.date_range('1990-01-01', '2023-12-31', freq='Y')
    
    # 模拟全球变暖趋势
    base_temp = 14.0
    warming_trend = np.linspace(0, 1.5, len(time_points))  # 1.5°C升温
    annual_cycle = 0.5 * np.sin(2 * np.pi * np.arange(len(time_points)) / 10)  # 10年周期
    noise = np.random.normal(0, 0.3, len(time_points))
    temperature = base_temp + warming_trend + annual_cycle + noise
    
    # 模拟极端事件
    extreme_years = [1998, 2003, 2010, 2016, 2020, 2023]
    extreme_indices = [i for i, year in enumerate(time_points.year) if year in extreme_years]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('全球气候变化分析报告 (1990-2023)', fontsize=16, fontweight='bold')
    
    # 1. 温度趋势分析
    ax1 = axes[0, 0]
    ax1.plot(time_points, temperature, 'b-', linewidth=1.5, alpha=0.7, label='年平均温度')
    
    # 添加趋势线
    z = np.polyfit(range(len(time_points)), temperature, 1)
    trend_line = np.poly1d(z)
    ax1.plot(time_points, trend_line(range(len(time_points))), 'r--', linewidth=2, 
             label=f'线性趋势 (+{z[0]:.3f}°C/年)')
    
    # 标记极端年份
    for idx in extreme_indices:
        ax1.scatter(time_points[idx], temperature[idx], color='red', s=80, zorder=5)
        ax1.annotate(f'{time_points[idx].year}', 
                    xy=(time_points[idx], temperature[idx]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='red')
    
    ax1.set_title('全球平均温度变化趋势')
    ax1.set_xlabel('年份')
    ax1.set_ylabel('温度 (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 温度异常分析
    ax2 = axes[0, 1]
    temp_anomaly = temperature - temperature.mean()
    colors = ['red' if x > 0 else 'blue' for x in temp_anomaly]
    
    bars = ax2.bar(time_points, temp_anomaly, color=colors, alpha=0.7, width=300)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('温度异常值分析')
    ax2.set_xlabel('年份')
    ax2.set_ylabel('温度异常 (°C)')
    ax2.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='正异常（偏暖）'),
                      Patch(facecolor='blue', alpha=0.7, label='负异常（偏冷）')]
    ax2.legend(handles=legend_elements)
    
    # 3. 年代际对比
    ax3 = axes[1, 0]
    
    # 按年代分组
    decades = ['1990s', '2000s', '2010s', '2020s']
    decade_temps = []
    
    for i, decade in enumerate(decades):
        start_year = 1990 + i * 10
        end_year = min(1999 + i * 10, 2023)
        decade_mask = (time_points.year >= start_year) & (time_points.year <= end_year)
        decade_temp = temperature[decade_mask].mean()
        decade_temps.append(decade_temp)
    
    bars = ax3.bar(decades, decade_temps, color=['lightblue', 'skyblue', 'orange', 'red'], alpha=0.8)
    ax3.set_title('年代际平均温度对比')
    ax3.set_xlabel('年代')
    ax3.set_ylabel('平均温度 (°C)')
    
    # 添加数值标签
    for bar, temp in zip(bars, decade_temps):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{temp:.2f}°C', ha='center', va='bottom', fontweight='bold')
    
    # 4. 关键统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')  # 隐藏坐标轴
    
    # 计算统计信息
    total_warming = temperature[-1] - temperature[0]
    avg_warming_rate = z[0]
    hottest_year = time_points[np.argmax(temperature)].year
    coldest_year = time_points[np.argmin(temperature)].year
    extreme_count = len(extreme_indices)
    
    # 创建统计信息文本
    stats_text = f"""
    📊 关键统计信息
    
    🌡️ 总升温幅度: {total_warming:.2f}°C
    📈 平均升温速率: {avg_warming_rate:.3f}°C/年
    🔥 最热年份: {hottest_year}年
    🧊 最冷年份: {coldest_year}年
    ⚠️ 极端事件: {extreme_count}次
    
    📅 分析时间段: 1990-2023年
    📏 数据点数: {len(time_points)}个
    
    🔍 主要发现:
    • 明显的全球变暖趋势
    • 2010年代后升温加速
    • 极端高温事件增多
    • 年际变化显著
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('outputs/font_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'climate_analysis_chinese_test.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"✅ 气候分析测试图表已保存到: {output_file}")
    
    try:
        plt.show()
    except:
        pass
    
    plt.close()
    
    return str(output_file)

def main():
    """
    主测试函数
    """
    print("🔤 中文字体显示修复测试")
    print("=" * 40)
    
    # 1. 打印字体配置信息
    print("\n📋 当前字体配置:")
    print_font_info()
    
    # 2. 基础字体测试
    print("\n🧪 执行基础字体测试...")
    if test_chinese_font_display():
        print("✅ 基础字体测试通过")
    else:
        print("❌ 基础字体测试失败")
        return
    
    # 3. 创建测试可视化
    print("\n🎨 创建测试可视化图表...")
    try:
        test_file = create_test_visualization()
        print(f"✅ 基础测试图表创建成功: {test_file}")
    except Exception as e:
        print(f"❌ 基础测试图表创建失败: {e}")
        return
    
    # 4. 创建气候分析测试
    print("\n🌡️ 创建气候分析测试图表...")
    try:
        climate_file = create_climate_analysis_test()
        print(f"✅ 气候分析测试图表创建成功: {climate_file}")
    except Exception as e:
        print(f"❌ 气候分析测试图表创建失败: {e}")
        return
    
    print("\n🎉 所有测试完成！")
    print("\n📁 测试结果文件位置:")
    print(f"  • 基础测试: {test_file}")
    print(f"  • 气候分析: {climate_file}")
    print("\n💡 请检查生成的图片文件，确认中文字符显示是否正常。")

if __name__ == '__main__':
    main()