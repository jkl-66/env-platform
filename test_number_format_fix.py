#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字格式化修复测试脚本

测试修复后的可视化代码是否能正确显示数字。
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

from src.utils.font_config import configure_chinese_fonts, format_number, format_percentage
from src.visualization.charts import ChartGenerator, ChartConfig

def create_number_format_test():
    """
    创建数字格式化测试图表
    """
    print("🔢 创建数字格式化测试图表...")
    
    # 配置中文字体
    configure_chinese_fonts()
    
    # 创建测试数据
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
    temperature = 15.1234 + 10.5678 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12) + np.random.normal(0, 2.3456, len(dates))
    precipitation = 50.9876 + 30.1234 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12 + np.pi/2) + np.random.normal(0, 10.5432, len(dates))
    
    # 创建DataFrame
    data = pd.DataFrame({
        'time': dates,
        'temperature': temperature,
        'precipitation': precipitation
    })
    
    # 创建测试图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('数字格式化修复测试 - 确保数字正确显示', fontsize=16, fontweight='bold')
    
    # 1. 时间序列图 - 测试趋势线斜率显示
    ax1 = axes[0, 0]
    ax1.plot(data['time'], data['temperature'], 'b-', linewidth=2, label='温度数据')
    
    # 计算趋势线
    x_numeric = np.arange(len(data))
    z = np.polyfit(x_numeric, data['temperature'], 1)
    trend_line = z[0] * x_numeric + z[1]
    
    # 使用格式化函数显示斜率
    slope_str = format_number(z[0], 4)
    ax1.plot(data['time'], trend_line, 'r--', linewidth=2, 
             label=f'趋势线 (斜率: {slope_str}°C/月)')
    
    ax1.set_title('温度趋势分析 - 斜率数字测试')
    ax1.set_xlabel('时间')
    ax1.set_ylabel('温度 (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 散点图 - 测试数值标签
    ax2 = axes[0, 1]
    scatter = ax2.scatter(data['temperature'], data['precipitation'], 
                         c=data.index, cmap='viridis', alpha=0.6)
    
    # 添加一些数值标签
    for i in range(0, len(data), 6):  # 每6个点标注一次
        temp_str = format_number(data['temperature'].iloc[i], 1)
        prec_str = format_number(data['precipitation'].iloc[i], 1)
        ax2.annotate(f'({temp_str}, {prec_str})', 
                    (data['temperature'].iloc[i], data['precipitation'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_title('温度-降水关系 - 坐标数字测试')
    ax2.set_xlabel('温度 (°C)')
    ax2.set_ylabel('降水量 (mm)')
    plt.colorbar(scatter, ax=ax2, label='时间序列')
    
    # 3. 柱状图 - 测试百分比显示
    ax3 = axes[1, 0]
    
    # 计算季节统计
    data['season'] = data['time'].dt.month % 12 // 3
    season_names = ['冬季', '春季', '夏季', '秋季']
    season_stats = data.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()
    season_stats['season_name'] = [season_names[i] for i in season_stats['season']]
    
    bars = ax3.bar(season_stats['season_name'], season_stats['mean'], 
                   yerr=season_stats['std'], capsize=5, alpha=0.7)
    
    # 添加数值标签
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, season_stats['mean'], season_stats['std'])):
        mean_str = format_number(mean_val, 2)
        std_str = format_number(std_val, 2)
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.5,
                f'{mean_str}±{std_str}°C', ha='center', va='bottom', fontsize=10)
    
    ax3.set_title('季节温度统计 - 均值±标准差数字测试')
    ax3.set_ylabel('温度 (°C)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 相关性热图 - 测试相关系数显示
    ax4 = axes[1, 1]
    
    # 创建更多变量用于相关性分析
    corr_data = pd.DataFrame({
        '温度': data['temperature'],
        '降水': data['precipitation'],
        '湿度': 60 + 20 * np.sin(np.arange(len(data)) * 2 * np.pi / 12) + np.random.normal(0, 5, len(data)),
        '风速': 5 + 3 * np.cos(np.arange(len(data)) * 2 * np.pi / 12) + np.random.normal(0, 1, len(data))
    })
    
    corr_matrix = corr_data.corr()
    
    # 创建热图
    im = ax4.imshow(corr_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # 设置标签
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_yticklabels(corr_matrix.columns)
    
    # 添加数值标签 - 使用格式化函数
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            corr_str = format_number(corr_val, 3)
            color = 'white' if abs(corr_val) > 0.5 else 'black'
            ax4.text(j, i, corr_str, ha="center", va="center", 
                    color=color, fontweight='bold')
    
    ax4.set_title('变量相关性矩阵 - 相关系数数字测试')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('相关系数')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('outputs/number_format_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'number_format_test.png', dpi=300, bbox_inches='tight')
    print(f"✅ 数字格式化测试图表已保存到: {output_dir / 'number_format_test.png'}")
    
    plt.close()

def test_format_functions():
    """
    测试格式化函数
    """
    print("\n🧪 测试数字格式化函数...")
    
    test_numbers = [1.23456, 0.00123, 123.456789, 0, -45.67890, 100.0]
    test_percentages = [12.3456, 0.123, 99.9876, 0.0, 100.0]
    
    print("数字格式化测试:")
    for num in test_numbers:
        formatted = format_number(num)
        formatted_2 = format_number(num, 2)
        print(f"  {num} -> 自动: '{formatted}', 2位小数: '{formatted_2}'")
    
    print("\n百分比格式化测试:")
    for pct in test_percentages:
        formatted = format_percentage(pct)
        formatted_3 = format_percentage(pct, 3)
        print(f"  {pct} -> 1位小数: '{formatted}', 3位小数: '{formatted_3}'")

def create_chart_generator_test():
    """
    测试ChartGenerator的数字格式化
    """
    print("\n📊 测试ChartGenerator数字格式化...")
    
    # 创建测试数据
    data = pd.DataFrame({
        'var1': np.random.normal(10.12345, 2.6789, 50),
        'var2': np.random.normal(20.98765, 3.4321, 50),
        'var3': np.random.normal(15.55555, 1.7777, 50)
    })
    
    # 创建图表生成器
    chart_gen = ChartGenerator()
    
    # 创建相关性矩阵
    config = ChartConfig(
        title="相关性矩阵 - 数字格式化测试",
        chart_type="correlation",
        width=800,
        height=600
    )
    
    try:
        chart_path = chart_gen.create_correlation_matrix(data, config)
        print(f"✅ 相关性矩阵图表已生成: {chart_path}")
    except Exception as e:
        print(f"❌ 生成相关性矩阵失败: {e}")

def main():
    """
    主函数
    """
    print("🚀 开始数字格式化修复测试...")
    
    try:
        # 测试格式化函数
        test_format_functions()
        
        # 创建测试图表
        create_number_format_test()
        
        # 测试图表生成器
        create_chart_generator_test()
        
        print("\n🎉 数字格式化修复测试完成！")
        print("请检查生成的图表，确认数字是否正确显示。")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()