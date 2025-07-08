#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化融合模型结果可视化
创建散点图展示准确率 vs 召回率的性能对比
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
from datetime import datetime

def find_latest_results():
    """查找最新的结果文件"""
    output_dir = Path('outputs/recall_comparison')
    csv_files = list(output_dir.glob('traditional_vs_ai_methods_*.csv'))
    
    if not csv_files:
        raise FileNotFoundError("未找到结果文件")
    
    # 按修改时间排序，获取最新文件
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    return latest_file

def create_precision_recall_scatter():
    """创建准确率vs召回率散点图"""
    print("正在创建优化融合模型的散点图可视化...")
    
    # 读取最新结果
    results_file = find_latest_results()
    print(f"读取结果文件: {results_file}")
    
    df = pd.read_csv(results_file)
    
    # 设置图形样式
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })
    
    # 专业配色方案
    colors = {
        'traditional': '#E74C3C',      # 红色
        'ai': '#3498DB',               # 蓝色
        'fusion': '#9B59B6',           # 紫色
        'autoencoder': '#F39C12',      # 橙色
        'sigma': '#27AE60',            # 绿色
        'grid': '#BDC3C7',             # 浅灰色
        'text': '#2C3E50'              # 深灰色
    }
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.suptitle('Climate Anomaly Detection: Precision vs Recall Performance\n(Optimized Fusion Model Results)', 
                 fontsize=18, fontweight='bold', color=colors['text'], y=0.95)
    
    # 分离不同类型的方法
    traditional_methods = df[df['method_type'] == 'Traditional']
    ai_methods = df[df['method_type'] == 'AI/ML']
    
    # 进一步细分AI方法
    fusion_methods = ai_methods[ai_methods['method'].str.contains('Fusion')]
    autoencoder_methods = ai_methods[ai_methods['method'].str.contains('AutoEncoder')]
    sigma_methods = ai_methods[ai_methods['method'].str.contains('3-Sigma')]
    other_ai_methods = ai_methods[~ai_methods['method'].str.contains('Fusion|AutoEncoder|3-Sigma')]
    
    # 绘制传统方法
    if len(traditional_methods) > 0:
        scatter1 = ax.scatter(traditional_methods['precision'], traditional_methods['recall'], 
                             c=colors['traditional'], s=120, alpha=0.8, 
                             label='Traditional Methods', marker='o', 
                             edgecolors='white', linewidth=2)
    
    # 绘制AutoEncoder方法
    if len(autoencoder_methods) > 0:
        scatter2 = ax.scatter(autoencoder_methods['precision'], autoencoder_methods['recall'], 
                             c=colors['autoencoder'], s=140, alpha=0.8, 
                             label='AutoEncoder', marker='s', 
                             edgecolors='white', linewidth=2)
    
    # 绘制3-Sigma方法
    if len(sigma_methods) > 0:
        scatter3 = ax.scatter(sigma_methods['precision'], sigma_methods['recall'], 
                             c=colors['sigma'], s=140, alpha=0.8, 
                             label='3-Sigma Method', marker='^', 
                             edgecolors='white', linewidth=2)
    
    # 绘制融合方法（重点突出）
    if len(fusion_methods) > 0:
        scatter4 = ax.scatter(fusion_methods['precision'], fusion_methods['recall'], 
                             c=colors['fusion'], s=250, alpha=0.9, 
                             label='Optimized Fusion Methods', marker='D', 
                             edgecolors='white', linewidth=3)
    
    # 添加方法标签
    for _, row in df.iterrows():
        # 为融合方法使用特殊样式
        if 'Fusion' in row['method']:
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor=colors['fusion'], alpha=0.2, edgecolor=colors['fusion'])
            fontweight = 'bold'
            fontsize = 11
        else:
            bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            fontweight = 'normal'
            fontsize = 10
        
        # 简化方法名称以便显示
        method_name = row['method']
        if 'Fusion_3Sigma_AE' in method_name:
            method_name = 'Optimized Fusion'
        elif 'Ensemble_Fusion' in method_name:
            method_name = 'Ensemble Fusion'
        elif 'AutoEncoder' in method_name:
            method_name = 'AutoEncoder'
        elif '3-Sigma' in method_name and 'AI/ML' in str(row['method_type']):
            method_name = '3-Sigma (AI)'
        
        ax.annotate(method_name, 
                   (row['precision'], row['recall']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=fontsize, ha='left', va='bottom', fontweight=fontweight,
                   bbox=bbox_props)
    
    # 设置坐标轴
    ax.set_xlabel('Precision (精确率)', color=colors['text'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall (召回率)', color=colors['text'], fontsize=14, fontweight='bold')
    ax.set_title('Precision vs Recall: Optimized Performance Comparison', 
                fontweight='bold', color=colors['text'], pad=20, fontsize=16)
    
    # 设置坐标轴范围
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # 添加网格
    ax.grid(True, alpha=0.3, color=colors['grid'], linestyle='--')
    ax.set_facecolor('#FAFAFA')
    
    # 添加性能区域标识
    # 高性能区域 (召回率 > 0.75, 精确率 > 0.75)
    ax.axhline(y=0.75, color='green', linestyle=':', alpha=0.6, linewidth=2)
    ax.axvline(x=0.75, color='green', linestyle=':', alpha=0.6, linewidth=2)
    ax.text(0.77, 0.77, 'High Performance\nZone\n(Recall > 0.75)', fontsize=10, color='green', 
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    
    # 目标召回率线
    ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax.text(0.02, 0.76, 'Target Recall = 0.75', fontsize=10, color='red', fontweight='bold')
    
    # 添加F1等值线
    precision_range = np.linspace(0.01, 1, 100)
    for f1_val in [0.5, 0.7, 0.8]:
        recall_line = (f1_val * precision_range) / (2 * precision_range - f1_val)
        recall_line = np.clip(recall_line, 0, 1)
        valid_mask = (recall_line >= 0) & (recall_line <= 1) & (precision_range >= f1_val/2)
        if np.any(valid_mask):
            ax.plot(precision_range[valid_mask], recall_line[valid_mask], 
                   '--', alpha=0.4, color='gray', linewidth=1)
            # 添加F1标签
            if f1_val == 0.7:
                ax.text(0.85, 0.58, f'F1={f1_val}', fontsize=9, color='gray', alpha=0.7)
            elif f1_val == 0.8:
                ax.text(0.9, 0.72, f'F1={f1_val}', fontsize=9, color='gray', alpha=0.7)
    
    # 添加图例
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # 添加性能统计信息
    stats_text = f"""Performance Summary:
• Best Recall: {df['recall'].max():.3f} ({df.loc[df['recall'].idxmax(), 'method']})
• Best F1: {df['f1_score'].max():.3f} ({df.loc[df['f1_score'].idxmax(), 'method']})
• Best Precision: {df['precision'].max():.3f} ({df.loc[df['precision'].idxmax(), 'method']})"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('outputs/recall_comparison')
    plot_path = output_dir / f'optimized_fusion_precision_recall_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', 
               edgecolor='none', format='png')
    print(f"优化融合模型散点图已保存至: {plot_path}")
    
    # 显示图表
    plt.show()
    
    # 打印优化结果总结
    print("\n" + "="*80)
    print("优化融合模型性能总结")
    print("="*80)
    
    fusion_results = df[df['method'].str.contains('Fusion')]
    if len(fusion_results) > 0:
        print("\n融合模型性能:")
        for _, row in fusion_results.iterrows():
            print(f"• {row['method']}:")
            print(f"  - 召回率: {row['recall']:.3f} {'✓' if row['recall'] >= 0.75 else '✗'} (目标: ≥0.75)")
            print(f"  - 精确率: {row['precision']:.3f}")
            print(f"  - F1分数: {row['f1_score']:.3f}")
            print(f"  - 准确率: {row['accuracy']:.3f}")
            print()
    
    # 对比传统方法和AI方法
    traditional_avg_recall = df[df['method_type'] == 'Traditional']['recall'].mean()
    ai_avg_recall = df[df['method_type'] == 'AI/ML']['recall'].mean()
    improvement = ai_avg_recall - traditional_avg_recall
    
    print(f"性能提升分析:")
    print(f"• 传统方法平均召回率: {traditional_avg_recall:.3f}")
    print(f"• AI/ML方法平均召回率: {ai_avg_recall:.3f}")
    print(f"• 召回率提升: +{improvement:.3f} ({improvement/traditional_avg_recall*100:+.1f}%)")
    
    return plot_path

if __name__ == "__main__":
    try:
        plot_path = create_precision_recall_scatter()
        print(f"\n可视化完成！图表已保存至: {plot_path}")
    except Exception as e:
        print(f"错误: {e}")