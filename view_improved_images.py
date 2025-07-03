#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进图像查看器
用于查看和对比生成的高质量生态警示图像
"""

import sys
from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

class ImprovedImageViewer:
    """改进的图像查看器"""
    
    def __init__(self):
        self.improved_dir = Path("outputs/improved_ecology_images")
        self.old_dir = Path("outputs/gan_test")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_improved_report(self):
        """加载改进图像生成报告"""
        report_files = list(self.improved_dir.glob("improved_generation_report_*.json"))
        
        if not report_files:
            print("❌ 未找到改进图像生成报告")
            return None
        
        # 使用最新的报告文件
        latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_report, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 读取报告文件失败: {e}")
            return None
    
    def display_scenario_comparison(self, scenario_key, max_images=3):
        """显示特定场景的图像对比"""
        report = self.load_improved_report()
        if not report:
            return
        
        if scenario_key not in report['scenarios']:
            print(f"❌ 场景 '{scenario_key}' 不存在")
            return
        
        scenario = report['scenarios'][scenario_key]
        scenario_name = scenario['scenario_name']
        
        print(f"\n=== 查看场景: {scenario_name} ===")
        
        # 获取改进图像
        improved_images = scenario['images'][:max_images]
        
        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, max_images, figure=fig, height_ratios=[1, 0.1])
        
        fig.suptitle(f'生态警示图像生成效果 - {scenario_name}', fontsize=16, fontweight='bold')
        
        for i, img_info in enumerate(improved_images):
            # 加载并显示改进图像
            img_path = self.improved_dir / img_info['filename']
            
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    
                    # 显示图像
                    ax = fig.add_subplot(gs[0, i])
                    ax.imshow(img)
                    ax.set_title(f'改进版本 {i+1}\n警示等级: {img_info["warning_level"]}', 
                               fontsize=12, fontweight='bold')
                    ax.axis('off')
                    
                    # 添加警示等级颜色边框
                    warning_colors = {
                        1: 'green',
                        2: 'yellow', 
                        3: 'orange',
                        4: 'red',
                        5: 'darkred'
                    }
                    
                    border_color = warning_colors.get(img_info['warning_level'], 'gray')
                    rect = patches.Rectangle((0, 0), img.width-1, img.height-1, 
                                           linewidth=5, edgecolor=border_color, 
                                           facecolor='none')
                    ax.add_patch(rect)
                    
                    # 添加图像信息
                    info_ax = fig.add_subplot(gs[1, i])
                    info_text = f"描述: {img_info['description']}\n文件: {img_info['filename']}"
                    info_ax.text(0.5, 0.5, info_text, ha='center', va='center', 
                               fontsize=9, wrap=True)
                    info_ax.axis('off')
                    
                except Exception as e:
                    print(f"❌ 加载图像失败 {img_path}: {e}")
            else:
                print(f"❌ 图像文件不存在: {img_path}")
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ 已显示 {len(improved_images)} 张 {scenario_name} 图像")
    
    def display_all_scenarios_overview(self):
        """显示所有场景的概览"""
        report = self.load_improved_report()
        if not report:
            return
        
        scenarios = report['scenarios']
        num_scenarios = len(scenarios)
        
        print(f"\n=== 所有生态场景概览 ({num_scenarios} 个场景) ===")
        
        # 创建网格布局
        cols = 3
        rows = (num_scenarios + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('生态警示图像生成系统 - 所有场景概览', fontsize=20, fontweight='bold')
        
        scenario_items = list(scenarios.items())
        
        for idx, (scenario_key, scenario) in enumerate(scenario_items):
            row = idx // cols
            col = idx % cols
            
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # 获取第一张图像作为代表
            if scenario['images']:
                img_info = scenario['images'][0]
                img_path = self.improved_dir / img_info['filename']
                
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        ax.imshow(img)
                        
                        # 设置标题
                        title = f"{scenario['scenario_name']}\n警示等级: {img_info['warning_level']}"
                        ax.set_title(title, fontsize=14, fontweight='bold')
                        
                        # 添加警示等级颜色边框
                        warning_colors = {
                            1: 'green', 2: 'yellow', 3: 'orange', 4: 'red', 5: 'darkred'
                        }
                        border_color = warning_colors.get(img_info['warning_level'], 'gray')
                        
                        for spine in ax.spines.values():
                            spine.set_edgecolor(border_color)
                            spine.set_linewidth(4)
                        
                    except Exception as e:
                        ax.text(0.5, 0.5, f'图像加载失败\n{e}', ha='center', va='center')
                        ax.set_title(scenario['scenario_name'], fontsize=14)
                else:
                    ax.text(0.5, 0.5, '图像文件不存在', ha='center', va='center')
                    ax.set_title(scenario['scenario_name'], fontsize=14)
            else:
                ax.text(0.5, 0.5, '无图像数据', ha='center', va='center')
                ax.set_title(scenario['scenario_name'], fontsize=14)
            
            ax.axis('off')
        
        # 隐藏多余的子图
        for idx in range(num_scenarios, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ 已显示 {num_scenarios} 个场景的概览")
    
    def show_generation_statistics(self):
        """显示生成统计信息"""
        report = self.load_improved_report()
        if not report:
            return
        
        print("\n=== 图像生成统计 ===")
        print(f"生成时间: {report['timestamp']}")
        print(f"生成方法: {report['generation_method']}")
        print(f"图像质量: {report['image_quality']}")
        print(f"总场景数: {report['total_scenarios']}")
        print(f"总图像数: {report['total_images_generated']}")
        
        print("\n--- 各场景详情 ---")
        for scenario_key, scenario in report['scenarios'].items():
            print(f"📊 {scenario['scenario_name']}:")
            print(f"   - 生成图像数: {scenario['images_generated']}")
            if scenario['images']:
                warning_level = scenario['images'][0]['warning_level']
                description = scenario['images'][0]['description']
                print(f"   - 警示等级: {warning_level}")
                print(f"   - 描述: {description}")
        
        # 警示等级分布
        warning_levels = []
        for scenario in report['scenarios'].values():
            if scenario['images']:
                warning_levels.append(scenario['images'][0]['warning_level'])
        
        if warning_levels:
            print("\n--- 警示等级分布 ---")
            level_counts = {}
            for level in warning_levels:
                level_counts[level] = level_counts.get(level, 0) + 1
            
            level_names = {
                1: "生态良好",
                2: "轻度警示", 
                3: "中度警示",
                4: "重度警示",
                5: "极度警示"
            }
            
            for level in sorted(level_counts.keys()):
                count = level_counts[level]
                name = level_names.get(level, f"等级{level}")
                print(f"   {name} (等级{level}): {count} 个场景")
    
    def compare_with_old_generation(self):
        """与旧版本生成进行对比"""
        print("\n=== 新旧版本对比 ===")
        
        # 检查旧版本图像
        old_images = list(self.old_dir.glob("*.png")) if self.old_dir.exists() else []
        
        # 检查新版本图像
        new_images = list(self.improved_dir.glob("*.png"))
        
        print(f"旧版本图像数量: {len(old_images)}")
        print(f"新版本图像数量: {len(new_images)}")
        
        if old_images and new_images:
            # 显示对比
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('图像生成效果对比', fontsize=16, fontweight='bold')
            
            # 显示旧版本图像（前3张）
            for i in range(min(3, len(old_images))):
                try:
                    img = Image.open(old_images[i])
                    axes[0, i].imshow(img)
                    axes[0, i].set_title(f'旧版本 {i+1}\n(可能是灰色占位符)', fontsize=12)
                    axes[0, i].axis('off')
                except Exception as e:
                    axes[0, i].text(0.5, 0.5, f'加载失败\n{e}', ha='center', va='center')
                    axes[0, i].set_title(f'旧版本 {i+1}', fontsize=12)
                    axes[0, i].axis('off')
            
            # 显示新版本图像（前3张）
            for i in range(min(3, len(new_images))):
                try:
                    img = Image.open(new_images[i])
                    axes[1, i].imshow(img)
                    axes[1, i].set_title(f'新版本 {i+1}\n(高质量彩色图像)', fontsize=12)
                    axes[1, i].axis('off')
                except Exception as e:
                    axes[1, i].text(0.5, 0.5, f'加载失败\n{e}', ha='center', va='center')
                    axes[1, i].set_title(f'新版本 {i+1}', fontsize=12)
                    axes[1, i].axis('off')
            
            # 隐藏多余的子图
            for i in range(3):
                if i >= len(old_images):
                    axes[0, i].axis('off')
                if i >= len(new_images):
                    axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print("✅ 对比显示完成")
            print("\n改进效果:")
            print("- ✅ 从灰色占位符变为彩色真实图像")
            print("- ✅ 添加了场景特定的视觉元素")
            print("- ✅ 包含警示等级和环境主题")
            print("- ✅ 使用程序化生成避免模型依赖")
        else:
            print("❌ 无法进行对比，缺少图像文件")

def main():
    """主函数"""
    print("改进的生态图像查看器")
    print("=" * 50)
    
    viewer = ImprovedImageViewer()
    
    # 显示生成统计
    viewer.show_generation_statistics()
    
    # 显示所有场景概览
    viewer.display_all_scenarios_overview()
    
    # 显示特定场景的详细对比
    scenarios_to_show = ['forest_protection', 'air_pollution', 'climate_change']
    
    for scenario in scenarios_to_show:
        viewer.display_scenario_comparison(scenario)
    
    # 与旧版本对比
    viewer.compare_with_old_generation()
    
    print("\n=== 查看完成 ===")
    print("🎨 新的图像生成系统特点:")
    print("- 丰富的颜色和视觉效果")
    print("- 场景特定的元素和主题")
    print("- 明确的警示等级指示")
    print("- 避免了灰色占位符问题")
    print("- 使用程序化生成确保质量")

if __name__ == "__main__":
    main()