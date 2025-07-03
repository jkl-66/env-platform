#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于生成式AI的生态警示图像系统演示

该脚本演示如何使用GAN和扩散模型根据环境危害数据生成具有视觉冲击力的警示图像，
旨在提升公众的生态环保意识。

功能特点：
1. 支持多种环境指标输入（碳排放量、污染指数等）
2. 基于条件GAN和扩散模型的图像生成
3. 预设环境场景模板
4. 可视化对比和分析
5. 教育意义的警示图像生成
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ecology_image_generator import EcologyImageGenerator
from src.utils.logger import setup_logger, get_logger
from src.utils.font_config import format_number, format_percentage

# 设置日志
setup_logger()
logger = get_logger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EcologyImageSystemDemo:
    """生态警示图像系统演示类"""
    
    def __init__(self):
        """初始化演示系统"""
        self.output_dir = Path("outputs/ecology_image_system")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化图像生成器
        self.image_generator = EcologyImageGenerator()
        
        logger.info("生态警示图像系统演示初始化完成")
    
    def run_complete_demo(self):
        """运行完整演示"""
        print("🌍 基于生成式AI的生态警示图像系统演示")
        print("=" * 80)
        print("本演示将展示如何根据环境危害数据生成具有视觉冲击力的警示图像")
        print("\n演示内容：")
        print("1. 环境指标输入与处理")
        print("2. GAN模型图像生成")
        print("3. 扩散模型图像生成")
        print("4. 预设场景模板演示")
        print("5. 教育意义图像对比")
        print("6. 系统性能评估")
        
        # 执行各个演示模块
        self.demo_environmental_indicators()
        self.demo_gan_generation()
        self.demo_diffusion_generation()
        self.demo_predefined_templates()
        self.demo_educational_comparison()
        self.demo_system_performance()
        
        print("\n🎉 生态警示图像系统演示完成！")
        print(f"📁 所有结果已保存至: {self.output_dir}")
    
    def demo_environmental_indicators(self):
        """演示环境指标输入与处理"""
        print("\n" + "=" * 60)
        print("📊 演示1: 环境指标输入与处理")
        print("=" * 60)
        
        # 定义不同严重程度的环境指标
        indicator_scenarios = {
            "轻度环境问题": {
                "co2_level": 380,  # ppm
                "pm25_level": 35,  # μg/m³
                "temperature": 26,  # °C
                "humidity": 65,    # %
                "forest_coverage": 45,  # %
                "water_quality": 8,     # 1-10分
                "air_quality": 7,       # 1-10分
                "biodiversity": 8,      # 1-10分
                "pollution_level": 2,   # 1-10分
                "warning_level": 1      # 1-5分
            },
            "中度环境问题": {
                "co2_level": 420,
                "pm25_level": 75,
                "temperature": 30,
                "humidity": 45,
                "forest_coverage": 25,
                "water_quality": 5,
                "air_quality": 4,
                "biodiversity": 5,
                "pollution_level": 6,
                "warning_level": 3
            },
            "重度环境危机": {
                "co2_level": 500,
                "pm25_level": 150,
                "temperature": 38,
                "humidity": 25,
                "forest_coverage": 10,
                "water_quality": 2,
                "air_quality": 2,
                "biodiversity": 2,
                "pollution_level": 9,
                "warning_level": 5
            }
        }
        
        # 处理和可视化环境指标
        self._visualize_environmental_indicators(indicator_scenarios)
        
        # 保存指标数据
        indicators_file = self.output_dir / "environmental_indicators.json"
        with open(indicators_file, 'w', encoding='utf-8') as f:
            json.dump(indicator_scenarios, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 环境指标数据已保存至: {indicators_file}")
        
        return indicator_scenarios
    
    def demo_gan_generation(self):
        """演示GAN模型图像生成"""
        print("\n" + "=" * 60)
        print("🎨 演示2: GAN模型图像生成")
        print("=" * 60)
        
        # 设置GAN生成模式
        self.image_generator.set_generation_mode("gan")
        
        # 定义测试场景
        test_scenarios = [
            {
                "name": "工业污染场景",
                "indicators": {
                    "co2_level": 450,
                    "pm25_level": 120,
                    "pollution_level": 8,
                    "air_quality": 2,
                    "warning_level": 4
                }
            },
            {
                "name": "森林砍伐场景",
                "indicators": {
                    "forest_coverage": 15,
                    "biodiversity": 3,
                    "co2_level": 430,
                    "warning_level": 4
                }
            },
            {
                "name": "极端气候场景",
                "indicators": {
                    "temperature": 42,
                    "humidity": 15,
                    "warning_level": 5,
                    "pollution_level": 6
                }
            }
        ]
        
        gan_results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n🖼️  生成场景 {i}: {scenario['name']}")
            
            try:
                # 使用GAN生成图像
                result = self.image_generator.generate_warning_image(
                    environmental_indicators=scenario['indicators'],
                    style="realistic",
                    num_images=2
                )
                
                if "error" not in result:
                    print(f"   ✅ GAN生成成功")
                    print(f"   📊 生成模式: {result.get('generation_mode', 'unknown')}")
                    print(f"   🖼️  图像数量: {len(result.get('generated_images', []))}")
                    
                    # 创建可视化
                    viz_path = self._create_gan_visualization(scenario, result, i)
                    
                    gan_results.append({
                        "scenario": scenario['name'],
                        "result": result,
                        "visualization_path": str(viz_path)
                    })
                else:
                    print(f"   ❌ GAN生成失败: {result['error']}")
                    
            except Exception as e:
                print(f"   ❌ 生成异常: {e}")
                logger.error(f"GAN生成失败: {e}")
        
        # 保存GAN生成结果
        gan_file = self.output_dir / "gan_generation_results.json"
        with open(gan_file, 'w', encoding='utf-8') as f:
            # 移除不可序列化的数据
            serializable_results = []
            for result in gan_results:
                serializable_result = {
                    "scenario": result["scenario"],
                    "visualization_path": result["visualization_path"],
                    "generation_mode": result["result"].get("generation_mode", "unknown")
                }
                serializable_results.append(serializable_result)
            
            json.dump({
                "generation_time": datetime.now().isoformat(),
                "total_scenarios": len(test_scenarios),
                "successful_generations": len(gan_results),
                "results": serializable_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 GAN生成结果已保存至: {gan_file}")
        return gan_results
    
    def demo_diffusion_generation(self):
        """演示扩散模型图像生成"""
        print("\n" + "=" * 60)
        print("🌟 演示3: 扩散模型图像生成")
        print("=" * 60)
        
        # 设置扩散生成模式
        self.image_generator.set_generation_mode("diffusion")
        
        # 定义文本提示场景
        prompt_scenarios = [
            {
                "name": "海平面上升警示",
                "prompt": "rising sea levels flooding coastal cities, melting glaciers, climate change disaster, dramatic lighting, photorealistic",
                "indicators": {"temperature": 35, "warning_level": 4}
            },
            {
                "name": "空气污染警示",
                "prompt": "heavy smog covering city skyline, industrial pollution, poor air quality, health warning, dystopian atmosphere",
                "indicators": {"pm25_level": 200, "air_quality": 1, "warning_level": 5}
            },
            {
                "name": "生物多样性丧失",
                "prompt": "deforestation and habitat destruction, endangered wildlife, biodiversity loss, environmental crisis, emotional impact",
                "indicators": {"forest_coverage": 5, "biodiversity": 1, "warning_level": 5}
            }
        ]
        
        diffusion_results = []
        
        for i, scenario in enumerate(prompt_scenarios, 1):
            print(f"\n🎭 生成场景 {i}: {scenario['name']}")
            print(f"   📝 提示词: {scenario['prompt'][:50]}...")
            
            try:
                # 使用扩散模型生成图像
                result = self.image_generator._generate_with_diffusion(
                    input_data={"prompt": scenario['prompt']},
                    num_images=1
                )
                
                if "error" not in result:
                    print(f"   ✅ 扩散模型生成成功")
                    print(f"   📊 生成模式: {result.get('generation_mode', 'unknown')}")
                    print(f"   🖼️  图像数量: {len(result.get('generated_images', []))}")
                    
                    # 创建可视化
                    viz_path = self._create_diffusion_visualization(scenario, result, i)
                    
                    diffusion_results.append({
                        "scenario": scenario['name'],
                        "prompt": scenario['prompt'],
                        "result": result,
                        "visualization_path": str(viz_path)
                    })
                else:
                    print(f"   ❌ 扩散模型生成失败: {result['error']}")
                    
            except Exception as e:
                print(f"   ❌ 生成异常: {e}")
                logger.error(f"扩散模型生成失败: {e}")
        
        # 保存扩散模型生成结果
        diffusion_file = self.output_dir / "diffusion_generation_results.json"
        with open(diffusion_file, 'w', encoding='utf-8') as f:
            # 移除不可序列化的数据
            serializable_results = []
            for result in diffusion_results:
                serializable_result = {
                    "scenario": result["scenario"],
                    "prompt": result["prompt"],
                    "visualization_path": result["visualization_path"],
                    "generation_mode": result["result"].get("generation_mode", "unknown")
                }
                serializable_results.append(serializable_result)
            
            json.dump({
                "generation_time": datetime.now().isoformat(),
                "total_scenarios": len(prompt_scenarios),
                "successful_generations": len(diffusion_results),
                "results": serializable_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 扩散模型生成结果已保存至: {diffusion_file}")
        return diffusion_results
    
    def demo_predefined_templates(self):
        """演示预设场景模板"""
        print("\n" + "=" * 60)
        print("📋 演示4: 预设场景模板")
        print("=" * 60)
        
        # 获取预设模板
        templates = self.image_generator.get_condition_templates()
        
        print(f"📚 可用模板数量: {len(templates)}")
        
        template_results = []
        
        for i, (template_name, conditions) in enumerate(templates.items(), 1):
            print(f"\n🏷️  模板 {i}: {template_name}")
            
            # 显示模板条件
            print("   📊 环境条件:")
            for key, value in conditions.items():
                if key == 'co2_level':
                    print(f"      • CO₂浓度: {format_number(value)} ppm")
                elif key == 'pm25_level':
                    print(f"      • PM2.5浓度: {format_number(value)} μg/m³")
                elif key == 'temperature':
                    print(f"      • 温度: {format_number(value)}°C")
                elif key == 'forest_coverage':
                    print(f"      • 森林覆盖率: {format_percentage(value/100)}")
                elif key == 'water_quality':
                    print(f"      • 水质指数: {format_number(value)}/10")
                elif key == 'air_quality':
                    print(f"      • 空气质量: {format_number(value)}/10")
                elif key == 'biodiversity':
                    print(f"      • 生物多样性: {format_number(value)}/10")
                elif key == 'pollution_level':
                    print(f"      • 污染程度: {format_number(value)}/10")
                elif key == 'warning_level':
                    print(f"      • 警示等级: {format_number(value)}/5")
            
            try:
                # 使用模板生成图像
                result = self.image_generator.generate_warning_image(
                    environmental_indicators=conditions,
                    style="photographic",
                    num_images=1
                )
                
                if "error" not in result:
                    print(f"   ✅ 模板图像生成成功")
                    
                    # 创建模板可视化
                    viz_path = self._create_template_visualization(template_name, conditions, result, i)
                    
                    template_results.append({
                        "template_name": template_name,
                        "conditions": conditions,
                        "result": result,
                        "visualization_path": str(viz_path)
                    })
                else:
                    print(f"   ❌ 模板图像生成失败: {result['error']}")
                    
            except Exception as e:
                print(f"   ❌ 生成异常: {e}")
                logger.error(f"模板生成失败: {e}")
        
        # 创建模板对比图
        self._create_template_comparison(template_results)
        
        # 保存模板结果
        template_file = self.output_dir / "template_generation_results.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            # 移除不可序列化的数据
            serializable_results = []
            for result in template_results:
                serializable_result = {
                    "template_name": result["template_name"],
                    "conditions": result["conditions"],
                    "visualization_path": result["visualization_path"]
                }
                serializable_results.append(serializable_result)
            
            json.dump({
                "generation_time": datetime.now().isoformat(),
                "total_templates": len(templates),
                "successful_generations": len(template_results),
                "results": serializable_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 模板生成结果已保存至: {template_file}")
        return template_results
    
    def demo_educational_comparison(self):
        """演示教育意义图像对比"""
        print("\n" + "=" * 60)
        print("🎓 演示5: 教育意义图像对比")
        print("=" * 60)
        
        # 定义对比场景：现在 vs 未来
        comparison_scenarios = [
            {
                "title": "森林保护的重要性",
                "current": {
                    "name": "健康森林",
                    "forest_coverage": 80,
                    "biodiversity": 9,
                    "co2_level": 350,
                    "warning_level": 1
                },
                "future": {
                    "name": "森林砍伐后果",
                    "forest_coverage": 10,
                    "biodiversity": 2,
                    "co2_level": 500,
                    "warning_level": 5
                }
            },
            {
                "title": "减少碳排放的必要性",
                "current": {
                    "name": "低碳生活",
                    "co2_level": 380,
                    "air_quality": 8,
                    "temperature": 25,
                    "warning_level": 1
                },
                "future": {
                    "name": "高碳排放后果",
                    "co2_level": 550,
                    "air_quality": 2,
                    "temperature": 40,
                    "warning_level": 5
                }
            }
        ]
        
        comparison_results = []
        
        for i, scenario in enumerate(comparison_scenarios, 1):
            print(f"\n📚 对比场景 {i}: {scenario['title']}")
            
            scenario_result = {
                "title": scenario['title'],
                "current": None,
                "future": None
            }
            
            # 生成当前状态图像
            print(f"   🌱 生成图像: {scenario['current']['name']}")
            try:
                current_result = self.image_generator.generate_warning_image(
                    environmental_indicators=scenario['current'],
                    style="realistic",
                    num_images=1
                )
                if "error" not in current_result:
                    scenario_result['current'] = current_result
                    print(f"      ✅ 当前状态图像生成成功")
                else:
                    print(f"      ❌ 当前状态图像生成失败")
            except Exception as e:
                print(f"      ❌ 生成异常: {e}")
            
            # 生成未来状态图像
            print(f"   ⚠️  生成图像: {scenario['future']['name']}")
            try:
                future_result = self.image_generator.generate_warning_image(
                    environmental_indicators=scenario['future'],
                    style="realistic",
                    num_images=1
                )
                if "error" not in future_result:
                    scenario_result['future'] = future_result
                    print(f"      ✅ 未来状态图像生成成功")
                else:
                    print(f"      ❌ 未来状态图像生成失败")
            except Exception as e:
                print(f"      ❌ 生成异常: {e}")
            
            # 创建对比可视化
            if scenario_result['current'] and scenario_result['future']:
                viz_path = self._create_comparison_visualization(scenario, scenario_result, i)
                scenario_result['visualization_path'] = str(viz_path)
                print(f"      📊 对比图表已生成: {viz_path.name}")
            
            comparison_results.append(scenario_result)
        
        # 保存对比结果
        comparison_file = self.output_dir / "educational_comparison_results.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            # 移除不可序列化的数据
            serializable_results = []
            for result in comparison_results:
                serializable_result = {
                    "title": result["title"],
                    "visualization_path": result.get("visualization_path", "")
                }
                serializable_results.append(serializable_result)
            
            json.dump({
                "generation_time": datetime.now().isoformat(),
                "total_comparisons": len(comparison_scenarios),
                "successful_comparisons": len([r for r in comparison_results if r.get('visualization_path')]),
                "results": serializable_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 教育对比结果已保存至: {comparison_file}")
        return comparison_results
    
    def demo_system_performance(self):
        """演示系统性能评估"""
        print("\n" + "=" * 60)
        print("⚡ 演示6: 系统性能评估")
        print("=" * 60)
        
        # 性能测试场景
        performance_tests = [
            {"name": "单图像生成", "num_images": 1, "iterations": 3},
            {"name": "批量生成", "num_images": 3, "iterations": 2},
            {"name": "高分辨率生成", "num_images": 1, "iterations": 2}
        ]
        
        performance_results = []
        
        test_indicators = {
            "co2_level": 450,
            "pollution_level": 7,
            "warning_level": 4
        }
        
        for test in performance_tests:
            print(f"\n🔬 性能测试: {test['name']}")
            print(f"   📊 图像数量: {test['num_images']}")
            print(f"   🔄 测试轮次: {test['iterations']}")
            
            test_times = []
            
            for i in range(test['iterations']):
                print(f"   ⏱️  第 {i+1} 轮测试...")
                
                start_time = datetime.now()
                
                try:
                    result = self.image_generator.generate_warning_image(
                        environmental_indicators=test_indicators,
                        num_images=test['num_images']
                    )
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    test_times.append(duration)
                    
                    if "error" not in result:
                        print(f"      ✅ 生成成功，耗时: {format_number(duration)}秒")
                    else:
                        print(f"      ❌ 生成失败: {result['error']}")
                        
                except Exception as e:
                    print(f"      ❌ 测试异常: {e}")
                    continue
            
            if test_times:
                avg_time = np.mean(test_times)
                min_time = np.min(test_times)
                max_time = np.max(test_times)
                
                print(f"   📈 性能统计:")
                print(f"      • 平均耗时: {format_number(avg_time)}秒")
                print(f"      • 最短耗时: {format_number(min_time)}秒")
                print(f"      • 最长耗时: {format_number(max_time)}秒")
                print(f"      • 平均每图耗时: {format_number(avg_time/test['num_images'])}秒")
                
                performance_results.append({
                    "test_name": test['name'],
                    "num_images": test['num_images'],
                    "iterations": test['iterations'],
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "avg_time_per_image": avg_time / test['num_images']
                })
        
        # 创建性能对比图
        self._create_performance_chart(performance_results)
        
        # 保存性能结果
        performance_file = self.output_dir / "system_performance_results.json"
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_time": datetime.now().isoformat(),
                "total_tests": len(performance_tests),
                "results": performance_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 性能测试结果已保存至: {performance_file}")
        return performance_results
    
    def _visualize_environmental_indicators(self, scenarios):
        """可视化环境指标"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('环境指标对比分析', fontsize=16, fontweight='bold')
        
        # 提取数据
        scenario_names = list(scenarios.keys())
        indicators = ['co2_level', 'pm25_level', 'temperature', 'forest_coverage']
        indicator_labels = ['CO₂浓度 (ppm)', 'PM2.5浓度 (μg/m³)', '温度 (°C)', '森林覆盖率 (%)']
        
        for i, (indicator, label) in enumerate(zip(indicators, indicator_labels)):
            ax = axes[i//2, i%2]
            
            values = [scenarios[name].get(indicator, 0) for name in scenario_names]
            colors = ['green', 'orange', 'red']
            
            bars = ax.bar(scenario_names, values, color=colors)
            ax.set_title(label, fontweight='bold')
            ax.set_ylabel('数值')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       format_number(value), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = self.output_dir / "environmental_indicators_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 环境指标对比图已保存: {chart_path.name}")
        return chart_path
    
    def _create_gan_visualization(self, scenario, result, index):
        """创建GAN生成结果可视化"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建模拟的生成结果展示
        ax.text(0.5, 0.7, f"GAN生成结果", ha='center', va='center', 
                fontsize=20, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.5, f"场景: {scenario['name']}", ha='center', va='center', 
                fontsize=14, transform=ax.transAxes)
        ax.text(0.5, 0.3, f"生成模式: {result.get('generation_mode', 'GAN')}", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.1, f"图像数量: {len(result.get('generated_images', []))}", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 保存可视化
        viz_path = self.output_dir / f"gan_generation_{index}_{scenario['name']}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def _create_diffusion_visualization(self, scenario, result, index):
        """创建扩散模型生成结果可视化"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建模拟的生成结果展示
        ax.text(0.5, 0.7, f"扩散模型生成结果", ha='center', va='center', 
                fontsize=20, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.5, f"场景: {scenario['name']}", ha='center', va='center', 
                fontsize=14, transform=ax.transAxes)
        ax.text(0.5, 0.3, f"生成模式: {result.get('generation_mode', 'Diffusion')}", 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.1, f"提示词: {scenario['prompt'][:30]}...", 
                ha='center', va='center', fontsize=10, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 保存可视化
        viz_path = self.output_dir / f"diffusion_generation_{index}_{scenario['name']}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def _create_template_visualization(self, template_name, conditions, result, index):
        """创建模板生成结果可视化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左侧：条件雷达图
        categories = list(conditions.keys())
        values = list(conditions.values())
        
        # 标准化数值到0-1范围
        normalized_values = []
        for key, value in conditions.items():
            if key in ['co2_level']:
                normalized_values.append(min(value / 500, 1.0))
            elif key in ['pm25_level']:
                normalized_values.append(min(value / 200, 1.0))
            elif key in ['temperature']:
                normalized_values.append(min(value / 50, 1.0))
            else:
                normalized_values.append(min(value / 10, 1.0))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        normalized_values += normalized_values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax1.plot(angles, normalized_values, 'o-', linewidth=2, color='red')
        ax1.fill(angles, normalized_values, alpha=0.25, color='red')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories, fontsize=8)
        ax1.set_ylim(0, 1)
        ax1.set_title(f'环境条件雷达图\n{template_name}', fontweight='bold')
        ax1.grid(True)
        
        # 右侧：生成结果信息
        ax2.text(0.5, 0.7, f"模板生成结果", ha='center', va='center', 
                fontsize=16, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.5, 0.5, f"模板: {template_name}", ha='center', va='center', 
                fontsize=12, transform=ax2.transAxes)
        ax2.text(0.5, 0.3, f"生成模式: {result.get('generation_mode', 'unknown')}", 
                ha='center', va='center', fontsize=10, transform=ax2.transAxes)
        ax2.text(0.5, 0.1, f"图像数量: {len(result.get('generated_images', []))}", 
                ha='center', va='center', fontsize=10, transform=ax2.transAxes)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        
        # 保存可视化
        viz_path = self.output_dir / f"template_generation_{index}_{template_name}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def _create_template_comparison(self, template_results):
        """创建模板对比图"""
        if not template_results:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        template_names = [r['template_name'] for r in template_results]
        warning_levels = [r['conditions'].get('warning_level', 0) for r in template_results]
        
        colors = ['green' if w <= 2 else 'orange' if w <= 3 else 'red' for w in warning_levels]
        
        bars = ax.bar(template_names, warning_levels, color=colors)
        ax.set_title('预设模板警示等级对比', fontsize=16, fontweight='bold')
        ax.set_ylabel('警示等级')
        ax.set_ylim(0, 5)
        
        # 添加数值标签
        for bar, level in zip(bars, warning_levels):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   format_number(level), ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存图表
        chart_path = self.output_dir / "template_warning_levels_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 模板对比图已保存: {chart_path.name}")
        return chart_path
    
    def _create_comparison_visualization(self, scenario, result, index):
        """创建教育对比可视化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左侧：当前状态
        ax1.text(0.5, 0.7, scenario['current']['name'], ha='center', va='center', 
                fontsize=16, fontweight='bold', color='green', transform=ax1.transAxes)
        ax1.text(0.5, 0.5, "✅ 环境友好", ha='center', va='center', 
                fontsize=14, transform=ax1.transAxes)
        ax1.text(0.5, 0.3, f"警示等级: {scenario['current']['warning_level']}/5", 
                ha='center', va='center', fontsize=12, transform=ax1.transAxes)
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_facecolor('#e8f5e8')
        ax1.axis('off')
        ax1.set_title('现状', fontweight='bold')
        
        # 右侧：未来状态
        ax2.text(0.5, 0.7, scenario['future']['name'], ha='center', va='center', 
                fontsize=16, fontweight='bold', color='red', transform=ax2.transAxes)
        ax2.text(0.5, 0.5, "⚠️ 环境危机", ha='center', va='center', 
                fontsize=14, transform=ax2.transAxes)
        ax2.text(0.5, 0.3, f"警示等级: {scenario['future']['warning_level']}/5", 
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_facecolor('#f5e8e8')
        ax2.axis('off')
        ax2.set_title('未来风险', fontweight='bold')
        
        fig.suptitle(f'教育对比: {scenario["title"]}', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # 保存可视化
        viz_path = self.output_dir / f"educational_comparison_{index}_{scenario['title']}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def _create_performance_chart(self, performance_results):
        """创建性能对比图表"""
        if not performance_results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        test_names = [r['test_name'] for r in performance_results]
        avg_times = [r['avg_time'] for r in performance_results]
        times_per_image = [r['avg_time_per_image'] for r in performance_results]
        
        # 左侧：总耗时对比
        bars1 = ax1.bar(test_names, avg_times, color=['blue', 'orange', 'green'])
        ax1.set_title('平均生成耗时对比', fontweight='bold')
        ax1.set_ylabel('耗时 (秒)')
        
        for bar, time in zip(bars1, avg_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{format_number(time)}s', ha='center', va='bottom')
        
        # 右侧：单图耗时对比
        bars2 = ax2.bar(test_names, times_per_image, color=['blue', 'orange', 'green'])
        ax2.set_title('平均单图生成耗时对比', fontweight='bold')
        ax2.set_ylabel('耗时 (秒/图)')
        
        for bar, time in zip(bars2, times_per_image):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{format_number(time)}s', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存图表
        chart_path = self.output_dir / "system_performance_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 性能对比图已保存: {chart_path.name}")
        return chart_path


def main():
    """主函数"""
    try:
        # 创建演示实例
        demo = EcologyImageSystemDemo()
        
        # 运行完整演示
        demo.run_complete_demo()
        
    except Exception as e:
        logger.error(f"演示运行失败: {e}")
        print(f"❌ 演示失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())