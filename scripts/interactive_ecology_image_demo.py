#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式生态警示图像生成系统

这是一个用户友好的交互式脚本，让您可以轻松使用生态警示图像生成系统。
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ecology_image_generator import EcologyImageGenerator
from src.utils.logger import setup_logger, get_logger

# 设置日志
setup_logger()
logger = get_logger(__name__)

class InteractiveEcologyImageSystem:
    """交互式生态图像生成系统"""
    
    def __init__(self):
        """初始化系统"""
        self.generator = EcologyImageGenerator()
        self.output_dir = Path("outputs/interactive_ecology_demo")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🌍 欢迎使用生态警示图像生成系统！")
        print("=" * 50)
    
    def show_menu(self):
        """显示主菜单"""
        print("\n📋 请选择功能：")
        print("1. 🎨 生成环境警示图像")
        print("2. 📊 查看预设环境场景")
        print("3. 🔧 自定义环境指标")
        print("4. 📈 批量生成对比图像")
        print("5. 💬 自然语言生成图像")
        print("6. 📚 查看使用指南")
        print("7. 🚪 退出系统")
        print("-" * 30)
    
    def get_environmental_indicators(self) -> Dict[str, float]:
        """获取环境指标输入"""
        print("\n🌡️ 请输入环境指标（直接回车使用默认值）：")
        
        indicators = {}
        
        # CO2排放量 (ppm)
        co2_input = input("CO2排放量 (ppm, 默认400): ").strip()
        indicators['co2_level'] = float(co2_input) if co2_input else 400.0
        
        # PM2.5浓度 (μg/m³)
        pm25_input = input("PM2.5浓度 (μg/m³, 默认50): ").strip()
        indicators['pm25_level'] = float(pm25_input) if pm25_input else 50.0
        
        # 温度变化 (°C)
        temp_input = input("温度变化 (°C, 默认25): ").strip()
        indicators['temperature'] = float(temp_input) if temp_input else 25.0
        
        # 森林覆盖率 (%)
        forest_input = input("森林覆盖率 (%, 默认60): ").strip()
        indicators['forest_coverage'] = float(forest_input) if forest_input else 60.0
        
        # 水质指数 (1-10)
        water_input = input("水质指数 (1-10, 默认7): ").strip()
        indicators['water_quality'] = float(water_input) if water_input else 7.0
        
        # 空气质量指数 (1-10)
        air_input = input("空气质量指数 (1-10, 默认6): ").strip()
        indicators['air_quality'] = float(air_input) if air_input else 6.0
        
        return indicators
    
    def generate_single_image(self):
        """生成单张警示图像"""
        print("\n🎨 生成环境警示图像")
        print("=" * 30)
        
        # 获取环境指标
        indicators = self.get_environmental_indicators()
        
        # 选择生成模式
        print("\n🤖 选择生成模式：")
        print("1. GAN模式 (快速生成)")
        print("2. 扩散模式 (高质量)")
        print("3. 混合模式 (平衡质量与速度)")
        
        mode_choice = input("请选择模式 (1-3, 默认1): ").strip()
        mode_map = {'1': 'gan', '2': 'diffusion', '3': 'hybrid'}
        generation_mode = mode_map.get(mode_choice, 'gan')
        
        # 设置生成模式
        self.generator.set_generation_mode(generation_mode)
        
        # 选择风格
        print("\n🎭 选择图像风格：")
        print("1. 写实风格")
        print("2. 艺术风格")
        print("3. 科幻风格")
        print("4. 教育风格")
        
        style_choice = input("请选择风格 (1-4, 默认1): ").strip()
        style_map = {
            '1': 'realistic',
            '2': 'artistic', 
            '3': 'sci-fi',
            '4': 'educational'
        }
        style = style_map.get(style_choice, 'realistic')
        
        # 生成图像数量
        num_images_input = input("\n生成图像数量 (1-5, 默认1): ").strip()
        num_images = int(num_images_input) if num_images_input.isdigit() else 1
        num_images = max(1, min(5, num_images))
        
        print(f"\n🚀 开始生成 {num_images} 张图像...")
        print(f"📊 环境指标: {indicators}")
        print(f"🤖 生成模式: {generation_mode}")
        print(f"🎭 图像风格: {style}")
        
        try:
            # 生成图像
            result = self.generator.generate_warning_image(
                environmental_indicators=indicators,
                style=style,
                num_images=num_images
            )
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = self.output_dir / f"generation_result_{timestamp}.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n✅ 图像生成完成！")
            print(f"📁 结果已保存至: {result_file}")
            print(f"⚠️  警示等级: {result['warning_level']}/5")
            print(f"🏷️  使用模板: {result['template_used']}")
            print(f"🔍 环境评估: {result['environmental_assessment']['overall_risk']}")
            
            # 显示生成的图像信息
            print(f"\n📸 生成的图像信息:")
            for i, img_info in enumerate(result['generated_images'], 1):
                print(f"  图像 {i}:")
                print(f"    - 描述: {img_info['description']}")
                print(f"    - 风格: {img_info['style']}")
                print(f"    - 质量评分: {img_info['quality_score']:.2f}")
                print(f"    - 生成时间: {img_info['generation_time']}秒")
            
            # 生成可视化图像并自动显示
            image_file = self._create_visualization_image(result, timestamp)
            if image_file:
                self._open_result_file(image_file)
            else:
                # 如果图像生成失败，则打开JSON文件
                self._open_result_file(result_file)
            
        except Exception as e:
            logger.error(f"图像生成失败: {e}")
            print(f"❌ 图像生成失败: {e}")
    
    def show_preset_scenarios(self):
        """显示预设环境场景"""
        print("\n📊 预设环境场景模板")
        print("=" * 30)
        
        try:
            templates = self.generator.get_condition_templates()
            
            for i, (name, template) in enumerate(templates.items(), 1):
                print(f"\n{i}. {name}")
                # 安全地访问模板字段，提供默认值
                description = template.get('description', '环境场景模板')
                warning_level = template.get('warning_level', 3)
                visual_elements = template.get('visual_elements', ['环境要素'])
                color_scheme = template.get('color_scheme', ['自然色彩'])
                
                print(f"   描述: {description}")
                print(f"   警示等级: {warning_level}/5")
                print(f"   视觉元素: {', '.join(visual_elements)}")
                print(f"   色彩方案: {', '.join(color_scheme)}")
        
            # 让用户选择模板生成图像
            choice = input(f"\n选择模板生成图像 (1-{len(templates)}, 回车跳过): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(templates):
                template_name = list(templates.keys())[int(choice) - 1]
                template = templates[template_name]
                
                print(f"\n🎨 使用模板 '{template_name}' 生成图像...")
                
                # 根据模板生成合适的环境指标
                indicators = self._generate_indicators_from_template(template_name)
                
                try:
                    result = self.generator.generate_warning_image(
                        environmental_indicators=indicators,
                        style='realistic',
                        num_images=1
                    )
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result_file = self.output_dir / f"template_{template_name}_{timestamp}.json"
                    
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
                    
                    print(f"✅ 模板图像生成完成！")
                    print(f"📁 结果已保存至: {result_file}")
                    print(f"⚠️  警示等级: {result['warning_level']}/5")
                    
                    # 自动打开结果文件
                    self._open_result_file(result_file)
                    
                except Exception as e:
                    print(f"❌ 模板图像生成失败: {e}")
                    
        except Exception as e:
            print(f"❌ 获取模板失败: {e}")
            print("请检查模型是否正确初始化")
    
    def _generate_indicators_from_template(self, template_name: str) -> Dict[str, float]:
        """根据模板生成对应的环境指标"""
        template_indicators = {
            "冰川融化": {
                "co2_level": 450.0,
                "temperature": 35.0,
                "forest_coverage": 40.0,
                "water_quality": 6.0,
                "air_quality": 5.0,
                "pm25_level": 80.0
            },
            "森林砍伐": {
                "co2_level": 420.0,
                "temperature": 28.0,
                "forest_coverage": 15.0,
                "water_quality": 4.0,
                "air_quality": 6.0,
                "pm25_level": 60.0
            },
            "空气污染": {
                "co2_level": 480.0,
                "temperature": 30.0,
                "forest_coverage": 30.0,
                "water_quality": 5.0,
                "air_quality": 2.0,
                "pm25_level": 150.0
            },
            "水质污染": {
                "co2_level": 410.0,
                "temperature": 26.0,
                "forest_coverage": 50.0,
                "water_quality": 2.0,
                "air_quality": 6.0,
                "pm25_level": 70.0
            },
            "极端天气": {
                "co2_level": 460.0,
                "temperature": 40.0,
                "forest_coverage": 35.0,
                "water_quality": 5.0,
                "air_quality": 4.0,
                "pm25_level": 90.0
            }
        }
        
        return template_indicators.get(template_name, {
            "co2_level": 400.0,
            "temperature": 25.0,
            "forest_coverage": 60.0,
            "water_quality": 7.0,
            "air_quality": 6.0,
            "pm25_level": 50.0
        })
    
    def custom_indicators_demo(self):
        """自定义环境指标演示"""
        print("\n🔧 自定义环境指标演示")
        print("=" * 30)
        
        print("\n📝 环境指标说明：")
        print("• CO2排放量: 大气中二氧化碳浓度 (ppm)")
        print("  - 正常值: 350-400 ppm")
        print("  - 警戒值: 400-450 ppm")
        print("  - 危险值: >450 ppm")
        
        print("\n• PM2.5浓度: 细颗粒物浓度 (μg/m³)")
        print("  - 优良: 0-35 μg/m³")
        print("  - 轻度污染: 35-75 μg/m³")
        print("  - 重度污染: >75 μg/m³")
        
        print("\n• 温度变化: 相对于基准温度的变化 (°C)")
        print("  - 正常: 20-25°C")
        print("  - 偏高: 25-35°C")
        print("  - 极端: >35°C")
        
        print("\n• 森林覆盖率: 地区森林覆盖百分比 (%)")
        print("  - 良好: >60%")
        print("  - 一般: 30-60%")
        print("  - 较差: <30%")
        
        print("\n• 水质指数: 水体质量评分 (1-10)")
        print("  - 优秀: 8-10")
        print("  - 良好: 6-8")
        print("  - 较差: <6")
        
        print("\n• 空气质量指数: 空气质量评分 (1-10)")
        print("  - 优秀: 8-10")
        print("  - 良好: 6-8")
        print("  - 较差: <6")
        
        input("\n按回车键继续...")
        self.generate_single_image()
    
    def batch_generation_demo(self):
        """批量生成对比图像"""
        print("\n📈 批量生成对比图像")
        print("=" * 30)
        
        # 预定义几个对比场景
        scenarios = {
            "轻度污染": {
                "co2_level": 380.0,
                "pm25_level": 40.0,
                "temperature": 24.0,
                "forest_coverage": 70.0,
                "water_quality": 8.0,
                "air_quality": 7.0
            },
            "中度污染": {
                "co2_level": 420.0,
                "pm25_level": 80.0,
                "temperature": 30.0,
                "forest_coverage": 45.0,
                "water_quality": 5.0,
                "air_quality": 4.0
            },
            "重度污染": {
                "co2_level": 480.0,
                "pm25_level": 150.0,
                "temperature": 38.0,
                "forest_coverage": 20.0,
                "water_quality": 2.0,
                "air_quality": 2.0
            }
        }
        
        print("\n🔄 将生成以下对比场景的图像：")
        for name, indicators in scenarios.items():
            print(f"• {name}: CO2={indicators['co2_level']}ppm, PM2.5={indicators['pm25_level']}μg/m³")
        
        confirm = input("\n确认开始批量生成？(y/N): ").strip().lower()
        if confirm != 'y':
            return
        
        results = {}
        
        for scenario_name, indicators in scenarios.items():
            print(f"\n🎨 正在生成 '{scenario_name}' 场景图像...")
            
            try:
                result = self.generator.generate_warning_image(
                    environmental_indicators=indicators,
                    style='realistic',
                    num_images=1
                )
                
                results[scenario_name] = result
                print(f"✅ '{scenario_name}' 生成完成，警示等级: {result['warning_level']}/5")
                
            except Exception as e:
                print(f"❌ '{scenario_name}' 生成失败: {e}")
                results[scenario_name] = {"error": str(e)}
        
        # 保存批量结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_result_file = self.output_dir / f"batch_comparison_{timestamp}.json"
        
        with open(batch_result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📊 批量生成完成！")
        print(f"📁 对比结果已保存至: {batch_result_file}")
        
        # 显示对比总结
        print("\n📈 对比总结：")
        for scenario_name, result in results.items():
            if "error" not in result:
                print(f"• {scenario_name}: 警示等级 {result['warning_level']}/5, 模板 '{result['template_used']}'")
            else:
                print(f"• {scenario_name}: 生成失败")
        
        # 自动打开批量结果文件
        self._open_result_file(batch_result_file)
    
    def _create_visualization_image(self, result: Dict[str, Any], timestamp: str) -> Path:
        """创建AI生成图像的展示界面"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.font_manager import FontProperties
            import numpy as np
            from PIL import Image
            import io
            import base64
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            generated_images = result.get('generated_images', [])
            
            # 如果有实际生成的图像数据，尝试显示
            if generated_images and len(generated_images) > 0:
                img_info = generated_images[0]
                image_data = img_info.get('image_data')
                
                # 检查是否有真实的图像数据
                if image_data and not isinstance(image_data, str):
                    try:
                        # 尝试处理图像数据
                        if isinstance(image_data, list):
                            # 转换为numpy数组
                            img_array = np.array(image_data)
                            if img_array.ndim == 3 and img_array.shape[2] == 3:
                                # 创建展示界面
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                                fig.suptitle(f'🎨 AI生成的环境警示图像作品 - 警示等级: {result["warning_level"]}/5', 
                                            fontsize=18, fontweight='bold', y=0.95)
                                
                                # 显示生成的图像
                                ax1.imshow(img_array)
                                ax1.set_title(f'生成图像: {img_info.get("description", "环境警示图像")}', 
                                            fontsize=14, fontweight='bold')
                                ax1.axis('off')
                                
                                # 显示图像信息
                                info_text = f"""🎨 图像作品信息
                                
📝 原始描述: {result.get('original_prompt', img_info.get('description', 'N/A'))}

🔧 增强提示: {result.get('enhanced_prompt', 'N/A')[:100]}...

🎭 生成风格: {img_info.get('style', 'N/A')}

⚡ 生成模式: {result.get('generation_mode', 'N/A')}

⭐ 质量评分: {img_info.get('quality_score', 0):.2f}/1.0

⏱️ 生成时间: {img_info.get('generation_time', 0):.1f}秒

🚨 警示等级: {result['warning_level']}/5

🌍 环境主题: {', '.join(result.get('text_analysis', {}).get('detected_themes', ['未知']))}

💥 环境影响: {result.get('text_analysis', {}).get('environmental_impact', '未知')}"""
                                
                                ax2.text(0.05, 0.95, info_text, fontsize=11, 
                                        transform=ax2.transAxes, verticalalignment='top',
                                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
                                ax2.set_xlim(0, 1)
                                ax2.set_ylim(0, 1)
                                ax2.axis('off')
                                
                                # 保存图像
                                image_file = self.output_dir / f"ai_artwork_{timestamp}.png"
                                plt.tight_layout()
                                plt.savefig(image_file, dpi=300, bbox_inches='tight')
                                plt.close()
                                
                                print(f"🎨 AI生成图像作品已保存: {image_file}")
                                return image_file
                    except Exception as e:
                        print(f"⚠️ 处理AI生成图像数据时出错: {e}")
            
            # 如果没有真实图像数据，创建概念展示图
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle(f'🎨 环境警示图像概念展示 - 警示等级: {result["warning_level"]}/5', 
                        fontsize=18, fontweight='bold', y=0.95)
            
            # 创建概念图像占位符
            concept_colors = {
                1: ['lightgreen', 'green'],
                2: ['yellow', 'orange'], 
                3: ['orange', 'darkorange'],
                4: ['red', 'darkred'],
                5: ['darkred', 'black']
            }
            
            warning_level = result['warning_level']
            colors = concept_colors.get(warning_level, ['gray', 'darkgray'])
            
            # 创建渐变背景表示环境状态
            x = np.linspace(0, 10, 100)
            y = np.linspace(0, 10, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X/2) * np.cos(Y/2) + warning_level
            
            im = ax.contourf(X, Y, Z, levels=20, cmap='Reds' if warning_level >= 3 else 'YlOrRd')
            
            # 添加概念性元素
            if generated_images:
                img_info = generated_images[0]
                description = img_info.get('description', result.get('original_prompt', '环境警示场景'))
                
                # 在图像中央显示描述
                ax.text(5, 8, f'🎨 AI图像作品概念', fontsize=16, fontweight='bold', 
                       ha='center', va='center', 
                       bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.9))
                
                ax.text(5, 6.5, f'📝 "{description}"', fontsize=14, 
                       ha='center', va='center', style='italic',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
                
                ax.text(5, 5, f'🚨 警示等级: {warning_level}/5', fontsize=14, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor=colors[0], alpha=0.8))
                
                # 显示生成模式和风格
                mode_style = f"🔧 {result.get('generation_mode', 'AI')}模式 | 🎭 {img_info.get('style', '真实')}风格"
                ax.text(5, 3.5, mode_style, fontsize=12,
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
                
                # 显示环境主题
                themes = result.get('text_analysis', {}).get('detected_themes', [])
                if themes:
                    theme_text = f"🌍 环境主题: {', '.join(themes)}"
                    ax.text(5, 2, theme_text, fontsize=11,
                           ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
            
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_title('AI生成的环境警示图像作品（概念展示）', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # 保存图像
            image_file = self.output_dir / f"concept_artwork_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(image_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🎨 环境警示图像概念展示已生成: {image_file}")
            print(f"💡 注意: 当前显示的是概念展示图，实际的AI图像生成需要配置真实的图像生成模型")
            return image_file
            
        except Exception as e:
            logger.error(f"创建图像展示失败: {e}")
            print(f"⚠️  创建图像展示失败: {e}")
            return None
    
    def _open_result_file(self, file_path: Path):
        """自动打开结果文件"""
        try:
            print(f"\n🖼️  正在打开结果文件...")
            
            # 如果是JSON文件，尝试创建并打开可视化图像
            if file_path.suffix.lower() == '.json':
                try:
                    # 读取JSON文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    
                    # 从文件名中提取时间戳
                    timestamp = file_path.stem.split('_')[-1]
                    
                    # 创建可视化图像
                    vis_image = self._create_visualization_image(result, timestamp)
                    
                    # 如果可视化图像创建成功，打开图像而不是JSON
                    if vis_image and vis_image.exists():
                        file_path = vis_image
                except Exception as e:
                    logger.error(f"创建可视化图像失败，将打开原始JSON文件: {e}")
                    print(f"⚠️ 无法创建可视化图像，将打开原始JSON文件: {e}")
            
            # 根据操作系统选择合适的打开方式
            system = platform.system().lower()
            
            if system == "windows":
                # Windows系统使用默认程序打开
                os.startfile(str(file_path))
            elif system == "darwin":  # macOS
                subprocess.run(["open", str(file_path)])
            elif system == "linux":
                subprocess.run(["xdg-open", str(file_path)])
            else:
                print(f"⚠️  无法自动打开文件，请手动打开: {file_path}")
                return
            
            print(f"✅ 已在默认程序中打开结果文件")
            
        except Exception as e:
            logger.warning(f"无法自动打开文件: {e}")
            print(f"⚠️  无法自动打开文件: {e}")
            print(f"📁 请手动打开: {file_path}")
    
    def show_usage_guide(self):
        """显示使用指南"""
        print("\n📚 生态警示图像生成系统使用指南")
        print("=" * 50)
        
        print("\n🎯 系统功能：")
        print("• 根据环境指标生成警示图像")
        print("• 支持多种生成模式（GAN、扩散、混合）")
        print("• 提供多种图像风格选择")
        print("• 智能评估环境风险等级")
        print("• 支持批量生成和对比分析")
        
        print("\n🔧 使用步骤：")
        print("1. 选择功能模式")
        print("2. 输入环境指标数据")
        print("3. 选择生成模式和风格")
        print("4. 等待图像生成完成")
        print("5. 查看生成结果和分析报告")
        
        print("\n📊 环境指标说明：")
        print("• CO2排放量: 反映温室气体排放水平")
        print("• PM2.5浓度: 反映空气污染程度")
        print("• 温度变化: 反映气候变化影响")
        print("• 森林覆盖率: 反映生态保护状况")
        print("• 水质指数: 反映水环境质量")
        print("• 空气质量指数: 反映大气环境质量")
        
        print("\n🎨 生成模式说明：")
        print("• GAN模式: 快速生成，适合实时预览")
        print("• 扩散模式: 高质量生成，适合最终展示")
        print("• 混合模式: 平衡质量与速度")
        
        print("\n🎭 风格选择：")
        print("• 写实风格: 真实感强，适合科学展示")
        print("• 艺术风格: 视觉冲击力强，适合宣传")
        print("• 科幻风格: 未来感强，适合警示教育")
        print("• 教育风格: 简洁明了，适合教学使用")
        
        print("\n💡 使用建议：")
        print("• 首次使用建议从预设场景开始")
        print("• 根据实际需求选择合适的生成模式")
        print("• 可以多次调整参数进行对比")
        print("• 生成结果会自动保存，便于后续查看")
        
        input("\n按回车键返回主菜单...")
    
    def natural_language_generation(self):
        """自然语言生成图像"""
        print("\n💬 自然语言生成图像")
        print("=" * 30)
        
        print("\n📝 请用自然语言描述您想要生成的环境场景：")
        print("例如：'一个严重污染的城市，空气中充满雾霾，河流被污染'")
        print("或者：'全球变暖导致的冰川融化和海平面上升场景'")
        
        description = input("\n请输入场景描述: ").strip()
        
        if not description:
            print("❌ 描述不能为空！")
            return
        
        print(f"\n🤖 正在分析描述: {description}")
        
        # 基于自然语言描述推断环境指标
        indicators = self._parse_natural_language_to_indicators(description)
        
        print(f"\n📊 推断的环境指标:")
        for key, value in indicators.items():
            print(f"  • {key}: {value}")
        
        # 确认是否使用推断的指标
        confirm = input("\n是否使用这些推断的指标生成图像？(Y/n): ").strip().lower()
        if confirm == 'n':
            print("已取消生成")
            return
        
        # 选择风格
        print("\n🎭 选择图像风格：")
        print("1. 写实风格")
        print("2. 艺术风格")
        print("3. 科幻风格")
        print("4. 教育风格")
        
        style_choice = input("请选择风格 (1-4, 默认1): ").strip()
        style_map = {
            '1': 'realistic',
            '2': 'artistic', 
            '3': 'sci-fi',
            '4': 'educational'
        }
        style = style_map.get(style_choice, 'realistic')
        
        print(f"\n🚀 开始生成图像...")
        print(f"📝 场景描述: {description}")
        print(f"🎭 图像风格: {style}")
        
        try:
            # 生成图像
            result = self.generator.generate_warning_image(
                environmental_indicators=indicators,
                style=style,
                num_images=1
            )
            
            # 添加原始描述到结果中
            result['natural_language_description'] = description
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = self.output_dir / f"natural_language_result_{timestamp}.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n✅ 图像生成完成！")
            print(f"📁 结果已保存至: {result_file}")
            print(f"⚠️  警示等级: {result['warning_level']}/5")
            print(f"🏷️  使用模板: {result['template_used']}")
            print(f"🔍 环境评估: {result['environmental_assessment']['overall_risk']}")
            
            # 生成可视化图像并自动显示
            image_file = self._create_visualization_image(result, timestamp)
            if image_file:
                self._open_result_file(image_file)
            else:
                self._open_result_file(result_file)
            
        except Exception as e:
            logger.error(f"自然语言图像生成失败: {e}")
            print(f"❌ 图像生成失败: {e}")
    
    def _parse_natural_language_to_indicators(self, description: str) -> Dict[str, float]:
        """将自然语言描述转换为环境指标"""
        # 默认指标
        indicators = {
            "co2_level": 400.0,
            "pm25_level": 50.0,
            "temperature": 25.0,
            "forest_coverage": 60.0,
            "water_quality": 7.0,
            "air_quality": 6.0
        }
        
        description_lower = description.lower()
        
        # 空气污染相关关键词
        if any(word in description_lower for word in ['雾霾', '空气污染', '烟雾', '灰尘', '颗粒物']):
            indicators['pm25_level'] = 120.0
            indicators['air_quality'] = 2.0
            indicators['co2_level'] = 450.0
        
        # 水污染相关关键词
        if any(word in description_lower for word in ['水污染', '河流污染', '海洋污染', '废水', '污水']):
            indicators['water_quality'] = 2.0
        
        # 森林砍伐相关关键词
        if any(word in description_lower for word in ['砍伐', '森林破坏', '树木减少', '荒漠化']):
            indicators['forest_coverage'] = 20.0
            indicators['co2_level'] = 430.0
        
        # 全球变暖相关关键词
        if any(word in description_lower for word in ['全球变暖', '气候变化', '温度上升', '冰川融化']):
            indicators['temperature'] = 35.0
            indicators['co2_level'] = 480.0
        
        # 极端天气相关关键词
        if any(word in description_lower for word in ['极端天气', '暴雨', '干旱', '台风', '洪水']):
            indicators['temperature'] = 38.0
            indicators['co2_level'] = 460.0
            indicators['air_quality'] = 4.0
        
        # 严重程度修饰词
        if any(word in description_lower for word in ['严重', '极度', '非常', '巨大']):
            # 加重所有负面指标
            indicators['pm25_level'] = min(200.0, indicators['pm25_level'] * 1.5)
            indicators['co2_level'] = min(500.0, indicators['co2_level'] * 1.2)
            indicators['temperature'] = min(45.0, indicators['temperature'] * 1.3)
            indicators['forest_coverage'] = max(10.0, indicators['forest_coverage'] * 0.5)
            indicators['water_quality'] = max(1.0, indicators['water_quality'] * 0.5)
            indicators['air_quality'] = max(1.0, indicators['air_quality'] * 0.5)
        
        # 轻微程度修饰词
        elif any(word in description_lower for word in ['轻微', '少量', '一点']):
            # 减轻负面指标
            indicators['pm25_level'] = max(30.0, indicators['pm25_level'] * 0.7)
            indicators['co2_level'] = max(380.0, indicators['co2_level'] * 0.9)
            indicators['temperature'] = max(22.0, indicators['temperature'] * 0.9)
            indicators['forest_coverage'] = min(80.0, indicators['forest_coverage'] * 1.2)
            indicators['water_quality'] = min(9.0, indicators['water_quality'] * 1.2)
            indicators['air_quality'] = min(8.0, indicators['air_quality'] * 1.2)
        
        return indicators
    
    def run(self):
        """运行交互式系统"""
        while True:
            self.show_menu()
            choice = input("请选择功能 (1-7): ").strip()
            
            if choice == '1':
                self.generate_image_demo()
            elif choice == '2':
                self.show_preset_scenarios()
            elif choice == '3':
                self.custom_indicators_demo()
            elif choice == '4':
                self.batch_generation_demo()
            elif choice == '5':
                self.natural_language_generation()
            elif choice == '6':
                self.show_usage_guide()
            elif choice == '7':
                print("\n👋 感谢使用生态警示图像生成系统！")
                print("🌍 让我们一起保护地球环境！")
                break
            else:
                print("\n❌ 无效选择，请重新输入！")
            
            input("\n按回车键继续...")


def main():
    """主函数"""
    try:
        system = InteractiveEcologyImageSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出系统")
    except Exception as e:
        logger.error(f"系统运行错误: {e}")
        print(f"\n❌ 系统运行错误: {e}")


if __name__ == "__main__":
    main()