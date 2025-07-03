#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版交互式生态警示图像生成系统

修复了warning_level错误，添加了更好的错误处理和模型支持检查。
"""

import os
import sys
import json
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

class ImprovedInteractiveEcologyImageSystem:
    """改进版交互式生态图像生成系统"""
    
    def __init__(self):
        """初始化系统"""
        self.generator = None
        self.output_dir = Path("outputs/improved_interactive_ecology_demo")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🌍 欢迎使用改进版生态警示图像生成系统！")
        print("=" * 50)
        
        # 检查依赖和初始化模型
        self._check_dependencies()
        self._initialize_model()
    
    def _check_dependencies(self):
        """检查依赖库"""
        print("\n🔍 检查系统依赖...")
        
        # 检查基础依赖
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
        except ImportError:
            print("❌ PyTorch 未安装")
            return False
        
        try:
            import numpy
            print(f"✅ NumPy: {numpy.__version__}")
        except ImportError:
            print("❌ NumPy 未安装")
            return False
        
        # 检查Hugging Face库
        try:
            import transformers
            print(f"✅ Transformers: {transformers.__version__}")
        except ImportError:
            print("⚠️  Transformers 未安装 - 扩散模型功能将不可用")
            print("   安装命令: pip install transformers")
        
        try:
            import diffusers
            print(f"✅ Diffusers: {diffusers.__version__}")
        except ImportError:
            print("⚠️  Diffusers 未安装 - 扩散模型功能将不可用")
            print("   安装命令: pip install diffusers")
        
        return True
    
    def _initialize_model(self):
        """初始化模型"""
        print("\n🤖 初始化生态图像生成模型...")
        
        try:
            self.generator = EcologyImageGenerator()
            print("✅ 模型初始化成功")
            
            # 检查可用的生成模式
            print("\n📋 可用的生成模式:")
            print("• GAN模式: ✅ 可用 (快速生成)")
            
            # 检查扩散模型是否可用
            try:
                from diffusers import StableDiffusionPipeline
                print("• 扩散模式: ✅ 可用 (高质量生成)")
                print("• 混合模式: ✅ 可用 (平衡质量与速度)")
            except ImportError:
                print("• 扩散模式: ❌ 不可用 (需要安装diffusers库)")
                print("• 混合模式: ❌ 不可用 (需要安装diffusers库)")
                print("\n💡 提示: 安装Hugging Face库以启用所有功能:")
                print("   pip install transformers diffusers accelerate")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            print(f"❌ 模型初始化失败: {e}")
            print("\n🔧 可能的解决方案:")
            print("1. 检查Python环境和依赖库")
            print("2. 确保有足够的内存和存储空间")
            print("3. 检查网络连接（如需下载模型）")
            return False
        
        return True
    
    def run(self):
        """运行主程序"""
        if self.generator is None:
            print("\n❌ 系统初始化失败，无法继续")
            return
        
        while True:
            try:
                self.show_menu()
                choice = input("\n请选择功能 (1-6): ").strip()
                
                if choice == '1':
                    self.generate_single_image()
                elif choice == '2':
                    self.show_preset_scenarios()
                elif choice == '3':
                    self.custom_indicators_demo()
                elif choice == '4':
                    self.batch_generation_demo()
                elif choice == '5':
                    self.show_usage_guide()
                elif choice == '6':
                    print("\n👋 感谢使用生态警示图像生成系统！")
                    break
                else:
                    print("\n❌ 无效选择，请重新输入")
                
                input("\n按回车键继续...")
                
            except KeyboardInterrupt:
                print("\n\n👋 程序已退出")
                break
            except Exception as e:
                logger.error(f"程序运行错误: {e}")
                print(f"\n❌ 发生错误: {e}")
                print("程序将继续运行...")
    
    def show_menu(self):
        """显示主菜单"""
        print("\n" + "=" * 50)
        print("📋 请选择功能：")
        print("1. 🎨 生成环境警示图像")
        print("2. 📊 查看预设环境场景")
        print("3. 🔧 自定义环境指标")
        print("4. 📈 批量生成对比图像")
        print("5. 📚 查看使用指南")
        print("6. 🚪 退出系统")
        print("-" * 30)
    
    def get_environmental_indicators(self) -> Dict[str, float]:
        """获取环境指标输入"""
        print("\n🌡️ 请输入环境指标（直接回车使用默认值）：")
        
        indicators = {}
        
        # 定义指标配置
        indicator_configs = [
            ('co2_level', 'CO2排放量 (ppm)', 400, 350, 500),
            ('pm25_level', 'PM2.5浓度 (μg/m³)', 50, 0, 300),
            ('temperature', '温度变化 (°C)', 25, 15, 50),
            ('forest_coverage', '森林覆盖率 (%)', 60, 0, 100),
            ('water_quality', '水质指数 (1-10)', 7, 1, 10),
            ('air_quality', '空气质量指数 (1-10)', 6, 1, 10)
        ]
        
        for key, name, default, min_val, max_val in indicator_configs:
            while True:
                try:
                    user_input = input(f"{name} (默认{default}): ").strip()
                    if not user_input:
                        indicators[key] = float(default)
                        break
                    
                    value = float(user_input)
                    if min_val <= value <= max_val:
                        indicators[key] = value
                        break
                    else:
                        print(f"⚠️  值应在 {min_val}-{max_val} 范围内")
                        
                except ValueError:
                    print("⚠️  请输入有效的数字")
        
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
        
        # 检查扩散模型是否可用
        diffusion_available = True
        try:
            from diffusers import StableDiffusionPipeline
            print("2. 扩散模式 (高质量)")
            print("3. 混合模式 (平衡质量与速度)")
        except ImportError:
            diffusion_available = False
            print("2. 扩散模式 (不可用 - 需要安装diffusers)")
            print("3. 混合模式 (不可用 - 需要安装diffusers)")
        
        while True:
            mode_choice = input("请选择模式 (1-3, 默认1): ").strip()
            if not mode_choice or mode_choice == '1':
                generation_mode = 'gan'
                break
            elif mode_choice in ['2', '3'] and diffusion_available:
                generation_mode = 'diffusion' if mode_choice == '2' else 'hybrid'
                break
            elif mode_choice in ['2', '3'] and not diffusion_available:
                print("⚠️  该模式不可用，请安装diffusers库或选择GAN模式")
            else:
                print("⚠️  请选择有效的模式")
        
        # 设置生成模式
        try:
            self.generator.set_generation_mode(generation_mode)
        except Exception as e:
            print(f"⚠️  设置生成模式失败，使用默认GAN模式: {e}")
            generation_mode = 'gan'
            self.generator.set_generation_mode('gan')
        
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
        while True:
            try:
                num_images_input = input("\n生成图像数量 (1-3, 默认1): ").strip()
                if not num_images_input:
                    num_images = 1
                    break
                num_images = int(num_images_input)
                if 1 <= num_images <= 3:
                    break
                else:
                    print("⚠️  数量应在1-3之间")
            except ValueError:
                print("⚠️  请输入有效的数字")
        
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
            
            # 显示主要环境问题
            concerns = result['environmental_assessment']['primary_concerns']
            print(f"\n🚨 主要环境问题:")
            for concern in concerns:
                print(f"  • {concern}")
            
            # 显示改善建议
            recommendations = result['environmental_assessment']['recommendations']
            print(f"\n💡 改善建议:")
            for rec in recommendations:
                print(f"  • {rec}")
            
            # 显示生成的图像信息
            print(f"\n📸 生成的图像信息:")
            for i, img_info in enumerate(result['generated_images'], 1):
                print(f"  图像 {i}:")
                print(f"    - 描述: {img_info['description']}")
                print(f"    - 风格: {img_info['style']}")
                print(f"    - 质量评分: {img_info['quality_score']:.2f}")
                print(f"    - 生成时间: {img_info['generation_time']:.1f}秒")
            
        except Exception as e:
            logger.error(f"图像生成失败: {e}")
            print(f"❌ 图像生成失败: {e}")
            print("\n🔧 可能的解决方案:")
            print("1. 检查网络连接")
            print("2. 确保有足够的内存")
            print("3. 尝试使用GAN模式")
            if 'diffusion' in str(e).lower():
                print("4. 安装Hugging Face库: pip install transformers diffusers")
    
    def show_preset_scenarios(self):
        """显示预设环境场景"""
        print("\n📊 预设环境场景模板")
        print("=" * 30)
        
        try:
            templates = self.generator.get_condition_templates()
            
            for i, (name, template) in enumerate(templates.items(), 1):
                print(f"\n{i}. {name}")
                print(f"   描述: {template['description']}")
                print(f"   警示等级: {template['warning_level']}/5")
                print(f"   视觉元素: {', '.join(template['visual_elements'])}")
                print(f"   色彩方案: {', '.join(template['color_scheme'])}")
            
            # 让用户选择模板生成图像
            choice = input(f"\n选择模板生成图像 (1-{len(templates)}, 回车跳过): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(templates):
                template_name = list(templates.keys())[int(choice) - 1]
                template = templates[template_name]
                
                print(f"\n🎨 使用模板 '{template_name}' 生成图像...")
                
                # 根据模板生成合适的环境指标
                indicators = self._generate_indicators_from_template(template_name, template)
                
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
                    
                except Exception as e:
                    print(f"❌ 模板图像生成失败: {e}")
                    
        except Exception as e:
            logger.error(f"获取模板失败: {e}")
            print(f"❌ 获取模板失败: {e}")
    
    def _generate_indicators_from_template(self, template_name: str, template: Dict[str, Any]) -> Dict[str, float]:
        """根据模板生成对应的环境指标"""
        # 基础指标
        indicators = {
            "co2_level": 400.0,
            "pm25_level": 50.0,
            "temperature": 25.0,
            "forest_coverage": 60.0,
            "water_quality": 7.0,
            "air_quality": 6.0
        }
        
        # 从模板中更新指标
        for key, value in template.items():
            if key in indicators and isinstance(value, (int, float)):
                indicators[key] = float(value)
        
        return indicators
    
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
                print(f"✅ '{scenario_name}' 生成完成 - 警示等级: {result['warning_level']}/5")
                
            except Exception as e:
                logger.error(f"'{scenario_name}' 生成失败: {e}")
                print(f"❌ '{scenario_name}' 生成失败: {e}")
        
        # 保存批量结果
        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_file = self.output_dir / f"batch_generation_{timestamp}.json"
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n📁 批量生成结果已保存至: {batch_file}")
            
            # 显示对比分析
            print("\n📊 对比分析:")
            for name, result in results.items():
                risk = result['environmental_assessment']['overall_risk']
                print(f"• {name}: 警示等级 {result['warning_level']}/5, 风险评估: {risk}")
    
    def show_usage_guide(self):
        """显示使用指南"""
        print("\n📚 生态警示图像生成系统使用指南")
        print("=" * 50)
        
        print("\n🎯 系统功能:")
        print("• 根据环境指标生成警示图像")
        print("• 支持多种生成模式和图像风格")
        print("• 提供环境风险评估和改善建议")
        print("• 支持批量生成和对比分析")
        
        print("\n🤖 生成模式:")
        print("• GAN模式: 快速生成，适合实时应用")
        print("• 扩散模式: 高质量生成，需要更多时间")
        print("• 混合模式: 平衡质量与速度")
        
        print("\n🎭 图像风格:")
        print("• 写实风格: 真实感强，适合科学展示")
        print("• 艺术风格: 艺术化表现，适合教育宣传")
        print("• 科幻风格: 未来感强，适合警示展示")
        print("• 教育风格: 简洁明了，适合教学使用")
        
        print("\n📊 环境指标:")
        print("• CO2排放量: 影响全球变暖")
        print("• PM2.5浓度: 影响空气质量")
        print("• 温度变化: 反映气候变化")
        print("• 森林覆盖率: 影响生态平衡")
        print("• 水质指数: 反映水环境状况")
        print("• 空气质量指数: 反映大气环境")
        
        print("\n💡 使用建议:")
        print("• 首次使用建议从预设场景开始")
        print("• 根据实际需求选择合适的生成模式")
        print("• 注意环境指标的合理范围")
        print("• 可以通过批量生成进行对比分析")
        
        print("\n🔧 故障排除:")
        print("• 如果扩散模式不可用，请安装: pip install transformers diffusers")
        print("• 如果生成失败，请检查网络连接和内存")
        print("• 如果结果不理想，请调整环境指标")
        
        print("\n📁 输出文件:")
        print(f"• 结果保存在: {self.output_dir}")
        print("• JSON格式包含完整的生成信息")
        print("• 可用于后续分析和展示")


def main():
    """主函数"""
    try:
        system = ImprovedInteractiveEcologyImageSystem()
        system.run()
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        print(f"❌ 系统启动失败: {e}")
        print("\n🔧 请检查:")
        print("1. Python环境和依赖库")
        print("2. 系统权限和存储空间")
        print("3. 网络连接")


if __name__ == "__main__":
    main()