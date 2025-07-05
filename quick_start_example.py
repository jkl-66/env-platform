#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境图像生成器快速开始示例
展示基本使用方法
"""

import sys
from pathlib import Path
from environmental_image_generator import EnvironmentalImageGenerator

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 初始化生成器 (API版本)
    generator = EnvironmentalImageGenerator()
    
    # 测试API连接
    if not generator.test_api_connection():
        print("❌ API连接失败，请检查HF_TOKEN环境变量设置")
        return
    
    print("✅ API连接成功！")
    
    # 中文自然语言输入
    user_input = "工厂烟囱冒出黑烟，城市被雾霾笼罩"
    print(f"\n用户输入: {user_input}")
    
    try:
        # 生成图像
        results = generator.generate_image(
            user_input=user_input
        )
        
        if results['success']:
            print(f"✅ 生成成功!")
            print(f"保存路径: {results['output_path']}")
            print(f"图像文件: {results['image_paths'][0]}")
            print(f"生成时间: {results['generation_time']:.2f} 秒")
            print(f"使用提示词: {results['prompt'][:100]}...")
        else:
            print(f"❌ 生成失败: {results['error']}")
            
    except Exception as e:
        print(f"❌ 发生错误: {e}")

def example_multiple_themes():
    """多主题生成示例"""
    print("\n=== 多主题生成示例 ===")
    
    generator = EnvironmentalImageGenerator()
    
    # 不同环境主题的输入
    themes = [
        "海洋中漂浮着大量塑料垃圾",
        "大片森林被砍伐，只剩下光秃秃的树桩", 
        "太阳能板和风力发电机的清洁能源景观",
        "河流被工业废水污染，鱼类死亡"
    ]
    
    for i, theme in enumerate(themes, 1):
        print(f"\n{i}. 生成主题: {theme}")
        
        try:
            results = generator.generate_image(
                user_input=theme
            )
            
            if results['success']:
                print(f"   ✅ 成功: {Path(results['image_paths'][0]).name}")
            else:
                print(f"   ❌ 失败: {results['error']}")
                
        except Exception as e:
            print(f"   ❌ 错误: {e}")

def example_custom_parameters():
    """自定义参数示例"""
    print("\n=== 自定义参数示例 ===")
    
    generator = EnvironmentalImageGenerator()
    
    user_input = "气候变化导致冰川融化，北极熊栖息地缩小"
    print(f"\n用户输入: {user_input}")
    
    # API版本生成示例
    print("\nAPI版本生成 (使用云端模型):")
    print("  模型: stabilityai/stable-diffusion-3.5-large-turbo")
    print("  无需本地GPU资源")
    print("  自动优化参数")
    
    try:
        results = generator.generate_image(
            user_input=user_input
        )
            
            if results['success']:
                print(f"  ✅ 成功，耗时: {results['generation_time']:.2f} 秒")
            else:
                print(f"  ❌ 失败: {results['error']}")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")

def example_english_input():
    """英文输入示例"""
    print("\n=== 英文输入示例 ===")
    
    generator = EnvironmentalImageGenerator()
    
    english_inputs = [
        "industrial air pollution with thick smog covering the city",
        "plastic waste floating in the ocean affecting marine life",
        "renewable energy landscape with solar panels and wind turbines",
        "deforestation showing cut down trees and environmental destruction"
    ]
    
    for i, english_input in enumerate(english_inputs, 1):
        print(f"\n{i}. English input: {english_input}")
        
        try:
            results = generator.generate_image(
                user_input=english_input
            )
            
            if results['success']:
                print(f"   ✅ Success: {Path(results['image_paths'][0]).name}")
            else:
                print(f"   ❌ Failed: {results['error']}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def show_project_info():
    """显示项目信息"""
    print("🌍 环境保护图像生成器 - API版本快速开始示例")
    print("基于 Hugging Face Inference API")
    print("=" * 60)
    
    print("\n📋 功能特点:")
    features = [
        "支持中文和英文自然语言输入",
        "专门针对环境保护主题优化",
        "10+ 种环境主题的内置模板",
        "云端生成，无需本地GPU",
        "自动提示词增强",
        "快速启动，无需模型下载"
    ]
    
    for feature in features:
        print(f"  • {feature}")
    
    print("\n⚠️  注意事项:")
    notes = [
        "需要设置 HF_TOKEN 环境变量",
        "需要稳定的网络连接",
        "生成时间取决于API响应速度",
        "请遵守Hugging Face使用条款"
    ]
    
    for note in notes:
        print(f"  • {note}")

def main():
    """主函数"""
    show_project_info()
    
    print("\n🚀 开始运行示例...")
    print("\n注意: 请确保已设置 HF_TOKEN 环境变量")
    
    try:
        # 运行示例
        example_basic_usage()
        
        # 询问是否继续运行更多示例
        print("\n" + "=" * 60)
        choice = input("是否运行更多示例？(y/n): ").strip().lower()
        
        if choice in ['y', 'yes', '是', '是的']:
            example_multiple_themes()
            example_custom_parameters()
            example_english_input()
        
        print("\n🎉 示例运行完成！")
        print("\n📖 更多信息:")
        print("  • 查看 README_Environmental_Generator.md 了解详细用法")
        print("  • 运行 python demo_environmental_generator.py 使用交互式界面")
        print("  • 编辑 config/environmental_prompts.json 自定义提示词模板")
        
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")
    except Exception as e:
        print(f"\n❌ 运行示例时发生错误: {e}")
        print("\n🔧 故障排除:")
        print("  1. 检查网络连接")
        print("  2. 确保有足够的磁盘空间 (至少 10GB)")
        print("  3. 检查 GPU 内存是否充足")
        print("  4. 运行 python verify_project_setup.py 检查项目设置")

if __name__ == "__main__":
    main()