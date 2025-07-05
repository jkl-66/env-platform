#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示API版本环境图像生成器的基本功能（无需真实Token）

这个脚本展示了API版本的代码结构和基本功能，
即使没有真实的HF_TOKEN也可以查看代码逻辑。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environmental_image_generator import EnvironmentalImageGenerator

def demo_initialization():
    """演示初始化过程"""
    print("🚀 演示API版本初始化...")
    
    # 使用虚拟token进行演示
    generator = EnvironmentalImageGenerator(hf_token="demo_token_placeholder")
    
    print("✅ 生成器初始化完成")
    print(f"📋 模型ID: {generator.model_id}")
    print(f"🔗 API URL: {generator.api_url}")
    
    return generator

def demo_prompt_enhancement():
    """演示提示词增强功能"""
    print("\n🎨 演示提示词增强功能...")
    
    generator = EnvironmentalImageGenerator(hf_token="demo_token")
    
    test_inputs = [
        "森林砍伐",
        "海洋污染",
        "城市雾霾",
        "垃圾堆积"
    ]
    
    for user_input in test_inputs:
        enhanced = generator.enhance_prompt(user_input)
        category = generator._detect_environmental_category(user_input)
        
        print(f"\n📝 原始输入: {user_input}")
        print(f"🏷️ 检测类别: {category}")
        print(f"✨ 增强提示: {enhanced[:100]}...")

def demo_environmental_categories():
    """演示环境类别功能"""
    print("\n🌍 演示环境类别功能...")
    
    generator = EnvironmentalImageGenerator(hf_token="demo_token")
    categories = generator.list_environmental_categories()
    
    print("支持的环境类别:")
    for category, description in categories.items():
        print(f"  🏷️ {category}: {description[:50]}...")

def demo_model_info():
    """演示模型信息获取"""
    print("\n📊 演示模型信息获取...")
    
    generator = EnvironmentalImageGenerator(hf_token="demo_token")
    info = generator.get_model_info()
    
    print("模型信息:")
    for key, value in info.items():
        if key == "environmental_categories":
            print(f"  {key}: {len(value)} 个类别")
        else:
            print(f"  {key}: {value}")

def demo_api_call_structure():
    """演示API调用结构（不实际调用）"""
    print("\n🔧 演示API调用结构...")
    
    generator = EnvironmentalImageGenerator(hf_token="demo_token")
    
    # 展示API调用会使用的参数
    test_prompt = generator.enhance_prompt("森林砍伐导致的环境破坏")
    
    print("API调用参数结构:")
    print(f"  🎯 目标URL: {generator.api_url}")
    print(f"  📝 增强提示词: {test_prompt[:80]}...")
    print(f"  🔑 认证方式: Bearer Token")
    print(f"  📊 请求格式: JSON POST")
    
    # 展示预期的响应结构
    print("\n预期响应结构:")
    print("  ✅ 成功时: 图像数据 (binary 或 base64)")
    print("  ❌ 失败时: 错误信息和状态码")

def demo_file_operations():
    """演示文件操作功能"""
    print("\n📁 演示文件操作功能...")
    
    generator = EnvironmentalImageGenerator(hf_token="demo_token")
    
    # 演示文件名生成
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_prompt = "森林砍伐"
    safe_prompt = "".join(c for c in test_prompt[:20] if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_prompt = safe_prompt.replace(" ", "_")
    
    filename = f"{safe_prompt}_{timestamp}_1_api.png"
    
    print(f"生成的文件名示例: {filename}")
    print(f"输出目录: outputs/environmental_images/")
    print(f"报告文件: generation_report_{timestamp}.json")

def main():
    """主演示函数"""
    print("🎭 API版本环境图像生成器功能演示")
    print("=" * 60)
    print("📌 注意: 这是功能演示，不会实际调用API")
    print("🔑 实际使用需要设置有效的HF_TOKEN")
    
    # 演示各个功能模块
    demo_initialization()
    demo_environmental_categories()
    demo_prompt_enhancement()
    demo_model_info()
    demo_api_call_structure()
    demo_file_operations()
    
    print("\n🎉 功能演示完成！")
    print("\n📖 使用说明:")
    print("1. 获取Hugging Face Token")
    print("2. 设置环境变量: set HF_TOKEN=your_token")
    print("3. 运行: python environmental_image_generator.py")
    print("4. 或运行测试: python test_api_generator.py")

if __name__ == "__main__":
    main()