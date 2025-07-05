#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试API版本的环境保护警示图像生成器

这个脚本用于测试基于Hugging Face Inference API的环境图像生成器功能。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from environmental_image_generator import EnvironmentalImageGenerator

def test_api_connection():
    """测试API连接"""
    print("🔗 测试API连接...")
    
    generator = EnvironmentalImageGenerator()
    
    if generator.test_api_connection():
        print("✅ API连接成功！")
        return True
    else:
        print("❌ API连接失败")
        print("💡 请检查:")
        print("   1. 网络连接是否正常")
        print("   2. HF_TOKEN环境变量是否设置")
        print("   3. Token是否有效")
        return False

def test_image_generation():
    """测试图像生成"""
    print("\n🎨 测试图像生成...")
    
    generator = EnvironmentalImageGenerator()
    
    # 测试用例
    test_cases = [
        "森林砍伐导致的环境破坏",
        "海洋塑料污染",
        "城市空气污染"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {test_input}")
        
        try:
            result = generator.generate_and_save(
                user_input=test_input,
                output_dir="outputs/test_api_images"
            )
            
            if result["success"]:
                print(f"✅ 生成成功！")
                print(f"📁 保存位置: {result['saved_files'][0]}")
                print(f"🏷️ 类别: {result['category']}")
                print(f"⏱️ 生成时间: {result.get('generation_time', 'N/A')}秒")
            else:
                print(f"❌ 生成失败: {result.get('error', '未知错误')}")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            
        print("-" * 50)

def test_model_info():
    """测试模型信息获取"""
    print("\n📊 测试模型信息获取...")
    
    generator = EnvironmentalImageGenerator()
    info = generator.get_model_info()
    
    print("模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

def main():
    """主测试函数"""
    print("🧪 API版本环境图像生成器测试")
    print("=" * 60)
    
    # 检查HF_TOKEN
    if not os.getenv('HF_TOKEN'):
        print("⚠️ 警告: 未设置HF_TOKEN环境变量")
        print("💡 请设置HF_TOKEN以使用Hugging Face Inference API")
        print("   例如: set HF_TOKEN=your_token_here")
        return
    
    # 测试API连接
    if not test_api_connection():
        return
    
    # 测试模型信息
    test_model_info()
    
    # 测试图像生成
    test_image_generation()
    
    print("\n🎉 测试完成！")
    print("📁 生成的图像保存在: outputs/test_api_images/")

if __name__ == "__main__":
    main()