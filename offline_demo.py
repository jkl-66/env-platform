#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境保护图像生成器 - API版本演示
展示API版本的功能特性
"""

import os
import json
from datetime import datetime
from pathlib import Path

def load_config():
    """加载配置文件"""
    config_path = Path("config/environmental_prompts.json")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def enhance_prompt(user_input, config):
    """演示提示词增强功能"""
    if not config:
        return user_input
    
    # 检测环境主题
    detected_category = None
    for category, data in config['environmental_prompts'].items():
        keywords = data.get('keywords', [])
        if any(keyword in user_input.lower() for keyword in keywords):
            detected_category = category
            break
    
    if detected_category:
        theme_data = config['environmental_prompts'][detected_category]
        enhanced = f"{user_input}, {theme_data['style_suffix']}"
        return enhanced, detected_category
    
    return user_input, None

def simulate_api_generation(prompt, category=None):
    """模拟API图像生成过程"""
    print(f"🎨 模拟API生成图像...")
    print(f"📝 增强后的提示词: {prompt}")
    if category:
        print(f"🏷️  检测到的环境主题: {category}")
    
    # API生成信息
    api_info = {
        "model_id": "stabilityai/stable-diffusion-3.5-large-turbo",
        "api_url": "https://api-inference.huggingface.co/models/",
        "method": "POST",
        "content_type": "application/json"
    }
    
    print(f"🌐 API信息: {api_info['model_id']}")
    print(f"⏱️  预计生成时间: 5-15秒 (云端处理)")
    print(f"💾 输出路径: outputs/environmental_images/")
    print(f"☁️  无需本地GPU资源")
    
    return {
        "status": "success",
        "prompt": prompt,
        "category": category,
        "api_info": api_info,
        "timestamp": datetime.now().isoformat()
    }

def main():
    print("="*60)
    print("🌍 环境保护图像生成器 - API版本功能演示")
    print("="*60)
    print()
    
    # 加载配置
    print("📋 正在加载配置文件...")
    config = load_config()
    if config:
        print("✅ 配置文件加载成功")
        print(f"📊 支持的环境主题数量: {len(config['environmental_prompts'])}")
        print(f"🎨 可用风格预设: {len(config['style_presets'])}")
    else:
        print("❌ 配置文件加载失败")
        return
    
    print()
    print("🎯 支持的环境主题:")
    for i, (category, data) in enumerate(config['environmental_prompts'].items(), 1):
        print(f"  {i:2d}. {category} - {data['description']}")
    
    print()
    print("🚀 开始功能演示...")
    print()
    
    # 演示案例
    test_cases = [
        "工厂烟囱冒出黑烟，城市被雾霾笼罩",
        "河流被工业废水污染，鱼类死亡",
        "森林被大规模砍伐，动物失去家园",
        "海洋中漂浮着大量塑料垃圾",
        "A polluted city with smog and industrial waste"
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"📝 示例 {i}: {user_input}")
        print("-" * 50)
        
        # 提示词增强
        enhanced_prompt, category = enhance_prompt(user_input, config)
        
        # 模拟API生成
        result = simulate_api_generation(enhanced_prompt, category)
        
        print(f"✅ 模拟生成完成")
        print()
    
    print("🎉 API版本演示完成！")
    print()
    print("📌 实际使用说明:")
    print("  1. 获取Hugging Face Token")
    print("  2. 设置HF_TOKEN环境变量")
    print("  3. 确保网络连接正常")
    print("  4. 运行 environmental_image_generator.py 进行实际生成")
    print()
    print("🔧 项目文件结构:")
    print("  • environmental_image_generator.py - 核心生成器 (API版本)")
    print("  • config/environmental_prompts.json - 配置文件")
    print("  • test_api_generator.py - API功能测试")
    print("  • demo_without_token.py - 无Token演示")
    print("  • API_USAGE_GUIDE.md - API使用指南")
    print("  • README_Environmental_Generator.md - 详细文档")

if __name__ == "__main__":
    main()