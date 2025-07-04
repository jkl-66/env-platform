#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境图像生成器测试脚本
测试基本功能和模型加载
"""

import os
import sys
import json
from pathlib import Path
from environmental_image_generator import EnvironmentalImageGenerator

def test_model_loading():
    """测试模型加载"""
    print("=== 测试模型加载 ===")
    try:
        generator = EnvironmentalImageGenerator(
            model_id="stabilityai/stable-diffusion-3.5-large-turbo",
            device="auto"
        )
        print("✅ 模型加载成功")
        print(f"   模型ID: {generator.model_id}")
        print(f"   设备: {generator.device}")
        return generator
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def test_config_loading():
    """测试配置文件加载"""
    print("\n=== 测试配置文件加载 ===")
    config_path = Path("config/environmental_prompts.json")
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("✅ 配置文件加载成功")
        
        # 检查配置结构
        prompts = config.get('environmental_prompts', {})
        styles = config.get('style_presets', {})
        settings = config.get('generation_settings', {})
        
        print(f"   环境提示词模板: {len(prompts)} 个")
        print(f"   风格预设: {len(styles)} 个")
        print(f"   生成设置: {len(settings)} 个")
        
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return None

def test_simple_generation(generator):
    """测试简单图像生成"""
    print("\n=== 测试简单图像生成 ===")
    
    test_prompt = "industrial air pollution, smoggy city skyline, environmental warning, photorealistic"
    print(f"测试提示词: {test_prompt}")
    
    try:
        results = generator.generate_image(
            user_input=test_prompt,
            guidance_scale=7.5,
            num_inference_steps=20,  # 使用较少步数加快测试
            height=512,  # 使用较小尺寸加快测试
            width=512
        )
        
        if results['success']:
            print("✅ 图像生成成功")
            print(f"   保存路径: {results['output_path']}")
            print(f"   图像文件: {results['image_paths'][0]}")
            print(f"   生成时间: {results.get('generation_time', 'N/A')} 秒")
            return True
        else:
            print(f"❌ 图像生成失败: {results['error']}")
            return False
            
    except Exception as e:
        print(f"❌ 生成过程出错: {e}")
        return False

def test_prompt_enhancement(generator):
    """测试提示词增强功能"""
    print("\n=== 测试提示词增强功能 ===")
    
    # 测试中文输入
    chinese_input = "工厂排放黑烟污染空气"
    print(f"中文输入: {chinese_input}")
    
    try:
        enhanced_prompt = generator._enhance_environmental_prompt(chinese_input)
        print(f"增强后提示词: {enhanced_prompt}")
        
        # 测试英文输入
        english_input = "plastic pollution in ocean"
        print(f"\n英文输入: {english_input}")
        enhanced_prompt = generator._enhance_environmental_prompt(english_input)
        print(f"增强后提示词: {enhanced_prompt}")
        
        print("✅ 提示词增强功能正常")
        return True
        
    except Exception as e:
        print(f"❌ 提示词增强失败: {e}")
        return False

def test_output_directory():
    """测试输出目录创建"""
    print("\n=== 测试输出目录 ===")
    
    output_dir = Path("outputs/environmental_images")
    
    if output_dir.exists():
        print(f"✅ 输出目录存在: {output_dir}")
        
        # 检查目录内容
        files = list(output_dir.glob("*.png"))
        print(f"   PNG 文件数量: {len(files)}")
        
        if files:
            print("   最近的文件:")
            for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"     {file.name} ({size_mb:.2f} MB)")
        
        return True
    else:
        print(f"❌ 输出目录不存在: {output_dir}")
        return False

def run_comprehensive_test():
    """运行综合测试"""
    print("🧪 环境图像生成器综合测试")
    print("=" * 50)
    
    test_results = []
    
    # 测试配置文件加载
    config = test_config_loading()
    test_results.append(("配置文件加载", config is not None))
    
    # 测试模型加载
    generator = test_model_loading()
    test_results.append(("模型加载", generator is not None))
    
    if generator:
        # 测试提示词增强
        prompt_test = test_prompt_enhancement(generator)
        test_results.append(("提示词增强", prompt_test))
        
        # 测试图像生成
        generation_test = test_simple_generation(generator)
        test_results.append(("图像生成", generation_test))
        
        # 测试输出目录
        output_test = test_output_directory()
        test_results.append(("输出目录", output_test))
    
    # 显示测试总结
    print("\n" + "=" * 50)
    print("📊 测试总结")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15s}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！环境图像生成器工作正常")
    else:
        print("⚠️  部分测试失败，请检查相关配置")
    
    return passed == total

def main():
    """主函数"""
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程发生未预期错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()