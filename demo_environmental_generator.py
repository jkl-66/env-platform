#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境保护图像生成器演示脚本
基于 Stable Diffusion 3.5 Large Turbo 模型
支持自然语言输入和内置环境提示词模板
"""

import os
import sys
import json
from pathlib import Path
from environmental_image_generator import EnvironmentalImageGenerator

def load_config():
    """加载配置文件"""
    config_path = Path("config/environmental_prompts.json")
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return None

def display_available_prompts(config):
    """显示可用的环境提示词模板"""
    print("\n=== 可用的环境提示词模板 ===")
    prompts = config.get('environmental_prompts', {})
    for i, (key, value) in enumerate(prompts.items(), 1):
        print(f"{i:2d}. {key:20s} - {value.get('description', '无描述')}")
    print()

def display_style_presets(config):
    """显示可用的风格预设"""
    print("\n=== 可用的风格预设 ===")
    styles = config.get('style_presets', {})
    for i, (key, value) in enumerate(styles.items(), 1):
        print(f"{i}. {key:15s} - {value}")
    print()

def display_generation_settings(config):
    """显示生成设置选项"""
    print("\n=== 生成质量设置 ===")
    settings = config.get('generation_settings', {})
    for i, (key, value) in enumerate(settings.items(), 1):
        steps = value.get('num_inference_steps', 'N/A')
        size = f"{value.get('width', 'N/A')}x{value.get('height', 'N/A')}"
        num_images = value.get('num_images', 'N/A')
        print(f"{i}. {key:12s} - 步数:{steps:2s}, 尺寸:{size:8s}, 数量:{num_images}")
    print()

def get_user_choice(prompt, options):
    """获取用户选择"""
    while True:
        try:
            choice = input(prompt).strip()
            if choice == '':
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                return list(options.keys())[choice_num - 1]
            else:
                print(f"请输入 1-{len(options)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            return None

def demo_preset_generation(generator, config):
    """演示预设模板生成"""
    print("\n=== 预设模板生成演示 ===")
    
    # 显示可用选项
    display_available_prompts(config)
    display_style_presets(config)
    display_generation_settings(config)
    
    # 获取用户选择
    prompts = config.get('environmental_prompts', {})
    prompt_key = get_user_choice("选择环境主题 (输入数字，回车跳过): ", prompts)
    
    styles = config.get('style_presets', {})
    style_key = get_user_choice("选择风格预设 (输入数字，回车跳过): ", styles)
    
    settings = config.get('generation_settings', {})
    setting_key = get_user_choice("选择生成质量 (输入数字，回车使用默认): ", settings)
    
    if prompt_key:
        prompt_config = prompts[prompt_key]
        user_input = prompt_config['base_prompt']
        
        # 添加风格
        if style_key:
            style_suffix = styles[style_key]
            user_input += f", {style_suffix}"
        
        # 添加质量增强
        quality_enhancers = config.get('quality_enhancers', {})
        user_input += f", {quality_enhancers.get('base', '')}"
        user_input += f", {quality_enhancers.get('environmental', '')}"
        
        print(f"\n生成提示词: {user_input}")
        
        # 设置生成参数
        gen_settings = settings.get(setting_key or 'default', settings['default'])
        
        # 生成图像
        try:
            results = generator.generate_image(
                user_input=user_input,
                **gen_settings
            )
            
            if results['success']:
                print(f"\n✅ 图像生成成功!")
                print(f"保存路径: {results['output_path']}")
                for i, path in enumerate(results['image_paths'], 1):
                    print(f"  图像 {i}: {path}")
            else:
                print(f"\n❌ 图像生成失败: {results['error']}")
                
        except Exception as e:
            print(f"\n❌ 生成过程出错: {e}")
    else:
        print("未选择环境主题，跳过预设生成")

def demo_natural_language_generation(generator):
    """演示自然语言输入生成"""
    print("\n=== 自然语言输入生成演示 ===")
    print("请输入您想要生成的环境保护相关图像描述:")
    print("例如: '工厂排放黑烟污染空气的场景'")
    print("     '海洋中充满塑料垃圾的景象'")
    print("     '森林被大量砍伐的环境破坏'")
    
    try:
        user_input = input("\n输入描述 (回车跳过): ").strip()
        
        if user_input:
            print(f"\n正在生成图像: {user_input}")
            
            results = generator.generate_image(
                user_input=user_input,
                guidance_scale=7.5,
                num_inference_steps=28,
                height=1024,
                width=1024
            )
            
            if results['success']:
                print(f"\n✅ 图像生成成功!")
                print(f"保存路径: {results['output_path']}")
                for i, path in enumerate(results['image_paths'], 1):
                    print(f"  图像 {i}: {path}")
            else:
                print(f"\n❌ 图像生成失败: {results['error']}")
        else:
            print("未输入描述，跳过自然语言生成")
            
    except KeyboardInterrupt:
        print("\n用户取消输入")
    except Exception as e:
        print(f"\n❌ 生成过程出错: {e}")

def demo_batch_generation(generator, config):
    """演示批量生成"""
    print("\n=== 批量生成演示 ===")
    print("将生成多个不同主题的环境警示图像")
    
    # 选择几个代表性主题进行批量生成
    prompts = config.get('environmental_prompts', {})
    selected_themes = ['air_pollution', 'water_pollution', 'plastic_pollution']
    
    batch_results = []
    
    for theme in selected_themes:
        if theme in prompts:
            prompt_config = prompts[theme]
            user_input = prompt_config['base_prompt']
            
            # 添加风格和质量增强
            style = config.get('style_presets', {}).get('documentary', '')
            quality = config.get('quality_enhancers', {}).get('environmental', '')
            user_input += f", {style}, {quality}"
            
            print(f"\n正在生成: {prompt_config['description']}")
            
            try:
                results = generator.generate_image(
                    user_input=user_input,
                    guidance_scale=7.5,
                    num_inference_steps=25,
                    height=768,
                    width=768
                )
                
                batch_results.append({
                    'theme': theme,
                    'description': prompt_config['description'],
                    'results': results
                })
                
                if results['success']:
                    print(f"  ✅ 成功: {results['image_paths'][0]}")
                else:
                    print(f"  ❌ 失败: {results['error']}")
                    
            except Exception as e:
                print(f"  ❌ 错误: {e}")
                batch_results.append({
                    'theme': theme,
                    'description': prompt_config['description'],
                    'results': {'success': False, 'error': str(e)}
                })
    
    # 显示批量生成总结
    print("\n=== 批量生成总结 ===")
    successful = sum(1 for r in batch_results if r['results']['success'])
    total = len(batch_results)
    print(f"成功生成: {successful}/{total} 张图像")
    
    for result in batch_results:
        status = "✅" if result['results']['success'] else "❌"
        print(f"  {status} {result['description']}")

def main():
    """主函数"""
    print("🌍 环境保护图像生成器演示")
    print("基于 Stable Diffusion 3.5 Large Turbo 模型")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    if not config:
        print("无法加载配置文件，程序退出")
        return
    
    # 初始化生成器
    print("\n正在初始化图像生成器...")
    try:
        generator = EnvironmentalImageGenerator(
            model_id="stabilityai/stable-diffusion-3.5-large-turbo",
            device="auto"
        )
        print("✅ 生成器初始化成功")
    except Exception as e:
        print(f"❌ 生成器初始化失败: {e}")
        return
    
    # 显示菜单
    while True:
        print("\n=== 演示菜单 ===")
        print("1. 预设模板生成")
        print("2. 自然语言输入生成")
        print("3. 批量生成演示")
        print("4. 查看配置信息")
        print("5. 退出")
        
        try:
            choice = input("\n请选择 (1-5): ").strip()
            
            if choice == '1':
                demo_preset_generation(generator, config)
            elif choice == '2':
                demo_natural_language_generation(generator)
            elif choice == '3':
                demo_batch_generation(generator, config)
            elif choice == '4':
                display_available_prompts(config)
                display_style_presets(config)
                display_generation_settings(config)
            elif choice == '5':
                print("\n感谢使用环境保护图像生成器！")
                break
            else:
                print("请输入有效的选项 (1-5)")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    main()