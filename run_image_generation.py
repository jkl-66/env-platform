#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境图像生成演示脚本

运行环境保护警示图像生成器，生成示例图像
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from environmental_image_generator import EnvironmentalImageGenerator

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查HF_TOKEN
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print(f"✅ HF_TOKEN已设置: {hf_token[:10]}...")
    else:
        print("⚠️ HF_TOKEN未设置，将使用免费API（可能有限制）")
        print("💡 如需设置Token，请运行: set HF_TOKEN=your_token_here")
    
    # 检查输出目录
    output_dir = Path("outputs/environmental_images")
    if not output_dir.exists():
        print(f"📁 创建输出目录: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"📁 输出目录已存在: {output_dir}")
    
    return True

def run_demo_generation():
    """运行演示图像生成"""
    print("\n🌍 启动环境保护警示图像生成器")
    print("=" * 50)
    
    # 初始化生成器
    try:
        generator = EnvironmentalImageGenerator()
        print("✅ 生成器初始化成功")
    except Exception as e:
        print(f"❌ 生成器初始化失败: {e}")
        return False
    
    # 测试API连接
    print("\n🔗 测试API连接...")
    connection_result = generator.test_api_connection()
    
    if not connection_result["success"]:
        print(f"❌ API连接失败: {connection_result.get('message', '未知错误')}")
        print("💡 请检查网络连接和HF_TOKEN设置")
        return False
    
    print("✅ API连接成功！")
    
    # 演示场景列表
    demo_scenarios = [
        {
            "description": "工厂排放污染空气",
            "category": "air_pollution"
        },
        {
            "description": "海洋塑料垃圾污染",
            "category": "plastic_pollution"
        },
        {
            "description": "太阳能发电清洁能源",
            "category": "renewable_energy"
        }
    ]
    
    print("\n🎨 开始生成演示图像...")
    
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"\n--- 场景 {i}: {scenario['description']} ---")
        
        try:
            # 生成图像
            result = generator.generate_and_save(
                user_input=scenario["description"],
                category=scenario["category"],
                width=512,  # 使用较小尺寸以加快生成速度
                height=512,
                num_inference_steps=20  # 减少推理步数以加快速度
            )
            
            if result["success"]:
                print(f"✅ 生成成功！")
                print(f"📁 保存位置: {result['saved_files'][0]}")
                print(f"🏷️  检测类别: {result['category']}")
                print(f"⏱️  生成时间: {result.get('generation_time', 'N/A'):.2f}秒")
                print(f"📝 增强提示词: {result['enhanced_prompt'][:100]}...")
            else:
                print(f"❌ 生成失败: {result.get('error', '未知错误')}")
                
        except Exception as e:
            print(f"❌ 生成过程中发生错误: {e}")
            continue
    
    print("\n🎉 演示完成！")
    print(f"📁 所有生成的图像保存在: outputs/environmental_images/")
    return True

def run_interactive_mode():
    """运行交互模式"""
    print("\n🎮 进入交互模式")
    print("请输入您想要生成的环境场景描述（输入 'quit' 退出）:")
    
    generator = EnvironmentalImageGenerator()
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                break
            
            if not user_input:
                print("❌ 输入不能为空")
                continue
            
            print(f"\n🎨 正在生成图像: {user_input}")
            
            # 生成图像
            result = generator.generate_and_save(
                user_input=user_input,
                width=512,
                height=512,
                num_inference_steps=20
            )
            
            if result["success"]:
                print(f"✅ 生成成功！")
                print(f"📁 保存位置: {result['saved_files'][0]}")
                print(f"🏷️  检测类别: {result['category']}")
                print(f"⏱️  生成时间: {result.get('generation_time', 'N/A'):.2f}秒")
            else:
                print(f"❌ 生成失败: {result.get('error', '未知错误')}")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户取消操作")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
    
    print("\n感谢使用环境保护警示图像生成器！")

def main():
    """主函数"""
    print("🌍 环境保护警示图像生成器演示")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        return
    
    # 选择运行模式
    print("\n请选择运行模式:")
    print("1. 演示模式 (生成预设场景图像)")
    print("2. 交互模式 (自定义输入)")
    print("3. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-3): ").strip()
            
            if choice == '1':
                run_demo_generation()
                break
            elif choice == '2':
                run_interactive_mode()
                break
            elif choice == '3':
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请输入 1、2 或 3")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户取消操作")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            break

if __name__ == "__main__":
    main()