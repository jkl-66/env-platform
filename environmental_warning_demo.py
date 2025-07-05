#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境警示图像生成器演示脚本

使用阿里云 DashScope 服务生成专业的环境警示图像
支持教育者、家长等用户输入环境数据，自动生成警示图像
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dashscope_environmental_generator import DashScopeEnvironmentalGenerator

def interactive_demo():
    """
    交互式演示
    """
    print("🌍 DashScope 环境警示图像生成器 - 交互式演示")
    print("=" * 50)
    
    try:
        # 初始化生成器
        generator = DashScopeEnvironmentalGenerator()
        
        # 测试连接
        print("\n🔗 测试 DashScope 连接...")
        connection_result = generator.test_connection()
        
        if not connection_result["success"]:
            print("❌ DashScope 连接失败，请检查配置")
            print(f"聊天模型状态: {connection_result['chat_model_status']}")
            print(f"图像模型状态: {connection_result['image_model_status']}")
            return
        
        print("✅ DashScope 连接成功")
        print(f"聊天模型: {connection_result['chat_model']}")
        print(f"图像模型: {connection_result['image_model']}")
        
        # 显示支持的数据类型
        print("\n📊 支持的环境数据类型:")
        data_types = generator.get_supported_data_types()
        for i, (data_type, config) in enumerate(data_types.items(), 1):
            print(f"  {i}. {config['name']} ({config['unit']})")
        
        while True:
            print("\n" + "=" * 50)
            print("请选择操作:")
            print("1. 使用预设示例数据生成图像")
            print("2. 手动输入环境数据")
            print("3. 退出")
            
            choice = input("\n请输入选择 (1-3): ").strip()
            
            if choice == "1":
                demo_with_preset_data(generator)
            elif choice == "2":
                demo_with_custom_data(generator)
            elif choice == "3":
                break
            else:
                print("❌ 无效选择，请重新输入")
        
        print("\n👋 感谢使用环境保护警示图像生成器！")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")

def demo_with_preset_data(generator):
    """
    使用预设数据演示
    """
    print("\n🎯 预设示例场景:")
    
    scenarios = {
        "1": {
            "name": "工业污染严重区域",
            "data": {
                "carbon_emission": 1800,  # 高碳排放
                "air_quality_index": 220,  # 不健康
                "water_pollution_index": 85  # 重度污染
            },
            "description": "重工业区域，工厂排放大量污染物，严重影响空气和水质",
            "audience": "educators"
        },
        "2": {
            "name": "城市交通污染",
            "data": {
                "carbon_emission": 600,
                "air_quality_index": 140,
                "noise_level": 85
            },
            "description": "城市中心交通繁忙，汽车尾气和噪音污染严重",
            "audience": "parents"
        },
        "3": {
            "name": "森林砍伐危机",
            "data": {
                "deforestation_rate": 18000,
                "carbon_emission": 900
            },
            "description": "大规模森林砍伐，生态系统遭到严重破坏",
            "audience": "students"
        },
        "4": {
            "name": "海洋塑料污染",
            "data": {
                "plastic_waste": 1500,
                "water_pollution_index": 70
            },
            "description": "海洋中大量塑料垃圾，威胁海洋生物生存",
            "audience": "general"
        }
    }
    
    for key, scenario in scenarios.items():
        print(f"  {key}. {scenario['name']}")
        for data_type, value in scenario['data'].items():
            data_config = generator.get_supported_data_types()[data_type]
            print(f"     - {data_config['name']}: {value} {data_config['unit']}")
    
    choice = input("\n请选择场景 (1-4): ").strip()
    
    if choice in scenarios:
        scenario = scenarios[choice]
        print(f"\n🎨 正在生成 '{scenario['name']}' 的警示图像...")
        
        result = generator.generate_environmental_warning_image(
            environmental_data=scenario['data'],
            user_description=scenario['description'],
            target_audience=scenario['audience'],
            auto_open=True
        )
        
        display_result(result, scenario['name'])
    else:
        print("❌ 无效选择")

def demo_with_custom_data(generator):
    """
    使用自定义数据演示
    """
    print("\n📝 请输入环境数据:")
    
    data_types = generator.get_supported_data_types()
    environmental_data = {}
    
    print("\n可选的环境数据类型 (输入数值，留空使用默认值):")
    
    for data_type, config in data_types.items():
        default_value = config.get('default_value', 0)
        while True:
            try:
                value_input = input(f"  {config['name']} ({config['unit']}) [默认: {default_value}]: ").strip()
                if not value_input:
                    environmental_data[data_type] = default_value
                    break
                
                value = float(value_input)
                if value < 0:
                    print("    ❌ 数值不能为负数，请重新输入")
                    continue
                
                environmental_data[data_type] = value
                break
                
            except ValueError:
                print("    ❌ 请输入有效的数字")
    
    # 获取用户描述
    user_description = input("\n请描述具体的环境情况 (可选): ").strip()
    if not user_description:
        user_description = None
    
    # 选择目标受众
    print("\n目标受众:")
    audiences = {
        "1": "general",
        "2": "educators", 
        "3": "parents",
        "4": "students"
    }
    
    for key, audience in audiences.items():
        audience_names = {
            "general": "一般公众",
            "educators": "教育工作者",
            "parents": "家长",
            "students": "学生"
        }
        print(f"  {key}. {audience_names[audience]}")
    
    audience_choice = input("请选择目标受众 (1-4, 默认为1): ").strip() or "1"
    target_audience = audiences.get(audience_choice, "general")
    
    print(f"\n🎨 正在生成环境警示图像...")
    
    result = generator.generate_environmental_warning_image(
        environmental_data=environmental_data,
        user_description=user_description,
        target_audience=target_audience,
        auto_open=True
    )
    
    display_result(result, "自定义环境数据")

def display_result(result, scenario_name):
    """
    显示生成结果
    """
    if result["success"]:
        print(f"\n✅ '{scenario_name}' 图像生成成功！")
        print(f"📁 保存位置: {result['saved_paths'][0]}")
        print(f"⏱️  生成时间: {result['generation_time']:.2f} 秒")
        print(f"🎯 总体严重程度: {result['analysis']['overall_severity']}")
        
        if result['analysis']['critical_factors']:
            print(f"⚠️  关键问题:")
            for factor in result['analysis']['critical_factors']:
                print(f"     - {factor['name']}: {factor['value']} {factor['unit']} ({factor['severity']})")
        
        # 显示偏差分析
        analysis = result['analysis']
        if "deviation_analysis" in analysis:
            deviation = analysis["deviation_analysis"]
            if deviation.get("primary_concerns"):
                print(f"\n⚠️  主要关注点 (偏差>100%):")
                for concern in deviation["primary_concerns"][:3]:
                    deviation_pct = concern["deviation_ratio"] * 100
                    print(f"     • {concern['name']}: {concern['current_value']} {concern['unit']} (偏差: {deviation_pct:+.1f}%)")
            
            if deviation.get("secondary_concerns"):
                print(f"\n🔶 次要关注点 (偏差30-100%):")
                for concern in deviation["secondary_concerns"][:3]:
                    deviation_pct = concern["deviation_ratio"] * 100
                    print(f"     • {concern['name']}: {concern['current_value']} {concern['unit']} (偏差: {deviation_pct:+.1f}%)")
        
        print(f"\n📝 生成的专业 Prompt (前200字符):")
        print(f"   {result['professional_prompt'][:200]}...")
        
        # 显示目标用户群体信息
        if "target_audience" in result:
            audience_styles = {
                "general": "现实主义风格，专业氛围",
                "educators": "教育风格，清晰明了", 
                "parents": "温和风格，关怀氛围",
                "students": "卡通风格，生动有趣"
            }
            style_desc = audience_styles.get(result["target_audience"], "默认风格")
            print(f"\n🎨 图像风格: {style_desc}")
        
        # 显示使用的模型信息
        if "models_used" in result:
            print(f"\n🤖 使用模型:")
            print(f"   聊天模型: {result['models_used']['chat_model']}")
            print(f"   图像模型: {result['models_used']['image_model']}")
        
    else:
        print(f"❌ '{scenario_name}' 图像生成失败: {result.get('error', '未知错误')}")
        
        if "analysis" in result:
            analysis = result["analysis"]
            print(f"\n📈 环境数据分析:")
            print(f"  整体严重程度: {analysis['overall_severity']}")
            print(f"  关键因素数量: {len(analysis['critical_factors'])}")
        
        if "prompt" in result:
            print(f"\n🎨 生成的 Prompt:")
            print(f"  {result['prompt'][:200]}...")

def quick_demo():
    """
    快速演示 - 使用预设数据
    """
    print("\n🚀 快速演示模式")
    print("-" * 30)
    
    try:
        # 初始化生成器
        generator = DashScopeEnvironmentalGenerator()
        
        # 预设场景
        scenarios = {
            "1": {
                "name": "重度空气污染",
                "data": {
                    "carbon_emission": 2580,
                    "air_quality_index": 280,
                    "water_pollution_index": 45,
                    "noise_level": 75
                },
                "description": "城市工业区严重空气污染场景",
                "audience": "general"
            },
            "2": {
                "name": "水体污染危机", 
                "data": {
                    "carbon_emission": 1230,
                    "air_quality_index": 120,
                    "water_pollution_index": 85,
                    "noise_level": 65
                },
                "description": "河流湖泊受到工业废水污染",
                "audience": "educators"
            },
            "3": {
                "name": "学生环保教育场景",
                "data": {
                    "carbon_emission": 1850,
                    "air_quality_index": 180,
                    "water_pollution_index": 60,
                    "noise_level": 80
                },
                "description": "适合学生的环保教育内容",
                "audience": "students"
            },
            "4": {
                "name": "综合环境恶化",
                "data": {
                    "carbon_emission": 3520,
                    "air_quality_index": 320,
                    "water_pollution_index": 78,
                    "noise_level": 95
                },
                "description": "多种污染源造成的环境危机",
                "audience": "parents"
            }
        }
        
        print("\n选择预设场景：")
        for key, scenario in scenarios.items():
            print(f"{key}. {scenario['name']} (目标用户: {scenario['audience']})")
        
        choice = input("\n请选择场景 (1-4): ").strip()
        
        if choice not in scenarios:
            print("❌ 无效选择")
            return
        
        scenario = scenarios[choice]
        print(f"\n📋 选择场景: {scenario['name']}")
        print(f"📝 描述: {scenario['description']}")
        print(f"🎯 目标用户: {scenario['audience']}")
        
        # 显示数据偏差分析
        print("\n📊 数据偏差分析:")
        data_types = generator.get_supported_data_types()
        for data_type, value in scenario["data"].items():
            if data_type in data_types:
                config = data_types[data_type]
                default_value = config.get("default_value", 0)
                deviation = ((value - default_value) / default_value * 100) if default_value > 0 else 0
                status = "⚠️ 异常" if abs(deviation) > 30 else "✅ 正常"
                print(f"  {config['name']}: {value} {config['unit']} (偏差: {deviation:+.1f}%) {status}")
        
        # 生成图像
        print("\n🎨 正在生成环境警示图像...")
        result = generator.generate_environmental_warning_image(
            environmental_data=scenario["data"],
            user_description=scenario["description"],
            target_audience=scenario["audience"],
            auto_open=True
        )
        
        # 显示结果
        display_result(result, scenario['name'])
        
    except Exception as e:
        print(f"❌ 快速演示失败: {e}")

def main():
    """
    主函数
    """
    # 检查环境变量
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("❌ 未设置 DASHSCOPE_API_KEY 环境变量")
        print("请在 .env 文件中配置 DASHSCOPE_API_KEY")
        return
    
    print("请选择演示模式:")
    print("1. 交互式演示 (推荐)")
    print("2. 快速演示")
    
    choice = input("\n请输入选择 (1-2, 默认为1): ").strip() or "1"
    
    if choice == "1":
        interactive_demo()
    elif choice == "2":
        quick_demo()
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()