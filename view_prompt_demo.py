#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专业 Prompt 查看器

专门用于查看和分析 DashScope 环境图像生成器生成的专业 prompt
支持多种查看方式：实时生成、历史记录查看、详细分析等
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dashscope_environmental_generator import DashScopeEnvironmentalGenerator

def view_prompt_only():
    """
    仅生成和查看专业 prompt，不生成图像
    """
    print("🔍 专业 Prompt 查看器")
    print("=" * 50)
    
    try:
        generator = DashScopeEnvironmentalGenerator()
        
        # 测试连接
        print("🔗 正在测试连接...")
        test_result = generator.test_connection()
        if not test_result["success"]:
            print(f"❌ 连接失败: {test_result.get('error', '未知错误')}")
            return
        
        print("✅ 连接成功！")
        
        while True:
            print("\n" + "=" * 50)
            print("请选择操作:")
            print("1. 使用预设数据生成 prompt")
            print("2. 自定义环境数据生成 prompt")
            print("3. 查看历史生成报告中的 prompt")
            print("4. 批量生成多个场景的 prompt")
            print("5. 退出")
            
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == "1":
                generate_prompt_with_preset_data(generator)
            elif choice == "2":
                generate_prompt_with_custom_data(generator)
            elif choice == "3":
                view_historical_prompts()
            elif choice == "4":
                batch_generate_prompts(generator)
            elif choice == "5":
                break
            else:
                print("❌ 无效选择，请重新输入")
        
        print("\n👋 感谢使用专业 Prompt 查看器！")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")

def generate_prompt_with_preset_data(generator):
    """
    使用预设数据生成 prompt
    """
    print("\n🎯 预设示例场景:")
    
    scenarios = {
        "1": {
            "name": "轻度空气污染",
            "data": {"air_quality_index": 120, "carbon_emission": 300},
            "description": "城市轻度雾霾，能见度稍有下降",
            "audience": "general"
        },
        "2": {
            "name": "中度工业污染",
            "data": {"carbon_emission": 800, "air_quality_index": 180, "water_pollution_index": 60},
            "description": "工业区污染物排放，影响周边环境",
            "audience": "educators"
        },
        "3": {
            "name": "严重环境危机",
            "data": {"carbon_emission": 2500, "air_quality_index": 350, "water_pollution_index": 95, "deforestation_rate": 25000},
            "description": "多重环境问题叠加，生态系统面临崩溃",
            "audience": "students"
        },
        "4": {
            "name": "海洋塑料污染",
            "data": {"plastic_waste": 2000, "water_pollution_index": 80},
            "description": "海洋中大量塑料垃圾，海洋生物受到严重威胁",
            "audience": "parents"
        },
        "5": {
            "name": "噪音污染严重区域",
            "data": {"noise_level": 95, "air_quality_index": 140},
            "description": "城市交通和工业噪音严重超标",
            "audience": "general"
        }
    }
    
    data_types = generator.get_supported_data_types()
    
    for key, scenario in scenarios.items():
        print(f"  {key}. {scenario['name']} (目标用户: {scenario['audience']})")
        for data_type, value in scenario['data'].items():
            if data_type in data_types:
                data_config = data_types[data_type]
                default_value = data_config.get('default_value', 0)
                deviation = ((value - default_value) / default_value * 100) if default_value > 0 else 0
                status = "⚠️ 异常" if abs(deviation) > 30 else "✅ 正常"
                print(f"     - {data_config['name']}: {value} {data_config['unit']} (默认: {default_value}, 偏差: {deviation:+.1f}%) {status}")
    
    choice = input("\n请选择场景 (1-5): ").strip()
    
    if choice in scenarios:
        scenario = scenarios[choice]
        print(f"\n🤖 正在为 '{scenario['name']}' 生成专业 prompt...")
        print(f"🎯 目标用户: {scenario['audience']}")
        
        # 分析环境数据
        analysis = generator._analyze_environmental_data(scenario['data'])
        
        # 生成专业 prompt
        professional_prompt = generator._generate_professional_prompt(
            scenario['data'],
            scenario['description'],
            scenario['audience']
        )
        
        display_prompt_analysis(scenario, analysis, professional_prompt)
    else:
        print("❌ 无效选择")

def generate_prompt_with_custom_data(generator):
    """
    使用自定义数据生成 prompt
    """
    print("\n📝 请输入环境数据:")
    
    data_types = generator.get_supported_data_types()
    environmental_data = {}
    
    print("\n可选的环境数据类型 (输入数值，留空跳过):")
    
    for data_type, config in data_types.items():
        while True:
            try:
                value_input = input(f"  {config['name']} ({config['unit']}): ").strip()
                if not value_input:
                    break
                
                value = float(value_input)
                if value < 0:
                    print("    ❌ 数值不能为负数，请重新输入")
                    continue
                
                environmental_data[data_type] = value
                break
                
            except ValueError:
                print("    ❌ 请输入有效的数字")
    
    if not environmental_data:
        print("❌ 未输入任何环境数据")
        return
    
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
    
    print(f"\n🤖 正在生成专业 prompt...")
    
    # 分析环境数据
    analysis = generator._analyze_environmental_data(environmental_data)
    
    # 生成专业 prompt
    professional_prompt = generator._generate_professional_prompt(
        environmental_data,
        user_description,
        target_audience
    )
    
    scenario = {
        "name": "自定义环境数据",
        "data": environmental_data,
        "description": user_description,
        "audience": target_audience
    }
    
    display_prompt_analysis(scenario, analysis, professional_prompt)

def view_historical_prompts():
    """
    查看历史生成报告中的 prompt
    """
    print("\n📚 查看历史生成报告")
    
    reports_dir = Path("outputs/environmental_images")
    if not reports_dir.exists():
        print("❌ 未找到历史报告目录")
        return
    
    # 查找所有报告文件
    report_files = list(reports_dir.glob("environmental_report_*.json"))
    
    if not report_files:
        print("❌ 未找到历史报告文件")
        return
    
    # 按时间排序
    report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"\n找到 {len(report_files)} 个历史报告:")
    
    for i, report_file in enumerate(report_files[:10], 1):  # 只显示最近10个
        mtime = datetime.fromtimestamp(report_file.stat().st_mtime)
        print(f"  {i}. {report_file.name} ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    try:
        choice = input("\n请选择要查看的报告 (1-10): ").strip()
        index = int(choice) - 1
        
        if 0 <= index < min(len(report_files), 10):
            report_file = report_files[index]
            
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            display_historical_prompt_analysis(report_data)
        else:
            print("❌ 无效选择")
            
    except (ValueError, IndexError):
        print("❌ 请输入有效的数字")
    except Exception as e:
        print(f"❌ 读取报告文件时发生错误: {e}")

def batch_generate_prompts(generator):
    """
    批量生成多个场景的 prompt
    """
    print("\n🔄 批量生成 Prompt")
    
    batch_scenarios = [
        {
            "name": "轻度污染",
            "data": {"air_quality_index": 80, "carbon_emission": 200},
            "description": "轻微的空气质量问题",
            "audience": "general"
        },
        {
            "name": "中度污染",
            "data": {"air_quality_index": 150, "carbon_emission": 600, "water_pollution_index": 50},
            "description": "中等程度的环境污染",
            "audience": "educators"
        },
        {
            "name": "重度污染",
            "data": {"air_quality_index": 250, "carbon_emission": 1500, "water_pollution_index": 80},
            "description": "严重的环境污染问题",
            "audience": "students"
        },
        {
            "name": "极重污染",
            "data": {"air_quality_index": 400, "carbon_emission": 3000, "water_pollution_index": 95, "deforestation_rate": 30000},
            "description": "极其严重的环境危机",
            "audience": "parents"
        }
    ]
    
    print(f"\n将为 {len(batch_scenarios)} 个场景生成 prompt:")
    for i, scenario in enumerate(batch_scenarios, 1):
        print(f"  {i}. {scenario['name']}")
    
    confirm = input("\n确认开始批量生成？(y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 已取消批量生成")
        return
    
    print("\n🚀 开始批量生成...")
    
    for i, scenario in enumerate(batch_scenarios, 1):
        print(f"\n--- 场景 {i}: {scenario['name']} ---")
        
        try:
            # 分析环境数据
            analysis = generator._analyze_environmental_data(scenario['data'])
            
            # 生成专业 prompt
            professional_prompt = generator._generate_professional_prompt(
                scenario['data'],
                scenario['description'],
                scenario['audience']
            )
            
            print(f"✅ 生成成功")
            print(f"📊 严重程度: {analysis['overall_severity']}")
            print(f"📝 Prompt (前100字符): {professional_prompt[:100]}...")
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
    
    print("\n🎉 批量生成完成！")

def display_prompt_analysis(scenario, analysis, professional_prompt):
    """
    显示 prompt 分析结果
    """
    print(f"\n" + "=" * 60)
    print(f"📋 场景分析: {scenario['name']}")
    print("=" * 60)
    
    # 环境数据
    print("\n📊 环境数据:")
    for data_type, value in scenario['data'].items():
        if data_type in analysis['severity_scores']:
            score_info = analysis['severity_scores'][data_type]
            print(f"  • {score_info['unit']} 前的数值: {value} {score_info['unit']} (严重程度: {score_info['severity']})")
    
    # 分析结果
    print(f"\n🎯 总体严重程度: {analysis['overall_severity']}")
    
    if analysis['critical_factors']:
        print(f"\n⚠️  关键问题因素 ({len(analysis['critical_factors'])}个):")
        for factor in analysis['critical_factors']:
            print(f"  • {factor['name']}: {factor['value']} {factor['unit']} ({factor['severity']})")
    
    # 目标受众和描述
    print(f"\n👥 目标受众: {scenario['audience']}")
    if scenario.get('description'):
        print(f"📝 用户描述: {scenario['description']}")
    
    # 完整的专业 prompt
    print(f"\n" + "=" * 60)
    print("🤖 生成的专业 Prompt (完整版):")
    print("=" * 60)
    print(professional_prompt)
    print("=" * 60)
    
    # Prompt 分析
    analyze_prompt_content(professional_prompt, scenario['audience'])
    
    # 保存选项
    save_option = input("\n💾 是否保存此 prompt 到文件？(y/N): ").strip().lower()
    if save_option == 'y':
        save_prompt_to_file(scenario, analysis, professional_prompt)

def display_historical_prompt_analysis(report_data):
    """
    显示历史 prompt 分析
    """
    print(f"\n" + "=" * 60)
    print(f"📋 历史报告分析")
    print("=" * 60)
    
    # 基本信息
    print(f"\n⏰ 生成时间: {report_data.get('timestamp', '未知')}")
    print(f"🎯 目标受众: {report_data.get('target_audience', '未知')}")
    print(f"⏱️  生成耗时: {report_data.get('generation_time', 0):.2f} 秒")
    
    # 使用的模型
    if 'models_used' in report_data:
        models = report_data['models_used']
        print(f"🤖 聊天模型: {models.get('chat_model', '未知')}")
        print(f"🎨 图像模型: {models.get('image_model', '未知')}")
    
    # 环境数据
    if 'environmental_data' in report_data:
        print(f"\n📊 环境数据:")
        for data_type, value in report_data['environmental_data'].items():
            print(f"  • {data_type}: {value}")
    
    # 分析结果
    if 'analysis' in report_data:
        analysis = report_data['analysis']
        print(f"\n🎯 总体严重程度: {analysis.get('overall_severity', '未知')}")
        
        if analysis.get('critical_factors'):
            print(f"\n⚠️  关键问题因素:")
            for factor in analysis['critical_factors']:
                print(f"  • {factor.get('name', '未知')}: {factor.get('value', '未知')} {factor.get('unit', '')}")
    
    # 专业 prompt
    if 'professional_prompt' in report_data:
        print(f"\n" + "=" * 60)
        print("🤖 专业 Prompt:")
        print("=" * 60)
        print(report_data['professional_prompt'])
        print("=" * 60)
        
        analyze_prompt_content(report_data['professional_prompt'], report_data.get('target_audience', 'general'))
    
    # 保存的图像路径
    if 'saved_paths' in report_data and report_data['saved_paths']:
        print(f"\n📁 生成的图像: {report_data['saved_paths'][0]}")

def analyze_prompt_content(prompt, target_audience="general"):
    """
    分析 prompt 内容
    """
    print(f"\n📈 Prompt 内容分析:")
    print(f"  • 字符数: {len(prompt)}")
    print(f"  • 单词数: {len(prompt.split())}")
    print(f"  • 句子数: {prompt.count('.') + prompt.count('!') + prompt.count('?')}")
    print(f"  • 目标用户: {target_audience}")
    
    # 检查限制条件
    restrictions_check = {
        "无人物": not any(word in prompt.lower() for word in ['people', 'person', 'human', 'man', 'woman', 'child', 'face']),
        "无文字": not any(word in prompt.lower() for word in ['text', 'words', 'letters', 'sign', 'label', 'writing']),
        "环境焦点": any(word in prompt.lower() for word in ['environment', 'nature', 'landscape', 'ecosystem', 'wildlife'])
    }
    
    print(f"\n✅ 限制条件检查:")
    for condition, passed in restrictions_check.items():
        status = "✅ 通过" if passed else "❌ 未通过"
        print(f"  • {condition}: {status}")
    
    # 风格分析（基于目标用户）
    style_keywords = {
        "general": ['realistic', 'professional', 'detailed', 'documentary'],
        "educators": ['educational', 'clear', 'informative', 'scientific'],
        "parents": ['gentle', 'caring', 'soft', 'warm', 'hopeful'],
        "students": ['cartoon', 'animated', 'colorful', 'playful', 'bright']
    }
    
    expected_style = style_keywords.get(target_audience, [])
    found_style_keywords = [kw for kw in expected_style if kw.lower() in prompt.lower()]
    
    print(f"\n🎨 风格适配性 (目标: {target_audience}):")
    if found_style_keywords:
        print(f"  • 匹配的风格词: {', '.join(found_style_keywords)}")
    else:
        print(f"  • 未发现明显的目标风格词汇")
    
    # 关键词分析
    environmental_keywords = [
        'pollution', 'environmental', 'toxic', 'contaminated', 'emissions', 
        'smog', 'waste', 'degradation', 'crisis', 'damage', 'harmful',
        'industrial', 'factory', 'smoke', 'chemical', 'oil', 'plastic'
    ]
    
    visual_keywords = [
        'dramatic', 'lighting', 'contrast', 'photography', 'realistic',
        'documentary', 'professional', 'high quality', '4k', 'detailed'
    ]
    
    found_env_keywords = [kw for kw in environmental_keywords if kw.lower() in prompt.lower()]
    found_visual_keywords = [kw for kw in visual_keywords if kw.lower() in prompt.lower()]
    
    if found_env_keywords:
        print(f"  • 环境关键词: {', '.join(found_env_keywords[:5])}{'...' if len(found_env_keywords) > 5 else ''}")
    
    if found_visual_keywords:
        print(f"  • 视觉关键词: {', '.join(found_visual_keywords[:5])}{'...' if len(found_visual_keywords) > 5 else ''}")
    
    # 情感倾向分析
    positive_words = ['hope', 'clean', 'clear', 'beautiful', 'pristine', 'healthy']
    negative_words = ['polluted', 'contaminated', 'toxic', 'dangerous', 'harmful', 'dirty']
    
    positive_count = sum(1 for word in positive_words if word.lower() in prompt.lower())
    negative_count = sum(1 for word in negative_words if word.lower() in prompt.lower())
    
    print(f"\n😊 情感倾向:")
    print(f"  • 积极词汇: {positive_count}个")
    print(f"  • 消极词汇: {negative_count}个")
    
    if negative_count > positive_count:
        print(f"  • 倾向: 警示性 (突出环境问题)")
    elif positive_count > negative_count:
        print(f"  • 倾向: 希望性 (强调解决方案)")
    else:
        print(f"  • 倾向: 平衡性 (问题与希望并重)")

def save_prompt_to_file(scenario, analysis, professional_prompt):
    """
    保存 prompt 到文件
    """
    try:
        output_dir = Path("outputs/prompts")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prompt_{scenario['name'].replace(' ', '_')}_{timestamp}.txt"
        file_path = output_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"场景: {scenario['name']}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"目标受众: {scenario['audience']}\n")
            f.write(f"总体严重程度: {analysis['overall_severity']}\n")
            f.write("\n" + "=" * 50 + "\n")
            f.write("专业 Prompt:\n")
            f.write("=" * 50 + "\n")
            f.write(professional_prompt)
            f.write("\n\n" + "=" * 50 + "\n")
            f.write("环境数据:\n")
            for data_type, value in scenario['data'].items():
                f.write(f"  {data_type}: {value}\n")
        
        print(f"✅ Prompt 已保存到: {file_path}")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")

def main():
    """
    主函数
    """
    # 检查环境变量
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("❌ 未设置 DASHSCOPE_API_KEY 环境变量")
        print("请在 .env 文件中配置 DASHSCOPE_API_KEY")
        return
    
    view_prompt_only()

if __name__ == "__main__":
    main()