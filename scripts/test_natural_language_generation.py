#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自然语言图像生成功能测试脚本

测试新增的自然语言输入生成环境警示图像功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ecology_image_generator import EcologyImageGenerator
import json
from datetime import datetime

def test_natural_language_generation():
    """测试自然语言图像生成功能"""
    print("🧪 测试自然语言图像生成功能")
    print("=" * 50)
    
    # 初始化生成器
    generator = EcologyImageGenerator()
    
    # 测试用例
    test_cases = [
        {
            "description": "烟雾笼罩的城市，空气中充满有毒气体",
            "expected_themes": ["air_pollution"],
            "expected_warning_level": 4
        },
        {
            "description": "干涸的河床，鱼类死亡，水源枯竭",
            "expected_themes": ["water_pollution"],
            "expected_warning_level": 4
        },
        {
            "description": "失去栖息地的北极熊，冰川快速融化",
            "expected_themes": ["climate_change", "wildlife_threat"],
            "expected_warning_level": 5
        },
        {
            "description": "大规模森林砍伐，动物无家可归",
            "expected_themes": ["deforestation", "wildlife_threat"],
            "expected_warning_level": 4
        },
        {
            "description": "轻微的空气污染，需要改善",
            "expected_themes": ["air_pollution"],
            "expected_warning_level": 2
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {test_case['description']}")
        
        try:
            # 生成图像
            result = generator.generate_from_text(
                text_prompt=test_case['description'],
                style="realistic",
                num_images=1
            )
            
            # 验证结果
            success = True
            issues = []
            
            # 检查警示等级
            if abs(result['warning_level'] - test_case['expected_warning_level']) > 1:
                success = False
                issues.append(f"警示等级不匹配: 期望{test_case['expected_warning_level']}, 实际{result['warning_level']}")
            
            # 检查主题检测
            detected_themes = result['text_analysis']['detected_themes']
            for expected_theme in test_case['expected_themes']:
                if expected_theme not in detected_themes:
                    success = False
                    issues.append(f"未检测到期望主题: {expected_theme}")
            
            # 检查生成的图像
            if not result['generated_images']:
                success = False
                issues.append("未生成图像")
            
            # 记录结果
            test_result = {
                "test_case": i,
                "description": test_case['description'],
                "success": success,
                "issues": issues,
                "warning_level": result['warning_level'],
                "detected_themes": detected_themes,
                "enhanced_prompt": result['enhanced_prompt'],
                "environmental_impact": result['text_analysis']['environmental_impact']
            }
            
            results.append(test_result)
            
            if success:
                print(f"✅ 测试通过")
            else:
                print(f"❌ 测试失败: {', '.join(issues)}")
            
            print(f"   警示等级: {result['warning_level']}/5")
            print(f"   检测主题: {detected_themes}")
            print(f"   环境影响: {result['text_analysis']['environmental_impact']}")
            print(f"   增强提示: {result['enhanced_prompt'][:100]}...")
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append({
                "test_case": i,
                "description": test_case['description'],
                "success": False,
                "issues": [f"异常: {str(e)}"],
                "error": str(e)
            })
    
    # 生成测试报告
    print("\n📊 测试总结")
    print("=" * 30)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"总测试用例: {total}")
    print(f"通过测试: {passed}")
    print(f"失败测试: {total - passed}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    # 保存测试结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/natural_language_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / f"test_results_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "summary": {
                "total_tests": total,
                "passed_tests": passed,
                "failed_tests": total - passed,
                "success_rate": passed/total*100
            },
            "test_results": results
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n📁 测试结果已保存至: {result_file}")
    
    return passed == total

def test_prompt_enhancement():
    """测试提示词增强功能"""
    print("\n🔧 测试提示词增强功能")
    print("=" * 30)
    
    generator = EcologyImageGenerator()
    
    test_prompts = [
        "烟雾笼罩的城市",
        "干涸的河床",
        "冰川融化",
        "森林砍伐",
        "海洋塑料污染"
    ]
    
    for prompt in test_prompts:
        enhanced = generator._enhance_warning_prompt(prompt, "realistic")
        print(f"原始: {prompt}")
        print(f"增强: {enhanced}")
        print("-" * 50)

def test_text_analysis():
    """测试文本分析功能"""
    print("\n🔍 测试文本分析功能")
    print("=" * 30)
    
    generator = EcologyImageGenerator()
    
    test_texts = [
        "严重的空气污染导致雾霾天气",
        "轻微的水质问题需要关注",
        "大规模森林砍伐威胁生物多样性",
        "极端天气频发，气候变化加剧"
    ]
    
    for text in test_texts:
        warning_level = generator._analyze_text_warning_level(text)
        analysis = generator._analyze_environmental_text(text)
        
        print(f"文本: {text}")
        print(f"警示等级: {warning_level}/5")
        print(f"检测主题: {analysis['detected_themes']}")
        print(f"严重性指标: {analysis['severity_indicators']}")
        print(f"环境影响: {analysis['environmental_impact']}")
        print("-" * 50)

if __name__ == "__main__":
    print("🌍 自然语言图像生成功能测试")
    print("=" * 60)
    
    try:
        # 运行所有测试
        test_prompt_enhancement()
        test_text_analysis()
        success = test_natural_language_generation()
        
        if success:
            print("\n🎉 所有测试通过！自然语言图像生成功能正常工作。")
        else:
            print("\n⚠️  部分测试失败，请检查相关功能。")
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()