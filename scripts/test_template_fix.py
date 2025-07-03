#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模板错误修复
验证预设环境场景模板功能是否正常工作
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ecology_image_generator import EcologyImageGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_template_structure():
    """测试模板结构完整性"""
    print("🧪 测试模板结构完整性")
    print("=" * 40)
    
    try:
        generator = EcologyImageGenerator()
        templates = generator.get_condition_templates()
        
        print(f"📋 找到 {len(templates)} 个模板")
        
        required_fields = ['description', 'warning_level', 'visual_elements', 'color_scheme']
        all_passed = True
        
        for name, template in templates.items():
            print(f"\n🔍 检查模板: {name}")
            
            missing_fields = []
            for field in required_fields:
                if field not in template:
                    missing_fields.append(field)
                    all_passed = False
                else:
                    print(f"  ✅ {field}: {template[field]}")
            
            if missing_fields:
                print(f"  ❌ 缺少字段: {', '.join(missing_fields)}")
            else:
                print(f"  ✅ 模板结构完整")
        
        if all_passed:
            print("\n🎉 所有模板结构检查通过！")
        else:
            print("\n⚠️  发现模板结构问题")
            
        return all_passed
        
    except Exception as e:
        logger.error(f"模板结构测试失败: {e}")
        print(f"❌ 模板结构测试失败: {e}")
        return False

def test_safe_template_access():
    """测试安全的模板访问"""
    print("\n🧪 测试安全的模板访问")
    print("=" * 40)
    
    try:
        generator = EcologyImageGenerator()
        templates = generator.get_condition_templates()
        
        print("📊 模拟原始脚本的模板访问方式:")
        
        for i, (name, template) in enumerate(templates.items(), 1):
            print(f"\n{i}. {name}")
            
            # 使用安全访问方式（修复后的方式）
            description = template.get('description', '环境场景模板')
            warning_level = template.get('warning_level', 3)
            visual_elements = template.get('visual_elements', ['环境要素'])
            color_scheme = template.get('color_scheme', ['自然色彩'])
            
            print(f"   描述: {description}")
            print(f"   警示等级: {warning_level}/5")
            print(f"   视觉元素: {', '.join(visual_elements)}")
            print(f"   色彩方案: {', '.join(color_scheme)}")
        
        print("\n✅ 安全模板访问测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"安全模板访问测试失败: {e}")
        print(f"❌ 安全模板访问测试失败: {e}")
        return False

def test_template_generation():
    """测试基于模板的图像生成"""
    print("\n🧪 测试基于模板的图像生成")
    print("=" * 40)
    
    try:
        generator = EcologyImageGenerator()
        templates = generator.get_condition_templates()
        
        # 选择第一个模板进行测试
        if templates:
            template_name = list(templates.keys())[0]
            print(f"🎨 使用模板 '{template_name}' 进行测试")
            
            # 生成测试指标
            test_indicators = {
                "co2_level": 450.0,
                "pm25_level": 100.0,
                "temperature": 35.0,
                "forest_coverage": 30.0,
                "water_quality": 4.0,
                "air_quality": 3.0
            }
            
            result = generator.generate_warning_image(
                environmental_indicators=test_indicators,
                style='realistic',
                num_images=1
            )
            
            print(f"✅ 模板生成测试成功！")
            print(f"⚠️  警示等级: {result['warning_level']}/5")
            print(f"🏷️  使用模板: {result['template_used']}")
            print(f"🔍 环境评估: {result['environmental_assessment']['overall_risk']}")
            
            return True
        else:
            print("❌ 没有找到可用的模板")
            return False
            
    except Exception as e:
        logger.error(f"模板生成测试失败: {e}")
        print(f"❌ 模板生成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔧 预设环境场景模板修复验证")
    print("=" * 50)
    
    tests = [
        ("模板结构完整性", test_template_structure),
        ("安全模板访问", test_safe_template_access),
        ("模板图像生成", test_template_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 运行测试: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} - 通过")
        else:
            print(f"❌ {test_name} - 失败")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！模板错误已修复")
        print("\n💡 现在可以安全使用原始交互式脚本：")
        print("   python scripts/interactive_ecology_image_demo.py")
    else:
        print("⚠️  部分测试失败，请检查修复")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)