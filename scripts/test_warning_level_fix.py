#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试warning_level错误修复

这个脚本用于验证generate_warning_image方法是否正确返回warning_level字段。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ecology_image_generator import EcologyImageGenerator
from src.utils.logger import setup_logger, get_logger

# 设置日志
setup_logger()
logger = get_logger(__name__)

def test_warning_level_fix():
    """测试warning_level错误修复"""
    print("🧪 测试warning_level错误修复")
    print("=" * 40)
    
    try:
        # 初始化生成器
        print("\n1. 初始化生态图像生成器...")
        generator = EcologyImageGenerator()
        print("✅ 生成器初始化成功")
        
        # 测试环境指标
        test_indicators = {
            "co2_level": 450.0,
            "pm25_level": 80.0,
            "temperature": 30.0,
            "forest_coverage": 40.0,
            "water_quality": 5.0,
            "air_quality": 4.0
        }
        
        print(f"\n2. 测试环境指标: {test_indicators}")
        
        # 生成警示图像
        print("\n3. 生成警示图像...")
        result = generator.generate_warning_image(
            environmental_indicators=test_indicators,
            style='realistic',
            num_images=1
        )
        
        print("✅ 图像生成成功")
        
        # 检查返回结果的结构
        print("\n4. 检查返回结果结构:")
        required_fields = [
            'warning_level',
            'template_used', 
            'environmental_assessment',
            'generation_mode',
            'style',
            'generated_images',
            'environmental_indicators'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field in result:
                print(f"✅ {field}: {type(result[field]).__name__}")
            else:
                print(f"❌ {field}: 缺失")
                missing_fields.append(field)
        
        if missing_fields:
            print(f"\n❌ 缺失字段: {missing_fields}")
            return False
        
        # 详细检查warning_level
        print("\n5. 详细检查warning_level:")
        warning_level = result['warning_level']
        print(f"   类型: {type(warning_level)}")
        print(f"   值: {warning_level}")
        
        if isinstance(warning_level, int) and 1 <= warning_level <= 5:
            print("✅ warning_level格式正确")
        else:
            print("❌ warning_level格式错误")
            return False
        
        # 检查environmental_assessment
        print("\n6. 检查environmental_assessment:")
        assessment = result['environmental_assessment']
        assessment_fields = ['overall_risk', 'risk_score', 'primary_concerns', 'recommendations']
        
        for field in assessment_fields:
            if field in assessment:
                print(f"✅ {field}: {assessment[field]}")
            else:
                print(f"❌ {field}: 缺失")
                return False
        
        # 检查generated_images
        print("\n7. 检查generated_images:")
        images = result['generated_images']
        if isinstance(images, list) and len(images) > 0:
            print(f"✅ 生成了 {len(images)} 张图像")
            
            # 检查第一张图像的结构
            first_image = images[0]
            image_fields = ['description', 'style', 'quality_score', 'generation_time']
            
            for field in image_fields:
                if field in first_image:
                    print(f"✅ 图像.{field}: {first_image[field]}")
                else:
                    print(f"❌ 图像.{field}: 缺失")
                    return False
        else:
            print("❌ generated_images格式错误")
            return False
        
        print("\n🎉 所有测试通过！warning_level错误已修复")
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_scenarios():
    """测试多个场景"""
    print("\n\n🧪 测试多个环境场景")
    print("=" * 40)
    
    scenarios = {
        "低风险环境": {
            "co2_level": 380.0,
            "pm25_level": 30.0,
            "temperature": 22.0,
            "forest_coverage": 70.0,
            "water_quality": 8.0,
            "air_quality": 7.0
        },
        "中风险环境": {
            "co2_level": 420.0,
            "pm25_level": 80.0,
            "temperature": 30.0,
            "forest_coverage": 45.0,
            "water_quality": 5.0,
            "air_quality": 4.0
        },
        "高风险环境": {
            "co2_level": 480.0,
            "pm25_level": 150.0,
            "temperature": 38.0,
            "forest_coverage": 20.0,
            "water_quality": 2.0,
            "air_quality": 2.0
        }
    }
    
    try:
        generator = EcologyImageGenerator()
        
        for scenario_name, indicators in scenarios.items():
            print(f"\n测试场景: {scenario_name}")
            
            result = generator.generate_warning_image(
                environmental_indicators=indicators,
                style='realistic',
                num_images=1
            )
            
            warning_level = result['warning_level']
            risk = result['environmental_assessment']['overall_risk']
            template = result['template_used']
            
            print(f"  警示等级: {warning_level}/5")
            print(f"  风险评估: {risk}")
            print(f"  使用模板: {template}")
            print(f"  ✅ 场景测试通过")
        
        print("\n🎉 多场景测试全部通过！")
        return True
        
    except Exception as e:
        print(f"❌ 多场景测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始测试warning_level错误修复")
    
    # 基础功能测试
    basic_test_passed = test_warning_level_fix()
    
    if basic_test_passed:
        # 多场景测试
        scenario_test_passed = test_multiple_scenarios()
        
        if scenario_test_passed:
            print("\n" + "=" * 50)
            print("🎉 所有测试通过！")
            print("✅ warning_level错误已完全修复")
            print("✅ 生态警示图像生成系统工作正常")
            print("\n💡 现在可以正常使用交互式系统了：")
            print("   python scripts/improved_interactive_ecology_demo.py")
        else:
            print("\n❌ 多场景测试失败")
    else:
        print("\n❌ 基础测试失败")

if __name__ == "__main__":
    main()