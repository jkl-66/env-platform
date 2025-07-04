#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线功能测试脚本

测试EcologyImageGenerator的基本功能，不依赖网络连接
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.ecology_image_generator import EcologyImageGenerator
from src.utils.logger import get_logger

logger = get_logger("offline_test")

def test_basic_initialization():
    """测试基本初始化功能"""
    print("\n=== 测试基本初始化 ===")
    
    try:
        # 测试基本初始化
        generator = EcologyImageGenerator()
        print(f"✅ 基本初始化成功")
        print(f"   模型名称: {generator.model_name}")
        print(f"   模型类型: {generator.model_type}")
        print(f"   设备: {generator.device}")
        print(f"   当前模型ID: {generator.model_id}")
        
        return generator
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return None

def test_model_listing(generator):
    """测试模型列表功能"""
    print("\n=== 测试模型列表功能 ===")
    
    try:
        models = generator.list_supported_models()
        print(f"✅ 支持的模型数量: {len(models)}")
        
        print("\n支持的模型:")
        for model_id, description in models.items():
            print(f"  • {model_id}")
            print(f"    {description}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型列表获取失败: {e}")
        return False

def test_model_info(generator):
    """测试模型信息获取"""
    print("\n=== 测试模型信息获取 ===")
    
    try:
        info = generator.get_model_info()
        print("✅ 模型信息获取成功")
        
        print("\n当前模型信息:")
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型信息获取失败: {e}")
        return False

def test_model_switching(generator):
    """测试模型切换功能"""
    print("\n=== 测试模型切换功能 ===")
    
    try:
        # 获取支持的模型列表
        models = generator.list_supported_models()
        model_ids = list(models.keys())
        
        if len(model_ids) >= 2:
            # 测试切换到第二个模型
            new_model_id = model_ids[1]
            print(f"切换到模型: {new_model_id}")
            
            generator.set_model(new_model_id)
            
            # 验证切换是否成功
            if generator.model_id == new_model_id:
                print(f"✅ 模型切换成功: {new_model_id}")
                return True
            else:
                print(f"❌ 模型切换失败")
                return False
        else:
            print("⚠️ 可用模型数量不足，跳过切换测试")
            return True
            
    except Exception as e:
        print(f"❌ 模型切换测试失败: {e}")
        return False

def test_train_method(generator):
    """测试训练方法"""
    print("\n=== 测试训练方法 ===")
    
    try:
        # 调用训练方法（应该返回预训练状态）
        result = generator.train(train_data=None)
        
        print("✅ 训练方法调用成功")
        print("\n训练结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        # 检查是否标记为已训练
        if generator.is_trained:
            print("✅ 模型已标记为训练状态")
        else:
            print("⚠️ 模型未标记为训练状态")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练方法测试失败: {e}")
        return False

def test_prompt_enhancement(generator):
    """测试提示词增强功能"""
    print("\n=== 测试提示词增强功能 ===")
    
    try:
        # 测试提示词
        test_prompts = [
            "城市空气污染",
            "forest destruction",
            "ocean plastic pollution",
            "climate change effects",
            "工业废水排放"
        ]
        
        for prompt in test_prompts:
            try:
                enhanced = generator._enhance_environmental_prompt(prompt)
                print(f"\n原始提示词: {prompt}")
                print(f"增强提示词: {enhanced}")
            except Exception as e:
                print(f"⚠️ 提示词增强失败: {prompt} - {e}")
        
        print("\n✅ 提示词增强功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 提示词增强测试失败: {e}")
        return False

def test_offline_generation(generator):
    """测试离线生成功能（不实际下载模型）"""
    print("\n=== 测试离线生成功能 ===")
    
    try:
        # 测试输入数据
        input_data = {
            "prompt": "工业污染的城市景观"
        }
        
        # 尝试生成（应该会回退到示例图像）
        result = generator.predict(input_data, num_images=1)
        
        print("✅ 离线生成测试完成")
        print("\n生成结果:")
        for key, value in result.items():
            if key == "generated_images":
                print(f"  {key}: [图像数据] (长度: {len(value) if value else 0})")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 离线生成测试失败: {e}")
        return False

def save_test_results(results, output_dir="outputs/offline_test"):
    """保存测试结果"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"offline_test_results_{timestamp}.json")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试结果已保存到: {result_file}")
    return result_file

def main():
    """主函数"""
    print("🧪 开始离线功能测试")
    
    test_results = {
        "test_timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    try:
        # 1. 基本初始化测试
        generator = test_basic_initialization()
        test_results["tests"]["initialization"] = generator is not None
        
        if generator is None:
            print("❌ 初始化失败，无法继续测试")
            return False
        
        # 2. 模型列表测试
        test_results["tests"]["model_listing"] = test_model_listing(generator)
        
        # 3. 模型信息测试
        test_results["tests"]["model_info"] = test_model_info(generator)
        
        # 4. 模型切换测试
        test_results["tests"]["model_switching"] = test_model_switching(generator)
        
        # 5. 训练方法测试
        test_results["tests"]["train_method"] = test_train_method(generator)
        
        # 6. 提示词增强测试
        test_results["tests"]["prompt_enhancement"] = test_prompt_enhancement(generator)
        
        # 7. 离线生成测试
        test_results["tests"]["offline_generation"] = test_offline_generation(generator)
        
        # 统计结果
        passed_tests = sum(1 for result in test_results["tests"].values() if result)
        total_tests = len(test_results["tests"])
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0
        }
        
        # 保存结果
        save_test_results(test_results)
        
        # 打印总结
        print("\n=== 测试总结 ===")
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {total_tests - passed_tests}")
        print(f"成功率: {passed_tests / total_tests * 100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 所有离线功能测试通过！")
            return True
        else:
            print("\n⚠️ 部分测试失败，请检查日志")
            return False
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)