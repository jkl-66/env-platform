#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境图像生成项目验证脚本
验证项目结构和配置是否正确
"""

import os
import sys
import json
from pathlib import Path

def check_project_structure():
    """检查项目结构"""
    print("=== 检查项目结构 ===")
    
    required_files = [
        "environmental_image_generator.py",
        "demo_environmental_generator.py",
        "test_environmental_generator.py",
        "config/environmental_prompts.json",
        "README_Environmental_Generator.md"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
            print(f"  ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path}")
    
    print(f"\n项目文件: {len(existing_files)}/{len(required_files)} 存在")
    return len(missing_files) == 0

def check_dependencies():
    """检查依赖包"""
    print("\n=== 检查依赖包 ===")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy")
    ]
    
    missing_packages = []
    available_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            available_packages.append(name)
            print(f"  ✅ {name}")
        except ImportError:
            missing_packages.append(name)
            print(f"  ❌ {name}")
    
    print(f"\n依赖包: {len(available_packages)}/{len(required_packages)} 可用")
    
    if missing_packages:
        print("\n缺少的依赖包安装命令:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("pip install diffusers transformers accelerate pillow numpy")
    
    return len(missing_packages) == 0

def check_configuration():
    """检查配置文件"""
    print("\n=== 检查配置文件 ===")
    
    config_path = Path("config/environmental_prompts.json")
    
    if not config_path.exists():
        print(f"  ❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"  ✅ 配置文件加载成功")
        
        # 检查配置结构
        required_sections = [
            "environmental_prompts",
            "generation_settings", 
            "style_presets",
            "quality_enhancers",
            "negative_prompts"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section in config:
                count = len(config[section])
                print(f"    ✅ {section}: {count} 项")
            else:
                missing_sections.append(section)
                print(f"    ❌ {section}: 缺失")
        
        # 显示环境主题
        if "environmental_prompts" in config:
            print("\n  环境主题列表:")
            for i, (key, value) in enumerate(config["environmental_prompts"].items(), 1):
                desc = value.get("description", "无描述")
                print(f"    {i:2d}. {key:20s} - {desc}")
        
        return len(missing_sections) == 0
        
    except Exception as e:
        print(f"  ❌ 配置文件解析失败: {e}")
        return False

def check_gpu_availability():
    """检查GPU可用性"""
    print("\n=== 检查GPU可用性 ===")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            
            print(f"  ✅ CUDA 可用")
            print(f"    GPU 数量: {gpu_count}")
            print(f"    当前设备: {current_device}")
            print(f"    GPU 名称: {gpu_name}")
            print(f"    GPU 内存: {gpu_memory:.1f} GB")
            
            if gpu_memory < 6:
                print(f"    ⚠️  GPU内存较少，建议使用较小的图像尺寸")
            
            return True
        else:
            print(f"  ❌ CUDA 不可用，将使用CPU（速度较慢）")
            return False
            
    except ImportError:
        print(f"  ❌ PyTorch 未安装")
        return False
    except Exception as e:
        print(f"  ❌ GPU检查失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    try:
        # 导入主模块
        from environmental_image_generator import EnvironmentalImageGenerator
        print("  ✅ 主模块导入成功")
        
        # 创建生成器实例（不加载模型）
        generator = EnvironmentalImageGenerator()
        print("  ✅ 生成器实例创建成功")
        
        # 测试提示词增强
        test_input = "工厂排放黑烟污染空气"
        enhanced = generator._enhance_environmental_prompt(test_input)
        print(f"  ✅ 提示词增强功能正常")
        print(f"    原始: {test_input}")
        print(f"    增强: {enhanced[:100]}...")
        
        # 测试类别检测
        category = generator._detect_environmental_category(test_input)
        print(f"  ✅ 类别检测功能正常: {category}")
        
        # 测试环境类别列表
        categories = generator.list_environmental_categories()
        print(f"  ✅ 环境类别列表: {len(categories)} 个类别")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 基本功能测试失败: {e}")
        return False

def show_usage_examples():
    """显示使用示例"""
    print("\n=== 使用示例 ===")
    
    examples = [
        "# 基本使用",
        "from environmental_image_generator import EnvironmentalImageGenerator",
        "",
        "generator = EnvironmentalImageGenerator()",
        "results = generator.generate_image(user_input='工厂排放黑烟污染空气')",
        "",
        "# 运行演示",
        "python demo_environmental_generator.py",
        "",
        "# 运行测试",
        "python test_environmental_generator.py"
    ]
    
    for example in examples:
        print(f"  {example}")

def main():
    """主函数"""
    print("🔍 环境图像生成项目验证")
    print("=" * 50)
    
    # 运行所有检查
    checks = [
        ("项目结构", check_project_structure),
        ("依赖包", check_dependencies),
        ("配置文件", check_configuration),
        ("GPU可用性", check_gpu_availability),
        ("基本功能", test_basic_functionality)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ {check_name}检查失败: {e}")
            results.append((check_name, False))
    
    # 显示总结
    print("\n" + "=" * 50)
    print("📊 验证总结")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name:15s}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项检查通过")
    
    if passed >= 4:  # GPU不是必需的
        print("\n🎉 项目设置基本完成！")
        print("\n📝 下一步:")
        print("  1. 确保网络连接正常，以便下载模型")
        print("  2. 运行 python demo_environmental_generator.py 开始使用")
        print("  3. 查看 README_Environmental_Generator.md 了解详细用法")
    else:
        print("\n⚠️  项目设置不完整，请检查失败的项目")
    
    # 显示使用示例
    show_usage_examples()
    
    return passed >= 4

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n验证被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 验证过程发生未预期错误: {e}")
        sys.exit(1)