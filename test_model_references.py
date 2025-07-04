#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有模型引用是否已正确更新为 Stable Diffusion 3.5
"""

import os
import sys
from pathlib import Path

def test_model_references():
    """测试所有文件中的模型引用"""
    print("🔍 检查模型引用更新情况...")
    print("=" * 50)
    
    # 要检查的文件列表
    files_to_check = [
        "src/models/ecology_image_generator.py",
        "download_sd35.py", 
        "retry_download_models.py",
        "test_simple_sd.py",
        ".env",
        "src/utils/config.py",
        "src/utils/deploy_image_generation.py",
        "test_partial_download.py",
        "test_real_huggingface_generation.py",
        "test_huggingface_models.py",
        "fix_huggingface_connection.py",
        "scripts/setup_real_image_generation.py"
    ]
    
    # 预期的新模型
    expected_models = [
        "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-2-1"
    ]
    
    # 不应该存在的旧模型
    old_models = [
        "runwayml/stable-diffusion-v1-5",
        "CompVis/stable-diffusion-v1-4"
    ]
    
    results = {}
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 检查是否包含新模型
            has_new_models = any(model in content for model in expected_models)
            
            # 检查是否还包含旧模型
            has_old_models = any(model in content for model in old_models)
            
            results[file_path] = {
                'exists': True,
                'has_new_models': has_new_models,
                'has_old_models': has_old_models,
                'status': 'updated' if has_new_models and not has_old_models else 'needs_update'
            }
        else:
            results[file_path] = {
                'exists': False,
                'status': 'missing'
            }
    
    # 显示结果
    print("\n📊 检查结果:")
    updated_count = 0
    total_count = 0
    
    for file_path, result in results.items():
        total_count += 1
        if result['exists']:
            if result['status'] == 'updated':
                print(f"✅ {file_path} - 已更新")
                updated_count += 1
            else:
                print(f"⚠️ {file_path} - 需要更新")
                if result['has_old_models']:
                    print(f"   └─ 仍包含旧模型引用")
                if not result['has_new_models']:
                    print(f"   └─ 缺少新模型引用")
        else:
            print(f"❌ {file_path} - 文件不存在")
    
    print(f"\n📈 总结: {updated_count}/{total_count} 个文件已正确更新")
    
    if updated_count == total_count:
        print("🎉 所有文件都已成功更新为 Stable Diffusion 3.5!")
    else:
        print("⚠️ 还有文件需要更新")
    
    return results

def test_environment_setup():
    """测试环境配置"""
    print("\n🔧 检查环境配置...")
    print("=" * 30)
    
    # 检查环境变量
    diffusion_model = os.getenv('DIFFUSION_MODEL_PATH')
    if diffusion_model:
        print(f"环境变量 DIFFUSION_MODEL_PATH: {diffusion_model}")
        if 'stable-diffusion-3.5-large' in diffusion_model:
            print("✅ 环境变量已正确设置")
        else:
            print("⚠️ 环境变量需要更新")
    else:
        print("❌ 未设置 DIFFUSION_MODEL_PATH 环境变量")
    
    # 检查缓存目录
    cache_dir = "cache/huggingface"
    if os.path.exists(cache_dir):
        cache_contents = os.listdir(cache_dir)
        print(f"\n缓存目录内容: {cache_contents}")
        
        # 检查是否有 SD3.5 相关的缓存
        sd35_cache = any('stable-diffusion-3' in item for item in cache_contents)
        if sd35_cache:
            print("✅ 发现 Stable Diffusion 3.x 缓存")
        else:
            print("⚠️ 未发现 Stable Diffusion 3.x 缓存")
    else:
        print("❌ 缓存目录不存在")

def main():
    """主函数"""
    print("🧪 Stable Diffusion 3.5 模型引用测试")
    print("=" * 60)
    
    # 测试模型引用
    results = test_model_references()
    
    # 测试环境配置
    test_environment_setup()
    
    print("\n✨ 测试完成!")
    
    return results

if __name__ == "__main__":
    main()