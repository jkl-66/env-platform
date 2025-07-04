#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试部分下载的模型文件
检查当前缓存状态并尝试使用可用的模型
"""

import os
import sys
from pathlib import Path

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_CACHE'] = str(Path.cwd() / 'cache' / 'huggingface')
os.environ['TRANSFORMERS_CACHE'] = str(Path.cwd() / 'cache' / 'huggingface')
os.environ['HF_HOME'] = str(Path.cwd() / 'cache' / 'huggingface')

print("🔍 检查当前模型下载状态")
print("=" * 50)
print(f"📁 缓存目录: {os.environ['HF_HUB_CACHE']}")
print()

def check_cache_status():
    """检查缓存目录状态"""
    cache_dir = Path(os.environ['HF_HUB_CACHE'])
    
    if not cache_dir.exists():
        print("❌ 缓存目录不存在")
        return
    
    print("📂 缓存目录内容:")
    
    # 检查 hub 目录
    hub_dir = cache_dir / 'hub'
    if hub_dir.exists():
        print(f"  📁 hub/ 目录存在")
        
        # 列出所有模型目录
        model_dirs = [d for d in hub_dir.iterdir() if d.is_dir() and d.name.startswith('models--')]
        
        for model_dir in model_dirs:
            model_name = model_dir.name.replace('models--', '').replace('--', '/')
            print(f"    🤖 模型: {model_name}")
            
            # 检查模型文件
            snapshots_dir = model_dir / 'snapshots'
            if snapshots_dir.exists():
                snapshot_dirs = list(snapshots_dir.iterdir())
                if snapshot_dirs:
                    latest_snapshot = snapshot_dirs[0]  # 取第一个快照
                    files = list(latest_snapshot.iterdir())
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    print(f"      📄 文件数量: {len(files)}")
                    print(f"      💾 总大小: {total_size / (1024*1024):.1f} MB")
                    
                    # 检查关键文件
                    key_files = ['config.json', 'model.safetensors', 'pytorch_model.bin']
                    for key_file in key_files:
                        file_path = latest_snapshot / key_file
                        if file_path.exists():
                            size_mb = file_path.stat().st_size / (1024*1024)
                            print(f"        ✅ {key_file}: {size_mb:.1f} MB")
                        else:
                            print(f"        ❌ {key_file}: 缺失")
    else:
        print("  ❌ hub/ 目录不存在")

def test_model_loading():
    """测试模型加载"""
    print("\n🧪 测试模型加载")
    print("-" * 30)
    
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        print("✅ 依赖库导入成功")
        
        # 尝试加载模型（仅本地文件）
        models_to_test = [
            "stabilityai/stable-diffusion-3.5-large",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-2-1"
        ]
        
        for model_id in models_to_test:
            print(f"\n🔧 测试模型: {model_id}")
            try:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    cache_dir=os.environ['HF_HUB_CACHE'],
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    local_files_only=True  # 仅使用本地文件
                )
                print(f"✅ {model_id} 加载成功！")
                
                # 简单测试生成
                print("🎨 测试图像生成...")
                with torch.no_grad():
                    image = pipeline(
                        "a simple test image",
                        num_inference_steps=1,  # 最少步数
                        guidance_scale=1.0,
                        height=64,  # 小尺寸
                        width=64
                    ).images[0]
                print("✅ 图像生成测试成功！")
                return True
                
            except Exception as e:
                print(f"❌ {model_id} 加载失败: {str(e)[:100]}...")
                continue
        
        print("❌ 所有模型加载失败")
        return False
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False

def test_alternative_approach():
    """测试替代方案"""
    print("\n🔄 测试替代方案")
    print("-" * 30)
    
    try:
        # 尝试使用更轻量级的方法
        from transformers import pipeline
        
        # 测试文本生成（更小的模型）
        print("📝 测试文本生成管道...")
        text_generator = pipeline(
            "text-generation",
            model="gpt2",  # 更小的模型
            cache_dir=os.environ['HF_HUB_CACHE']
        )
        
        result = text_generator("Environmental protection is", max_length=30, num_return_sequences=1)
        print(f"✅ 文本生成成功: {result[0]['generated_text']}")
        return True
        
    except Exception as e:
        print(f"❌ 替代方案失败: {e}")
        return False

def main():
    """主函数"""
    # 检查缓存状态
    check_cache_status()
    
    # 测试模型加载
    model_success = test_model_loading()
    
    # 如果模型加载失败，尝试替代方案
    if not model_success:
        alt_success = test_alternative_approach()
        if alt_success:
            print("\n💡 建议: 可以使用文本生成等替代功能")
    
    print("\n📊 测试总结")
    print("=" * 30)
    if model_success:
        print("✅ Stable Diffusion 模型可用")
        print("🎉 可以进行图像生成")
    else:
        print("❌ Stable Diffusion 模型暂不可用")
        print("💡 建议继续等待下载完成或检查网络连接")
        print("🔄 当前下载可能仍在进行中")

if __name__ == "__main__":
    main()