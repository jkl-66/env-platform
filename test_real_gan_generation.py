#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试真正的GAN模型图像生成
解决PyTorch版本兼容性问题并验证模型是否真正工作
"""

import sys
import torch
import numpy as np
from pathlib import Path
import json
import datetime
from typing import List, Dict, Any
from PIL import Image

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.models.ecology_image_generator import EcologyImageGenerator

def check_pytorch_compatibility():
    """检查PyTorch兼容性"""
    print("=== PyTorch 兼容性检查 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_capability = torch.cuda.get_device_capability(i)
            print(f"GPU {i}: {gpu_name}")
            print(f"计算能力: {gpu_capability[0]}.{gpu_capability[1]} (sm_{gpu_capability[0]}{gpu_capability[1]})")
            
            # 测试简单的CUDA操作
            try:
                test_tensor = torch.randn(10, 10).cuda(i)
                result = test_tensor @ test_tensor.T
                print(f"GPU {i} 基础操作测试: ✅ 成功")
            except Exception as e:
                print(f"GPU {i} 基础操作测试: ❌ 失败 - {e}")
                return False
    
    return True

def test_gan_model_creation():
    """测试GAN模型创建"""
    print("\n=== 测试GAN模型创建 ===")
    
    # 强制使用CPU避免CUDA兼容性问题
    device = "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 创建生成器
        generator = EcologyImageGenerator(device=device)
        generator.build_model()
        
        print("✅ GAN模型创建成功")
        print(f"生成器参数数量: {sum(p.numel() for p in generator.generator.parameters()):,}")
        print(f"判别器参数数量: {sum(p.numel() for p in generator.discriminator.parameters()):,}")
        
        return generator
        
    except Exception as e:
        print(f"❌ GAN模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_gan_generation(generator, num_tests=3):
    """测试GAN图像生成"""
    print(f"\n=== 测试GAN图像生成 (生成{num_tests}张图像) ===")
    
    results = []
    
    for i in range(num_tests):
        print(f"\n--- 测试 {i+1}/{num_tests} ---")
        
        # 创建随机条件
        conditions = np.random.rand(10).tolist()
        print(f"环境条件: {[f'{x:.3f}' for x in conditions]}")
        
        try:
            # 生成图像
            input_data = {"conditions": conditions}
            result = generator.predict(input_data, num_images=1)
            
            if "error" in result:
                print(f"❌ 生成失败: {result['error']}")
                continue
            
            # 检查生成结果
            if "generated_images" in result and len(result["generated_images"]) > 0:
                img_data = result["generated_images"][0]
                img_array = np.array(img_data)
                
                print(f"✅ 生成成功")
                print(f"图像形状: {img_array.shape}")
                print(f"数据类型: {img_array.dtype}")
                print(f"数值范围: [{img_array.min():.3f}, {img_array.max():.3f}]")
                print(f"平均值: {img_array.mean():.3f}")
                print(f"标准差: {img_array.std():.3f}")
                
                # 检查图像是否有变化（不是全零或全一样的值）
                unique_values = len(np.unique(img_array.flatten()))
                print(f"唯一像素值数量: {unique_values}")
                
                if unique_values > 100:  # 如果有足够的变化
                    print("✅ 图像具有足够的变化，可能是真实生成")
                    quality = "good"
                elif unique_values > 10:
                    print("⚠️ 图像变化较少，可能是简单模式")
                    quality = "medium"
                else:
                    print("❌ 图像几乎没有变化，可能是占位符")
                    quality = "poor"
                
                results.append({
                    "test_id": i+1,
                    "success": True,
                    "conditions": conditions,
                    "image_shape": img_array.shape,
                    "value_range": [float(img_array.min()), float(img_array.max())],
                    "mean_value": float(img_array.mean()),
                    "std_value": float(img_array.std()),
                    "unique_values": int(unique_values),
                    "quality": quality,
                    "image_data": img_data
                })
                
            else:
                print(f"❌ 生成结果为空")
                results.append({
                    "test_id": i+1,
                    "success": False,
                    "error": "Empty generation result"
                })
                
        except Exception as e:
            print(f"❌ 生成过程出错: {e}")
            results.append({
                "test_id": i+1,
                "success": False,
                "error": str(e)
            })
    
    return results

def save_test_images(results, output_dir="outputs/gan_test"):
    """保存测试图像"""
    print(f"\n=== 保存测试图像到 {output_dir} ===")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    for result in results:
        if result["success"] and "image_data" in result:
            try:
                # 转换图像数据
                img_array = np.array(result["image_data"])
                
                # 确保数据在正确范围内
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
                # 创建PIL图像
                img = Image.fromarray(img_array)
                
                # 保存图像
                filename = f"gan_test_{timestamp}_{result['test_id']}_quality_{result['quality']}.png"
                filepath = output_path / filename
                img.save(filepath)
                
                saved_files.append(str(filepath))
                print(f"✅ 图像已保存: {filepath}")
                
            except Exception as e:
                print(f"❌ 保存图像 {result['test_id']} 失败: {e}")
    
    # 保存测试报告
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": "cpu",
        "total_tests": len(results),
        "successful_tests": sum(1 for r in results if r["success"]),
        "test_results": results,
        "saved_files": saved_files
    }
    
    report_file = output_path / f"gan_test_report_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 测试报告已保存: {report_file}")
    return report

def analyze_results(results):
    """分析测试结果"""
    print("\n=== 测试结果分析 ===")
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    
    print(f"总测试数: {total_tests}")
    print(f"成功测试数: {successful_tests}")
    print(f"成功率: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests > 0:
        # 分析图像质量
        quality_counts = {}
        for result in results:
            if result["success"]:
                quality = result.get("quality", "unknown")
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        print("\n图像质量分布:")
        for quality, count in quality_counts.items():
            print(f"  {quality}: {count} 张")
        
        # 分析数值特征
        mean_values = [r["mean_value"] for r in results if r["success"]]
        std_values = [r["std_value"] for r in results if r["success"]]
        unique_counts = [r["unique_values"] for r in results if r["success"]]
        
        print(f"\n数值特征:")
        print(f"  平均像素值范围: [{min(mean_values):.3f}, {max(mean_values):.3f}]")
        print(f"  标准差范围: [{min(std_values):.3f}, {max(std_values):.3f}]")
        print(f"  唯一值数量范围: [{min(unique_counts)}, {max(unique_counts)}]")
        
        # 判断是否真正使用了GAN模型
        avg_unique_values = sum(unique_counts) / len(unique_counts)
        avg_std = sum(std_values) / len(std_values)
        
        print(f"\n模型使用评估:")
        if avg_unique_values > 1000 and avg_std > 0.1:
            print("✅ 很可能使用了真正的GAN模型生成")
        elif avg_unique_values > 100 and avg_std > 0.05:
            print("⚠️ 可能使用了简化的生成方法")
        else:
            print("❌ 很可能使用了占位符或示例图像")
    
    return {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": successful_tests/total_tests if total_tests > 0 else 0
    }

def main():
    """主函数"""
    print("GAN模型真实性测试")
    print("=" * 50)
    
    # 检查PyTorch兼容性
    if not check_pytorch_compatibility():
        print("\n❌ PyTorch兼容性检查失败，建议更新到支持sm_120的版本")
        print("建议运行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    
    # 测试GAN模型创建
    generator = test_gan_model_creation()
    if generator is None:
        print("\n❌ 无法创建GAN模型，测试终止")
        return
    
    # 测试GAN图像生成
    results = test_gan_generation(generator, num_tests=5)
    
    # 保存测试图像和报告
    report = save_test_images(results)
    
    # 分析结果
    analysis = analyze_results(results)
    
    print("\n=== 总结 ===")
    if analysis["success_rate"] > 0.8:
        print("✅ GAN模型工作正常")
    elif analysis["success_rate"] > 0.5:
        print("⚠️ GAN模型部分工作，可能需要优化")
    else:
        print("❌ GAN模型存在问题，需要检查配置")
    
    print(f"\n测试完成！查看生成的图像: outputs/gan_test/")

if __name__ == "__main__":
    main()