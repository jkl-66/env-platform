#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU图像生成测试脚本
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.models.ecology_image_generator import EcologyImageGenerator

def test_gpu_image_generation():
    """测试GPU图像生成功能"""
    print("=== GPU图像生成测试 ===")
    
    # 检查GPU状态
    print(f"GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 初始化生成器
    print("\n初始化生态图像生成器...")
    generator = EcologyImageGenerator()
    print(f"生成器设备: {generator.device}")
    
    # 构建模型
    print("\n构建模型...")
    generator.build_model()
    print("模型构建完成")
    
    # 测试GAN生成
    print("\n=== 测试GAN图像生成 ===")
    try:
        # 设置生成模式为GAN
        generator.generation_mode = "gan"
        
        # 准备输入条件
        conditions = [0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4, 0.5, 0.8]
        input_data = {
            "conditions": conditions
        }
        
        print(f"输入条件: {conditions}")
        print("开始生成图像...")
        
        # 生成图像
        result = generator.predict(input_data, num_images=2)
        
        if "error" in result:
            print(f"生成失败: {result['error']}")
        else:
            print(f"生成成功!")
            print(f"生成模式: {result.get('generation_mode', 'unknown')}")
            print(f"生成的图像数量: {len(result.get('generated_images', []))}")
            
            # 检查图像数据
            images = result.get('generated_images', [])
            if images:
                image_shape = np.array(images[0]).shape
                print(f"图像形状: {image_shape}")
                print(f"图像数据类型: {type(images[0])}")
                
                # 检查像素值范围
                img_array = np.array(images[0])
                print(f"像素值范围: [{img_array.min():.3f}, {img_array.max():.3f}]")
    
    except Exception as e:
        print(f"GAN生成测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试扩散模型生成
    print("\n=== 测试扩散模型图像生成 ===")
    try:
        # 设置生成模式为扩散模型
        generator.generation_mode = "diffusion"
        
        # 准备输入提示
        input_data = {
            "prompt": "A beautiful forest with clean air and wildlife, environmental protection concept",
            "negative_prompt": "pollution, smog, industrial waste"
        }
        
        print(f"输入提示: {input_data['prompt']}")
        print("开始生成图像...")
        
        # 生成图像
        result = generator.predict(input_data, num_images=1)
        
        if "error" in result:
            print(f"生成失败: {result['error']}")
        else:
            print(f"生成成功!")
            print(f"生成模式: {result.get('generation_mode', 'unknown')}")
            print(f"生成的图像数量: {len(result.get('generated_images', []))}")
    
    except Exception as e:
        print(f"扩散模型生成测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # GPU内存状态
    if torch.cuda.is_available():
        print("\n=== GPU内存状态 ===")
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"已分配内存: {allocated:.1f}MB")
        print(f"已缓存内存: {cached:.1f}MB")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        print("GPU内存已清理")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_gpu_image_generation()