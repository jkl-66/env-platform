#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU图像生成测试脚本
由于GPU兼容性问题，使用CPU进行图像生成测试
"""

import sys
import torch
import numpy as np
from pathlib import Path
import os

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.models.ecology_image_generator import EcologyImageGenerator

def test_cpu_image_generation():
    """测试CPU图像生成功能"""
    print("=== CPU图像生成测试 ===")
    
    # 强制使用CPU
    print("强制使用CPU设备进行图像生成")
    
    # 初始化生成器（强制使用CPU）
    print("\n初始化生态图像生成器...")
    generator = EcologyImageGenerator(device="cpu")
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
                
                # 保存生成的图像
                save_generated_images(images, "gan_generated")
    
    except Exception as e:
        print(f"GAN生成测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试扩散模型生成（如果可用）
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
            
            # 保存生成的图像
            images = result.get('generated_images', [])
            if images:
                save_generated_images(images, "diffusion_generated")
    
    except Exception as e:
        print(f"扩散模型生成测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 测试完成 ===")

def save_generated_images(images, prefix="generated"):
    """保存生成的图像"""
    try:
        from PIL import Image
        import datetime
        
        # 创建输出目录
        output_dir = Path("outputs/generated_images")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, img_data in enumerate(images):
            # 转换图像数据
            img_array = np.array(img_data)
            
            # 确保数据在[0, 1]范围内
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
            
            # 如果是灰度图像，转换为RGB
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # 创建PIL图像
            img = Image.fromarray(img_array)
            
            # 保存图像
            filename = f"{prefix}_{timestamp}_{i+1}.png"
            filepath = output_dir / filename
            img.save(filepath)
            
            print(f"图像已保存: {filepath}")
    
    except Exception as e:
        print(f"保存图像失败: {e}")

def test_simple_generation():
    """测试简单的图像生成功能"""
    print("\n=== 简单图像生成测试 ===")
    
    try:
        # 创建简单的随机图像
        print("生成随机测试图像...")
        
        # 生成64x64的RGB图像
        random_image = np.random.rand(64, 64, 3)
        
        # 保存测试图像
        save_generated_images([random_image], "test_random")
        
        print("随机图像生成完成")
        
    except Exception as e:
        print(f"简单生成测试失败: {e}")

if __name__ == "__main__":
    test_cpu_image_generation()
    test_simple_generation()