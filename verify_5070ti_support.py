#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 PyTorch 安装是否支持 RTX 5070 Ti 显卡
"""

import torch
import sys

def check_5070ti_support():
    print("=== RTX 5070 Ti 支持验证 ===")
    print()
    
    # 检查 PyTorch 版本
    import pkg_resources
    try:
        pytorch_version = pkg_resources.get_distribution('torch').version
    except:
        pytorch_version = 'Unknown'
    print(f"PyTorch 版本: {pytorch_version}")
    
    # 检查 CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")
    
    if cuda_available:
        # 检查 CUDA 版本
        cuda_version = torch.version.cuda
        print(f"CUDA 版本: {cuda_version}")
        
        # 检查 GPU 数量
        gpu_count = torch.cuda.device_count()
        print(f"GPU 数量: {gpu_count}")
        
        if gpu_count > 0:
            # 检查每个 GPU 的信息
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_capability = torch.cuda.get_device_capability(i)
                print(f"GPU {i}: {gpu_name}")
                print(f"GPU {i} 计算能力: {gpu_capability[0]}.{gpu_capability[1]}")
                
                # 检查是否为 RTX 5070 Ti
                if "5070" in gpu_name.upper() or "RTX 50" in gpu_name.upper():
                    print(f"✓ 检测到 RTX 50 系列显卡: {gpu_name}")
                    
                    # RTX 5070 Ti 需要计算能力 sm_120 (12.0)
                    if gpu_capability[0] >= 12:
                        print("✓ GPU 计算能力支持 (>= sm_120)")
                    else:
                        print("✗ GPU 计算能力不足 (需要 >= sm_120)")
                        return False
                        
        # 检查 CUDA 版本是否支持 RTX 5070 Ti
        if cuda_version:
            cuda_major = int(cuda_version.split('.')[0])
            cuda_minor = int(cuda_version.split('.')[1])
            
            # RTX 5070 Ti 需要 CUDA 12.8+
            if cuda_major > 12 or (cuda_major == 12 and cuda_minor >= 8):
                print(f"✓ CUDA 版本支持 RTX 5070 Ti (>= 12.8): {cuda_version}")
            else:
                print(f"✗ CUDA 版本不支持 RTX 5070 Ti (需要 >= 12.8): {cuda_version}")
                return False
                
        # 测试简单的 GPU 操作
        try:
            device = torch.device('cuda:0')
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.matmul(test_tensor, test_tensor.T)
            print("✓ GPU 基本操作测试通过")
            
            # 测试内存
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2   # MB
            print(f"GPU 内存使用: {memory_allocated:.1f} MB (已分配), {memory_reserved:.1f} MB (已保留)")
            
        except Exception as e:
            print(f"✗ GPU 操作测试失败: {e}")
            return False
            
    else:
        print("✗ CUDA 不可用")
        print("\n可能的原因:")
        print("1. 没有安装 NVIDIA 驱动")
        print("2. 安装的是 CPU 版本的 PyTorch")
        print("3. CUDA 驱动版本不兼容")
        return False
    
    print("\n=== 总结 ===")
    if cuda_available and gpu_count > 0:
        print("✓ 系统支持 GPU 计算")
        
        # 检查是否为 nightly 版本 (RTX 5070 Ti 需要)
        if "dev" in str(pytorch_version) or "nightly" in str(pytorch_version).lower():
            print("✓ 使用 PyTorch nightly 版本 (RTX 5070 Ti 推荐)")
            return True
        else:
            print("⚠ 使用稳定版 PyTorch，RTX 5070 Ti 可能需要 nightly 版本")
            return False
    else:
        print("✗ 系统不支持 GPU 计算")
        return False

if __name__ == "__main__":
    try:
        success = check_5070ti_support()
        if success:
            print("\n🎉 恭喜！您的系统支持 RTX 5070 Ti")
        else:
            print("\n❌ 您的系统暂不支持 RTX 5070 Ti，请按照之前的指南安装相应版本")
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        sys.exit(1)