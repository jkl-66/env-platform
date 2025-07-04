#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速检查 GPU 支持情况
"""

try:
    import torch
    print("✓ PyTorch 导入成功")
    
    # 检查 CUDA
    if hasattr(torch, 'cuda'):
        print("✓ torch.cuda 模块存在")
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 可用: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                capability = torch.cuda.get_device_capability(i)
                print(f"GPU {i}: {gpu_name} (计算能力: {capability[0]}.{capability[1]})")
                
                # 检查是否支持 RTX 5070 Ti
                if "5070" in gpu_name.upper():
                    print(f"🎉 检测到 RTX 5070 Ti: {gpu_name}")
                    if capability[0] >= 12:  # sm_120
                        print("✓ 计算能力支持 RTX 5070 Ti")
                    else:
                        print("✗ 计算能力不足")
        else:
            print("❌ CUDA 不可用")
    else:
        print("❌ torch.cuda 模块不存在")
        
except ImportError as e:
    print(f"❌ PyTorch 导入失败: {e}")
except Exception as e:
    print(f"❌ 检查过程出错: {e}")