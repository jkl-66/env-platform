#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查PyTorch版本和CUDA支持
"""

import torch
import sys

def check_pytorch_info():
    print("=== PyTorch版本信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Python版本: {sys.version}")
    
    if hasattr(torch.version, 'cuda') and torch.version.cuda:
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("CUDA版本: 未安装CUDA版本")
    
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - 计算能力: {props.major}.{props.minor}")
            print(f"  - 总内存: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("GPU数量: 0 (CUDA不可用)")
    
    # 检查是否支持sm_120架构
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        compute_capability = f"{props.major}{props.minor}"
        print(f"\n当前GPU计算能力: sm_{compute_capability}")
        
        if compute_capability == "120":
            print("✅ 当前GPU为sm_120架构 (RTX 5070 Ti)")
            print("需要PyTorch 2.5+和CUDA 12.8+支持")
        else:
            print(f"当前GPU架构: sm_{compute_capability}")

if __name__ == "__main__":
    check_pytorch_info()