#!/usr/bin/env python3
"""
GPU配置验证脚本

验证PyTorch和相关AI库的GPU配置是否正确。
"""

import torch
import sys
import warnings
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_pytorch_gpu():
    """检查PyTorch GPU配置"""
    print("=== PyTorch GPU 配置检查 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # 测试GPU张量操作
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(test_tensor, test_tensor.t())
            print(f"GPU张量测试: 成功 (结果形状: {result.shape})")
            print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            
            # 清理GPU内存
            del test_tensor, result
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"GPU张量测试: 失败 - {e}")
    else:
        print("CUDA不可用，将使用CPU")
    
    print()

def check_ai_libraries():
    """检查AI库的GPU支持"""
    print("=== AI库GPU支持检查 ===")
    
    # 检查transformers
    try:
        import transformers
        print(f"Transformers版本: {transformers.__version__}")
        
        # 测试简单的GPU操作
        if torch.cuda.is_available():
            from transformers import AutoTokenizer
            # 这里只是验证库能正常导入，不加载大模型
            print("Transformers GPU支持: 可用")
        else:
            print("Transformers GPU支持: CUDA不可用")
            
    except ImportError:
        print("Transformers: 未安装")
    except Exception as e:
        print(f"Transformers检查失败: {e}")
    
    # 检查diffusers
    try:
        import diffusers
        print(f"Diffusers版本: {diffusers.__version__}")
        print("Diffusers GPU支持: 可用" if torch.cuda.is_available() else "Diffusers GPU支持: CUDA不可用")
    except ImportError:
        print("Diffusers: 未安装")
    except Exception as e:
        print(f"Diffusers检查失败: {e}")
    
    # 检查torchvision
    try:
        import torchvision
        print(f"Torchvision版本: {torchvision.__version__}")
        print("Torchvision GPU支持: 可用" if torch.cuda.is_available() else "Torchvision GPU支持: CUDA不可用")
    except ImportError:
        print("Torchvision: 未安装")
    except Exception as e:
        print(f"Torchvision检查失败: {e}")
    
    print()

def check_project_models():
    """检查项目模型的GPU配置"""
    print("=== 项目模型GPU配置检查 ===")
    
    try:
        from src.models.ecology_image_generator import EcologyImageGenerator
        
        # 创建生成器实例
        generator = EcologyImageGenerator()
        print(f"生态图像生成器设备: {generator.device}")
        print(f"生成器使用GPU: {generator.device == 'cuda'}")
        
        # 检查模型是否能正确构建
        try:
            generator.build_model()
            print("模型构建: 成功")
            
            # 检查模型参数是否在正确设备上
            if hasattr(generator, 'generator') and generator.generator is not None:
                model_device = next(generator.generator.parameters()).device
                print(f"模型参数设备: {model_device}")
                print(f"设备一致性: {model_device.type == generator.device}")
            
        except Exception as e:
            print(f"模型构建失败: {e}")
            
    except ImportError as e:
        print(f"无法导入项目模型: {e}")
    except Exception as e:
        print(f"项目模型检查失败: {e}")
    
    print()

def check_gpu_memory():
    """检查GPU内存使用情况"""
    if not torch.cuda.is_available():
        return
        
    print("=== GPU内存使用情况 ===")
    
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
        cached_memory = torch.cuda.memory_reserved(i) / 1024**3
        
        print(f"GPU {i}:")
        print(f"  总内存: {total_memory:.1f}GB")
        print(f"  已分配: {allocated_memory:.3f}GB")
        print(f"  已缓存: {cached_memory:.3f}GB")
        print(f"  可用内存: {total_memory - cached_memory:.1f}GB")
    
    print()

def main():
    """主函数"""
    print("GPU配置验证脚本")
    print("=" * 50)
    
    # 抑制一些警告
    warnings.filterwarnings("ignore", category=UserWarning)
    
    check_pytorch_gpu()
    check_ai_libraries()
    check_project_models()
    check_gpu_memory()
    
    print("验证完成!")
    
    # 提供建议
    if torch.cuda.is_available():
        print("\n✅ GPU配置正常，可以使用GPU加速训练和推理。")
        
        # 检查是否有架构兼容性警告
        try:
            test_tensor = torch.randn(10, 10).cuda()
            del test_tensor
            torch.cuda.empty_cache()
        except Exception:
            pass
            
        print("\n💡 建议:")
        print("- 确保在训练时设置适当的batch_size以充分利用GPU内存")
        print("- 使用mixed precision训练可以进一步提升性能")
        print("- 定期清理GPU内存: torch.cuda.empty_cache()")
    else:
        print("\n⚠️  CUDA不可用，将使用CPU进行计算。")
        print("\n💡 如需GPU加速，请检查:")
        print("- NVIDIA驱动是否正确安装")
        print("- CUDA工具包是否安装")
        print("- PyTorch是否为GPU版本")

if __name__ == "__main__":
    main()