#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实AI图像生成配置指南

本脚本展示如何配置真正的AI图像生成模型来替代示例图像
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def show_setup_guide():
    """显示真实AI图像生成的配置指南"""
    print("🎨 真实AI图像生成配置指南")
    print("=" * 60)
    
    print("\n📋 当前状态:")
    print("• 系统目前使用示例图像数据来模拟AI图像生成")
    print("• 要获得真正的AI生成图像作品，需要配置以下模型之一:")
    
    print("\n🔧 配置选项:")
    
    print("\n1️⃣ 使用Stable Diffusion (推荐)")
    print("   • 安装: pip install diffusers transformers accelerate")
    print("   • 需要: 4GB+ GPU显存")
    print("   • 优点: 高质量图像，开源免费")
    print("   • 示例代码:")
    print("     ```python")
    print("     from diffusers import StableDiffusionPipeline")
    print("     pipe = StableDiffusionPipeline.from_pretrained(")
    print("         'runwayml/stable-diffusion-v1-5'")
    print("     )")
    print("     ```")
    
    print("\n2️⃣ 使用DALL-E API")
    print("   • 安装: pip install openai")
    print("   • 需要: OpenAI API密钥")
    print("   • 优点: 高质量，无需本地GPU")
    print("   • 缺点: 需要付费API")
    
    print("\n3️⃣ 使用本地GAN模型")
    print("   • 需要: 训练好的GAN模型")
    print("   • 优点: 完全本地化")
    print("   • 缺点: 需要大量训练数据")
    
    print("\n🛠️ 修改步骤:")
    print("1. 选择上述配置选项之一")
    print("2. 安装相应的依赖包")
    print("3. 修改 ecology_image_generator.py 中的 _generate_with_diffusion 方法")
    print("4. 确保 diffusion_pipeline 正确初始化")
    
    print("\n📁 相关文件:")
    print(f"• 主要生成器: {project_root}/src/models/ecology_image_generator.py")
    print(f"• 交互界面: {project_root}/scripts/interactive_ecology_image_demo.py")
    print(f"• 配置示例: {project_root}/docs/huggingface_setup_guide.md")
    
    print("\n⚠️ 重要提示:")
    print("• 真实的AI图像生成需要较大的计算资源")
    print("• 建议使用GPU加速以获得更好的性能")
    print("• 生成时间可能需要几秒到几分钟不等")
    print("• 确保有足够的磁盘空间存储模型文件")

def create_stable_diffusion_example():
    """创建Stable Diffusion配置示例"""
    example_code = '''
# Stable Diffusion 配置示例
# 在 ecology_image_generator.py 的 _build_model 方法中添加:

try:
    from diffusers import StableDiffusionPipeline
    import torch
    
    # 检查GPU可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载Stable Diffusion模型
    self.diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,  # 可选：禁用安全检查器
        requires_safety_checker=False
    )
    
    self.diffusion_pipeline = self.diffusion_pipeline.to(device)
    
    # 启用内存优化（如果使用GPU）
    if device == "cuda":
        self.diffusion_pipeline.enable_attention_slicing()
        self.diffusion_pipeline.enable_memory_efficient_attention()
    
    logger.info(f"Stable Diffusion模型已加载到 {device}")
    
except ImportError:
    logger.warning("diffusers库未安装，无法使用Stable Diffusion")
    self.diffusion_pipeline = None
except Exception as e:
    logger.warning(f"Stable Diffusion加载失败: {e}")
    self.diffusion_pipeline = None
'''
    
    example_file = project_root / "examples" / "stable_diffusion_setup.py"
    example_file.parent.mkdir(exist_ok=True)
    
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print(f"\n📄 Stable Diffusion配置示例已保存到: {example_file}")

def check_current_setup():
    """检查当前的图像生成配置"""
    print("\n🔍 检查当前配置...")
    
    try:
        # 检查diffusers
        import diffusers
        print(f"✅ diffusers 已安装 (版本: {diffusers.__version__})")
    except ImportError:
        print("❌ diffusers 未安装")
    
    try:
        # 检查transformers
        import transformers
        print(f"✅ transformers 已安装 (版本: {transformers.__version__})")
    except ImportError:
        print("❌ transformers 未安装")
    
    try:
        # 检查torch
        import torch
        print(f"✅ torch 已安装 (版本: {torch.__version__})")
        print(f"   GPU可用: {'是' if torch.cuda.is_available() else '否'}")
        if torch.cuda.is_available():
            print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"   GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    except ImportError:
        print("❌ torch 未安装")
    
    try:
        # 检查PIL
        from PIL import Image
        print("✅ PIL 已安装")
    except ImportError:
        print("❌ PIL 未安装")
    
    # 检查生成器配置
    try:
        from src.models.ecology_image_generator import EcologyImageGenerator
        generator = EcologyImageGenerator()
        
        if hasattr(generator, 'diffusion_pipeline') and generator.diffusion_pipeline is not None:
            print("✅ 扩散模型已配置")
        else:
            print("⚠️ 扩散模型未配置，将使用示例图像")
            
    except Exception as e:
        print(f"❌ 生成器检查失败: {e}")

def install_dependencies():
    """安装必要的依赖"""
    print("\n📦 安装AI图像生成依赖...")
    
    dependencies = [
        "diffusers>=0.21.0",
        "transformers>=4.25.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
        "xformers; sys_platform != 'darwin'",  # 不在macOS上安装xformers
    ]
    
    print("将安装以下包:")
    for dep in dependencies:
        print(f"  • {dep}")
    
    confirm = input("\n是否继续安装? (y/N): ").strip().lower()
    if confirm == 'y':
        import subprocess
        import sys
        
        for dep in dependencies:
            try:
                print(f"\n安装 {dep}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"✅ {dep} 安装成功")
            except subprocess.CalledProcessError as e:
                print(f"❌ {dep} 安装失败: {e}")
    else:
        print("安装已取消")

def main():
    """主函数"""
    print("🎨 AI图像生成配置工具")
    print("=" * 40)
    
    while True:
        print("\n请选择操作:")
        print("1. 📋 查看配置指南")
        print("2. 🔍 检查当前配置")
        print("3. 📦 安装依赖包")
        print("4. 📄 创建配置示例")
        print("5. 🚪 退出")
        
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == '1':
            show_setup_guide()
        elif choice == '2':
            check_current_setup()
        elif choice == '3':
            install_dependencies()
        elif choice == '4':
            create_stable_diffusion_example()
        elif choice == '5':
            print("\n👋 再见！")
            break
        else:
            print("\n❌ 无效选择，请重新输入")

if __name__ == "__main__":
    main()