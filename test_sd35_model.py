#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SD3.5模型测试脚本
验证模型是否能正常加载和生成图像
"""

import os
import torch
import logging
from pathlib import Path
from datetime import datetime
from environmental_image_generator import EnvironmentalImageGenerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_loading():
    """测试模型加载"""
    logger.info("🧪 测试SD3.5模型加载...")
    
    try:
        # 设置离线模式
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # 创建生成器实例
        generator = EnvironmentalImageGenerator(
            model_id="stabilityai/stable-diffusion-3.5-large-turbo",
            cache_dir="./cache/huggingface",
            device="auto"
        )
        
        logger.info("✅ 环境图像生成器初始化成功")
        
        # 尝试加载模型
        success = generator.load_model()
        
        if success:
            logger.info("✅ SD3.5模型加载成功！")
            return True, generator
        else:
            logger.error("❌ SD3.5模型加载失败")
            return False, None
            
    except Exception as e:
        logger.error(f"❌ 模型加载过程中发生错误: {e}")
        return False, None

def test_image_generation(generator):
    """测试图像生成"""
    logger.info("🎨 测试图像生成...")
    
    try:
        # 测试用例
        test_prompts = [
            "工厂烟囱冒出黑烟，城市被雾霾笼罩",
            "polluted river with industrial waste"
        ]
        
        results = []
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"测试 {i}/{len(test_prompts)}: {prompt}")
            
            try:
                # 生成图像
                result = generator.generate_and_save(
                    user_input=prompt,
                    num_images=1,
                    guidance_scale=7.5,
                    num_inference_steps=20,
                    height=512,
                    width=512
                )
                
                if result['status'] == 'success':
                    logger.info(f"✅ 图像生成成功: {result['saved_paths']}")
                    results.append(result)
                else:
                    logger.error(f"❌ 图像生成失败: {result.get('error', '未知错误')}")
                    
            except Exception as e:
                logger.error(f"❌ 生成过程中发生错误: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 图像生成测试失败: {e}")
        return []

def test_configuration():
    """测试配置文件"""
    logger.info("⚙️  测试配置文件...")
    
    config_path = Path("config/environmental_prompts.json")
    
    if config_path.exists():
        logger.info("✅ 配置文件存在")
        
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            themes_count = len(config.get('environmental_prompts', {}))
            styles_count = len(config.get('style_presets', {}))
            
            logger.info(f"✅ 配置文件加载成功")
            logger.info(f"   - 环境主题: {themes_count}")
            logger.info(f"   - 风格预设: {styles_count}")
            
            return True, config
            
        except Exception as e:
            logger.error(f"❌ 配置文件解析失败: {e}")
            return False, None
    else:
        logger.error("❌ 配置文件不存在")
        return False, None

def check_system_requirements():
    """检查系统要求"""
    logger.info("🔍 检查系统要求...")
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ GPU: {gpu_name}")
        logger.info(f"   显存: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 8:
            logger.info("✅ 显存充足")
        else:
            logger.warning("⚠️  显存可能不足，建议8GB+")
    else:
        logger.warning("⚠️  未检测到GPU，将使用CPU模式")
    
    # 检查磁盘空间
    cache_dir = Path("./cache/huggingface")
    if cache_dir.exists():
        # 简单估算缓存大小
        total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        size_gb = total_size / 1024**3
        logger.info(f"📁 模型缓存大小: {size_gb:.1f} GB")
    
    # 检查输出目录
    output_dir = Path("outputs/environmental_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 输出目录: {output_dir}")

def main():
    """主测试函数"""
    logger.info("="*60)
    logger.info("🧪 SD3.5环境图像生成器完整测试")
    logger.info("="*60)
    
    # 检查系统要求
    check_system_requirements()
    print()
    
    # 测试配置文件
    config_ok, config = test_configuration()
    print()
    
    # 测试模型加载
    model_ok, generator = test_model_loading()
    print()
    
    if model_ok and generator:
        # 测试图像生成
        results = test_image_generation(generator)
        print()
        
        # 总结
        logger.info("📊 测试总结:")
        logger.info(f"   - 配置文件: {'✅ 正常' if config_ok else '❌ 异常'}")
        logger.info(f"   - 模型加载: {'✅ 正常' if model_ok else '❌ 异常'}")
        logger.info(f"   - 图像生成: {'✅ 正常' if results else '❌ 异常'}")
        
        if results:
            logger.info(f"   - 成功生成: {len(results)} 张图像")
            for result in results:
                for path in result.get('saved_paths', []):
                    logger.info(f"     📸 {path}")
        
        if config_ok and model_ok and results:
            print("\n" + "="*60)
            print("🎉 所有测试通过！SD3.5环境图像生成器已就绪")
            print("📋 可以开始使用:")
            print("   1. python demo_environmental_generator.py")
            print("   2. python environmental_image_generator.py")
            print("="*60)
            return True
        else:
            print("\n" + "="*60)
            print("⚠️  部分测试失败，请检查相关问题")
            print("="*60)
            return False
    else:
        print("\n" + "="*60)
        print("❌ 模型加载失败，无法进行图像生成测试")
        print("🔧 建议:")
        print("   1. 检查网络连接")
        print("   2. 重新运行 python force_download_sd35.py")
        print("   3. 或使用离线模式")
        print("="*60)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)