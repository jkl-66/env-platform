#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强制下载 Stable Diffusion 3.5 Large Turbo 模型
确保模型能够成功下载并缓存到本地
"""

import os
import time
import logging
from pathlib import Path
from huggingface_hub import snapshot_download, login
from diffusers import StableDiffusion3Pipeline
import torch

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SD35ModelDownloader:
    def __init__(self, cache_dir=None):
        self.model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
        self.cache_dir = cache_dir or "./cache/huggingface"
        self.max_retries = 5
        self.retry_delay = 10  # 秒
        
        # 确保缓存目录存在
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # 设置环境变量
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['TRANSFORMERS_CACHE'] = self.cache_dir
        
    def check_network_connection(self):
        """检查网络连接"""
        try:
            import requests
            response = requests.get('https://huggingface.co', timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"网络连接检查失败: {e}")
            return False
    
    def download_with_retry(self):
        """带重试机制的模型下载"""
        logger.info(f"开始下载 {self.model_id} 模型...")
        logger.info(f"缓存目录: {self.cache_dir}")
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"第 {attempt}/{self.max_retries} 次尝试下载...")
                
                # 检查网络连接
                if not self.check_network_connection():
                    logger.warning("网络连接不稳定，等待后重试...")
                    time.sleep(self.retry_delay)
                    continue
                
                # 使用 snapshot_download 下载完整模型
                logger.info("正在下载模型文件...")
                local_dir = snapshot_download(
                    repo_id=self.model_id,
                    cache_dir=self.cache_dir,
                    resume_download=True,
                    local_files_only=False,
                    force_download=False
                )
                
                logger.info(f"✅ 模型下载成功！本地路径: {local_dir}")
                return local_dir
                
            except Exception as e:
                logger.error(f"第 {attempt} 次下载失败: {e}")
                if attempt < self.max_retries:
                    logger.info(f"等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("所有下载尝试都失败了")
                    raise
        
        return None
    
    def verify_model_download(self):
        """验证模型是否下载完整"""
        try:
            logger.info("验证模型完整性...")
            
            # 尝试加载模型以验证完整性
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            logger.info("✅ 模型验证成功！")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型验证失败: {e}")
            return False
    
    def get_model_info(self):
        """获取模型信息"""
        try:
            from huggingface_hub import model_info
            info = model_info(self.model_id)
            
            logger.info(f"模型信息:")
            logger.info(f"  - 模型ID: {info.modelId}")
            logger.info(f"  - 下载次数: {info.downloads}")
            logger.info(f"  - 最后修改: {info.lastModified}")
            logger.info(f"  - 模型大小: {info.safetensors}")
            
            return info
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return None
    
    def force_download(self):
        """强制下载模型的主方法"""
        logger.info("="*60)
        logger.info("🚀 Stable Diffusion 3.5 Large Turbo 强制下载器")
        logger.info("="*60)
        
        # 检查GPU
        if torch.cuda.is_available():
            logger.info(f"✅ 检测到GPU: {torch.cuda.get_device_name()}")
            logger.info(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.warning("⚠️  未检测到GPU，将使用CPU模式")
        
        # 获取模型信息
        self.get_model_info()
        
        # 下载模型
        try:
            local_path = self.download_with_retry()
            if local_path:
                # 验证模型
                if self.verify_model_download():
                    logger.info("🎉 SD3.5模型下载和验证完成！")
                    logger.info(f"📁 模型缓存位置: {self.cache_dir}")
                    logger.info("现在可以运行 demo_environmental_generator.py 进行图像生成")
                    return True
                else:
                    logger.error("模型验证失败，请重新下载")
                    return False
            else:
                logger.error("模型下载失败")
                return False
                
        except Exception as e:
            logger.error(f"下载过程中发生错误: {e}")
            return False

def main():
    """主函数"""
    # 创建下载器
    downloader = SD35ModelDownloader()
    
    # 强制下载
    success = downloader.force_download()
    
    if success:
        print("\n" + "="*60)
        print("✅ SD3.5模型准备就绪！")
        print("📋 下一步操作:")
        print("   1. 运行 python demo_environmental_generator.py")
        print("   2. 或运行 python environmental_image_generator.py")
        print("   3. 开始生成环境保护警示图像")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 模型下载失败")
        print("🔧 故障排除建议:")
        print("   1. 检查网络连接")
        print("   2. 检查磁盘空间（需要约8GB）")
        print("   3. 尝试使用VPN或更换网络")
        print("   4. 重新运行此脚本")
        print("="*60)

if __name__ == "__main__":
    main()