#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动清理不完整的模型缓存并重新下载SD3.5（无需用户确认）
"""

import os
import shutil
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoModelCacheCleaner:
    def __init__(self):
        self.cache_dir = Path("./cache/huggingface")
        self.model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
        self.model_cache_path = self.cache_dir / "models--stabilityai--stable-diffusion-3.5-large-turbo"
        
    def clean_entire_cache(self):
        """完全清理模型缓存"""
        logger.info("🗑️  完全清理模型缓存...")
        
        if self.model_cache_path.exists():
            try:
                shutil.rmtree(self.model_cache_path)
                logger.info("✅ 模型缓存已完全清理")
                return True
            except Exception as e:
                logger.error(f"❌ 清理失败: {e}")
                return False
        else:
            logger.info("✅ 缓存目录不存在，无需清理")
            return True
    
    def download_model_with_retry(self, max_retries=5):
        """重新下载模型"""
        logger.info(f"📥 开始重新下载模型 (最多重试 {max_retries} 次)...")
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"🔄 第 {attempt}/{max_retries} 次尝试...")
                
                # 确保缓存目录存在
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                
                # 下载模型
                local_dir = snapshot_download(
                    repo_id=self.model_id,
                    cache_dir=str(self.cache_dir),
                    resume_download=True,
                    local_files_only=False,
                    force_download=True,  # 强制重新下载
                    token=None
                )
                
                logger.info(f"✅ 模型下载成功: {local_dir}")
                return True, local_dir
                
            except Exception as e:
                logger.error(f"❌ 第 {attempt} 次下载失败: {e}")
                if attempt < max_retries:
                    wait_time = 30 * attempt  # 递增等待时间
                    logger.info(f"⏳ 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error("❌ 所有下载尝试都失败了")
        
        return False, None
    
    def verify_download(self):
        """验证下载完整性"""
        logger.info("🔍 验证下载完整性...")
        
        # 检查关键目录
        required_dirs = [
            "scheduler",
            "text_encoder",
            "text_encoder_2", 
            "text_encoder_3",
            "tokenizer",
            "tokenizer_2",
            "tokenizer_3",
            "transformer",
            "vae"
        ]
        
        snapshots_dir = self.model_cache_path / "snapshots"
        if not snapshots_dir.exists():
            logger.error("❌ snapshots目录不存在")
            return False
        
        # 找到最新的snapshot
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshot_dirs:
            logger.error("❌ 未找到snapshot目录")
            return False
        
        latest_snapshot = snapshot_dirs[0]  # 假设只有一个
        logger.info(f"📁 检查snapshot: {latest_snapshot.name}")
        
        missing_dirs = []
        empty_dirs = []
        
        for req_dir in required_dirs:
            dir_path = latest_snapshot / req_dir
            if not dir_path.exists():
                missing_dirs.append(req_dir)
            else:
                # 检查目录是否为空
                files_in_dir = list(dir_path.iterdir())
                if not files_in_dir:
                    empty_dirs.append(req_dir)
                else:
                    logger.info(f"✅ {req_dir}: {len(files_in_dir)} 个文件")
        
        if missing_dirs:
            logger.error(f"❌ 缺少关键目录: {missing_dirs}")
            return False
            
        if empty_dirs:
            logger.error(f"❌ 空目录: {empty_dirs}")
            return False
        
        # 检查是否还有不完整文件
        blobs_dir = self.model_cache_path / "blobs"
        if blobs_dir.exists():
            incomplete_files = list(blobs_dir.glob("*.incomplete"))
            if incomplete_files:
                logger.error(f"❌ 仍有不完整文件: {len(incomplete_files)}")
                return False
        
        logger.info("✅ 下载验证通过")
        return True
    
    def run_auto_clean_and_redownload(self):
        """执行自动清理和重新下载"""
        logger.info("="*60)
        logger.info("🔧 SD3.5模型自动清理和重新下载工具")
        logger.info("="*60)
        
        # 强制完全清理
        logger.info("🗑️  强制完全清理模式（自动确认）")
        if not self.clean_entire_cache():
            return False
        
        # 重新下载
        success, local_dir = self.download_model_with_retry()
        
        if success:
            # 验证下载
            if self.verify_download():
                logger.info("🎉 模型下载和验证完成！")
                return True
            else:
                logger.error("❌ 下载验证失败")
                return False
        else:
            logger.error("❌ 模型下载失败")
            return False

def main():
    """主函数"""
    cleaner = AutoModelCacheCleaner()
    
    success = cleaner.run_auto_clean_and_redownload()
    
    if success:
        print("\n" + "="*60)
        print("✅ SD3.5模型准备就绪！")
        print("📋 下一步:")
        print("   python test_sd35_model.py  # 测试模型")
        print("   python demo_environmental_generator.py  # 开始使用")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 清理和下载失败")
        print("🔧 建议:")
        print("   1. 检查网络连接")
        print("   2. 检查磁盘空间（需要约8GB）")
        print("   3. 尝试使用VPN或代理")
        print("   4. 检查防火墙设置")
        print("="*60)

if __name__ == "__main__":
    main()