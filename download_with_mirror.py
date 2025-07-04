#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用镜像站点下载SD3.5模型
"""

import os
import shutil
import logging
from pathlib import Path
import requests
import time
from urllib.parse import urljoin
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MirrorDownloader:
    def __init__(self):
        self.cache_dir = Path("./cache/huggingface")
        self.model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
        self.model_cache_path = self.cache_dir / "models--stabilityai--stable-diffusion-3.5-large-turbo"
        
        # 镜像站点列表
        self.mirrors = [
            "https://hf-mirror.com",
            "https://huggingface.co",  # 原站作为备选
        ]
        
        # 必需的文件列表
        self.required_files = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json", 
            "text_encoder_2/model.safetensors",
            "text_encoder_3/config.json",
            "text_encoder_3/model.safetensors",
            "tokenizer/tokenizer_config.json",
            "tokenizer/tokenizer.json",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/tokenizer.json",
            "tokenizer_3/tokenizer_config.json",
            "tokenizer_3/tokenizer.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors"
        ]
    
    def test_mirror_connection(self, mirror_url):
        """测试镜像站点连接"""
        try:
            test_url = f"{mirror_url}/{self.model_id}/resolve/main/model_index.json"
            response = requests.head(test_url, timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ 镜像可用: {mirror_url}")
                return True
            else:
                logger.warning(f"❌ 镜像不可用: {mirror_url} (状态码: {response.status_code})")
                return False
        except Exception as e:
            logger.warning(f"❌ 镜像连接失败: {mirror_url} - {e}")
            return False
    
    def download_file(self, mirror_url, file_path, local_path, max_retries=3):
        """下载单个文件"""
        download_url = f"{mirror_url}/{self.model_id}/resolve/main/{file_path}"
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"📥 下载 {file_path} (尝试 {attempt}/{max_retries})")
                
                # 确保目录存在
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 下载文件
                response = requests.get(download_url, stream=True, timeout=30)
                response.raise_for_status()
                
                # 写入文件
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 验证文件大小
                file_size = local_path.stat().st_size
                if file_size > 0:
                    logger.info(f"✅ {file_path} 下载完成 ({file_size:,} bytes)")
                    return True
                else:
                    logger.error(f"❌ {file_path} 文件为空")
                    local_path.unlink(missing_ok=True)
                    
            except Exception as e:
                logger.error(f"❌ {file_path} 下载失败 (尝试 {attempt}): {e}")
                local_path.unlink(missing_ok=True)
                
                if attempt < max_retries:
                    time.sleep(5 * attempt)
        
        return False
    
    def create_snapshot_structure(self):
        """创建snapshot目录结构"""
        import hashlib
        import time
        
        # 生成一个简单的commit hash
        commit_hash = hashlib.sha1(str(time.time()).encode()).hexdigest()
        
        snapshot_dir = self.model_cache_path / "snapshots" / commit_hash
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        return snapshot_dir
    
    def download_model_from_mirror(self):
        """从镜像站点下载模型"""
        logger.info("🔍 测试镜像站点连接...")
        
        # 找到可用的镜像
        working_mirror = None
        for mirror in self.mirrors:
            if self.test_mirror_connection(mirror):
                working_mirror = mirror
                break
        
        if not working_mirror:
            logger.error("❌ 所有镜像站点都不可用")
            return False
        
        logger.info(f"🌐 使用镜像: {working_mirror}")
        
        # 清理现有缓存
        if self.model_cache_path.exists():
            logger.info("🗑️  清理现有缓存...")
            shutil.rmtree(self.model_cache_path)
        
        # 创建snapshot目录
        snapshot_dir = self.create_snapshot_structure()
        logger.info(f"📁 创建snapshot目录: {snapshot_dir.name}")
        
        # 下载所有必需文件
        success_count = 0
        total_files = len(self.required_files)
        
        for file_path in self.required_files:
            local_path = snapshot_dir / file_path
            
            if self.download_file(working_mirror, file_path, local_path):
                success_count += 1
            else:
                logger.error(f"❌ 关键文件下载失败: {file_path}")
        
        logger.info(f"📊 下载统计: {success_count}/{total_files} 文件成功")
        
        if success_count == total_files:
            logger.info("✅ 所有文件下载完成")
            return True
        else:
            logger.error(f"❌ 下载不完整，缺少 {total_files - success_count} 个文件")
            return False
    
    def verify_download(self):
        """验证下载完整性"""
        logger.info("🔍 验证下载完整性...")
        
        snapshots_dir = self.model_cache_path / "snapshots"
        if not snapshots_dir.exists():
            logger.error("❌ snapshots目录不存在")
            return False
        
        # 找到snapshot目录
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshot_dirs:
            logger.error("❌ 未找到snapshot目录")
            return False
        
        snapshot_dir = snapshot_dirs[0]
        logger.info(f"📁 验证snapshot: {snapshot_dir.name}")
        
        # 检查所有必需文件
        missing_files = []
        for file_path in self.required_files:
            full_path = snapshot_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            elif full_path.stat().st_size == 0:
                missing_files.append(f"{file_path} (空文件)")
        
        if missing_files:
            logger.error(f"❌ 缺少文件: {missing_files[:5]}...")  # 只显示前5个
            return False
        
        logger.info("✅ 所有必需文件验证通过")
        return True
    
    def run_mirror_download(self):
        """执行镜像下载"""
        logger.info("="*60)
        logger.info("🪞 SD3.5模型镜像下载工具")
        logger.info("="*60)
        
        # 下载模型
        if self.download_model_from_mirror():
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
    downloader = MirrorDownloader()
    
    success = downloader.run_mirror_download()
    
    if success:
        print("\n" + "="*60)
        print("✅ SD3.5模型准备就绪！")
        print("📋 下一步:")
        print("   python test_sd35_model.py  # 测试模型")
        print("   python demo_environmental_generator.py  # 开始使用")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 镜像下载失败")
        print("🔧 建议:")
        print("   1. 检查网络连接")
        print("   2. 尝试使用VPN")
        print("   3. 检查防火墙设置")
        print("   4. 稍后重试")
        print("="*60)

if __name__ == "__main__":
    main()