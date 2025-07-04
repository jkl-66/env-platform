#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SD3.5模型离线下载指南和工具
当网络连接有问题时的解决方案
"""

import os
import shutil
import logging
from pathlib import Path
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OfflineDownloadGuide:
    def __init__(self):
        self.cache_dir = Path("./cache/huggingface")
        self.model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
        self.model_cache_path = self.cache_dir / "models--stabilityai--stable-diffusion-3.5-large-turbo"
        
        # 文件下载链接（多个镜像源）
        self.download_links = {
            "hf-mirror.com": "https://hf-mirror.com/stabilityai/stable-diffusion-3.5-large-turbo/tree/main",
            "huggingface.co": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo/tree/main",
            "modelscope.cn": "https://modelscope.cn/models/AI-ModelScope/stable-diffusion-3.5-large-turbo"
        }
        
        # 必需的文件列表和大小（估算）
        self.required_files = {
            "model_index.json": "1KB",
            "scheduler/scheduler_config.json": "1KB",
            "text_encoder/config.json": "2KB",
            "text_encoder/model.safetensors": "246MB",
            "text_encoder_2/config.json": "1KB", 
            "text_encoder_2/model.safetensors": "5.1GB",
            "text_encoder_3/config.json": "1KB",
            "text_encoder_3/model.safetensors": "4.7GB",
            "tokenizer/tokenizer_config.json": "2KB",
            "tokenizer/tokenizer.json": "17MB",
            "tokenizer_2/tokenizer_config.json": "1KB",
            "tokenizer_2/tokenizer.json": "2.4MB",
            "tokenizer_3/tokenizer_config.json": "1KB",
            "tokenizer_3/tokenizer.json": "587KB",
            "transformer/config.json": "1KB",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": "4.9GB",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": "4.9GB",
            "transformer/diffusion_pytorch_model.safetensors.index.json": "25KB",
            "vae/config.json": "1KB",
            "vae/diffusion_pytorch_model.safetensors": "335MB"
        }
    
    def print_download_guide(self):
        """打印下载指南"""
        print("="*80)
        print("🔧 SD3.5模型离线下载指南")
        print("="*80)
        print()
        
        print("📋 方案一：手动下载（推荐）")
        print("-" * 40)
        print("1. 打开以下任一网站（建议按顺序尝试）：")
        for name, url in self.download_links.items():
            print(f"   • {name}: {url}")
        print()
        
        print("2. 下载以下文件到指定目录：")
        print(f"   目标目录: {self.model_cache_path / 'snapshots' / 'ec07796fc06b096cc56de9762974a28f4c632eda'}")
        print()
        
        # 按目录分组显示文件
        dirs = {}
        for file_path, size in self.required_files.items():
            dir_name = str(Path(file_path).parent) if '/' in file_path else '.'
            if dir_name not in dirs:
                dirs[dir_name] = []
            dirs[dir_name].append((Path(file_path).name, size))
        
        for dir_name, files in sorted(dirs.items()):
            if dir_name == '.':
                print("   📁 根目录:")
            else:
                print(f"   📁 {dir_name}/:")
            
            for filename, size in files:
                print(f"      • {filename} ({size})")
            print()
        
        total_size = "约20GB"
        print(f"   💾 总大小: {total_size}")
        print()
        
        print("📋 方案二：使用下载工具")
        print("-" * 40)
        print("1. 安装git-lfs:")
        print("   git lfs install")
        print()
        print("2. 克隆仓库:")
        print("   git clone https://hf-mirror.com/stabilityai/stable-diffusion-3.5-large-turbo")
        print("   或")
        print("   git clone https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo")
        print()
        
        print("📋 方案三：使用huggingface-hub")
        print("-" * 40)
        print("1. 安装依赖:")
        print("   pip install huggingface-hub")
        print()
        print("2. 设置镜像（可选）:")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        print()
        print("3. 下载模型:")
        print("   python -c \"from huggingface_hub import snapshot_download; snapshot_download('stabilityai/stable-diffusion-3.5-large-turbo', local_dir='./model')\"")
        print()
        
        print("📋 方案四：使用本地模型")
        print("-" * 40)
        print("如果您已经有其他SD3.5模型文件，可以：")
        print("1. 将模型文件复制到缓存目录")
        print("2. 运行 python setup_local_model.py 来配置")
        print()
    
    def create_directory_structure(self):
        """创建目录结构"""
        commit_hash = "ec07796fc06b096cc56de9762974a28f4c632eda"
        snapshot_dir = self.model_cache_path / "snapshots" / commit_hash
        
        # 创建所有必需的目录
        dirs_to_create = set()
        for file_path in self.required_files.keys():
            if '/' in file_path:
                dir_path = snapshot_dir / Path(file_path).parent
                dirs_to_create.add(dir_path)
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 创建目录: {dir_path.relative_to(self.cache_dir)}")
        
        return snapshot_dir
    
    def create_download_script(self):
        """创建下载脚本"""
        script_content = '''#!/bin/bash
# SD3.5模型下载脚本

echo "开始下载SD3.5模型文件..."

# 设置变量
MODEL_DIR="./cache/huggingface/models--stabilityai--stable-diffusion-3.5-large-turbo/snapshots/ec07796fc06b096cc56de9762974a28f4c632eda"
BASE_URL="https://hf-mirror.com/stabilityai/stable-diffusion-3.5-large-turbo/resolve/main"

# 创建目录
mkdir -p "$MODEL_DIR/scheduler"
mkdir -p "$MODEL_DIR/text_encoder"
mkdir -p "$MODEL_DIR/text_encoder_2"
mkdir -p "$MODEL_DIR/text_encoder_3"
mkdir -p "$MODEL_DIR/tokenizer"
mkdir -p "$MODEL_DIR/tokenizer_2"
mkdir -p "$MODEL_DIR/tokenizer_3"
mkdir -p "$MODEL_DIR/transformer"
mkdir -p "$MODEL_DIR/vae"

# 下载函数
download_file() {
    local file_path="$1"
    local url="$BASE_URL/$file_path"
    local output="$MODEL_DIR/$file_path"
    
    echo "下载: $file_path"
    curl -L -o "$output" "$url" || wget -O "$output" "$url"
    
    if [ $? -eq 0 ]; then
        echo "✅ $file_path 下载完成"
    else
        echo "❌ $file_path 下载失败"
    fi
}

# 下载所有文件
download_file "model_index.json"
download_file "scheduler/scheduler_config.json"
download_file "text_encoder/config.json"
download_file "text_encoder/model.safetensors"
download_file "text_encoder_2/config.json"
download_file "text_encoder_2/model.safetensors"
download_file "text_encoder_3/config.json"
download_file "text_encoder_3/model.safetensors"
download_file "tokenizer/tokenizer_config.json"
download_file "tokenizer/tokenizer.json"
download_file "tokenizer_2/tokenizer_config.json"
download_file "tokenizer_2/tokenizer.json"
download_file "tokenizer_3/tokenizer_config.json"
download_file "tokenizer_3/tokenizer.json"
download_file "transformer/config.json"
download_file "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
download_file "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
download_file "transformer/diffusion_pytorch_model.safetensors.index.json"
download_file "vae/config.json"
download_file "vae/diffusion_pytorch_model.safetensors"

echo "下载完成！"
'''
        
        script_path = Path("download_sd35.sh")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 创建Windows批处理文件
        bat_content = '''@echo off
echo 开始下载SD3.5模型文件...

set MODEL_DIR=.\cache\huggingface\models--stabilityai--stable-diffusion-3.5-large-turbo\snapshots\ec07796fc06b096cc56de9762974a28f4c632eda
set BASE_URL=https://hf-mirror.com/stabilityai/stable-diffusion-3.5-large-turbo/resolve/main

REM 创建目录
mkdir "%MODEL_DIR%\scheduler" 2>nul
mkdir "%MODEL_DIR%\text_encoder" 2>nul
mkdir "%MODEL_DIR%\text_encoder_2" 2>nul
mkdir "%MODEL_DIR%\text_encoder_3" 2>nul
mkdir "%MODEL_DIR%\tokenizer" 2>nul
mkdir "%MODEL_DIR%\tokenizer_2" 2>nul
mkdir "%MODEL_DIR%\tokenizer_3" 2>nul
mkdir "%MODEL_DIR%\transformer" 2>nul
mkdir "%MODEL_DIR%\vae" 2>nul

REM 下载文件（需要安装curl或wget）
echo 请手动下载以下文件到对应目录：
echo.
echo 根目录 (%MODEL_DIR%):
echo   - model_index.json
echo.
echo scheduler目录:
echo   - scheduler_config.json
echo.
echo text_encoder目录:
echo   - config.json
echo   - model.safetensors
echo.
echo text_encoder_2目录:
echo   - config.json
echo   - model.safetensors
echo.
echo text_encoder_3目录:
echo   - config.json
echo   - model.safetensors
echo.
echo tokenizer目录:
echo   - tokenizer_config.json
echo   - tokenizer.json
echo.
echo tokenizer_2目录:
echo   - tokenizer_config.json
echo   - tokenizer.json
echo.
echo tokenizer_3目录:
echo   - tokenizer_config.json
echo   - tokenizer.json
echo.
echo transformer目录:
echo   - config.json
echo   - diffusion_pytorch_model-00001-of-00002.safetensors
echo   - diffusion_pytorch_model-00002-of-00002.safetensors
echo   - diffusion_pytorch_model.safetensors.index.json
echo.
echo vae目录:
echo   - config.json
echo   - diffusion_pytorch_model.safetensors
echo.
echo 从以下地址下载: %BASE_URL%/[文件路径]
echo.
pause
'''
        
        bat_path = Path("download_sd35.bat")
        with open(bat_path, 'w', encoding='gbk') as f:
            f.write(bat_content)
        
        logger.info(f"✅ 下载脚本已创建: {script_path.name} 和 {bat_path.name}")
    
    def check_existing_files(self):
        """检查已存在的文件"""
        logger.info("🔍 检查现有文件...")
        
        snapshots_dir = self.model_cache_path / "snapshots"
        if not snapshots_dir.exists():
            logger.info("📁 snapshots目录不存在，需要下载所有文件")
            return []
        
        # 找到snapshot目录
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshot_dirs:
            logger.info("📁 未找到snapshot目录，需要下载所有文件")
            return []
        
        snapshot_dir = snapshot_dirs[0]
        existing_files = []
        
        for file_path in self.required_files.keys():
            full_path = snapshot_dir / file_path
            if full_path.exists() and full_path.stat().st_size > 0:
                existing_files.append(file_path)
        
        if existing_files:
            logger.info(f"✅ 找到 {len(existing_files)} 个现有文件")
            missing_files = set(self.required_files.keys()) - set(existing_files)
            if missing_files:
                logger.info(f"❌ 缺少 {len(missing_files)} 个文件:")
                for file_path in sorted(missing_files):
                    logger.info(f"   • {file_path}")
        else:
            logger.info("❌ 未找到任何有效文件")
        
        return existing_files
    
    def run_offline_guide(self):
        """运行离线指南"""
        logger.info("🔧 SD3.5模型离线下载指南")
        
        # 检查现有文件
        existing_files = self.check_existing_files()
        
        # 创建目录结构
        snapshot_dir = self.create_directory_structure()
        logger.info(f"📁 目标目录: {snapshot_dir}")
        
        # 创建下载脚本
        self.create_download_script()
        
        # 打印指南
        self.print_download_guide()
        
        return True

def main():
    """主函数"""
    guide = OfflineDownloadGuide()
    guide.run_offline_guide()
    
    print("\n" + "="*80)
    print("📝 总结")
    print("="*80)
    print("由于网络连接问题，建议使用以下方法：")
    print()
    print("🥇 最佳方案: 手动下载")
    print("   1. 访问 https://hf-mirror.com/stabilityai/stable-diffusion-3.5-large-turbo")
    print("   2. 下载所有必需文件到指定目录")
    print("   3. 运行 python test_sd35_model.py 验证")
    print()
    print("🥈 备选方案: 使用VPN + 自动下载")
    print("   1. 连接VPN")
    print("   2. 运行 python download_with_china_mirror.py")
    print()
    print("🥉 其他方案: 使用git或huggingface-hub")
    print("   详见上方指南")
    print()
    print("📁 目标目录已创建，可以开始手动下载文件")
    print("="*80)

if __name__ == "__main__":
    main()