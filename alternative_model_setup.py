#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
替代模型设置方案
当无法直接从Hugging Face下载时的解决方案
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlternativeModelSetup:
    def __init__(self):
        self.cache_dir = Path("./cache/huggingface")
        self.model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
        
    def check_existing_cache(self):
        """检查是否已有缓存的模型文件"""
        logger.info("检查现有模型缓存...")
        
        model_cache_path = self.cache_dir / "models--stabilityai--stable-diffusion-3.5-large-turbo"
        
        if model_cache_path.exists():
            logger.info(f"✅ 发现现有模型缓存: {model_cache_path}")
            
            # 检查关键文件
            key_files = [
                "snapshots",
                "refs",
                "blobs"
            ]
            
            missing_files = []
            for file in key_files:
                if not (model_cache_path / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                logger.warning(f"⚠️  缺少关键文件: {missing_files}")
                return False, model_cache_path
            else:
                logger.info("✅ 模型缓存完整")
                return True, model_cache_path
        else:
            logger.info("❌ 未发现模型缓存")
            return False, None
    
    def setup_offline_mode(self):
        """设置离线模式配置"""
        logger.info("设置离线模式...")
        
        # 设置环境变量强制离线模式
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        logger.info("✅ 离线模式已启用")
    
    def create_mock_model_config(self):
        """创建模拟模型配置用于测试"""
        logger.info("创建模拟模型配置...")
        
        mock_config = {
            "model_type": "stable-diffusion-3",
            "model_id": self.model_id,
            "cache_dir": str(self.cache_dir),
            "offline_mode": True,
            "mock_mode": True,
            "created_at": datetime.now().isoformat(),
            "note": "这是一个模拟配置，用于在无法下载真实模型时进行功能测试"
        }
        
        config_path = Path("config/mock_model_config.json")
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(mock_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 模拟配置已创建: {config_path}")
        return config_path
    
    def suggest_manual_download(self):
        """提供手动下载建议"""
        logger.info("提供手动下载建议...")
        
        suggestions = [
            "🔧 手动下载建议:",
            "",
            "1. 使用镜像站点:",
            "   - HuggingFace镜像: https://hf-mirror.com",
            "   - ModelScope: https://modelscope.cn",
            "",
            "2. 使用git clone:",
            "   git clone https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo",
            "",
            "3. 使用代理或VPN:",
            "   - 配置HTTP代理",
            "   - 使用科学上网工具",
            "",
            "4. 离线传输:",
            "   - 从其他机器下载后传输",
            "   - 使用移动存储设备",
            "",
            "5. 使用替代模型:",
            "   - stable-diffusion-2.1",
            "   - stable-diffusion-xl-base-1.0",
        ]
        
        for suggestion in suggestions:
            print(suggestion)
        
        return suggestions
    
    def create_network_test_script(self):
        """创建网络测试脚本"""
        script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络连接测试脚本
"""

import requests
import time

def test_connections():
    """测试各种连接"""
    test_urls = [
        ("Hugging Face", "https://huggingface.co"),
        ("HF Mirror", "https://hf-mirror.com"),
        ("ModelScope", "https://modelscope.cn"),
        ("GitHub", "https://github.com"),
        ("Google", "https://google.com")
    ]
    
    print("🌐 网络连接测试")
    print("=" * 40)
    
    for name, url in test_urls:
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                print(f"✅ {name}: 连接成功 ({end_time - start_time:.2f}s)")
            else:
                print(f"⚠️  {name}: 状态码 {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: 连接失败 - {e}")
    
    print("\n💡 建议:")
    print("- 如果所有连接都失败，检查网络设置")
    print("- 如果只有HuggingFace失败，尝试使用镜像站点")
    print("- 考虑使用代理或VPN")

if __name__ == "__main__":
    test_connections()
'''
        
        script_path = Path("network_test.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        logger.info(f"✅ 网络测试脚本已创建: {script_path}")
        return script_path
    
    def run_alternative_setup(self):
        """运行替代设置方案"""
        logger.info("="*60)
        logger.info("🔧 SD3.5 替代模型设置方案")
        logger.info("="*60)
        
        # 检查现有缓存
        has_cache, cache_path = self.check_existing_cache()
        
        if has_cache:
            logger.info("🎉 发现完整的模型缓存，可以直接使用！")
            self.setup_offline_mode()
            return True
        
        # 创建模拟配置
        mock_config_path = self.create_mock_model_config()
        
        # 创建网络测试脚本
        network_script = self.create_network_test_script()
        
        # 提供建议
        self.suggest_manual_download()
        
        print("\n" + "="*60)
        print("📋 下一步操作:")
        print("1. 运行网络测试: python network_test.py")
        print("2. 尝试手动下载方法")
        print("3. 或使用离线模式进行功能测试")
        print("4. 模型下载成功后运行: python demo_environmental_generator.py")
        print("="*60)
        
        return False

def main():
    setup = AlternativeModelSetup()
    setup.run_alternative_setup()

if __name__ == "__main__":
    main()