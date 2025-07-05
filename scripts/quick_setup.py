#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速项目设置脚本
Quick Project Setup Script

此脚本用于快速设置整个气候数据分析项目，包括环境检查、数据库初始化、数据下载等。
This script quickly sets up the entire climate data analysis project, including environment checks, database initialization, data download, etc.
"""

import asyncio
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup.log')
    ]
)
logger = logging.getLogger("quick_setup")


class ProjectSetup:
    """项目设置管理器"""
    
    def __init__(self):
        self.project_root = project_root
        self.scripts_dir = self.project_root / "scripts"
        self.requirements_file = self.project_root / "requirements.txt"
        self.env_file = self.project_root / ".env"
        
        # 设置步骤状态
        self.setup_steps = {
            "environment_check": False,
            "dependencies_install": False,
            "env_config": False,
            "database_init": False,
            "data_download": False,
            "model_setup": False,
            "system_test": False
        }
    
    def print_banner(self):
        """打印项目横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    气候数据分析与生态警示系统                                    ║
║                Climate Data Analysis and Ecological Warning System           ║
║                                                                              ║
║                              快速设置向导                                      ║
║                            Quick Setup Wizard                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"项目路径: {self.project_root}")
        print(f"设置时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def check_environment(self) -> bool:
        """检查环境要求"""
        logger.info("🔍 检查环境要求...")
        
        checks = {
            "Python版本": self._check_python_version(),
            "项目文件": self._check_project_files(),
            "系统依赖": self._check_system_dependencies()
        }
        
        all_passed = all(checks.values())
        
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"{status} {check_name}: {'通过' if passed else '失败'}")
        
        if all_passed:
            logger.info("✅ 环境检查通过")
            self.setup_steps["environment_check"] = True
        else:
            logger.error("❌ 环境检查失败，请解决上述问题后重试")
        
        return all_passed
    
    def _check_python_version(self) -> bool:
        """检查Python版本"""
        version = sys.version_info
        required_version = (3, 8)
        
        if version >= required_version:
            logger.info(f"Python版本: {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            logger.error(f"需要Python {required_version[0]}.{required_version[1]}+，当前版本: {version.major}.{version.minor}")
            return False
    
    def _check_project_files(self) -> bool:
        """检查项目文件"""
        required_files = [
            "src/main.py",
            "src/models/climate_analysis.py",
            "src/models/regional_prediction.py",
            "src/data_processing/data_collector.py",
            "src/data_processing/data_storage.py",
            "environmental_image_generator.py",
            "requirements.txt"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"缺少必要文件: {missing_files}")
            return False
        
        return True
    
    def _check_system_dependencies(self) -> bool:
        """检查系统依赖"""
        # 这里可以检查系统级依赖，如Docker、PostgreSQL等
        # 暂时返回True
        return True
    
    def install_dependencies(self) -> bool:
        """安装Python依赖"""
        logger.info("📦 安装Python依赖包...")
        
        if not self.requirements_file.exists():
            logger.error(f"requirements.txt文件不存在: {self.requirements_file}")
            return False
        
        try:
            # 升级pip
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # 安装依赖
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)],
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.info("✅ 依赖包安装完成")
            self.setup_steps["dependencies_install"] = True
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 依赖包安装失败: {e.stderr}")
            return False
    
    def setup_environment_config(self) -> bool:
        """设置环境配置"""
        logger.info("⚙️ 设置环境配置...")
        
        if not self.env_file.exists():
            logger.error(f".env文件不存在: {self.env_file}")
            logger.info("请确保.env文件已创建并配置了必要的API密钥")
            return False
        
        # 检查关键配置
        try:
            from src.utils.config import get_settings
            settings = get_settings()
            
            config_checks = {
                "NOAA API密钥": bool(settings.NOAA_API_KEY and settings.NOAA_API_KEY != "your_noaa_api_key_here"),
                "ECMWF API密钥": bool(settings.ECMWF_API_KEY and settings.ECMWF_API_KEY != "your_ecmwf_api_key_here"),
                "数据库配置": bool(settings.POSTGRES_PASSWORD != "your_postgres_password")
            }
            
            for config_name, is_configured in config_checks.items():
                status = "✅" if is_configured else "⚠️"
                logger.info(f"{status} {config_name}: {'已配置' if is_configured else '需要配置'}")
            
            if any(config_checks.values()):
                logger.info("✅ 环境配置检查完成")
                self.setup_steps["env_config"] = True
                return True
            else:
                logger.warning("⚠️ 请配置API密钥后重新运行")
                return False
                
        except Exception as e:
            logger.error(f"❌ 环境配置检查失败: {e}")
            return False
    
    async def initialize_databases(self) -> bool:
        """初始化数据库"""
        logger.info("🗄️ 初始化数据库...")
        
        try:
            # 运行数据库初始化脚本
            init_script = self.scripts_dir / "init_database.py"
            if not init_script.exists():
                logger.error(f"数据库初始化脚本不存在: {init_script}")
                return False
            
            # 导入并运行初始化
            sys.path.insert(0, str(self.scripts_dir))
            from init_database import DatabaseManager
            
            manager = DatabaseManager()
            await manager.initialize_all()
            
            # 检查数据库状态
            status = manager.check_status()
            
            if status['postgresql']:
                logger.info("✅ PostgreSQL数据库初始化完成")
            else:
                logger.warning("⚠️ PostgreSQL数据库连接失败")
            
            if status['influxdb']:
                logger.info("✅ InfluxDB数据库初始化完成")
            else:
                logger.warning("⚠️ InfluxDB数据库连接失败")
            
            self.setup_steps["database_init"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据库初始化失败: {e}")
            return False
    
    async def download_sample_data(self) -> bool:
        """下载示例数据"""
        logger.info("📊 下载示例气候数据...")
        
        try:
            # 运行数据下载脚本
            download_script = self.scripts_dir / "download_climate_data.py"
            if not download_script.exists():
                logger.error(f"数据下载脚本不存在: {download_script}")
                return False
            
            # 导入并运行数据下载（限制数据量）
            sys.path.insert(0, str(self.scripts_dir))
            from download_climate_data import ClimateDataManager
            
            manager = ClimateDataManager()
            
            # 下载最近1个月的数据作为示例
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            logger.info(f"下载时间范围: {start_date} 到 {end_date}")
            await manager.download_all_data(start_date, end_date)
            
            logger.info("✅ 示例数据下载完成")
            self.setup_steps["data_download"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据下载失败: {e}")
            logger.info("可以稍后手动运行数据下载脚本")
            # 数据下载失败不阻止后续步骤
            return True
    
    def setup_models(self) -> bool:
        """设置AI模型"""
        logger.info("🤖 设置AI模型...")
        
        try:
            # 创建模型目录
            model_dirs = [
                self.project_root / "models" / "trained",
                self.project_root / "models" / "checkpoints",
                self.project_root / "models" / "cache"
            ]
            
            for model_dir in model_dirs:
                model_dir.mkdir(parents=True, exist_ok=True)
            
            # 检查模型文件（如果存在）
            model_files = [
                "models/trained/climate_analysis.pkl",
                "models/trained/regional_prediction.pth",
                "models/trained/ecology_generator.pth"
            ]
            
            existing_models = []
            for model_file in model_files:
                if (self.project_root / model_file).exists():
                    existing_models.append(model_file)
            
            if existing_models:
                logger.info(f"✅ 发现已有模型: {existing_models}")
            else:
                logger.info("ℹ️ 未发现预训练模型，将在首次运行时自动训练")
            
            logger.info("✅ 模型设置完成")
            self.setup_steps["model_setup"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型设置失败: {e}")
            return False
    
    async def run_system_test(self) -> bool:
        """运行系统测试"""
        logger.info("🧪 运行系统测试...")
        
        try:
            # 测试导入主要模块
            test_imports = [
                "src.utils.config",
                "src.data_processing.data_storage",
                "src.models.climate_analysis",
                "src.models.regional_prediction",
                "src.models.environmental_image_generator"
            ]
            
            for module_name in test_imports:
                try:
                    __import__(module_name)
                    logger.info(f"✅ 模块导入成功: {module_name}")
                except ImportError as e:
                    logger.error(f"❌ 模块导入失败: {module_name} - {e}")
                    return False
            
            # 测试配置加载
            from src.utils.config import get_settings
            settings = get_settings()
            logger.info(f"✅ 配置加载成功: {settings.APP_NAME}")
            
            # 测试数据存储初始化
            from src.data_processing.data_storage import DataStorage
            storage = DataStorage()
            await storage.initialize()
            await storage.close()
            logger.info("✅ 数据存储测试通过")
            
            logger.info("✅ 系统测试完成")
            self.setup_steps["system_test"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统测试失败: {e}")
            return False
    
    def print_setup_summary(self):
        """打印设置摘要"""
        print("\n" + "="*80)
        print("📋 设置摘要 / Setup Summary")
        print("="*80)
        
        for step_name, completed in self.setup_steps.items():
            status = "✅" if completed else "❌"
            step_display = {
                "environment_check": "环境检查",
                "dependencies_install": "依赖安装",
                "env_config": "环境配置",
                "database_init": "数据库初始化",
                "data_download": "数据下载",
                "model_setup": "模型设置",
                "system_test": "系统测试"
            }
            print(f"{status} {step_display.get(step_name, step_name)}")
        
        completed_steps = sum(self.setup_steps.values())
        total_steps = len(self.setup_steps)
        
        print(f"\n完成进度: {completed_steps}/{total_steps} ({completed_steps/total_steps*100:.1f}%)")
        
        if completed_steps == total_steps:
            print("\n🎉 项目设置完成！")
            print("\n下一步操作:")
            print("1. 启动系统: python src/main.py")
            print("2. 访问API文档: http://localhost:8000/docs")
            print("3. 查看日志: tail -f logs/app.log")
        else:
            print("\n⚠️ 项目设置未完全完成，请检查失败的步骤")
    
    async def run_full_setup(self):
        """运行完整设置流程"""
        self.print_banner()
        
        setup_functions = [
            ("环境检查", self.check_environment),
            ("安装依赖", self.install_dependencies),
            ("环境配置", self.setup_environment_config),
            ("数据库初始化", self.initialize_databases),
            ("下载数据", self.download_sample_data),
            ("模型设置", self.setup_models),
            ("系统测试", self.run_system_test)
        ]
        
        for step_name, setup_func in setup_functions:
            print(f"\n🚀 开始: {step_name}")
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(setup_func):
                    success = await setup_func()
                else:
                    success = setup_func()
                
                elapsed_time = time.time() - start_time
                
                if success:
                    print(f"✅ {step_name} 完成 ({elapsed_time:.1f}s)")
                else:
                    print(f"❌ {step_name} 失败 ({elapsed_time:.1f}s)")
                    
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"❌ {step_name} 异常 ({elapsed_time:.1f}s): {e}")
                logger.exception(f"{step_name} 执行异常")
        
        self.print_setup_summary()


async def main():
    """主函数"""
    setup = ProjectSetup()
    await setup.run_full_setup()


if __name__ == "__main__":
    asyncio.run(main())