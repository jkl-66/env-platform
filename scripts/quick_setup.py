#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Project Setup Script

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
    """Project Setup Manager"""
    
    def __init__(self):
        self.project_root = project_root
        self.scripts_dir = self.project_root / "scripts"
        self.requirements_file = self.project_root / "requirements.txt"
        self.env_file = self.project_root / ".env"
        
        # Setup step status
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
        """Print project banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                Climate Data Analysis and Ecological Warning System           ║
║                                                                              ║
║                            Quick Setup Wizard                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"Project Path: {self.project_root}")
        print(f"Setup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def check_environment(self) -> bool:
        """Check environment requirements"""
        logger.info("🔍 Checking environment requirements...")
        
        checks = {
            "Python Version": self._check_python_version(),
            "Project Files": self._check_project_files(),
            "System Dependencies": self._check_system_dependencies()
        }
        
        all_passed = all(checks.values())
        
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"{status} {check_name}: {'Passed' if passed else 'Failed'}")
        
        if all_passed:
            logger.info("✅ Environment check passed")
            self.setup_steps["environment_check"] = True
        else:
            logger.error("❌ Environment check failed, please resolve the above issues and try again")
        
        return all_passed
    
    def _check_python_version(self) -> bool:
        """Check Python version"""
        version = sys.version_info
        required_version = (3, 8)
        
        if version >= required_version:
            logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            logger.error(f"Python {required_version[0]}.{required_version[1]}+ required, current version: {version.major}.{version.minor}")
            return False
    
    def _check_project_files(self) -> bool:
        """Check project files"""
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
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        return True
    
    def _check_system_dependencies(self) -> bool:
        """Check system dependencies"""
        # Here you can check system-level dependencies like Docker, PostgreSQL, etc.
        # Currently returns True
        return True
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")
        
        if not self.requirements_file.exists():
            logger.error(f"requirements.txt file does not exist: {self.requirements_file}")
            return False
        
        try:
            # Upgrade pip
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install dependencies
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)],
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.info("✅ Dependencies installation completed")
            self.setup_steps["dependencies_install"] = True
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Dependencies installation failed: {e.stderr}")
            return False
    
    def setup_environment_config(self) -> bool:
        """Setup environment configuration"""
        logger.info("⚙️ Setting up environment configuration...")
        
        if not self.env_file.exists():
            logger.error(f".env file does not exist: {self.env_file}")
            logger.info("Please ensure .env file is created and configured with necessary API keys")
            return False
        
        # Check key configurations
        try:
            from src.utils.config import get_settings
            settings = get_settings()
            
            config_checks = {
                "NOAA API Key": bool(settings.NOAA_API_KEY and settings.NOAA_API_KEY != "your_noaa_api_key_here"),
                "ECMWF API Key": bool(settings.ECMWF_API_KEY and settings.ECMWF_API_KEY != "your_ecmwf_api_key_here"),
                "Database Config": bool(settings.POSTGRES_PASSWORD != "your_postgres_password")
            }
            
            for config_name, is_configured in config_checks.items():
                status = "✅" if is_configured else "⚠️"
                logger.info(f"{status} {config_name}: {'Configured' if is_configured else 'Needs configuration'}")
            
            if any(config_checks.values()):
                logger.info("✅ Environment configuration check completed")
                self.setup_steps["env_config"] = True
                return True
            else:
                logger.warning("⚠️ Please configure API keys and run again")
                return False
                
        except Exception as e:
            logger.error(f"❌ Environment configuration check failed: {e}")
            return False
    
    async def initialize_databases(self) -> bool:
        """Initialize databases"""
        logger.info("🗄️ Initializing databases...")
        
        try:
            # Run database initialization script
            init_script = self.scripts_dir / "init_database.py"
            if not init_script.exists():
                logger.error(f"Database initialization script does not exist: {init_script}")
                return False
            
            # Import and run initialization
            sys.path.insert(0, str(self.scripts_dir))
            from init_database import DatabaseManager
            
            manager = DatabaseManager()
            await manager.initialize_all()
            
            # Check database status
            status = manager.check_status()
            
            if status['postgresql']:
                logger.info("✅ PostgreSQL database initialization completed")
            else:
                logger.warning("⚠️ PostgreSQL database connection failed")
            
            if status['influxdb']:
                logger.info("✅ InfluxDB database initialization completed")
            else:
                logger.warning("⚠️ InfluxDB database connection failed")
            
            self.setup_steps["database_init"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
    
    async def download_sample_data(self) -> bool:
        """Download sample data"""
        logger.info("📊 Downloading sample climate data...")
        
        try:
            # Run data download script
            download_script = self.scripts_dir / "download_climate_data.py"
            if not download_script.exists():
                logger.error(f"Data download script does not exist: {download_script}")
                return False
            
            # Import and run data download (limited data volume)
            sys.path.insert(0, str(self.scripts_dir))
            from download_climate_data import ClimateDataManager
            
            manager = ClimateDataManager()
            
            # Download last 1 month of data as example
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            logger.info(f"Download time range: {start_date} to {end_date}")
            await manager.download_all_data(start_date, end_date)
            
            logger.info("✅ Sample data download completed")
            self.setup_steps["data_download"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Data download failed: {e}")
            logger.info("You can manually run the data download script later")
            # Data download failure does not block subsequent steps
            return True
    
    def setup_models(self) -> bool:
        """Setup AI models"""
        logger.info("🤖 Setting up AI models...")
        
        try:
            # Create model directories
            model_dirs = [
                self.project_root / "models" / "trained",
                self.project_root / "models" / "checkpoints",
                self.project_root / "models" / "cache"
            ]
            
            for model_dir in model_dirs:
                model_dir.mkdir(parents=True, exist_ok=True)
            
            # Check model files (if they exist)
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
                logger.info(f"✅ Found existing models: {existing_models}")
            else:
                logger.info("ℹ️ No pre-trained models found, will auto-train on first run")
            
            logger.info("✅ Model setup completed")
            self.setup_steps["model_setup"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Model setup failed: {e}")
            return False
    
    async def run_system_test(self) -> bool:
        """Run system test"""
        logger.info("🧪 Running system test...")
        
        try:
            # Test importing main modules
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
                    logger.info(f"✅ Module import successful: {module_name}")
                except ImportError as e:
                    logger.error(f"❌ Module import failed: {module_name} - {e}")
                    return False
            
            # Test configuration loading
            from src.utils.config import get_settings
            settings = get_settings()
            logger.info(f"✅ Configuration loading successful: {settings.APP_NAME}")
            
            # Test data storage initialization
            from src.data_processing.data_storage import DataStorage
            storage = DataStorage()
            await storage.initialize()
            await storage.close()
            logger.info("✅ Data storage test passed")
            
            logger.info("✅ System test completed")
            self.setup_steps["system_test"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ System test failed: {e}")
            return False
    
    def print_setup_summary(self):
        """Print setup summary"""
        print("\n" + "="*80)
        print("📋 Setup Summary")
        print("="*80)
        
        for step_name, completed in self.setup_steps.items():
            status = "✅" if completed else "❌"
            step_display = {
                "environment_check": "Environment Check",
                "dependencies_install": "Dependencies Installation",
                "env_config": "Environment Configuration",
                "database_init": "Database Initialization",
                "data_download": "Data Download",
                "model_setup": "Model Setup",
                "system_test": "System Test"
            }
            print(f"{status} {step_display.get(step_name, step_name)}")
        
        completed_steps = sum(self.setup_steps.values())
        total_steps = len(self.setup_steps)
        
        print(f"\nCompletion Progress: {completed_steps}/{total_steps} ({completed_steps/total_steps*100:.1f}%)")
        
        if completed_steps == total_steps:
            print("\n🎉 Project setup completed!")
            print("\nNext steps:")
            print("1. Start system: python src/main.py")
            print("2. Access API docs: http://localhost:8000/docs")
            print("3. View logs: tail -f logs/app.log")
        else:
            print("\n⚠️ Project setup not fully completed, please check failed steps")
    
    async def run_full_setup(self):
        """Run full setup process"""
        self.print_banner()
        
        setup_functions = [
            ("Environment Check", self.check_environment),
            ("Install Dependencies", self.install_dependencies),
            ("Environment Configuration", self.setup_environment_config),
            ("Database Initialization", self.initialize_databases),
            ("Download Data", self.download_sample_data),
            ("Model Setup", self.setup_models),
            ("System Test", self.run_system_test)
        ]
        
        for step_name, setup_func in setup_functions:
            print(f"\n🚀 Starting: {step_name}")
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(setup_func):
                    success = await setup_func()
                else:
                    success = setup_func()
                
                elapsed_time = time.time() - start_time
                
                if success:
                    print(f"✅ {step_name} completed ({elapsed_time:.1f}s)")
                else:
                    print(f"❌ {step_name} failed ({elapsed_time:.1f}s)")
                    
            except Exception as e:
                elapsed_time = time.time() - start_time
                print(f"❌ {step_name} exception ({elapsed_time:.1f}s): {e}")
                logger.exception(f"{step_name} execution exception")
        
        self.print_setup_summary()


async def main():
    """Main function"""
    setup = ProjectSetup()
    await setup.run_full_setup()


if __name__ == "__main__":
    asyncio.run(main())