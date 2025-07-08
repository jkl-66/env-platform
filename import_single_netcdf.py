#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单个NetCDF文件导入测试脚本
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 尝试导入项目模块
try:
    from src.data_processing.data_storage import DataStorage
    from src.utils.logger import get_logger
    from src.utils.config import get_settings
    USE_PROJECT_MODULES = True
except ImportError as e:
    print(f"警告: 无法导入项目模块 ({e})，将使用简化版本")
    USE_PROJECT_MODULES = False
    
    # 简化的日志记录器
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    # 简化的设置
    class SimpleSettings:
        def __init__(self):
            self.database_url = "sqlite:///climate_data.db"
    
    def get_settings():
        return SimpleSettings()

logger = get_logger(__name__)
settings = get_settings()

class NetCDFImporter:
    """NetCDF文件导入器"""
    
    def __init__(self):
        if USE_PROJECT_MODULES:
            self.data_storage = DataStorage()
        else:
            self.data_storage = None
            # 创建输出目录
            self.output_dir = Path('outputs') / 'netcdf_import'
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """初始化数据存储"""
        if USE_PROJECT_MODULES and self.data_storage:
            await self.data_storage.initialize()
            logger.info("数据存储初始化完成")
        else:
            logger.info("使用简化模式，跳过数据存储初始化")
    
    def load_netcdf_file(self, file_path: Path) -> pd.DataFrame:
        """加载NetCDF文件"""
        logger.info(f"正在加载NetCDF文件: {file_path}")
        
        try:
            import xarray as xr
            
            # 打开NetCDF文件
            ds = xr.open_dataset(file_path)
            
            logger.info(f"NetCDF文件信息:")
            logger.info(f"  维度: {dict(ds.dims)}")
            logger.info(f"  变量: {list(ds.data_vars)}")
            logger.info(f"  坐标: {list(ds.coords)}")
            
            # 转换为DataFrame
            df = ds.to_dataframe().reset_index()
            
            # 移除NaN值
            df = df.dropna()
            
            # 添加文件信息
            df['source_file'] = file_path.name
            df['dataset_type'] = self._extract_dataset_type(file_path.name)
            df['year'] = self._extract_year(file_path.name)
            
            logger.info(f"NetCDF文件加载完成: {file_path.name}")
            logger.info(f"数据形状: {df.shape}")
            logger.info(f"列名: {list(df.columns)}")
            
            # 显示数据样本
            if not df.empty:
                logger.info(f"数据样本:")
                logger.info(f"{df.head()}")
            
            return df
            
        except ImportError:
            logger.error("需要安装xarray库来处理NetCDF文件")
            logger.info("请运行: pip install xarray netcdf4")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"加载NetCDF文件时发生错误: {e}")
            return pd.DataFrame()
    
    def _extract_dataset_type(self, filename: str) -> str:
        """从文件名提取数据集类型"""
        if 'hot-dry-windy' in filename:
            return 'hot-dry-windy'
        elif 'hot-dry' in filename:
            return 'hot-dry'
        elif 'hot-wet' in filename:
            return 'hot-wet'
        elif 'wet-windy' in filename:
            return 'wet-windy'
        else:
            return 'unknown'
    
    def _extract_year(self, filename: str) -> int:
        """从文件名提取年份"""
        import re
        match = re.search(r'_(\d{4})_', filename)
        if match:
            return int(match.group(1))
        return None
    
    async def import_netcdf_file(self, file_path: Path) -> str:
        """导入NetCDF文件到数据库"""
        logger.info(f"开始导入NetCDF文件: {file_path}")
        
        # 加载数据
        data = self.load_netcdf_file(file_path)
        
        if data.empty:
            logger.warning(f"数据为空，跳过导入: {file_path}")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_type = self._extract_dataset_type(file_path.name)
        year = self._extract_year(file_path.name)
        
        if USE_PROJECT_MODULES and self.data_storage:
            # 使用完整的数据存储系统
            filename = f"climate_netcdf_{dataset_type}_{year}_{timestamp}"
            
            # 保存为parquet格式
            saved_file_path = self.data_storage.save_dataframe(
                data, 
                filename, 
                data_category="processed", 
                format="parquet"
            )
            
            # 创建数据记录
            record_id = await self.data_storage.save_data_record(
                source=f"Global Climate NetCDF - {dataset_type}",
                data_type="extreme_events",
                location="Global",
                start_time=datetime(year, 1, 1) if year else datetime(1951, 1, 1),
                end_time=datetime(year, 12, 31) if year else datetime(2022, 12, 31),
                file_path=saved_file_path,
                file_format="parquet",
                file_size=os.path.getsize(saved_file_path),
                variables=list(data.columns),
                data_metadata={
                    "description": f"全球0.5°极端气候事件数据 - {dataset_type} ({year}年)",
                    "resolution": "0.5度",
                    "year": year,
                    "data_type": "extreme_climate_events",
                    "event_type": dataset_type,
                    "original_file": str(file_path),
                    "data_shape": data.shape
                }
            )
            
            logger.info(f"NetCDF文件已导入数据库，记录ID: {record_id}")
            return record_id
        else:
            # 简化模式：直接保存到本地文件
            filename = f"climate_netcdf_{dataset_type}_{year}_{timestamp}.parquet"
            saved_file_path = self.output_dir / filename
            
            data.to_parquet(saved_file_path)
            
            # 保存元数据
            metadata = {
                "source": f"Global Climate NetCDF - {dataset_type}",
                "data_type": "extreme_events",
                "location": "Global",
                "year": year,
                "file_path": str(saved_file_path),
                "file_format": "parquet",
                "file_size": os.path.getsize(saved_file_path),
                "variables": list(data.columns),
                "description": f"全球0.5°极端气候事件数据 - {dataset_type} ({year}年)",
                "resolution": "0.5度",
                "event_type": dataset_type,
                "original_file": str(file_path),
                "data_shape": list(data.shape)
            }
            
            metadata_path = self.output_dir / f"metadata_{dataset_type}_{year}_{timestamp}.json"
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"NetCDF文件已保存到本地: {saved_file_path}")
            logger.info(f"元数据已保存: {metadata_path}")
            return f"local_file_{dataset_type}_{year}_{timestamp}"

async def main():
    """主函数"""
    # 测试单个NetCDF文件
    test_file = r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]_1_1~12\global_hot-dry_1951_1~12.nc"
    
    print("🚀 开始NetCDF文件导入测试...")
    print(f"📂 测试文件: {test_file}")
    
    # 检查文件是否存在
    if not Path(test_file).exists():
        print(f"❌ 错误: 测试文件不存在: {test_file}")
        return
    
    # 创建导入器并运行导入
    importer = NetCDFImporter()
    
    try:
        await importer.initialize()
        
        record_id = await importer.import_netcdf_file(Path(test_file))
        
        if record_id:
            print(f"\n✅ NetCDF文件导入成功!")
            print(f"🗄️ 记录ID: {record_id}")
        else:
            print(f"\n❌ NetCDF文件导入失败")
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        logger.error(f"导入失败: {e}", exc_info=True)
    finally:
        if USE_PROJECT_MODULES and importer.data_storage:
            await importer.data_storage.close()

if __name__ == "__main__":
    asyncio.run(main())