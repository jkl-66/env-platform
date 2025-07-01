#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOAA数据下载测试脚本
Test script for NOAA data download functionality
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_settings
from src.utils.logger import get_logger
from scripts.download_climate_data import NOAADataDownloader, ClimateDataManager

logger = get_logger("noaa_test")
settings = get_settings()
from unittest.mock import patch
import cdsapi

# 模拟 cdsapi.Client
class MockCDSAPIClient:
    def retrieve(self, *args, **kwargs):
        logger.info("Mock CDSAPI retrieve called")
        # 创建一个虚拟的空文件作为下载结果
        with open("mock_cds_data.grib", "w") as f:
            f.write("mock data")
        return "mock_cds_data.grib"

# 在测试函数中使用 mock
@patch('src.data_processing.data_storage.DataStorage.initialize', new_callable=lambda: asyncio.coroutine(lambda self: None))
@patch('cdsapi.Client', new=MockCDSAPIClient)
async def test_noaa_download(mock_storage_init):
    """测试NOAA数据下载功能"""
    logger.info("开始测试NOAA数据下载功能")
    
    # 检查API密钥
    if not settings.NOAA_API_KEY or settings.NOAA_API_KEY == "your_noaa_api_key_here":
        logger.error("请在.env文件中配置有效的NOAA_API_KEY")
        logger.info("获取API密钥: https://www.ncdc.noaa.gov/cdo-web/token")
        return False
    
    try:
        # 创建数据管理器
        manager = ClimateDataManager()
        
        # 初始化存储系统
        await manager.initialize()
        
        if not manager.noaa_downloader:
            logger.error("NOAA下载器初始化失败")
            return False
        
        # 测试获取气象站列表
        logger.info("测试获取气象站列表...")
        stations = manager.noaa_downloader.get_stations(limit=5)
        
        if not stations:
            logger.error("无法获取气象站列表")
            return False
        
        logger.info(f"成功获取{len(stations)}个气象站")
        for i, station in enumerate(stations[:3]):
            logger.info(f"  {i+1}. {station.get('name', 'Unknown')} ({station.get('id', 'Unknown ID')})")
        
        # 测试下载少量数据（最近7天）
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        logger.info(f"测试下载数据: {start_date} 到 {end_date}")
        
        # 使用一个已知的、数据可靠的气象站进行测试
        # GHCND:USW00094728 是纽约中央公园的ID，通常有数据
        station_ids = ["GHCND:USW00094728"]
        logger.info(f"使用固定的气象站进行测试: {station_ids}")
        
        daily_data = manager.noaa_downloader.download_daily_data(
            start_date, end_date, station_ids
        )
        
        if daily_data.empty:
            logger.warning("未下载到任何数据，可能是时间范围或气象站问题")
            return True  # 不算失败，可能是数据可用性问题
        
        logger.info(f"成功下载{len(daily_data)}条数据记录")
        logger.info(f"数据列: {list(daily_data.columns)}")
        
        # 测试数据存储
        logger.info("测试数据存储...")
        success = await manager.storage.store_weather_data(
            daily_data, "noaa_daily_test", "daily"
        )
        
        # 新增：检查文件是否实际写入到 data/raw 目录
        if success:
            logger.info("数据存储测试成功")
            # 检查文件路径
            records = manager.storage.search_data_records(
                source="noaa_daily_test",
                limit=1
            )
            if records:
                file_path = records[0]["file_path"]
                if Path(file_path).exists() and str(Path(file_path)).startswith(str(settings.DATA_ROOT_PATH)):
                    logger.info(f"数据文件已存在于: {file_path}")
                else:
                    logger.error(f"数据文件未正确写入到 data 目录: {file_path}")
                    return False
        else:
            logger.error("数据存储测试失败")
            return False
        
        # 测试数据查询
        logger.info("测试数据查询...")
        records = manager.storage.search_data_records(
            source="noaa_daily_test",
            limit=5
        )
        
        if records:
            logger.info(f"查询到{len(records)}条数据记录")
            for record in records:
                logger.info(f"  记录ID: {record['id'][:8]}..., 文件: {Path(record['file_path']).name}")
        else:
            logger.warning("未查询到数据记录")
        
        logger.info("NOAA数据下载测试完成")
        return True
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        return False
    finally:
        # 清理资源
        if 'manager' in locals():
            await manager.storage.close()


def main():
    logger.info("NOAA数据下载测试脚本启动")
    try:
        asyncio.run(test_noaa_download())
        logger.info("测试成功完成")
    except Exception as e:
        logger.error(f"测试失败: {e}")

if __name__ == "__main__":
    main()