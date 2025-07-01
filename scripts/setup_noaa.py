#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOAA数据下载快速设置脚本
Quick setup script for NOAA data download
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_settings
from src.utils.logger import get_logger
from scripts.download_climate_data import ClimateDataManager

logger = get_logger("noaa_setup")
settings = get_settings()


async def setup_and_download_noaa():
    """设置并下载NOAA数据"""
    logger.info("开始NOAA数据下载设置")
    
    # 检查配置
    if not settings.NOAA_API_KEY or settings.NOAA_API_KEY == "your_noaa_api_key_here":
        logger.error("请先配置NOAA API密钥")
        logger.info("1. 访问 https://www.ncdc.noaa.gov/cdo-web/token 获取API密钥")
        logger.info("2. 在.env文件中设置 NOAA_API_KEY=你的密钥")
        return False
    
    try:
        # 创建必要的目录
        data_root = Path(settings.DATA_ROOT_PATH)
        for subdir in ["raw", "processed", "cache"]:
            (data_root / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"数据目录已创建: {data_root}")
        
        # 初始化数据管理器
        manager = ClimateDataManager()
        await manager.initialize()
        
        if not manager.noaa_downloader:
            logger.error("NOAA下载器初始化失败")
            return False
        
        # 下载最近30天的数据作为示例
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        logger.info(f"开始下载NOAA数据: {start_date} 到 {end_date}")
        
        # 获取主要气象站（限制数量以加快下载）
        stations = manager.noaa_downloader.get_stations(limit=20)
        if not stations:
            logger.error("无法获取气象站列表")
            return False
        
        logger.info(f"找到{len(stations)}个气象站")
        
        # 选择前5个气象站进行下载
        station_ids = [station["id"] for station in stations[:5]]
        logger.info(f"选择{len(station_ids)}个气象站进行数据下载")
        
        # 下载日数据
        daily_data = manager.noaa_downloader.download_daily_data(
            start_date, end_date, station_ids
        )
        
        if not daily_data.empty:
            # 存储数据
            success = await manager.storage.store_weather_data(
                daily_data, "noaa_daily", "daily"
            )
            
            if success:
                logger.info(f"成功下载并存储{len(daily_data)}条NOAA日数据")
            else:
                logger.error("数据存储失败")
                return False
        else:
            logger.warning("未下载到日数据")
        
        # 下载月度汇总数据
        monthly_data = manager.noaa_downloader.download_monthly_summary(
            start_date[:7], end_date[:7]  # 转换为YYYY-MM格式
        )
        
        if not monthly_data.empty:
            # 存储数据
            success = await manager.storage.store_weather_data(
                monthly_data, "noaa_monthly", "monthly"
            )
            
            if success:
                logger.info(f"成功下载并存储{len(monthly_data)}条NOAA月度数据")
            else:
                logger.error("月度数据存储失败")
        else:
            logger.warning("未下载到月度数据")
        
        # 显示存储的数据记录
        records = manager.storage.search_data_records(limit=10)
        if records:
            logger.info(f"数据库中共有{len(records)}条数据记录:")
            for i, record in enumerate(records[:5], 1):
                logger.info(f"  {i}. {record['source']} - {record['data_type']} ({record['file_format']})")
        
        logger.info("NOAA数据下载设置完成")
        return True
        
    except Exception as e:
        logger.error(f"设置过程中出现错误: {e}")
        return False
    finally:
        # 清理资源
        if 'manager' in locals():
            await manager.storage.close()


async def main():
    """主函数"""
    logger.info("NOAA数据下载快速设置脚本启动")
    
    success = await setup_and_download_noaa()
    
    if success:
        logger.info("设置成功完成！")
        logger.info("你现在可以使用以下命令进行更多数据下载:")
        logger.info("  python scripts/download_climate_data.py")
    else:
        logger.error("设置失败")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())