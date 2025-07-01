#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气候数据下载脚本
Climate Data Download Script

此脚本用于从NOAA和ECMWF下载气候数据，并存储到本地数据库。
This script downloads climate data from NOAA and ECMWF and stores it in local database.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import requests
import cdsapi
import logging
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.data_processing.data_storage import DataStorage

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger("climate_data_downloader")
settings = get_settings()


class NOAADataDownloader:
    """NOAA数据下载器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2"
        self.session = requests.Session()
        self.session.headers.update({"token": api_key})
    
    def get_stations(self, dataset_id: str = "GHCND", limit: int = 1000) -> List[Dict]:
        """获取气象站列表"""
        logger.info(f"获取{dataset_id}数据集的气象站列表")
        
        url = f"{self.base_url}/stations"
        params = {
            "datasetid": dataset_id,
            "limit": limit,
            "offset": 1
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            stations = data.get("results", [])
            logger.info(f"成功获取{len(stations)}个气象站")
            return stations
            
        except Exception as e:
            logger.error(f"获取气象站列表失败: {e}")
            return []
    
    def download_daily_data(
        self,
        start_date: str,
        end_date: str,
        station_ids: List[str],
        datatypes: List[str] = None
    ) -> pd.DataFrame:
        """下载日气象数据"""
        
        if datatypes is None:
            datatypes = ["TMAX", "TMIN", "TAVG", "PRCP", "SNOW"]
        
        logger.info(f"下载日气象数据: {start_date} 到 {end_date}")
        
        all_data = []
        
        for station_id in tqdm(station_ids, desc="下载气象站数据"):
            try:
                url = f"{self.base_url}/data"
                params = {
                    "datasetid": "GHCND",
                    "stationid": station_id,
                    "startdate": start_date,
                    "enddate": end_date,
                    "datatypeid": ",".join(datatypes),
                    "limit": 1000,
                    "format": "json"
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "results" in data:
                    station_df = pd.DataFrame(data["results"])
                    all_data.append(station_df)
                    logger.debug(f"站点{station_id}: {len(station_df)}条记录")
                
            except Exception as e:
                logger.error(f"下载站点{station_id}数据失败: {e}")
                continue
        
        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"成功下载{len(result_df)}条NOAA数据记录")
            return result_df
        else:
            logger.warning("未下载到任何NOAA数据")
            return pd.DataFrame()
    
    def download_monthly_summary(
        self,
        start_date: str,
        end_date: str,
        location_ids: List[str] = None
    ) -> pd.DataFrame:
        """下载月度汇总数据"""
        
        if location_ids is None:
            location_ids = ["FIPS:01", "FIPS:02"]  # 示例：阿拉巴马州和阿拉斯加州
        
        logger.info(f"下载月度汇总数据: {start_date} 到 {end_date}")
        
        all_data = []
        
        for location_id in tqdm(location_ids, desc="下载月度数据"):
            try:
                url = f"{self.base_url}/data"
                params = {
                    "datasetid": "GSOM",
                    "locationid": location_id,
                    "startdate": start_date,
                    "enddate": end_date,
                    "limit": 1000,
                    "format": "json"
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "results" in data:
                    location_df = pd.DataFrame(data["results"])
                    all_data.append(location_df)
                    logger.debug(f"位置{location_id}: {len(location_df)}条记录")
                
            except Exception as e:
                logger.error(f"下载位置{location_id}数据失败: {e}")
                continue
        
        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"成功下载{len(result_df)}条月度汇总数据")
            return result_df
        else:
            logger.warning("未下载到任何月度汇总数据")
            return pd.DataFrame()


class ECMWFDataDownloader:
    """ECMWF数据下载器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # 创建CDS API客户端
        self.client = cdsapi.Client()
    
    def download_era5_reanalysis(
        self,
        start_date: str,
        end_date: str,
        variables: List[str] = None,
        area: List[float] = None,
        output_file: str = None
    ) -> str:
        """下载ERA5再分析数据"""
        
        if variables is None:
            variables = [
                '2m_temperature',
                'total_precipitation',
                'surface_pressure',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind'
            ]
        
        if area is None:
            # 默认全球范围
            area = [90, -180, -90, 180]  # North, West, South, East
        
        if output_file is None:
            output_file = f"era5_data_{start_date}_{end_date}.nc"
        
        output_path = settings.DATA_ROOT_PATH / "raw" / output_file
        
        logger.info(f"下载ERA5数据: {start_date} 到 {end_date}")
        
        try:
            self.client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': variables,
                    'year': self._get_years_range(start_date, end_date),
                    'month': list(range(1, 13)),
                    'day': list(range(1, 32)),
                    'time': [
                        '00:00', '06:00', '12:00', '18:00'
                    ],
                    'area': area,
                    'format': 'netcdf',
                },
                str(output_path)
            )
            
            logger.info(f"ERA5数据下载完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"下载ERA5数据失败: {e}")
            raise
    
    def download_seasonal_forecast(
        self,
        year: int,
        month: int,
        variables: List[str] = None,
        output_file: str = None
    ) -> str:
        """下载季节性预测数据"""
        
        if variables is None:
            variables = [
                '2m_temperature',
                'total_precipitation'
            ]
        
        if output_file is None:
            output_file = f"seasonal_forecast_{year}_{month:02d}.nc"
        
        output_path = settings.DATA_ROOT_PATH / "raw" / output_file
        
        logger.info(f"下载季节性预测数据: {year}-{month:02d}")
        
        try:
            self.client.retrieve(
                'seasonal-original-single-levels',
                {
                    'variable': variables,
                    'year': str(year),
                    'month': f"{month:02d}",
                    'leadtime_month': ['1', '2', '3', '4', '5', '6'],
                    'format': 'netcdf',
                },
                str(output_path)
            )
            
            logger.info(f"季节性预测数据下载完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"下载季节性预测数据失败: {e}")
            raise
    
    def _get_years_range(self, start_date: str, end_date: str) -> List[str]:
        """获取年份范围"""
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        return [str(year) for year in range(start_year, end_year + 1)]


class ClimateDataManager:
    """气候数据管理器"""
    
    def __init__(self):
        self.noaa_downloader = None
        self.ecmwf_downloader = None
        self.storage = DataStorage()
        
        # 初始化下载器
        if settings.NOAA_API_KEY:
            self.noaa_downloader = NOAADataDownloader(settings.NOAA_API_KEY)
            logger.info("NOAA下载器初始化成功")
        else:
            logger.warning("NOAA API密钥未配置")
        
        if settings.ECMWF_API_KEY:
            self.ecmwf_downloader = ECMWFDataDownloader(settings.ECMWF_API_KEY)
            logger.info("ECMWF下载器初始化成功")
        else:
            logger.warning("ECMWF API密钥未配置")
    
    async def initialize(self):
        """初始化数据存储系统"""
        await self.storage.initialize()
        logger.info("数据存储系统初始化完成")
    
    async def download_all_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31"
    ):
        """下载所有可用的气候数据"""
        
        logger.info("开始下载气候数据")
        
        # 1. 下载NOAA数据
        if self.noaa_downloader:
            await self._download_noaa_data(start_date, end_date)
        
        # 2. 下载ECMWF数据
        if self.ecmwf_downloader:
            await self._download_ecmwf_data(start_date, end_date)
        
        logger.info("气候数据下载完成")
    
    async def _download_noaa_data(self, start_date: str, end_date: str):
        """下载NOAA数据"""
        logger.info("开始下载NOAA数据")
        
        try:
            # 获取主要气象站
            stations = self.noaa_downloader.get_stations(limit=50)
            station_ids = [station["id"] for station in stations[:10]]  # 限制为前10个站点
            
            # 下载日数据
            daily_data = self.noaa_downloader.download_daily_data(
                start_date, end_date, station_ids
            )
            
            if not daily_data.empty:
                # 存储到数据库和文件系统
                success = await self.storage.store_weather_data(daily_data, "noaa_daily", "daily")
                if success:
                    logger.info(f"NOAA日数据存储成功: {len(daily_data)}条记录")
                else:
                    logger.error("NOAA日数据存储失败")
            
            # 下载月度汇总数据
            monthly_data = self.noaa_downloader.download_monthly_summary(
                start_date[:7], end_date[:7]  # 转换为YYYY-MM格式
            )
            
            if not monthly_data.empty:
                # 存储到数据库和文件系统
                success = await self.storage.store_weather_data(monthly_data, "noaa_monthly", "monthly")
                if success:
                    logger.info(f"NOAA月度数据存储成功: {len(monthly_data)}条记录")
                else:
                    logger.error("NOAA月度数据存储失败")
            
        except Exception as e:
            logger.error(f"下载NOAA数据失败: {e}")
    
    async def _download_ecmwf_data(self, start_date: str, end_date: str):
        """下载ECMWF数据"""
        logger.info("开始下载ECMWF数据")
        
        try:
            # 下载ERA5再分析数据
            era5_file = self.ecmwf_downloader.download_era5_reanalysis(
                start_date, end_date
            )
            logger.info(f"ERA5数据下载完成: {era5_file}")
            
            # 下载季节性预测数据（最近一年）
            current_year = datetime.now().year
            for month in range(1, 13, 3):  # 每季度下载一次
                try:
                    forecast_file = self.ecmwf_downloader.download_seasonal_forecast(
                        current_year, month
                    )
                    logger.info(f"季节性预测数据下载完成: {forecast_file}")
                except Exception as e:
                    logger.warning(f"下载{current_year}-{month:02d}季节性预测失败: {e}")
            
        except Exception as e:
            logger.error(f"下载ECMWF数据失败: {e}")


async def main():
    """主函数"""
    logger.info("气候数据下载脚本启动")
    
    # 检查API密钥
    if not settings.NOAA_API_KEY and not settings.ECMWF_API_KEY:
        logger.error("未配置任何API密钥，请检查.env文件")
        return
    
    # 创建数据管理器
    manager = ClimateDataManager()
    
    # 初始化数据存储系统
    await manager.initialize()
    
    # 设置下载时间范围（最近3年）
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
    
    logger.info(f"下载时间范围: {start_date} 到 {end_date}")
    
    # 开始下载
    await manager.download_all_data(start_date, end_date)
    
    logger.info("气候数据下载脚本完成")


if __name__ == "__main__":
    asyncio.run(main())