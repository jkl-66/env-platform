#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的NOAA数据下载测试脚本
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
import logging
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 直接导入配置，避免复杂的模型导入
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class SimpleSettings(BaseSettings):
    """简化的配置类"""
    NOAA_API_KEY: Optional[str] = Field(default=None, description="NOAA API密钥")
    DATA_ROOT_PATH: Path = Field(default=Path("data"), description="数据根目录")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore"  # 忽略额外的环境变量
    }

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("noaa_test")

class NOAADataDownloader:
    """NOAA数据下载器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2"
        self.session = requests.Session()
        self.session.headers.update({"token": api_key})
    
    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            url = f"{self.base_url}/datasets"
            response = self.session.get(url, params={"limit": 1})
            response.raise_for_status()
            logger.info("NOAA API连接测试成功")
            return True
        except Exception as e:
            logger.error(f"NOAA API连接测试失败: {e}")
            return False
    
    def get_stations(self, dataset_id: str = "GHCND", limit: int = 10) -> list:
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
    
    def download_sample_data(self, station_id: str, days: int = 7) -> pd.DataFrame:
        """下载样本数据"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        logger.info(f"下载站点{station_id}的数据: {start_date} 到 {end_date}")
        
        url = f"{self.base_url}/data"
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "datatypeid": "TMAX,TMIN,PRCP",
            "limit": 1000,
            "format": "json"
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "results" in data:
                df = pd.DataFrame(data["results"])
                logger.info(f"成功下载{len(df)}条记录")
                return df
            else:
                logger.warning("未获取到数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"下载数据失败: {e}")
            return pd.DataFrame()

def main():
    """主函数"""
    logger.info("开始NOAA数据下载测试")
    
    # 加载配置
    settings = SimpleSettings()
    
    # 检查API密钥
    if not settings.NOAA_API_KEY or settings.NOAA_API_KEY == "your_noaa_api_key_here":
        logger.error("NOAA API密钥未配置或使用默认值，请在.env文件中设置NOAA_API_KEY")
        logger.info("请访问 https://www.ncdc.noaa.gov/cdo-web/token 申请API密钥")
        return False
    
    # 创建下载器
    downloader = NOAADataDownloader(settings.NOAA_API_KEY)
    
    # 测试连接
    if not downloader.test_connection():
        logger.error("无法连接到NOAA API")
        return False
    
    # 获取气象站
    stations = downloader.get_stations(limit=5)
    if not stations:
        logger.error("无法获取气象站列表")
        return False
    
    # 显示气象站信息
    logger.info("可用的气象站:")
    for i, station in enumerate(stations[:3]):
        logger.info(f"  {i+1}. {station.get('name', 'Unknown')} ({station.get('id', 'Unknown ID')})")
    
    # 下载样本数据
    test_station = stations[0]
    station_id = test_station.get('id')
    
    if station_id:
        sample_data = downloader.download_sample_data(station_id, days=7)
        
        if not sample_data.empty:
            logger.info(f"样本数据预览:")
            logger.info(f"数据形状: {sample_data.shape}")
            logger.info(f"列名: {list(sample_data.columns)}")
            
            # 保存样本数据
            data_dir = settings.DATA_ROOT_PATH / "raw"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = data_dir / f"noaa_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            sample_data.to_csv(output_file, index=False)
            logger.info(f"样本数据已保存到: {output_file}")
            
            logger.info("NOAA数据下载功能测试成功！")
            return True
        else:
            logger.warning("未能下载到样本数据")
            return False
    else:
        logger.error("无法获取有效的气象站ID")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ NOAA数据下载功能正常工作")
    else:
        print("\n❌ NOAA数据下载功能存在问题")