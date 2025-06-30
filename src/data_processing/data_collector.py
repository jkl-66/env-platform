"""数据收集模块

从多个数据源收集气候数据，包括NOAA、ECMWF等。
"""

import asyncio
import aiohttp
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from ..utils.config import get_settings
from ..utils.logger import get_logger

logger = get_logger("data_collector")
settings = get_settings()


@dataclass
class DataSource:
    """数据源配置"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 10  # 每秒请求数
    timeout: int = 30


class DataCollector:
    """数据收集器
    
    负责从多个外部数据源收集气候数据。
    """
    
    def __init__(self):
        self.data_sources = self._init_data_sources()
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(5)  # 限制并发请求数
    
    def _init_data_sources(self) -> Dict[str, DataSource]:
        """初始化数据源配置"""
        return {
            "noaa": DataSource(
                name="NOAA",
                base_url="https://www.ncei.noaa.gov/data/global-summary-of-the-month/access",
                api_key=settings.NOAA_API_KEY
            ),
            "ecmwf": DataSource(
                name="ECMWF",
                base_url="https://cds.climate.copernicus.eu/api/v2",
                api_key=settings.ECMWF_API_KEY
            ),
            # 可以添加更多数据源
        }
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def collect_noaa_data(
        self,
        start_date: datetime,
        end_date: datetime,
        stations: Optional[List[str]] = None,
        variables: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """收集NOAA气象数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stations: 气象站列表
            variables: 变量列表
            
        Returns:
            包含气象数据的DataFrame
        """
        logger.info(f"开始收集NOAA数据: {start_date} 到 {end_date}")
        
        if not self.session:
            raise RuntimeError("请在异步上下文中使用DataCollector")
        
        # 默认变量
        if variables is None:
            variables = ["TAVG", "TMAX", "TMIN", "PRCP", "SNOW"]
        
        # 默认站点（示例）
        if stations is None:
            stations = ["USW00014734", "USW00023174"]  # 示例站点ID
        
        all_data = []
        
        async with self.semaphore:
            for station in stations:
                try:
                    station_data = await self._fetch_noaa_station_data(
                        station, start_date, end_date, variables
                    )
                    if not station_data.empty:
                        all_data.append(station_data)
                        
                except Exception as e:
                    logger.error(f"获取站点 {station} 数据失败: {e}")
                    continue
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"成功收集NOAA数据: {len(result)} 条记录")
            return result
        else:
            logger.warning("未收集到任何NOAA数据")
            return pd.DataFrame()
    
    async def _fetch_noaa_station_data(
        self,
        station_id: str,
        start_date: datetime,
        end_date: datetime,
        variables: List[str]
    ) -> pd.DataFrame:
        """获取单个站点的NOAA数据"""
        
        # 构建请求URL（这里是示例，实际需要根据NOAA API文档调整）
        params = {
            "dataset": "daily-summaries",
            "stations": station_id,
            "startdate": start_date.strftime("%Y-%m-%d"),
            "enddate": end_date.strftime("%Y-%m-%d"),
            "datatypeid": ",".join(variables),
            "format": "json",
            "limit": 1000
        }
        
        if self.data_sources["noaa"].api_key:
            params["token"] = self.data_sources["noaa"].api_key
        
        try:
            async with self.session.get(
                f"{self.data_sources['noaa'].base_url}/data",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_noaa_response(data, station_id)
                else:
                    logger.error(f"NOAA API请求失败: {response.status}")
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"请求NOAA数据异常: {e}")
            return pd.DataFrame()
    
    def _parse_noaa_response(self, data: dict, station_id: str) -> pd.DataFrame:
        """解析NOAA API响应"""
        if "results" not in data:
            return pd.DataFrame()
        
        records = []
        for item in data["results"]:
            record = {
                "station_id": station_id,
                "date": pd.to_datetime(item.get("date")),
                "datatype": item.get("datatype"),
                "value": item.get("value"),
                "attributes": item.get("attributes", "")
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    async def collect_satellite_data(
        self,
        start_date: datetime,
        end_date: datetime,
        region: Dict[str, float],
        product: str = "MOD11A1"  # MODIS地表温度产品
    ) -> xr.Dataset:
        """收集卫星遥感数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            region: 区域范围 {"north": lat, "south": lat, "east": lon, "west": lon}
            product: 卫星产品名称
            
        Returns:
            包含卫星数据的xarray Dataset
        """
        logger.info(f"开始收集卫星数据: {product}")
        
        # 这里是示例实现，实际需要根据具体的卫星数据API调整
        # 可以使用NASA的Earthdata API或其他卫星数据服务
        
        try:
            # 生成示例数据（实际应该从API获取）
            dates = pd.date_range(start_date, end_date, freq="D")
            lats = np.linspace(region["south"], region["north"], 100)
            lons = np.linspace(region["west"], region["east"], 100)
            
            # 创建示例温度数据
            temp_data = np.random.normal(20, 5, (len(dates), len(lats), len(lons)))
            
            dataset = xr.Dataset(
                {
                    "temperature": (["time", "lat", "lon"], temp_data),
                },
                coords={
                    "time": dates,
                    "lat": lats,
                    "lon": lons
                },
                attrs={
                    "product": product,
                    "source": "satellite",
                    "region": region
                }
            )
            
            logger.info(f"成功收集卫星数据: {dataset.sizes}")
            return dataset
            
        except Exception as e:
            logger.error(f"收集卫星数据失败: {e}")
            raise
    
    async def collect_reanalysis_data(
        self,
        start_date: datetime,
        end_date: datetime,
        variables: List[str],
        region: Optional[Dict[str, float]] = None
    ) -> xr.Dataset:
        """收集再分析数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            variables: 变量列表
            region: 区域范围
            
        Returns:
            包含再分析数据的xarray Dataset
        """
        logger.info(f"开始收集再分析数据: {variables}")
        
        # 这里是示例实现，实际需要连接到ERA5或其他再分析数据源
        try:
            dates = pd.date_range(start_date, end_date, freq="6H")
            
            if region:
                lats = np.linspace(region["south"], region["north"], 50)
                lons = np.linspace(region["west"], region["east"], 50)
            else:
                lats = np.linspace(-90, 90, 181)
                lons = np.linspace(-180, 180, 361)
            
            data_vars = {}
            for var in variables:
                # 生成示例数据
                if var == "temperature":
                    data = np.random.normal(15, 10, (len(dates), len(lats), len(lons)))
                elif var == "precipitation":
                    data = np.random.exponential(2, (len(dates), len(lats), len(lons)))
                elif var == "pressure":
                    data = np.random.normal(1013, 20, (len(dates), len(lats), len(lons)))
                else:
                    data = np.random.normal(0, 1, (len(dates), len(lats), len(lons)))
                
                data_vars[var] = (["time", "lat", "lon"], data)
            
            dataset = xr.Dataset(
                data_vars,
                coords={
                    "time": dates,
                    "lat": lats,
                    "lon": lons
                },
                attrs={
                    "source": "reanalysis",
                    "variables": variables,
                    "region": region or "global"
                }
            )
            
            logger.info(f"成功收集再分析数据: {dataset.sizes}")
            return dataset
            
        except Exception as e:
            logger.error(f"收集再分析数据失败: {e}")
            raise
    
    async def save_data(
        self,
        data: Union[pd.DataFrame, xr.Dataset],
        filename: str,
        format_type: str = "auto"
    ) -> Path:
        """保存数据到文件
        
        Args:
            data: 要保存的数据
            filename: 文件名
            format_type: 文件格式 (auto/csv/netcdf/hdf5)
            
        Returns:
            保存的文件路径
        """
        save_path = settings.DATA_ROOT_PATH / "raw" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(data, pd.DataFrame):
                if format_type == "auto" or format_type == "csv":
                    data.to_csv(save_path.with_suffix(".csv"), index=False)
                elif format_type == "hdf5":
                    data.to_hdf(save_path.with_suffix(".h5"), key="data")
                    
            elif isinstance(data, xr.Dataset):
                if format_type == "auto" or format_type == "netcdf":
                    data.to_netcdf(save_path.with_suffix(".nc"))
                elif format_type == "hdf5":
                    data.to_netcdf(save_path.with_suffix(".h5"), engine="h5netcdf")
            
            logger.info(f"数据已保存到: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            raise


# 便捷函数
async def collect_climate_data(
    start_date: datetime,
    end_date: datetime,
    data_types: List[str] = None,
    region: Dict[str, float] = None
) -> Dict[str, Union[pd.DataFrame, xr.Dataset]]:
    """收集多种类型的气候数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        data_types: 数据类型列表 ["noaa", "satellite", "reanalysis"]
        region: 区域范围
        
    Returns:
        包含各种数据的字典
    """
    if data_types is None:
        data_types = ["noaa", "reanalysis"]
    
    results = {}
    
    async with DataCollector() as collector:
        for data_type in data_types:
            try:
                if data_type == "noaa":
                    results["noaa"] = await collector.collect_noaa_data(
                        start_date, end_date
                    )
                elif data_type == "satellite":
                    if region:
                        results["satellite"] = await collector.collect_satellite_data(
                            start_date, end_date, region
                        )
                elif data_type == "reanalysis":
                    results["reanalysis"] = await collector.collect_reanalysis_data(
                        start_date, end_date, ["temperature", "precipitation"]
                    )
                    
            except Exception as e:
                logger.error(f"收集 {data_type} 数据失败: {e}")
                continue
    
    return results