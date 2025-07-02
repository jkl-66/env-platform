"""GRIB文件处理模块

该模块提供了处理GRIB格式气象数据文件的功能，包括：
- 读取和加载GRIB文件
- 转换GRIB文件为其他格式
- 提取特定变量和时间范围的数据
- 与现有数据处理流程集成
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import xarray as xr
import pandas as pd
import pygrib
from datetime import datetime, timedelta

from .data_processor import DataProcessor, ProcessingConfig
from .data_storage import DataStorage

logger = logging.getLogger(__name__)

class GRIBProcessor:
    """GRIB文件处理器
    
    提供GRIB文件的读取、处理、转换等功能
    """
    
    def __init__(self, data_storage: Optional[DataStorage] = None):
        """初始化GRIB处理器
        
        Args:
            data_storage: 数据存储实例，用于保存处理后的数据
        """
        self.data_storage = data_storage or DataStorage()
        self.data_processor = DataProcessor()
        
    def load_grib_file(self, file_path: Union[str, Path], 
                      engine: str = 'cfgrib',
                      backend_kwargs: Optional[Dict] = None) -> xr.Dataset:
        """加载单个GRIB文件
        
        Args:
            file_path: GRIB文件路径
            engine: 读取引擎，默认使用cfgrib
            backend_kwargs: 传递给xarray.open_dataset的额外参数
            
        Returns:
            xr.Dataset: 加载的数据集
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"GRIB文件不存在: {file_path}")
            
        if not file_path.suffix.lower() in ['.grib', '.grib2', '.grb', '.grb2']:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
            
        try:
            backend_kwargs = backend_kwargs or {}
            logger.info(f"正在加载GRIB文件: {file_path}")
            
            dataset = xr.open_dataset(
                file_path, 
                engine=engine,
                **backend_kwargs
            )
            
            logger.info(f"成功加载GRIB文件，包含变量: {list(dataset.data_vars.keys())}")
            return dataset
            
        except Exception as e:
            logger.error(f"加载GRIB文件失败: {e}")
            raise
    
    def load_multiple_grib_files(self, file_paths: List[Union[str, Path]], 
                                concat_dim: str = 'time',
                                engine: str = 'cfgrib') -> xr.Dataset:
        """加载多个GRIB文件并合并
        
        Args:
            file_paths: GRIB文件路径列表
            concat_dim: 合并维度，默认为时间维度
            engine: 读取引擎
            
        Returns:
            xr.Dataset: 合并后的数据集
        """
        datasets = []
        
        for file_path in file_paths:
            try:
                ds = self.load_grib_file(file_path, engine=engine)
                datasets.append(ds)
            except Exception as e:
                logger.warning(f"跳过文件 {file_path}: {e}")
                continue
                
        if not datasets:
            raise ValueError("没有成功加载任何GRIB文件")
            
        logger.info(f"合并 {len(datasets)} 个GRIB文件")
        return xr.concat(datasets, dim=concat_dim)
    
    def get_grib_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """获取GRIB文件信息
        
        Args:
            file_path: GRIB文件路径
            
        Returns:
            Dict: 包含文件信息的字典
        """
        dataset = self.load_grib_file(file_path)
        
        info = {
            'file_path': str(file_path),
            'file_size': os.path.getsize(file_path),
            'variables': list(dataset.data_vars.keys()),
            'coordinates': list(dataset.coords.keys()),
            'dimensions': dict(dataset.dims),
            'attributes': dict(dataset.attrs),
            'time_range': None,
            'spatial_extent': None
        }
        
        # 获取时间范围
        if 'time' in dataset.coords:
            time_coord = dataset.coords['time']
            info['time_range'] = {
                'start': str(time_coord.min().values),
                'end': str(time_coord.max().values),
                'count': len(time_coord)
            }
        
        # 获取空间范围
        spatial_coords = {}
        for coord in ['latitude', 'longitude', 'lat', 'lon']:
            if coord in dataset.coords:
                coord_data = dataset.coords[coord]
                spatial_coords[coord] = {
                    'min': float(coord_data.min().values),
                    'max': float(coord_data.max().values),
                    'count': len(coord_data)
                }
        
        if spatial_coords:
            info['spatial_extent'] = spatial_coords
            
        dataset.close()
        return info
    
    def extract_variables(self, file_path: Union[str, Path], 
                         variables: List[str],
                         time_range: Optional[tuple] = None,
                         spatial_bounds: Optional[Dict[str, tuple]] = None) -> xr.Dataset:
        """从GRIB文件中提取特定变量和范围的数据
        
        Args:
            file_path: GRIB文件路径
            variables: 要提取的变量名列表
            time_range: 时间范围 (start_time, end_time)
            spatial_bounds: 空间范围 {'lat': (min, max), 'lon': (min, max)}
            
        Returns:
            xr.Dataset: 提取的数据集
        """
        dataset = self.load_grib_file(file_path)
        
        # 选择变量
        available_vars = list(dataset.data_vars.keys())
        valid_vars = [var for var in variables if var in available_vars]
        
        if not valid_vars:
            raise ValueError(f"未找到指定变量。可用变量: {available_vars}")
            
        if len(valid_vars) != len(variables):
            missing_vars = set(variables) - set(valid_vars)
            logger.warning(f"未找到变量: {missing_vars}")
            
        extracted = dataset[valid_vars]
        
        # 时间范围筛选
        if time_range and 'time' in extracted.coords:
            start_time, end_time = time_range
            extracted = extracted.sel(time=slice(start_time, end_time))
            
        # 空间范围筛选
        if spatial_bounds:
            for coord, (min_val, max_val) in spatial_bounds.items():
                if coord in extracted.coords:
                    extracted = extracted.sel({coord: slice(min_val, max_val)})
                    
        dataset.close()
        return extracted

    def process_grib_to_dataframe(self, file_path: str) -> pd.DataFrame:
        """
        Processes a GRIB file and returns a pandas DataFrame in a memory-efficient way.
        """
        logger.info(f"Memory-efficiently processing GRIB file: {file_path}")
        try:
            grib_messages = pygrib.open(str(file_path))
        except Exception as e:
            logger.error(f"Error opening GRIB file {file_path}: {e}")
            return pd.DataFrame()

        records = {}  # Using a dictionary to store records: (time, lat, lon) -> {var: val}

        for msg in grib_messages:
            try:
                lats, lons = msg.latlons()
                values = msg.values
                variable_name = msg.shortName
                time = msg.validDate

                if lats.ndim != 2 or lons.ndim != 2 or values.ndim != 2:
                    logger.warning(f"Skipping message with non-2D data for variable {variable_name}")
                    continue

                # Flatten the grid data
                lat_flat = lats.flatten()
                lon_flat = lons.flatten()
                val_flat = values.flatten()

                for i in range(len(lat_flat)):
                    key = (time, lat_flat[i], lon_flat[i])
                    if key not in records:
                        records[key] = {}
                    records[key][variable_name] = val_flat[i]

            except Exception as e:
                logger.warning(f"Could not process a message in {file_path}: {e}")
                continue
        
        grib_messages.close()

        if not records:
            logger.warning(f"No data processed from GRIB file: {file_path}")
            return pd.DataFrame()
            
        # Convert the dictionary of records to a list of dictionaries for DataFrame creation
        data_list = []
        for key, variables in records.items():
            record = {'time': key[0], 'latitude': key[1], 'longitude': key[2]}
            record.update(variables)
            data_list.append(record)
            
        df = pd.DataFrame(data_list)
        logger.info(f"Successfully processed GRIB file {file_path} into a DataFrame with shape {df.shape}")
        return df
    
    def convert_to_netcdf(self, grib_path: Union[str, Path], 
                         output_path: Union[str, Path],
                         variables: Optional[List[str]] = None,
                         compression: bool = True) -> str:
        """将GRIB文件转换为NetCDF格式
        
        Args:
            grib_path: 输入GRIB文件路径
            output_path: 输出NetCDF文件路径
            variables: 要转换的变量列表，None表示转换所有变量
            compression: 是否启用压缩
            
        Returns:
            str: 输出文件路径
        """
        dataset = self.load_grib_file(grib_path)
        
        if variables:
            available_vars = list(dataset.data_vars.keys())
            valid_vars = [var for var in variables if var in available_vars]
            if valid_vars:
                dataset = dataset[valid_vars]
            else:
                raise ValueError(f"未找到指定变量。可用变量: {available_vars}")
        
        # 设置编码选项
        encoding = {}
        if compression:
            for var in dataset.data_vars:
                encoding[var] = {'zlib': True, 'complevel': 6}
                
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"转换GRIB文件 {grib_path} 为NetCDF格式: {output_path}")
        dataset.to_netcdf(output_path, encoding=encoding if compression else None)
        
        dataset.close()
        return str(output_path)
    
    def process_grib_data(self, file_path: Union[str, Path],
                         config: Optional[ProcessingConfig] = None,
                         variables: Optional[List[str]] = None) -> xr.Dataset:
        """处理GRIB数据
        
        Args:
            file_path: GRIB文件路径
            config: 处理配置
            variables: 要处理的变量列表
            
        Returns:
            xr.Dataset: 处理后的数据集
        """
        # 加载数据
        if variables:
            dataset = self.extract_variables(file_path, variables)
        else:
            dataset = self.load_grib_file(file_path)
            
        # 使用数据处理器处理数据
        if config:
            processed_dataset = self.data_processor.process_data(dataset, config)
        else:
            # 使用默认配置
            default_config = ProcessingConfig(
                remove_outliers=True,
                fill_missing=True,
                smooth_data=False,
                normalize=False
            )
            processed_dataset = self.data_processor.process_data(dataset, default_config)
            
        return processed_dataset
    
    def save_processed_data(self, dataset: xr.Dataset, 
                           output_path: Union[str, Path],
                           format: str = 'netcdf') -> str:
        """保存处理后的数据
        
        Args:
            dataset: 要保存的数据集
            output_path: 输出路径
            format: 输出格式 ('netcdf', 'zarr')
            
        Returns:
            str: 保存的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'netcdf':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.nc')
            self.data_storage.save_xarray(dataset, str(output_path))
        elif format.lower() == 'zarr':
            if not output_path.suffix:
                output_path = output_path.with_suffix('.zarr')
            dataset.to_zarr(output_path)
        else:
            raise ValueError(f"不支持的输出格式: {format}")
            
        logger.info(f"数据已保存到: {output_path}")
        return str(output_path)
    
    def batch_process_grib_files(self, input_dir: Union[str, Path],
                                output_dir: Union[str, Path],
                                pattern: str = '*.grib*',
                                config: Optional[ProcessingConfig] = None,
                                output_format: str = 'netcdf') -> List[str]:
        """批量处理GRIB文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            pattern: 文件匹配模式
            config: 处理配置
            output_format: 输出格式
            
        Returns:
            List[str]: 处理后的文件路径列表
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        grib_files = list(input_dir.glob(pattern))
        if not grib_files:
            raise ValueError(f"在目录 {input_dir} 中未找到匹配 {pattern} 的GRIB文件")
            
        processed_files = []
        
        for grib_file in grib_files:
            try:
                logger.info(f"处理文件: {grib_file}")
                
                # 处理数据
                processed_data = self.process_grib_data(grib_file, config)
                
                # 生成输出文件名
                output_file = output_dir / f"{grib_file.stem}_processed"
                
                # 保存处理后的数据
                saved_path = self.save_processed_data(
                    processed_data, output_file, output_format
                )
                processed_files.append(saved_path)
                
                processed_data.close()
                
            except Exception as e:
                logger.error(f"处理文件 {grib_file} 时出错: {e}")
                continue
                
        logger.info(f"批量处理完成，共处理 {len(processed_files)} 个文件")
        return processed_files
    
    @staticmethod
    def get_common_grib_variables() -> Dict[str, str]:
        """获取常见的GRIB变量名称和描述
        
        Returns:
            Dict[str, str]: 变量名到描述的映射
        """
        return {
            't2m': '2米气温 (Temperature at 2 meters)',
            'tp': '总降水量 (Total precipitation)',
            'sp': '地面气压 (Surface pressure)',
            'msl': '海平面气压 (Mean sea level pressure)',
            'u10': '10米U风分量 (10 metre U wind component)',
            'v10': '10米V风分量 (10 metre V wind component)',
            'd2m': '2米露点温度 (2 metre dewpoint temperature)',
            'r2': '2米相对湿度 (2 metre relative humidity)',
            'tcc': '总云量 (Total cloud cover)',
            'ssrd': '地面太阳辐射 (Surface solar radiation downwards)',
            'strd': '地面热辐射 (Surface thermal radiation downwards)',
            'ro': '径流 (Runoff)',
            'skt': '地表温度 (Skin temperature)',
            'swvl1': '土壤湿度层1 (Soil wetness level 1)',
            'swvl2': '土壤湿度层2 (Soil wetness level 2)',
            'swvl3': '土壤湿度层3 (Soil wetness level 3)',
            'swvl4': '土壤湿度层4 (Soil wetness level 4)'
        }
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 清理资源
        pass