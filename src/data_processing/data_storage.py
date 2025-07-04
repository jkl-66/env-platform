"""数据存储模块

实现多种数据库的统一存储接口，支持时序数据、文件数据和元数据管理。
"""

import asyncio
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
import warnings

try:
    import xarray as xr
except ImportError:
    warnings.warn("xarray未安装，NetCDF数据处理功能受限")
    xr = None



from ..utils.config import get_settings
from ..utils.logger import get_logger
logger = get_logger("data_storage")
settings = get_settings()

# SQLAlchemy基类
Base = declarative_base()


class ClimateDataRecord(Base):
    """气候数据记录表"""
    __tablename__ = "climate_data_records"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source = Column(String(100), nullable=False)  # 数据源
    data_type = Column(String(50), nullable=False)  # 数据类型
    location = Column(String(200))  # 位置信息
    latitude = Column(Float)
    longitude = Column(Float)
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    file_path = Column(String(500))  # 文件路径
    file_format = Column(String(20))  # 文件格式
    file_size = Column(Integer)  # 文件大小（字节）
    variables = Column(JSON)  # 变量列表
    data_metadata = Column(JSON)  # 元数据
    quality_score = Column(Float)  # 数据质量评分
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))


class ProcessingJob(Base):
    """数据处理任务表"""
    __tablename__ = "processing_jobs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_type = Column(String(50), nullable=False)  # 任务类型
    status = Column(String(20), default="pending")  # 状态
    input_data_id = Column(String(36))  # 输入数据ID
    output_data_id = Column(String(36))  # 输出数据ID
    parameters = Column(JSON)  # 处理参数
    progress = Column(Float, default=0.0)  # 进度
    error_message = Column(Text)  # 错误信息
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))


class ModelResult(Base):
    """模型结果表"""
    __tablename__ = "model_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    input_data_id = Column(String(36))
    result_type = Column(String(50))  # 结果类型
    result_data = Column(JSON)  # 结果数据
    confidence_score = Column(Float)  # 置信度
    execution_time = Column(Float)  # 执行时间（秒）
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))


class DataStorage:
    """数据存储管理器
    
    统一管理多种数据存储方式：MySQL、文件系统。
    """
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        
        # 文件存储路径
        self.data_root = Path(settings.DATA_ROOT_PATH)
        self.raw_data_path = self.data_root / "raw"
        self.processed_data_path = self.data_root / "processed"
        self.model_data_path = self.data_root / "models"
        self.cache_data_path = self.data_root / "cache"
        
        # 创建目录
        for path in [self.raw_data_path, self.processed_data_path, 
                     self.model_data_path, self.cache_data_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> None:
        """初始化数据存储"""
        logger.info("初始化数据存储系统...")
        
        # 设置MySQL
        try:
            logger.info(f"正在连接MySQL: {settings.mysql_url}")
            self.engine = create_engine(settings.mysql_url, connect_args={'connect_timeout': 5})
            logger.info("数据库引擎创建成功。")
            self.session_factory = sessionmaker(bind=self.engine)
            logger.info("数据库会话工厂创建成功。")
            
            # 创建表
            Base.metadata.create_all(self.engine)
            logger.info("MySQL数据库连接建立，表创建完成")
        except Exception as e:
            logger.error(f"无法连接到MySQL数据库: {e}", exc_info=True)
            raise RuntimeError(f"数据库初始化失败: {e}") from e
            
        logger.info("数据存储系统初始化完成")
    
    async def close(self) -> None:
        """关闭数据存储连接"""
        try:
            if self.engine:
                self.engine.dispose()
            logger.info("数据存储连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据存储连接时出错: {e}")

    async def fetch_data_as_dataframe(self, query: str, params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """从数据库获取数据并作为Pandas DataFrame返回

        Args:
            query: SQL查询语句
            params: 查询参数

        Returns:
            Pandas DataFrame, or None if an error occurs.
        """
        if not self.engine:
            logger.warning("数据库未初始化，无法获取数据")
            return None
        
        try:
            loop = asyncio.get_running_loop()
            # pd.read_sql_query is synchronous, run it in an executor
            df = await loop.run_in_executor(
                None, 
                lambda: pd.read_sql_query(query, self.engine, params=params)
            )
            return df
        except Exception as e:
            logger.error(f"执行数据库查询时出错: {e}", exc_info=True)
            return None
    
    # ==================== 元数据管理 ====================
    

    
    async def save_data_record(
        self,
        source: str,
        data_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        file_path: Optional[str] = None,
        file_format: Optional[str] = None,
        file_size: Optional[int] = None,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        variables: Optional[List[str]] = None,
        data_metadata: Optional[Dict[str, Any]] = None,
        quality_score: Optional[float] = None
    ) -> Optional[uuid.UUID]:
        """保存数据记录元数据
        
        Args:
            source: 数据源
            data_type: 数据类型
            file_path: 文件路径
            location: 位置描述
            coordinates: 坐标(纬度, 经度)
            time_range: 时间范围
            variables: 变量列表
            metadata: 元数据
            quality_score: 质量评分
            
        Returns:
            数据记录ID
        """
        if not self.session_factory:
            logger.warning("数据库未初始化，无法保存元数据记录")
            return None
        
        session = self.session_factory()
        try:
            # 创建记录
            record = ClimateDataRecord(
                source=source,
                data_type=data_type,
                location=location,
                latitude=latitude,
                longitude=longitude,
                start_time=start_time,
                end_time=end_time,
                file_path=file_path,
                file_format=file_format,
                file_size=file_size,
                variables=variables,
                data_metadata=data_metadata,
                quality_score=quality_score
            )
            
            session.add(record)
            session.commit()
            
            session.flush() # 获取生成的ID
            record_id = record.id
            session.commit()
            
            logger.info(f"数据记录已保存: {record_id}")
            return record_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"保存数据记录失败: {e}")
            raise
        finally:
            session.close()
    
    def get_data_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """获取数据记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            数据记录字典
        """
        if not self.session_factory:
            return None
        
        session = self.session_factory()
        try:
            record = session.query(ClimateDataRecord).filter(
                ClimateDataRecord.id == uuid.UUID(record_id)
            ).first()
            
            if record:
                return {
                    "id": str(record.id),
                    "source": record.source,
                    "data_type": record.data_type,
                    "location": record.location,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "start_time": record.start_time.isoformat() if record.start_time else None,
                    "end_time": record.end_time.isoformat() if record.end_time else None,
                    "file_path": record.file_path,
                    "file_format": record.file_format,
                    "file_size": record.file_size,
                    "variables": record.variables,
                    "metadata": record.data_metadata,
                    "quality_score": record.quality_score,
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"获取数据记录失败: {e}")
            return None
        finally:
            session.close()
    
    async def get_data_sources(
        self,
        source_type: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """获取可用的数据源列表"""
        # 在实际应用中，这些信息应该来自数据库或配置文件
        sources = [
            {
                "id": "noaa-ghcn-daily",
                "name": "NOAA GHCN-Daily",
                "type": "noaa",
                "description": "NOAA全球历史气候网络日度数据",
                "is_active": True,
                "created_at": datetime.now(),
                "last_updated": datetime.now()
            },
            {
                "id": "noaa-oisst-avhrr",
                "name": "NOAA OISST AVHRR-Only",
                "type": "noaa",
                "description": "NOAA最优插值海表温度数据",
                "is_active": True,
                "created_at": datetime.now(),
                "last_updated": datetime.now()
            },
            {
                "id": "ecmwf-era5",
                "name": "ECMWF ERA5",
                "type": "ecmwf",
                "description": "ECMWF第五代全球气候再分析数据",
                "is_active": True,
                "created_at": datetime.now(),
                "last_updated": datetime.now()
            }
        ]

        if source_type:
            sources = [s for s in sources if s.type == source_type]
        if is_active is not None:
            sources = [s for s in sources if s.is_active == is_active]

        return sources

    async def get_datasets(
        self,
        dataset_type: Optional[str] = None,
        data_source: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取数据集列表"""
        records = self.search_data_records(
            source=data_source,
            data_type=dataset_type,
            time_range=(start_date, end_date) if start_date and end_date else None,
            limit=limit
        )

        datasets = []
        for record in records:
            datasets.append({
                "id": record.get("id"),
                "name": f"{record.get('source')} - {record.get('data_type')}",
                "description": f"Dataset for {record.get('location')}",
                "data_source": record.get("source"),
                "dataset_type": record.get("data_type"),
                "variables": record.get("variables", []),
                "temporal_resolution": record.get("metadata", {}).get("temporal_resolution"),
                "spatial_resolution": record.get("metadata", {}).get("spatial_resolution"),
                "start_date": record.get("start_time"),
                "end_date": record.get("end_time"),
                "file_size": record.get("file_size"),
                "record_count": record.get("metadata", {}).get("record_count"),
                "is_processed": record.get("metadata", {}).get("is_processed", False),
                "created_at": record.get("created_at")
            })
        return datasets

    def search_data_records(
        self,
        source: Optional[str] = None,
        data_type: Optional[str] = None,
        location: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        variables: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """搜索数据记录
        
        Args:
            source: 数据源过滤
            data_type: 数据类型过滤
            location: 位置过滤
            time_range: 时间范围过滤
            variables: 变量过滤
            limit: 结果限制
            
        Returns:
            数据记录列表
        """
        if not self.session_factory:
            return []
        
        session = self.session_factory()
        try:
            query = session.query(ClimateDataRecord)
            
            # 应用过滤条件
            if source:
                query = query.filter(ClimateDataRecord.source == source)
            
            if data_type:
                query = query.filter(ClimateDataRecord.data_type == data_type)
            
            if location:
                query = query.filter(ClimateDataRecord.location.ilike(f"%{location}%"))
            
            if time_range:
                start_time, end_time = time_range
                query = query.filter(
                    ClimateDataRecord.start_time >= start_time,
                    ClimateDataRecord.end_time <= end_time
                )
            
            if variables:
                # 搜索包含指定变量的记录
                for var in variables:
                    query = query.filter(
                        ClimateDataRecord.variables.op('@>')([var])
                    )
            
            # 执行查询
            records = query.order_by(ClimateDataRecord.created_at.desc()).limit(limit).all()
            
            # 转换为字典列表
            result = []
            for record in records:
                result.append({
                    "id": str(record.id),
                    "source": record.source,
                    "data_type": record.data_type,
                    "location": record.location,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "start_time": record.start_time.isoformat() if record.start_time else None,
                    "end_time": record.end_time.isoformat() if record.end_time else None,
                    "file_path": record.file_path,
                    "file_format": record.file_format,
                    "variables": record.variables,
                    "quality_score": record.quality_score,
                    "created_at": record.created_at.isoformat()
                })
            
            return result
            
        except Exception as e:
            logger.error(f"搜索数据记录失败: {e}")
            return []
        finally:
            session.close()
    

    
    # ==================== 文件数据存储 ====================
    
    def save_dataframe(
        self,
        data: pd.DataFrame,
        filename: str,
        data_category: str = "processed",
        format: str = "parquet"
    ) -> str:
        """保存DataFrame到文件
        
        Args:
            data: DataFrame数据
            filename: 文件名
            data_category: 数据类别（raw/processed/models/cache）
            format: 文件格式（parquet/csv/json）
            
        Returns:
            文件路径
        """
        # 选择存储路径
        if data_category == "raw":
            base_path = self.raw_data_path
        elif data_category == "processed":
            base_path = self.processed_data_path
        elif data_category == "models":
            base_path = self.model_data_path
        elif data_category == "cache":
            base_path = self.cache_data_path
        else:
            base_path = self.data_root
        
        # 添加文件扩展名
        if not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"
        
        file_path = base_path / filename
        
        try:
            # 确保目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存文件
            if format == "parquet":
                data.to_parquet(file_path, index=True)
            elif format == "csv":
                data.to_csv(file_path, index=True)
            elif format == "json":
                data.to_json(file_path, orient="records", date_format="iso")
            else:
                raise ValueError(f"不支持的文件格式: {format}")
            
            logger.info(f"DataFrame已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"保存DataFrame失败: {e}")
            raise
    
    def load_dataframe(
        self,
        file_path: str,
        format: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """从文件加载DataFrame
        
        Args:
            file_path: 文件路径
            format: 文件格式（自动检测如果未指定）
            
        Returns:
            DataFrame数据
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            logger.error(f"文件不存在: {file_path}")
            return None
        
        # 自动检测格式
        if format is None:
            format = file_path_obj.suffix.lower().lstrip('.')
        
        try:
            if format == "parquet":
                data = pd.read_parquet(file_path)
            elif format == "csv":
                data = pd.read_csv(file_path, index_col=0)
            elif format == "json":
                data = pd.read_json(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {format}")
            
            logger.info(f"DataFrame已加载: {file_path}, 形状: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"加载DataFrame失败: {e}")
            return None
    
    def save_xarray(
        self,
        data: 'xr.Dataset',
        filename: str,
        data_category: str = "processed"
    ) -> str:
        """保存xarray数据集
        
        Args:
            data: xarray数据集
            filename: 文件名
            data_category: 数据类别
            
        Returns:
            文件路径
        """
        if xr is None:
            raise RuntimeError("xarray未安装")
        
        # 选择存储路径
        if data_category == "raw":
            base_path = self.raw_data_path
        elif data_category == "processed":
            base_path = self.processed_data_path
        else:
            base_path = self.data_root
        
        # 添加文件扩展名
        if not filename.endswith('.nc'):
            filename = f"{filename}.nc"
        
        file_path = base_path / filename
        
        try:
            # 确保目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存NetCDF文件
            data.to_netcdf(file_path)
            
            logger.info(f"xarray数据集已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"保存xarray数据集失败: {e}")
            raise
    
    def load_xarray(self, file_path: str) -> Optional['xr.Dataset']:
        """加载xarray数据集
        
        支持多种格式：NetCDF, GRIB, HDF5等
        
        Args:
            file_path: 文件路径
            
        Returns:
            xarray数据集
        """
        if xr is None:
            logger.error("xarray未安装")
            return None
        
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            logger.error(f"文件不存在: {file_path}")
            return None
        
        try:
            # 检测文件格式并选择合适的引擎
            file_suffix = file_path_obj.suffix.lower()
            
            if file_suffix in ['.grib', '.grib2', '.grb', '.grb2']:
                # GRIB文件使用cfgrib引擎
                logger.info(f"检测到GRIB文件，使用cfgrib引擎: {file_path}")
                data = xr.open_dataset(file_path, engine='cfgrib')
            elif file_suffix in ['.nc', '.netcdf', '.nc4']:
                # NetCDF文件使用默认引擎
                data = xr.open_dataset(file_path)
            elif file_suffix in ['.h5', '.hdf5']:
                # HDF5文件
                data = xr.open_dataset(file_path, engine='h5netcdf')
            else:
                # 其他格式，尝试默认引擎
                data = xr.open_dataset(file_path)
            
            logger.info(f"xarray数据集已加载: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"加载xarray数据集失败: {e}")
            return None
    
    def load_grib_file(self, file_path: str, 
                      backend_kwargs: Optional[Dict] = None) -> Optional['xr.Dataset']:
        """专门加载GRIB文件的方法
        
        Args:
            file_path: GRIB文件路径
            backend_kwargs: 传递给cfgrib的额外参数
            
        Returns:
            xarray数据集
        """
        if xr is None:
            logger.error("xarray未安装")
            return None
        
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            logger.error(f"GRIB文件不存在: {file_path}")
            return None
        
        if not file_path_obj.suffix.lower() in ['.grib', '.grib2', '.grb', '.grb2']:
            logger.warning(f"文件扩展名不是标准GRIB格式: {file_path_obj.suffix}")
        
        try:
            backend_kwargs = backend_kwargs or {}
            logger.info(f"正在加载GRIB文件: {file_path}")
            
            data = xr.open_dataset(file_path, engine='cfgrib', **backend_kwargs)
            
            logger.info(f"GRIB文件加载成功，包含变量: {list(data.data_vars.keys())}")
            return data
            
        except Exception as e:
            logger.error(f"加载GRIB文件失败: {e}")
            return None
    
    # ==================== 气象数据存储 ====================
    
    async def store_weather_data(
        self,
        data: pd.DataFrame,
        source: str,
        station_id: str,
        tags: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """存储天气数据到时序数据库和文件系统"""
        if data.empty:
            logger.warning("接收到空的数据框，不进行存储")
            return None

        logger.info(f"开始存储来自 {source} 的气象数据，站点: {station_id}")

        # 1. 保存为Parquet文件
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"{source.lower()}_{station_id}_{timestamp}"
        
        try:
            file_path_str = self.save_dataframe(data, file_name, data_category="processed", format="parquet")
        except Exception as e:
            logger.error(f"保存Parquet文件失败: {e}")
            return None

        # 2. 存储元数据记录
        if self.session_factory:
            try:
                start_time = data['DATE'].min().to_pydatetime()
                end_time = data['DATE'].max().to_pydatetime()
                variables = [col for col in data.columns if col not in ['DATE', 'STATION']]
                
                self.save_data_record(
                    source=source,
                    data_type="time_series_observation",
                    location=station_id,
                    start_time=start_time,
                    end_time=end_time,
                    file_path=file_path_str,
                    file_format="parquet",
                    file_size=Path(file_path_str).stat().st_size,
                    variables=variables,
                    data_metadata=tags
                )
                logger.info(f"成功在PostgreSQL中创建数据记录")

            except Exception as e:
                logger.error(f"在PostgreSQL中创建数据记录失败: {e}")


        
        return file_path_str
    

    
    # ==================== 缓存管理 ====================
    
    def cache_set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            expire: 过期时间（秒）
            
        Returns:
            是否成功
        """
        if not self.redis_client:
            # 使用文件缓存作为备选
            return self._file_cache_set(key, value, expire)
        
        try:
            # 序列化值
            serialized_value = pickle.dumps(value)
            
            # 设置缓存
            if expire:
                result = self.redis_client.setex(key, expire, serialized_value)
            else:
                result = self.redis_client.set(key, serialized_value)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"设置Redis缓存失败: {e}")
            return self._file_cache_set(key, value, expire)
    
    def cache_get(self, key: str) -> Any:
        """获取缓存
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值
        """
        if not self.redis_client:
            # 使用文件缓存作为备选
            return self._file_cache_get(key)
        
        try:
            serialized_value = self.redis_client.get(key)
            
            if serialized_value:
                return pickle.loads(serialized_value)
            
            return None
            
        except Exception as e:
            logger.error(f"获取Redis缓存失败: {e}")
            return self._file_cache_get(key)
    
    def cache_delete(self, key: str) -> bool:
        """删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功
        """
        if not self.redis_client:
            return self._file_cache_delete(key)
        
        try:
            result = self.redis_client.delete(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"删除Redis缓存失败: {e}")
            return self._file_cache_delete(key)
    
    def _file_cache_set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """文件缓存设置"""
        try:
            cache_file = self.cache_data_path / f"{key}.cache"
            
            cache_data = {
                "value": value,
                "created_at": datetime.now(timezone.utc).timestamp(),
                "expire": expire
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            return True
            
        except Exception as e:
            logger.error(f"文件缓存设置失败: {e}")
            return False
    
    def _file_cache_get(self, key: str) -> Any:
        """文件缓存获取"""
        try:
            cache_file = self.cache_data_path / f"{key}.cache"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 检查过期时间
            if cache_data.get("expire"):
                created_at = cache_data["created_at"]
                expire = cache_data["expire"]
                
                if datetime.now(timezone.utc).timestamp() - created_at > expire:
                    # 缓存已过期，删除文件
                    cache_file.unlink()
                    return None
            
            return cache_data["value"]
            
        except Exception as e:
            logger.error(f"文件缓存获取失败: {e}")
            return None
    
    def _file_cache_delete(self, key: str) -> bool:
        """文件缓存删除"""
        try:
            cache_file = self.cache_data_path / f"{key}.cache"
            
            if cache_file.exists():
                cache_file.unlink()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"文件缓存删除失败: {e}")
            return False
    
    # ==================== 任务管理 ====================
    
    def create_processing_job(
        self,
        job_type: str,
        input_data_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建处理任务
        
        Args:
            job_type: 任务类型
            input_data_id: 输入数据ID
            parameters: 处理参数
            
        Returns:
            任务ID
        """
        if not self.session_factory:
            raise RuntimeError("PostgreSQL未初始化")
        
        session = self.session_factory()
        try:
            job = ProcessingJob(
                job_type=job_type,
                input_data_id=uuid.UUID(input_data_id) if input_data_id else None,
                parameters=parameters or {}
            )
            
            session.add(job)
            session.commit()
            
            job_id = str(job.id)
            logger.info(f"处理任务已创建: {job_id}")
            
            return job_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"创建处理任务失败: {e}")
            raise
        finally:
            session.close()
    
    def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """更新任务状态
        
        Args:
            job_id: 任务ID
            status: 状态
            progress: 进度
            error_message: 错误信息
            
        Returns:
            是否成功
        """
        if not self.session_factory:
            return False
        
        session = self.session_factory()
        try:
            job = session.query(ProcessingJob).filter(
                ProcessingJob.id == uuid.UUID(job_id)
            ).first()
            
            if job:
                job.status = status
                if progress is not None:
                    job.progress = progress
                if error_message:
                    job.error_message = error_message
                
                if status == "running" and not job.started_at:
                    job.started_at = datetime.now(timezone.utc)
                elif status in ["completed", "failed"]:
                    job.completed_at = datetime.now(timezone.utc)
                
                session.commit()
                return True
            
            return False
            
        except Exception as e:
            session.rollback()
            logger.error(f"更新任务状态失败: {e}")
            return False
        finally:
            session.close()
    
    def save_model_result(
        self,
        model_name: str,
        result_data: Dict[str, Any],
        model_version: Optional[str] = None,
        input_data_id: Optional[str] = None,
        result_type: Optional[str] = None,
        confidence_score: Optional[float] = None,
        execution_time: Optional[float] = None
    ) -> str:
        """保存模型结果
        
        Args:
            model_name: 模型名称
            result_data: 结果数据
            model_version: 模型版本
            input_data_id: 输入数据ID
            result_type: 结果类型
            confidence_score: 置信度
            execution_time: 执行时间
            
        Returns:
            结果ID
        """
        if not self.session_factory:
            raise RuntimeError("PostgreSQL未初始化")
        
        session = self.session_factory()
        try:
            result = ModelResult(
                model_name=model_name,
                model_version=model_version,
                input_data_id=uuid.UUID(input_data_id) if input_data_id else None,
                result_type=result_type,
                result_data=result_data,
                confidence_score=confidence_score,
                execution_time=execution_time
            )
            
            session.add(result)
            session.commit()
            
            result_id = str(result.id)
            logger.info(f"模型结果已保存: {result_id}")
            
            return result_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"保存模型结果失败: {e}")
            raise
        finally:
            session.close()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息
        
        Returns:
            存储统计字典
        """
        stats = {
            "file_storage": {},
            "database_records": {},
            "cache_info": {}
        }
        
        # 文件存储统计
        for category, path in [
            ("raw", self.raw_data_path),
            ("processed", self.processed_data_path),
            ("models", self.model_data_path),
            ("cache", self.cache_data_path)
        ]:
            if path.exists():
                files = list(path.rglob("*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                stats["file_storage"][category] = {
                    "file_count": len([f for f in files if f.is_file()]),
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "path": str(path)
                }
        
        # 数据库记录统计
        if self.session_factory:
            session = self.session_factory()
            try:
                stats["database_records"] = {
                    "climate_data_records": session.query(ClimateDataRecord).count(),
                    "processing_jobs": session.query(ProcessingJob).count(),
                    "model_results": session.query(ModelResult).count()
                }
            except Exception as e:
                logger.error(f"获取数据库统计失败: {e}")
                stats["database_records"] = {"error": str(e)}
            finally:
                session.close()
        

        
        return stats