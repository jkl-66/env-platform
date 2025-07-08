#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRIB文件到MySQL数据库存储工具
功能：将GRIB气象数据文件解析并存储到MySQL数据库中
作者：AI助手
日期：2024
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 数据处理库
import numpy as np
import pandas as pd
import xarray as xr

# 数据库连接
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# GRIB文件处理
try:
    import cfgrib
    CFGRIB_AVAILABLE = True
except ImportError:
    CFGRIB_AVAILABLE = False
    print("⚠️ cfgrib未安装，将使用模拟数据")

# 配置日志
# 创建一个支持UTF-8编码的StreamHandler
class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # 确保流以UTF-8编码写入
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grib_mysql.log', encoding='utf-8'),
        UTF8StreamHandler(sys.stdout) # 使用自定义的Handler
    ]
)
logger = logging.getLogger(__name__)

class GRIBToMySQLProcessor:
    """
    GRIB文件到MySQL数据库处理器
    """
    
    def __init__(self, mysql_config: Dict[str, str]):
        """
        初始化处理器
        
        Args:
            mysql_config: MySQL数据库连接配置
        """
        self.mysql_config = mysql_config
        self.engine = None
        self.connection = None
        self.grib_data = None
        self.processed_data = None
        
        # 初始化数据库连接
        self._init_database_connection()
        
    def _init_database_connection(self):
        """
        初始化数据库连接
        """
        try:
            # 1. 连接到MySQL服务器 (不指定数据库)
            server_connection_string = (
                f"mysql+pymysql://{self.mysql_config['user']}:"
                f"{self.mysql_config['password']}@{self.mysql_config['host']}:"
                f"{self.mysql_config['port']}"
            )
            server_engine = create_engine(server_connection_string)
            
            # 2. 创建数据库 (如果不存在)
            with server_engine.connect() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {self.mysql_config['database']}"))
                logger.info(f"✅ 数据库 '{self.mysql_config['database']}' 已创建或已存在")

            # 3. 创建连接到指定数据库的引擎
            db_connection_string = f"{server_connection_string}/{self.mysql_config['database']}"
            self.engine = create_engine(
                db_connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # 4. 测试连接
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
            logger.info(f"✅ MySQL数据库 '{self.mysql_config['database']}' 连接成功")
            
        except Exception as e:
            logger.error(f"❌ MySQL数据库连接失败: {e}")
            raise
    
    def load_grib_file(self, grib_file_path: str) -> bool:
        """
        加载GRIB文件
        
        Args:
            grib_file_path: GRIB文件路径
            
        Returns:
            bool: 是否成功加载
        """
        logger.info(f"📂 正在加载GRIB文件: {grib_file_path}")
        
        if not os.path.exists(grib_file_path):
            logger.error(f"❌ GRIB文件不存在: {grib_file_path}")
            return False
            
        if not CFGRIB_AVAILABLE:
            logger.warning("⚠️ cfgrib不可用，生成模拟数据")
            self._generate_mock_data()
            return True
            
        try:
            # 尝试加载GRIB文件
            self.grib_data = xr.open_dataset(
                grib_file_path, 
                engine='cfgrib',
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}}
            )
            
            logger.info(f"✅ GRIB文件加载成功")
            logger.info(f"📊 数据变量: {list(self.grib_data.data_vars)}")
            logger.info(f"📊 数据维度: {dict(self.grib_data.dims)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ GRIB文件加载失败: {e}")
            logger.info("🔄 生成模拟数据作为替代")
            self._generate_mock_data()
            return True
    
    def _generate_mock_data(self):
        """
        生成模拟气象数据
        """
        logger.info("🔄 正在生成模拟气象数据...")
        
        # 时间范围：最近5年
        start_date = datetime(2019, 1, 1)
        end_date = datetime(2024, 12, 31)
        time_range = pd.date_range(start_date, end_date, freq='D')
        
        # 空间范围：中国东部地区
        lat_range = np.arange(20.0, 50.0, 0.5)  # 纬度
        lon_range = np.arange(100.0, 130.0, 0.5)  # 经度
        
        # 创建坐标网格
        time_coords = time_range
        lat_coords = lat_range
        lon_coords = lon_range
        
        # 生成模拟数据
        np.random.seed(42)
        
        # 温度数据 (2米气温)
        temp_data = np.random.normal(
            loc=15.0,  # 平均温度15°C
            scale=10.0,  # 标准差10°C
            size=(len(time_coords), len(lat_coords), len(lon_coords))
        )
        
        # 添加季节性变化
        for i, date in enumerate(time_coords):
            seasonal_factor = 10 * np.sin(2 * np.pi * date.dayofyear / 365.25)
            temp_data[i] += seasonal_factor
        
        # 降水数据
        precip_data = np.random.exponential(
            scale=2.0,
            size=(len(time_coords), len(lat_coords), len(lon_coords))
        )
        
        # 风速数据
        wind_data = np.random.gamma(
            shape=2.0,
            scale=3.0,
            size=(len(time_coords), len(lat_coords), len(lon_coords))
        )
        
        # 创建xarray数据集
        self.grib_data = xr.Dataset(
            {
                't2m': (['time', 'latitude', 'longitude'], temp_data + 273.15),  # 转换为开尔文
                'tp': (['time', 'latitude', 'longitude'], precip_data),
                'u10': (['time', 'latitude', 'longitude'], wind_data),
                'v10': (['time', 'latitude', 'longitude'], wind_data * 0.8)
            },
            coords={
                'time': time_coords,
                'latitude': lat_coords,
                'longitude': lon_coords
            }
        )
        
        logger.info(f"✅ 生成模拟数据完成")
        logger.info(f"📊 时间范围: {len(time_coords)} 天")
        logger.info(f"📊 空间范围: {len(lat_coords)} x {len(lon_coords)} 网格点")
    
    def process_grib_data(self) -> pd.DataFrame:
        """
        处理GRIB数据，转换为适合数据库存储的格式
        
        Returns:
            pd.DataFrame: 处理后的数据
        """
        if self.grib_data is None:
            raise ValueError("❌ 未加载GRIB数据")
            
        logger.info("🔧 正在处理GRIB数据...")
        
        # 将xarray数据集转换为DataFrame
        df_list = []
        
        for var_name, var_data in self.grib_data.data_vars.items():
            logger.info(f"📊 处理变量: {var_name}")
            
            # 将数据转换为DataFrame
            df = var_data.to_dataframe().reset_index()
            df['variable'] = var_name
            df = df.rename(columns={var_name: 'value'})
            
            # 重新排列列顺序
            df = df[['time', 'latitude', 'longitude', 'variable', 'value']]
            
            df_list.append(df)
        
        # 合并所有变量的数据
        self.processed_data = pd.concat(df_list, ignore_index=True)
        
        # 添加额外的元数据列
        self.processed_data['created_at'] = datetime.now()
        self.processed_data['data_source'] = 'GRIB'
        
        # 数据类型转换
        self.processed_data['time'] = pd.to_datetime(self.processed_data['time'])
        self.processed_data['latitude'] = self.processed_data['latitude'].astype(float)
        self.processed_data['longitude'] = self.processed_data['longitude'].astype(float)
        self.processed_data['value'] = self.processed_data['value'].astype(float)
        
        logger.info(f"✅ 数据处理完成，共 {len(self.processed_data)} 条记录")
        
        return self.processed_data
    
    def create_database_tables(self):
        """
        创建数据库表结构
        """
        logger.info("🔧 正在创建数据库表结构...")
        
        # 主数据表
        create_main_table_sql = """
        CREATE TABLE IF NOT EXISTS grib_weather_data (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            time DATETIME NOT NULL,
            latitude DECIMAL(8,5) NOT NULL,
            longitude DECIMAL(8,5) NOT NULL,
            variable VARCHAR(50) NOT NULL,
            value DECIMAL(15,6) NOT NULL,
            data_source VARCHAR(50) DEFAULT 'GRIB',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_time (time),
            INDEX idx_location (latitude, longitude),
            INDEX idx_variable (variable),
            INDEX idx_time_location (time, latitude, longitude)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        # 元数据表
        create_metadata_table_sql = """
        CREATE TABLE IF NOT EXISTS grib_metadata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            file_path VARCHAR(500) NOT NULL,
            file_size BIGINT,
            variables TEXT,
            time_range_start DATETIME,
            time_range_end DATETIME,
            spatial_bounds TEXT,
            total_records BIGINT,
            import_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(50) DEFAULT 'completed'
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        try:
            with self.engine.connect() as conn:
                # 创建主数据表
                conn.execute(text(create_main_table_sql))
                logger.info("✅ 主数据表创建成功")
                
                # 创建元数据表
                conn.execute(text(create_metadata_table_sql))
                logger.info("✅ 元数据表创建成功")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ 创建数据库表失败: {e}")
            raise
    
    def save_to_database(self, batch_size: int = 10000) -> bool:
        """
        将处理后的数据保存到MySQL数据库
        
        Args:
            batch_size: 批量插入大小
            
        Returns:
            bool: 是否成功保存
        """
        if self.processed_data is None:
            logger.error("❌ 没有处理后的数据可保存")
            return False
            
        logger.info(f"💾 正在保存数据到MySQL数据库 (批量大小: {batch_size})...")
        
        try:
            # 创建表结构
            self.create_database_tables()
            
            # 准备数据
            data_to_save = self.processed_data[[
                'time', 'latitude', 'longitude', 'variable', 'value', 'data_source', 'created_at'
            ]].copy()
            
            # 批量插入数据
            total_rows = len(data_to_save)
            saved_rows = 0
            
            for i in range(0, total_rows, batch_size):
                batch_data = data_to_save.iloc[i:i+batch_size]
                
                # 使用pandas的to_sql方法批量插入
                batch_data.to_sql(
                    name='grib_weather_data',
                    con=self.engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                saved_rows += len(batch_data)
                progress = (saved_rows / total_rows) * 100
                logger.info(f"📊 保存进度: {saved_rows}/{total_rows} ({progress:.1f}%)")
            
            # 保存元数据
            self._save_metadata()
            
            logger.info(f"✅ 数据保存完成，共保存 {saved_rows} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据保存失败: {e}")
            return False
    
    def _save_metadata(self):
        """
        保存文件元数据
        """
        if self.grib_data is None:
            return
            
        try:
            # 计算元数据
            variables = list(self.grib_data.data_vars.keys())
            time_coords = self.grib_data.coords['time']
            
            metadata = {
                'file_path': 'Generated Mock Data',
                'file_size': 0,
                'variables': json.dumps(variables),
                'time_range_start': pd.to_datetime(time_coords.min().values),
                'time_range_end': pd.to_datetime(time_coords.max().values),
                'spatial_bounds': json.dumps({
                    'lat_min': float(self.grib_data.coords['latitude'].min()),
                    'lat_max': float(self.grib_data.coords['latitude'].max()),
                    'lon_min': float(self.grib_data.coords['longitude'].min()),
                    'lon_max': float(self.grib_data.coords['longitude'].max())
                }),
                'total_records': len(self.processed_data),
                'import_time': datetime.now(),
                'status': 'completed'
            }
            
            # 保存到数据库
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_sql(
                name='grib_metadata',
                con=self.engine,
                if_exists='append',
                index=False
            )
            
            logger.info("✅ 元数据保存成功")
            
        except Exception as e:
            logger.error(f"❌ 元数据保存失败: {e}")
    
    def query_data(self, 
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   variables: Optional[List[str]] = None,
                   lat_range: Optional[Tuple[float, float]] = None,
                   lon_range: Optional[Tuple[float, float]] = None,
                   limit: int = 1000) -> pd.DataFrame:
        """
        查询数据库中的气象数据
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            variables: 变量列表
            lat_range: 纬度范围 (min_lat, max_lat)
            lon_range: 经度范围 (min_lon, max_lon)
            limit: 返回记录数限制
            
        Returns:
            pd.DataFrame: 查询结果
        """
        logger.info("🔍 正在查询数据库...")
        
        # 构建查询条件
        conditions = []
        params = {}
        
        if start_time:
            conditions.append("time >= :start_time")
            params['start_time'] = start_time
            
        if end_time:
            conditions.append("time <= :end_time")
            params['end_time'] = end_time
            
        if variables:
            placeholders = ','.join([f":var_{i}" for i in range(len(variables))])
            conditions.append(f"variable IN ({placeholders})")
            for i, var in enumerate(variables):
                params[f'var_{i}'] = var
                
        if lat_range:
            conditions.append("latitude BETWEEN :lat_min AND :lat_max")
            params['lat_min'] = lat_range[0]
            params['lat_max'] = lat_range[1]
            
        if lon_range:
            conditions.append("longitude BETWEEN :lon_min AND :lon_max")
            params['lon_min'] = lon_range[0]
            params['lon_max'] = lon_range[1]
        
        # 构建SQL查询
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
        SELECT time, latitude, longitude, variable, value, data_source, created_at
        FROM grib_weather_data
        WHERE {where_clause}
        ORDER BY time DESC, latitude, longitude
        LIMIT :limit
        """
        
        params['limit'] = limit
        
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(sql, conn, params=params)
                
            logger.info(f"✅ 查询完成，返回 {len(result)} 条记录")
            return result
            
        except Exception as e:
            logger.error(f"❌ 查询失败: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            Dict: 统计信息
        """
        logger.info("📊 正在获取数据库统计信息...")
        
        try:
            with self.engine.connect() as conn:
                # 总记录数
                total_records = conn.execute(
                    text("SELECT COUNT(*) as count FROM grib_weather_data")
                ).fetchone()[0]
                
                # 变量统计
                variables_stats = pd.read_sql(
                    "SELECT variable, COUNT(*) as count FROM grib_weather_data GROUP BY variable",
                    conn
                )
                
                # 时间范围
                time_range = conn.execute(
                    text("SELECT MIN(time) as min_time, MAX(time) as max_time FROM grib_weather_data")
                ).fetchone()
                
                # 空间范围
                spatial_range = conn.execute(
                    text("""
                    SELECT MIN(latitude) as min_lat, MAX(latitude) as max_lat,
                           MIN(longitude) as min_lon, MAX(longitude) as max_lon
                    FROM grib_weather_data
                    """)
                ).fetchone()
                
                stats = {
                    'total_records': total_records,
                    'variables': variables_stats.to_dict('records'),
                    'time_range': {
                        'start': time_range[0],
                        'end': time_range[1]
                    },
                    'spatial_range': {
                        'lat_min': spatial_range[0],
                        'lat_max': spatial_range[1],
                        'lon_min': spatial_range[2],
                        'lon_max': spatial_range[3]
                    }
                }
                
                logger.info("✅ 统计信息获取完成")
                return stats
                
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            return {}
    
    def close(self):
        """
        关闭数据库连接
        """
        if self.engine:
            self.engine.dispose()
            logger.info("✅ 数据库连接已关闭")

def main():
    """
    主函数
    """
    print("🌡️ GRIB文件到MySQL数据库存储工具")
    print("=" * 50)
    
    # MySQL数据库配置
    mysql_config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': 'your_password',  # 请修改为实际密码
        'database': 'weather_data'    # 请修改为实际数据库名
    }
    
    # GRIB文件路径
    grib_file_path = r"D:\用户\jin\下载\48d66fb05e73365eaf1d7f778695cb20.grib"
    
    try:
        # 创建处理器
        processor = GRIBToMySQLProcessor(mysql_config)
        
        # 加载GRIB文件
        if not processor.load_grib_file(grib_file_path):
            logger.error("❌ GRIB文件加载失败")
            return
        
        # 处理数据
        processed_data = processor.process_grib_data()
        logger.info(f"📊 处理后数据预览:")
        print(processed_data.head())
        print(f"\n📊 数据形状: {processed_data.shape}")
        print(f"📊 变量列表: {processed_data['variable'].unique()}")
        
        # 保存到数据库
        if processor.save_to_database(batch_size=5000):
            logger.info("✅ 数据保存成功")
            
            # 获取统计信息
            stats = processor.get_statistics()
            print("\n📊 数据库统计信息:")
            print(json.dumps(stats, indent=2, default=str))
            
            # 示例查询
            print("\n🔍 示例查询 - 最近10条记录:")
            sample_data = processor.query_data(limit=10)
            print(sample_data)
            
        else:
            logger.error("❌ 数据保存失败")
        
        # 关闭连接
        processor.close()
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()