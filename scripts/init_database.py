#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库初始化脚本
Database Initialization Script

此脚本用于初始化PostgreSQL和InfluxDB数据库，创建必要的表和存储桶。
This script initializes PostgreSQL and InfluxDB databases, creating necessary tables and buckets.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import asyncpg
    import influxdb_client
    from influxdb_client.client.write_api import SYNCHRONOUS
    from sqlalchemy import create_engine, text
    from sqlalchemy.ext.asyncio import create_async_engine
except ImportError as e:
    print(f"缺少必要的依赖包: {e}")
    print("请运行: pip install asyncpg influxdb-client sqlalchemy[asyncio]")
    sys.exit(1)

from src.utils.config import get_settings
from src.utils.logger import get_logger

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger("database_initializer")
settings = get_settings()


class PostgreSQLInitializer:
    """PostgreSQL数据库初始化器"""
    
    def __init__(self):
        self.settings = settings
        self.engine = None
    
    async def initialize(self):
        """初始化PostgreSQL数据库"""
        logger.info("开始初始化PostgreSQL数据库")
        
        try:
            # 创建数据库连接
            await self._create_database_if_not_exists()
            
            # 创建异步引擎
            self.engine = create_async_engine(
                self.settings.postgres_url.replace("postgresql://", "postgresql+asyncpg://"),
                echo=self.settings.DEBUG
            )
            
            # 创建表
            await self._create_tables()
            
            # 插入初始数据
            await self._insert_initial_data()
            
            logger.info("PostgreSQL数据库初始化完成")
            
        except Exception as e:
            logger.error(f"PostgreSQL数据库初始化失败: {e}")
            raise
        finally:
            if self.engine:
                await self.engine.dispose()
    
    async def _create_database_if_not_exists(self):
        """创建数据库（如果不存在）"""
        # 连接到默认的postgres数据库
        default_url = (
            f"postgresql://{self.settings.POSTGRES_USER}:{self.settings.POSTGRES_PASSWORD}@"
            f"{self.settings.POSTGRES_HOST}:{self.settings.POSTGRES_PORT}/postgres"
        )
        
        try:
            conn = await asyncpg.connect(default_url)
            
            # 检查数据库是否存在
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                self.settings.POSTGRES_DB
            )
            
            if not exists:
                # 创建数据库
                await conn.execute(f"CREATE DATABASE {self.settings.POSTGRES_DB}")
                logger.info(f"数据库 {self.settings.POSTGRES_DB} 创建成功")
            else:
                logger.info(f"数据库 {self.settings.POSTGRES_DB} 已存在")
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"创建数据库失败: {e}")
            raise
    
    async def _create_tables(self):
        """创建数据表"""
        logger.info("创建数据表")
        
        # 定义表结构
        tables_sql = [
            # 用户表
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                is_admin BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # 数据源表
            """
            CREATE TABLE IF NOT EXISTS data_sources (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                type VARCHAR(50) NOT NULL,
                url VARCHAR(500),
                api_key_required BOOLEAN DEFAULT FALSE,
                description TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # 气象站表
            """
            CREATE TABLE IF NOT EXISTS weather_stations (
                id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(200),
                latitude DECIMAL(10, 7),
                longitude DECIMAL(10, 7),
                elevation DECIMAL(10, 2),
                country VARCHAR(10),
                state VARCHAR(50),
                data_source_id INTEGER REFERENCES data_sources(id),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # 数据集表
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id SERIAL PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                data_source_id INTEGER REFERENCES data_sources(id),
                start_date DATE,
                end_date DATE,
                variables JSONB,
                metadata JSONB,
                file_path VARCHAR(500),
                file_size BIGINT,
                record_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # 模型表
            """
            CREATE TABLE IF NOT EXISTS models (
                id SERIAL PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                type VARCHAR(50) NOT NULL,
                version VARCHAR(50),
                description TEXT,
                model_path VARCHAR(500),
                parameters JSONB,
                metrics JSONB,
                training_dataset_id INTEGER REFERENCES datasets(id),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # 预测任务表
            """
            CREATE TABLE IF NOT EXISTS prediction_tasks (
                id SERIAL PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                model_id INTEGER REFERENCES models(id),
                input_data JSONB,
                output_data JSONB,
                status VARCHAR(50) DEFAULT 'pending',
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_by INTEGER REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # 系统日志表
            """
            CREATE TABLE IF NOT EXISTS system_logs (
                id SERIAL PRIMARY KEY,
                level VARCHAR(20) NOT NULL,
                message TEXT NOT NULL,
                module VARCHAR(100),
                function_name VARCHAR(100),
                user_id INTEGER REFERENCES users(id),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        # 创建索引
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_weather_stations_location ON weather_stations(latitude, longitude)",
            "CREATE INDEX IF NOT EXISTS idx_datasets_source ON datasets(data_source_id)",
            "CREATE INDEX IF NOT EXISTS idx_datasets_dates ON datasets(start_date, end_date)",
            "CREATE INDEX IF NOT EXISTS idx_models_type ON models(type)",
            "CREATE INDEX IF NOT EXISTS idx_prediction_tasks_status ON prediction_tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_prediction_tasks_created ON prediction_tasks(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level)",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_created ON system_logs(created_at)"
        ]
        
        async with self.engine.begin() as conn:
            # 创建表
            for table_sql in tables_sql:
                await conn.execute(text(table_sql))
                logger.debug("表创建成功")
            
            # 创建索引
            for index_sql in indexes_sql:
                await conn.execute(text(index_sql))
                logger.debug("索引创建成功")
        
        logger.info("所有数据表创建完成")
    
    async def _insert_initial_data(self):
        """插入初始数据"""
        logger.info("插入初始数据")
        
        # 初始数据源
        data_sources = [
            {
                'name': 'NOAA',
                'type': 'weather_api',
                'url': 'https://www.ncei.noaa.gov/cdo-web/api/v2',
                'api_key_required': True,
                'description': 'National Oceanic and Atmospheric Administration'
            },
            {
                'name': 'ECMWF',
                'type': 'reanalysis_api',
                'url': 'https://cds.climate.copernicus.eu/api/v2',
                'api_key_required': True,
                'description': 'European Centre for Medium-Range Weather Forecasts'
            },
            {
                'name': 'Satellite',
                'type': 'satellite_data',
                'url': 'https://earthdata.nasa.gov',
                'api_key_required': True,
                'description': 'NASA Earth Science Data'
            }
        ]
        
        # 默认管理员用户
        admin_user = {
            'username': 'admin',
            'email': 'admin@climate-system.com',
            'password_hash': '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6hsxq5/Caa',  # 密码: admin123
            'is_admin': True
        }
        
        async with self.engine.begin() as conn:
            # 插入数据源
            for source in data_sources:
                await conn.execute(
                    text("""
                        INSERT INTO data_sources (name, type, url, api_key_required, description)
                        VALUES (:name, :type, :url, :api_key_required, :description)
                        ON CONFLICT (name) DO NOTHING
                    """),
                    source
                )
            
            # 插入管理员用户
            await conn.execute(
                text("""
                    INSERT INTO users (username, email, password_hash, is_admin)
                    VALUES (:username, :email, :password_hash, :is_admin)
                    ON CONFLICT (username) DO NOTHING
                """),
                admin_user
            )
        
        logger.info("初始数据插入完成")


class InfluxDBInitializer:
    """InfluxDB数据库初始化器"""
    
    def __init__(self):
        self.settings = settings
        self.client = None
    
    def initialize(self):
        """初始化InfluxDB数据库"""
        logger.info("开始初始化InfluxDB数据库")
        
        try:
            # 创建客户端
            self.client = influxdb_client.InfluxDBClient(
                url=self.settings.INFLUXDB_URL,
                token=self.settings.INFLUXDB_TOKEN,
                org=self.settings.INFLUXDB_ORG
            )
            
            # 检查连接
            self._check_connection()
            
            # 创建存储桶
            self._create_buckets()
            
            # 创建数据保留策略
            self._create_retention_policies()
            
            logger.info("InfluxDB数据库初始化完成")
            
        except Exception as e:
            logger.error(f"InfluxDB数据库初始化失败: {e}")
            raise
        finally:
            if self.client:
                self.client.close()
    
    def _check_connection(self):
        """检查InfluxDB连接"""
        try:
            health = self.client.health()
            if health.status == "pass":
                logger.info("InfluxDB连接正常")
            else:
                raise Exception(f"InfluxDB健康检查失败: {health.message}")
        except Exception as e:
            logger.error(f"InfluxDB连接失败: {e}")
            raise
    
    def _create_buckets(self):
        """创建存储桶"""
        logger.info("创建InfluxDB存储桶")
        
        buckets_api = self.client.buckets_api()
        org_api = self.client.organizations_api()
        
        # 获取组织ID
        try:
            orgs = org_api.find_organizations(org=self.settings.INFLUXDB_ORG)
            if not orgs:
                raise Exception(f"组织 {self.settings.INFLUXDB_ORG} 不存在")
            org_id = orgs[0].id
        except Exception as e:
            logger.error(f"获取组织ID失败: {e}")
            raise
        
        # 定义存储桶
        buckets_to_create = [
            {
                'name': 'climate-data',
                'description': '气候数据存储桶',
                'retention_rules': [{'type': 'expire', 'everySeconds': 31536000}]  # 1年
            },
            {
                'name': 'weather-stations',
                'description': '气象站数据存储桶',
                'retention_rules': [{'type': 'expire', 'everySeconds': 63072000}]  # 2年
            },
            {
                'name': 'satellite-data',
                'description': '卫星数据存储桶',
                'retention_rules': [{'type': 'expire', 'everySeconds': 15768000}]  # 6个月
            },
            {
                'name': 'model-predictions',
                'description': '模型预测结果存储桶',
                'retention_rules': [{'type': 'expire', 'everySeconds': 7776000}]  # 3个月
            },
            {
                'name': 'system-metrics',
                'description': '系统监控指标存储桶',
                'retention_rules': [{'type': 'expire', 'everySeconds': 2592000}]  # 1个月
            }
        ]
        
        for bucket_config in buckets_to_create:
            try:
                # 检查存储桶是否已存在
                existing_bucket = buckets_api.find_bucket_by_name(bucket_config['name'])
                
                if existing_bucket:
                    logger.info(f"存储桶 {bucket_config['name']} 已存在")
                else:
                    # 创建存储桶
                    bucket = buckets_api.create_bucket(
                        bucket_name=bucket_config['name'],
                        org_id=org_id,
                        description=bucket_config['description'],
                        retention_rules=bucket_config['retention_rules']
                    )
                    logger.info(f"存储桶 {bucket_config['name']} 创建成功")
                    
            except Exception as e:
                logger.error(f"创建存储桶 {bucket_config['name']} 失败: {e}")
                continue
    
    def _create_retention_policies(self):
        """创建数据保留策略"""
        logger.info("配置数据保留策略")
        
        # 保留策略已在创建存储桶时设置
        logger.info("数据保留策略配置完成")


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.postgresql_init = PostgreSQLInitializer()
        self.influxdb_init = InfluxDBInitializer()
    
    async def initialize_all(self):
        """初始化所有数据库"""
        logger.info("开始初始化所有数据库")
        
        try:
            # 初始化PostgreSQL
            await self.postgresql_init.initialize()
            
            # 初始化InfluxDB
            if settings.INFLUXDB_TOKEN:
                self.influxdb_init.initialize()
            else:
                logger.warning("InfluxDB令牌未配置，跳过InfluxDB初始化")
            
            logger.info("所有数据库初始化完成")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def check_status(self):
        """检查数据库状态"""
        logger.info("检查数据库状态")
        
        status = {
            'postgresql': False,
            'influxdb': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # 检查PostgreSQL
        try:
            engine = create_engine(settings.postgres_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            status['postgresql'] = True
            logger.info("PostgreSQL连接正常")
        except Exception as e:
            logger.error(f"PostgreSQL连接失败: {e}")
        
        # 检查InfluxDB
        if settings.INFLUXDB_TOKEN:
            try:
                client = influxdb_client.InfluxDBClient(
                    url=settings.INFLUXDB_URL,
                    token=settings.INFLUXDB_TOKEN,
                    org=settings.INFLUXDB_ORG
                )
                health = client.health()
                if health.status == "pass":
                    status['influxdb'] = True
                    logger.info("InfluxDB连接正常")
                client.close()
            except Exception as e:
                logger.error(f"InfluxDB连接失败: {e}")
        
        return status


async def main():
    """主函数"""
    logger.info("数据库初始化脚本启动")
    
    # 创建数据库管理器
    manager = DatabaseManager()
    
    try:
        # 初始化所有数据库
        await manager.initialize_all()
        
        # 检查状态
        status = manager.check_status()
        logger.info(f"数据库状态: {status}")
        
        logger.info("数据库初始化脚本完成")
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())