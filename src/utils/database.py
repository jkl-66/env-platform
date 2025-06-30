"""数据库连接管理模块

管理PostgreSQL、InfluxDB、Redis等数据库的连接。
"""

import asyncio
from typing import Optional
import asyncpg
import aioredis
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

from .config import get_settings
from .logger import get_logger

logger = get_logger("database")
settings = get_settings()

# 全局数据库连接实例
postgres_engine = None
redis_client = None
influxdb_client = None
async_session_maker = None


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.postgres_engine = None
        self.redis_client = None
        self.influxdb_client = None
        self.async_session_maker = None
    
    async def init_postgres(self):
        """初始化PostgreSQL连接"""
        try:
            # 创建异步引擎
            database_url = settings.postgres_url.replace("postgresql://", "postgresql+asyncpg://")
            self.postgres_engine = create_async_engine(
                database_url,
                echo=settings.DEBUG,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # 创建会话工厂
            self.async_session_maker = sessionmaker(
                self.postgres_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # 测试连接
            async with self.postgres_engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            logger.info("PostgreSQL连接初始化成功")
            
        except Exception as e:
            logger.error(f"PostgreSQL连接初始化失败: {e}")
            raise
    
    async def init_redis(self):
        """初始化Redis连接"""
        try:
            self.redis_client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            
            # 测试连接
            await self.redis_client.ping()
            logger.info("Redis连接初始化成功")
            
        except Exception as e:
            logger.error(f"Redis连接初始化失败: {e}")
            raise
    
    async def init_influxdb(self):
        """初始化InfluxDB连接"""
        try:
            self.influxdb_client = InfluxDBClientAsync(
                url=settings.INFLUXDB_URL,
                token=settings.INFLUXDB_TOKEN,
                org=settings.INFLUXDB_ORG
            )
            
            # 测试连接
            health = await self.influxdb_client.health()
            if health.status == "pass":
                logger.info("InfluxDB连接初始化成功")
            else:
                raise Exception(f"InfluxDB健康检查失败: {health.status}")
                
        except Exception as e:
            logger.error(f"InfluxDB连接初始化失败: {e}")
            # InfluxDB可选，不抛出异常
            self.influxdb_client = None
    
    async def close_all(self):
        """关闭所有数据库连接"""
        if self.postgres_engine:
            await self.postgres_engine.dispose()
            logger.info("PostgreSQL连接已关闭")
        
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis连接已关闭")
        
        if self.influxdb_client:
            await self.influxdb_client.close()
            logger.info("InfluxDB连接已关闭")


# 全局数据库管理器实例
db_manager = DatabaseManager()


async def init_databases():
    """初始化所有数据库连接"""
    global postgres_engine, redis_client, influxdb_client, async_session_maker
    
    logger.info("开始初始化数据库连接...")
    
    # 初始化PostgreSQL
    await db_manager.init_postgres()
    postgres_engine = db_manager.postgres_engine
    async_session_maker = db_manager.async_session_maker
    
    # 初始化Redis
    await db_manager.init_redis()
    redis_client = db_manager.redis_client
    
    # 初始化InfluxDB (可选)
    await db_manager.init_influxdb()
    influxdb_client = db_manager.influxdb_client
    
    logger.info("所有数据库连接初始化完成")


async def close_databases():
    """关闭所有数据库连接"""
    global postgres_engine, redis_client, influxdb_client, async_session_maker
    
    await db_manager.close_all()
    
    postgres_engine = None
    redis_client = None
    influxdb_client = None
    async_session_maker = None


@asynccontextmanager
async def get_postgres_session():
    """获取PostgreSQL会话"""
    if not async_session_maker:
        raise RuntimeError("PostgreSQL未初始化")
    
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_redis() -> aioredis.Redis:
    """获取Redis客户端"""
    if not redis_client:
        raise RuntimeError("Redis未初始化")
    return redis_client


async def get_influxdb() -> Optional[InfluxDBClientAsync]:
    """获取InfluxDB客户端"""
    return influxdb_client


# 缓存装饰器
def cache_result(key_prefix: str, ttl: int = None):
    """Redis缓存装饰器
    
    Args:
        key_prefix: 缓存键前缀
        ttl: 过期时间(秒)，默认使用配置中的CACHE_TTL
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not redis_client:
                return await func(*args, **kwargs)
            
            # 生成缓存键
            cache_key = f"{key_prefix}:{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            try:
                cached_result = await redis_client.get(cache_key)
                if cached_result:
                    import json
                    return json.loads(cached_result)
            except Exception as e:
                logger.warning(f"缓存读取失败: {e}")
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 存储到缓存
            try:
                import json
                cache_ttl = ttl or settings.CACHE_TTL
                await redis_client.setex(
                    cache_key,
                    cache_ttl,
                    json.dumps(result, default=str)
                )
            except Exception as e:
                logger.warning(f"缓存写入失败: {e}")
            
            return result
        return wrapper
    return decorator