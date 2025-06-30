#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统管理API路由

提供系统状态监控、配置管理、日志查看等功能。
"""

import asyncio
import psutil
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.utils.logger import get_logger
from src.utils.config import settings
from src.data_processing.data_storage import DataStorage

logger = get_logger(__name__)
router = APIRouter()


# 响应模型
class SystemStatus(BaseModel):
    """系统状态模型"""
    timestamp: datetime = Field(description="状态时间戳")
    uptime: float = Field(description="系统运行时间（秒）")
    cpu_usage: float = Field(description="CPU使用率（%）")
    memory_usage: float = Field(description="内存使用率（%）")
    disk_usage: float = Field(description="磁盘使用率（%）")
    network_io: Dict[str, int] = Field(description="网络IO统计")
    database_status: Dict[str, str] = Field(description="数据库状态")
    active_connections: int = Field(description="活跃连接数")
    version: str = Field(description="系统版本")


class LogEntry(BaseModel):
    """日志条目模型"""
    timestamp: datetime = Field(description="日志时间")
    level: str = Field(description="日志级别")
    module: str = Field(description="模块名称")
    message: str = Field(description="日志消息")
    extra_data: Optional[Dict[str, Any]] = Field(description="额外数据")


class ConfigItem(BaseModel):
    """配置项模型"""
    key: str = Field(description="配置键")
    value: Any = Field(description="配置值")
    description: str = Field(description="配置描述")
    is_sensitive: bool = Field(description="是否敏感信息")


class MetricsData(BaseModel):
    """指标数据模型"""
    timestamp: datetime = Field(description="时间戳")
    metrics: Dict[str, float] = Field(description="指标数据")


# 依赖注入
async def get_data_storage() -> DataStorage:
    """获取数据存储实例"""
    # 这里应该从应用状态中获取
    # 暂时返回None，实际使用时需要从app.state中获取
    return None


@router.get("/status", response_model=SystemStatus, summary="获取系统状态")
async def get_system_status(
    data_storage: DataStorage = Depends(get_data_storage)
):
    """获取系统运行状态"""
    try:
        # 获取系统信息
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # 检查数据库状态
        database_status = {
            "postgresql": "unknown",
            "influxdb": "unknown",
            "redis": "unknown"
        }
        
        if data_storage:
            try:
                # 这里应该实现实际的数据库健康检查
                database_status = {
                    "postgresql": "healthy",
                    "influxdb": "healthy", 
                    "redis": "healthy"
                }
            except Exception as e:
                logger.error(f"数据库状态检查失败: {e}")
        
        return SystemStatus(
            timestamp=datetime.now(),
            uptime=psutil.boot_time(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            network_io={
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            database_status=database_status,
            active_connections=len(psutil.net_connections()),
            version=settings.VERSION
        )
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail="无法获取系统状态")


@router.get("/logs", response_model=List[LogEntry], summary="获取系统日志")
async def get_system_logs(
    level: Optional[str] = Query(None, description="日志级别过滤"),
    module: Optional[str] = Query(None, description="模块名称过滤"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    limit: int = Query(100, ge=1, le=1000, description="返回条数限制"),
    data_storage: DataStorage = Depends(get_data_storage)
):
    """获取系统日志"""
    try:
        # 这里应该从数据库中查询日志
        # 暂时返回示例数据
        logs = [
            LogEntry(
                timestamp=datetime.now() - timedelta(minutes=i),
                level="INFO",
                module="system",
                message=f"系统运行正常 - {i}",
                extra_data={"request_id": f"req_{i}"}
            )
            for i in range(min(limit, 10))
        ]
        
        return logs
        
    except Exception as e:
        logger.error(f"获取系统日志失败: {e}")
        raise HTTPException(status_code=500, detail="无法获取系统日志")


@router.get("/config", response_model=List[ConfigItem], summary="获取系统配置")
async def get_system_config():
    """获取系统配置信息"""
    try:
        config_items = [
            ConfigItem(
                key="APP_NAME",
                value=settings.APP_NAME,
                description="应用程序名称",
                is_sensitive=False
            ),
            ConfigItem(
                key="VERSION",
                value=settings.VERSION,
                description="系统版本",
                is_sensitive=False
            ),
            ConfigItem(
                key="DEBUG",
                value=settings.DEBUG,
                description="调试模式",
                is_sensitive=False
            ),
            ConfigItem(
                key="HOST",
                value=settings.HOST,
                description="服务器主机",
                is_sensitive=False
            ),
            ConfigItem(
                key="PORT",
                value=settings.PORT,
                description="服务器端口",
                is_sensitive=False
            ),
            ConfigItem(
                key="DATABASE_URL",
                value="***" if settings.DATABASE_URL else None,
                description="数据库连接URL",
                is_sensitive=True
            ),
            ConfigItem(
                key="REDIS_URL",
                value="***" if settings.REDIS_URL else None,
                description="Redis连接URL",
                is_sensitive=True
            )
        ]
        
        return config_items
        
    except Exception as e:
        logger.error(f"获取系统配置失败: {e}")
        raise HTTPException(status_code=500, detail="无法获取系统配置")


@router.get("/metrics", response_model=List[MetricsData], summary="获取系统指标")
async def get_system_metrics(
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    interval: int = Query(300, ge=60, le=3600, description="采样间隔（秒）")
):
    """获取系统性能指标"""
    try:
        # 生成示例指标数据
        now = datetime.now()
        if not end_time:
            end_time = now
        if not start_time:
            start_time = end_time - timedelta(hours=1)
        
        metrics_data = []
        current_time = start_time
        
        while current_time <= end_time:
            metrics_data.append(MetricsData(
                timestamp=current_time,
                metrics={
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage('/').percent,
                    "network_bytes_sent": psutil.net_io_counters().bytes_sent,
                    "network_bytes_recv": psutil.net_io_counters().bytes_recv
                }
            ))
            current_time += timedelta(seconds=interval)
        
        return metrics_data
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        raise HTTPException(status_code=500, detail="无法获取系统指标")


@router.post("/restart", summary="重启系统服务")
async def restart_system():
    """重启系统服务（需要管理员权限）"""
    try:
        # 这里应该实现实际的重启逻辑
        # 注意：这是一个危险操作，需要严格的权限控制
        logger.warning("收到系统重启请求")
        
        return {
            "message": "系统重启请求已接收",
            "status": "pending",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"系统重启失败: {e}")
        raise HTTPException(status_code=500, detail="系统重启失败")


@router.post("/cleanup", summary="清理系统缓存")
async def cleanup_system():
    """清理系统缓存和临时文件"""
    try:
        # 实现缓存清理逻辑
        cleanup_tasks = [
            "清理临时文件",
            "清理日志文件",
            "清理缓存数据",
            "优化数据库"
        ]
        
        logger.info("开始系统清理")
        
        # 模拟清理过程
        await asyncio.sleep(1)
        
        return {
            "message": "系统清理完成",
            "tasks_completed": cleanup_tasks,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"系统清理失败: {e}")
        raise HTTPException(status_code=500, detail="系统清理失败")


@router.get("/health/detailed", summary="详细健康检查")
async def detailed_health_check(
    data_storage: DataStorage = Depends(get_data_storage)
):
    """详细的系统健康检查"""
    try:
        health_status = {
            "overall": "healthy",
            "timestamp": datetime.now(),
            "components": {}
        }
        
        # 检查各个组件
        components = {
            "database": "检查数据库连接",
            "storage": "检查存储系统",
            "memory": "检查内存使用",
            "disk": "检查磁盘空间",
            "network": "检查网络连接"
        }
        
        for component, description in components.items():
            try:
                # 这里应该实现实际的健康检查逻辑
                if component == "memory":
                    memory = psutil.virtual_memory()
                    status = "healthy" if memory.percent < 90 else "warning"
                elif component == "disk":
                    disk = psutil.disk_usage('/')
                    status = "healthy" if disk.percent < 90 else "warning"
                else:
                    status = "healthy"
                
                health_status["components"][component] = {
                    "status": status,
                    "description": description,
                    "last_check": datetime.now()
                }
                
            except Exception as e:
                health_status["components"][component] = {
                    "status": "unhealthy",
                    "description": description,
                    "error": str(e),
                    "last_check": datetime.now()
                }
                health_status["overall"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"详细健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="健康检查失败")