#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理API路由

提供气候数据的上传、下载、查询、管理等功能。
"""

import asyncio
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field

from src.utils.logger import get_logger
from src.data_processing.data_collector import DataCollector
from src.data_processing.data_storage import DataStorage
from src.api.routes.auth import get_current_active_user

logger = get_logger(__name__)
router = APIRouter()


# 请求模型
class DataSourceConfig(BaseModel):
    """数据源配置"""
    name: str = Field(..., description="数据源名称")
    type: str = Field(..., description="数据源类型")
    api_endpoint: Optional[str] = Field(None, description="API端点")
    api_key: Optional[str] = Field(None, description="API密钥")
    config: Dict[str, Any] = Field(default_factory=dict, description="额外配置")


class DataCollectionRequest(BaseModel):
    """数据收集请求"""
    data_source: str = Field(..., description="数据源类型")
    start_date: date = Field(..., description="开始日期")
    end_date: date = Field(..., description="结束日期")
    variables: List[str] = Field(..., description="变量列表")
    location: Optional[Dict[str, float]] = Field(None, description="位置信息")
    stations: Optional[List[str]] = Field(None, description="气象站列表")


class DataQueryRequest(BaseModel):
    """数据查询请求"""
    dataset_id: Optional[str] = Field(None, description="数据集ID")
    data_type: Optional[str] = Field(None, description="数据类型")
    start_date: Optional[date] = Field(None, description="开始日期")
    end_date: Optional[date] = Field(None, description="结束日期")
    variables: Optional[List[str]] = Field(None, description="变量列表")
    location: Optional[Dict[str, float]] = Field(None, description="位置信息")
    limit: int = Field(100, ge=1, le=10000, description="返回条数限制")


# 响应模型
class DataSourceInfo(BaseModel):
    """数据源信息"""
    id: str = Field(description="数据源ID")
    name: str = Field(description="数据源名称")
    type: str = Field(description="数据源类型")
    description: Optional[str] = Field(description="描述")
    is_active: bool = Field(description="是否激活")
    created_at: datetime = Field(description="创建时间")
    last_updated: Optional[datetime] = Field(description="最后更新时间")


class DatasetInfo(BaseModel):
    """数据集信息"""
    id: str = Field(description="数据集ID")
    name: str = Field(description="数据集名称")
    description: Optional[str] = Field(description="描述")
    data_source: str = Field(description="数据源")
    dataset_type: str = Field(description="数据集类型")
    variables: List[str] = Field(description="变量列表")
    temporal_resolution: Optional[str] = Field(description="时间分辨率")
    spatial_resolution: Optional[str] = Field(description="空间分辨率")
    start_date: Optional[date] = Field(description="开始日期")
    end_date: Optional[date] = Field(description="结束日期")
    file_size: Optional[int] = Field(description="文件大小")
    record_count: Optional[int] = Field(description="记录数量")
    is_processed: bool = Field(description="是否已处理")
    created_at: datetime = Field(description="创建时间")


class CollectionTask(BaseModel):
    """数据收集任务"""
    task_id: str = Field(description="任务ID")
    status: str = Field(description="任务状态")
    progress: int = Field(description="进度百分比")
    message: str = Field(description="状态消息")
    created_at: datetime = Field(description="创建时间")
    started_at: Optional[datetime] = Field(description="开始时间")
    completed_at: Optional[datetime] = Field(description="完成时间")
    error_message: Optional[str] = Field(description="错误消息")


class DataRecord(BaseModel):
    """数据记录"""
    timestamp: datetime = Field(description="时间戳")
    location: Optional[Dict[str, float]] = Field(description="位置信息")
    variables: Dict[str, Any] = Field(description="变量数据")
    metadata: Optional[Dict[str, Any]] = Field(description="元数据")


# 依赖注入
async def get_data_collector() -> DataCollector:
    """获取数据收集器实例"""
    return DataCollector()


async def get_data_storage() -> DataStorage:
    """获取数据存储实例"""
    storage = DataStorage()
    if not storage.is_initialized():
        await storage.initialize()
    return storage


# API路由
@router.get("/sources", response_model=List[DataSourceInfo], summary="获取数据源列表")
async def get_data_sources(
    source_type: Optional[str] = Query(None, description="数据源类型过滤"),
    is_active: Optional[bool] = Query(None, description="是否激活过滤"),
    current_user: dict = Depends(get_current_active_user),
    data_storage: DataStorage = Depends(get_data_storage)
):
    """获取可用的数据源列表"""
    try:
        sources = await data_storage.get_data_sources(
            source_type=source_type,
            is_active=is_active
        )
        return sources
    except Exception as e:
        logger.error(f"获取数据源列表失败: {e}")
        raise HTTPException(status_code=500, detail="无法获取数据源列表")


@router.post("/sources", response_model=DataSourceInfo, summary="添加数据源")
async def add_data_source(
    source_config: DataSourceConfig,
    current_user: dict = Depends(get_current_active_user)
):
    """添加新的数据源"""
    try:
        # 这里应该保存到数据库
        source_info = DataSourceInfo(
            id=f"{source_config.type}-{datetime.now().timestamp()}",
            name=source_config.name,
            type=source_config.type,
            description=f"用户添加的{source_config.type}数据源",
            is_active=True,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        logger.info(f"添加数据源: {source_config.name} by {current_user['username']}")
        
        return source_info
        
    except Exception as e:
        logger.error(f"添加数据源失败: {e}")
        raise HTTPException(status_code=500, detail="添加数据源失败")


@router.get("/datasets", response_model=List[DatasetInfo], summary="获取数据集列表")
async def get_datasets(
    dataset_type: Optional[str] = Query(None, description="数据集类型过滤"),
    data_source: Optional[str] = Query(None, description="数据源过滤"),
    start_date: Optional[date] = Query(None, description="开始日期过滤"),
    end_date: Optional[date] = Query(None, description="结束日期过滤"),
    limit: int = Query(100, ge=1, le=1000, description="返回条数限制"),
    current_user: dict = Depends(get_current_active_user),
    data_storage: DataStorage = Depends(get_data_storage)
):
    """获取数据集列表"""
    try:
        datasets = await data_storage.get_datasets(
            dataset_type=dataset_type,
            data_source=data_source,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        return datasets
    except Exception as e:
        logger.error(f"获取数据集列表失败: {e}")
        raise HTTPException(status_code=500, detail="无法获取数据集列表")


@router.post("/collect", response_model=CollectionTask, summary="启动数据收集任务")
async def collect_data(
    request: DataCollectionRequest,
    background_tasks: BackgroundTasks,
    data_collector: DataCollector = Depends(get_data_collector),
    data_storage: DataStorage = Depends(get_data_storage),
    current_user: dict = Depends(get_current_active_user)
):
    """启动一个异步的数据收集任务"""
    try:
        task_id = await data_storage.create_task(
            task_type="data_collection",
            name=f"Collect {request.data_source} data",
            params=request.dict()
        )

        async def collection_task():
            try:
                await data_storage.update_task_status(task_id, "running")
                if request.data_source == "noaa":
                    await data_collector.collect_noaa_data(
                        start_date=request.start_date,
                        end_date=request.end_date,
                        stations=request.stations,
                        data_types=request.variables
                    )
                elif request.data_source == "ecmwf":
                    # ... ecmwf collection logic
                    pass
                await data_storage.update_task_status(task_id, "completed")
            except Exception as e:
                logger.error(f"Data collection task {task_id} failed: {e}")
                await data_storage.update_task_status(task_id, "failed", error_message=str(e))

        background_tasks.add_task(collection_task)

        logger.info(f"User {current_user['username']} started data collection task: {task_id}")

        return CollectionTask(
            task_id=task_id,
            status="pending",
            progress=0,
            message="任务已提交",
            created_at=datetime.now()
        )

    except Exception as e:
        logger.error(f"启动数据收集任务失败: {e}")
        raise HTTPException(status_code=500, detail="无法启动数据收集任务")


@router.get("/tasks/{task_id}", response_model=CollectionTask, summary="获取任务状态")
async def get_task_status(
    task_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """获取数据收集任务状态"""
    try:
        # 这里应该从数据库或缓存中查询任务状态
        task = CollectionTask(
            task_id=task_id,
            status="running",
            progress=50,
            message="正在收集数据...",
            created_at=datetime.now(),
            started_at=datetime.now()
        )
        
        return task
        
    except Exception as e:
        logger.error(f"获取任务状态失败: {e}")
        raise HTTPException(status_code=500, detail="无法获取任务状态")


@router.post("/query", response_model=List[DataRecord], summary="查询数据")
async def query_data(
    query_request: DataQueryRequest,
    current_user: dict = Depends(get_current_active_user),
    data_storage: DataStorage = Depends(get_data_storage)
):
    """查询气候数据"""
    try:
        # 这里应该实现实际的数据查询逻辑
        records = [
            DataRecord(
                timestamp=datetime.now(),
                location={"latitude": 40.7128, "longitude": -74.0060},
                variables={
                    "temperature": 25.5,
                    "humidity": 60.0,
                    "precipitation": 0.0
                },
                metadata={"station_id": "NYC001", "quality": "good"}
            )
        ]
        
        return records[:query_request.limit]
        
    except Exception as e:
        logger.error(f"数据查询失败: {e}")
        raise HTTPException(status_code=500, detail="数据查询失败")


@router.post("/upload", summary="上传数据文件")
async def upload_data_file(
    file: UploadFile = File(...),
    dataset_name: str = Query(..., description="数据集名称"),
    description: Optional[str] = Query(None, description="描述"),
    current_user: dict = Depends(get_current_active_user)
):
    """上传气候数据文件"""
    try:
        # 检查文件类型
        allowed_types = [".csv", ".nc", ".json", ".xlsx"]
        file_suffix = Path(file.filename).suffix.lower()
        
        if file_suffix not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file_suffix}"
            )
        
        # 保存文件
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{datetime.now().timestamp()}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"文件上传成功: {file.filename} by {current_user['username']}")
        
        return {
            "message": "文件上传成功",
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": len(content),
            "dataset_name": dataset_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail="文件上传失败")


@router.delete("/datasets/{dataset_id}", summary="删除数据集")
async def delete_dataset(
    dataset_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """删除数据集"""
    try:
        # 这里应该实现实际的删除逻辑
        # 1. 检查用户权限
        # 2. 删除数据库记录
        # 3. 删除文件
        
        logger.info(f"删除数据集: {dataset_id} by {current_user['username']}")
        
        return {"message": "数据集删除成功"}
        
    except Exception as e:
        logger.error(f"删除数据集失败: {e}")
        raise HTTPException(status_code=500, detail="删除数据集失败")


# 后台任务函数
async def run_data_collection(
    task_id: str,
    collection_request: DataCollectionRequest,
    data_collector: DataCollector
):
    """运行数据收集任务"""
    try:
        logger.info(f"开始执行数据收集任务: {task_id}")
        
        # 更新任务状态
        # 这里应该更新数据库中的任务状态
        
        # 执行数据收集
        if collection_request.data_source == "noaa":
            await data_collector.collect_noaa_data(
                start_date=collection_request.start_date,
                end_date=collection_request.end_date,
                variables=collection_request.variables,
                stations=collection_request.stations
            )
        elif collection_request.data_source == "ecmwf":
            await data_collector.collect_reanalysis_data(
                start_date=collection_request.start_date,
                end_date=collection_request.end_date,
                variables=collection_request.variables,
                location=collection_request.location
            )
        
        logger.info(f"数据收集任务完成: {task_id}")
        
    except Exception as e:
        logger.error(f"数据收集任务失败: {task_id} - {e}")
        # 更新任务状态为失败