"""气候数据分析API

提供气候数据查询、分析和处理的API接口。
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Query, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
import pandas as pd
import numpy as np
from datetime import datetime, date
import asyncio
import io
import json
import tempfile
import os

from .models import (
    ClimateDataQuery, ClimateDataResponse, ClimateDataRecord,
    ClimateAnalysisRequest, ClimateAnalysisResponse, ClimateAnalysisResult,
    BaseResponse, ResponseStatus, ErrorResponse, FileUploadResponse,
    DataSource, DataType, AnalysisType
)
from .dependencies import (
    DBSession, RedisClient, InfluxDBClient, CurrentUser, AuthenticatedUser,
    PaginationDep, CommonQueryDep, NormalRateLimit, validate_date_range,
    validate_coordinates, validate_file_upload
)
from ..data_processing import DataCollector, DataCleaner, DataStorage, DataPipeline
from ..models import ClimateAnalysisModel
from ..utils.logger import logger
from ..utils.config import get_settings


router = APIRouter(prefix="/climate", tags=["气候数据"])


# ==================== 数据查询接口 ====================

@router.get("/data", response_model=ClimateDataResponse, summary="查询气候数据")
async def query_climate_data(
    source: Optional[DataSource] = Query(None, description="数据源"),
    data_type: Optional[DataType] = Query(None, description="数据类型"),
    location: Optional[str] = Query(None, description="位置描述"),
    variables: Optional[str] = Query(None, description="变量列表，逗号分隔"),
    format: str = Query("json", regex="^(json|csv|netcdf)$", description="返回格式"),
    pagination: PaginationDep = None,
    date_range: dict = validate_date_range,
    coordinates: dict = validate_coordinates,
    db: DBSession = None,
    redis: RedisClient = None,
    current_user: CurrentUser = None,
    rate_limit: NormalRateLimit = None
):
    """查询气候数据记录"""
    try:
        # 构建查询条件
        query_conditions = {}
        
        if source:
            query_conditions["source"] = source.value
        if data_type:
            query_conditions["data_type"] = data_type.value
        if location:
            query_conditions["location"] = location
        
        # 处理坐标范围查询
        if coordinates.get("latitude") and coordinates.get("longitude"):
            # 可以扩展为范围查询
            query_conditions["latitude"] = coordinates["latitude"]
            query_conditions["longitude"] = coordinates["longitude"]
        
        # 处理时间范围
        if date_range.get("start_date"):
            query_conditions["start_date"] = date_range["start_date"]
        if date_range.get("end_date"):
            query_conditions["end_date"] = date_range["end_date"]
        
        # 处理变量列表
        if variables:
            query_conditions["variables"] = variables.split(",")
        
        # 使用数据存储类查询
        data_storage = DataStorage()
        
        # 查询数据记录
        records, total = await data_storage.search_data_records(
            conditions=query_conditions,
            offset=pagination.offset,
            limit=pagination.size
        )
        
        # 转换为响应模型
        data_records = [
            ClimateDataRecord(
                id=record["id"],
                source=record["source"],
                data_type=record["data_type"],
                location=record.get("location"),
                latitude=record.get("latitude"),
                longitude=record.get("longitude"),
                start_time=record.get("start_time"),
                end_time=record.get("end_time"),
                file_path=record["file_path"],
                file_format=record["file_format"],
                file_size=record["file_size"],
                variables=record.get("variables"),
                metadata=record.get("metadata"),
                quality_score=record.get("quality_score"),
                created_at=record["created_at"],
                updated_at=record["updated_at"]
            )
            for record in records
        ]
        
        return ClimateDataResponse(
            status=ResponseStatus.SUCCESS,
            message="查询成功",
            data=data_records,
            total=total,
            page=pagination.page,
            size=pagination.size
        )
        
    except Exception as e:
        logger.error(f"查询气候数据失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询失败: {str(e)}"
        )


@router.get("/data/{record_id}", response_model=Dict[str, Any], summary="获取具体数据内容")
async def get_climate_data_content(
    record_id: str,
    format: str = Query("json", regex="^(json|csv|netcdf)$", description="返回格式"),
    variables: Optional[str] = Query(None, description="指定变量，逗号分隔"),
    db: DBSession = None,
    current_user: CurrentUser = None,
    rate_limit: NormalRateLimit = None
):
    """获取具体的气候数据内容"""
    try:
        data_storage = DataStorage()
        
        # 获取数据记录信息
        record = await data_storage.get_data_record(record_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="数据记录不存在"
            )
        
        # 读取数据文件
        file_path = record["file_path"]
        file_format = record["file_format"]
        
        if file_format == "dataframe":
            # 读取DataFrame
            df = await data_storage.load_dataframe(file_path)
            
            # 过滤指定变量
            if variables:
                var_list = variables.split(",")
                available_vars = [var for var in var_list if var in df.columns]
                if available_vars:
                    df = df[available_vars]
            
            # 根据格式返回数据
            if format == "json":
                return {
                    "record_id": record_id,
                    "data": df.to_dict(orient="records"),
                    "columns": df.columns.tolist(),
                    "shape": df.shape,
                    "dtypes": df.dtypes.to_dict()
                }
            elif format == "csv":
                # 返回CSV格式
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_content = csv_buffer.getvalue()
                
                return StreamingResponse(
                    io.BytesIO(csv_content.encode()),
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={record_id}.csv"}
                )
        
        elif file_format == "xarray":
            # 读取xarray数据集
            ds = await data_storage.load_xarray(file_path)
            
            # 过滤指定变量
            if variables:
                var_list = variables.split(",")
                available_vars = [var for var in var_list if var in ds.data_vars]
                if available_vars:
                    ds = ds[available_vars]
            
            if format == "json":
                # 转换为JSON格式（简化版）
                data_dict = {}
                for var in ds.data_vars:
                    data_dict[var] = {
                        "data": ds[var].values.tolist(),
                        "dims": ds[var].dims,
                        "attrs": dict(ds[var].attrs)
                    }
                
                return {
                    "record_id": record_id,
                    "data_vars": data_dict,
                    "coords": {coord: ds.coords[coord].values.tolist() for coord in ds.coords},
                    "attrs": dict(ds.attrs)
                }
            elif format == "netcdf":
                # 返回NetCDF文件
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
                ds.to_netcdf(temp_file.name)
                
                return FileResponse(
                    temp_file.name,
                    media_type="application/x-netcdf",
                    filename=f"{record_id}.nc"
                )
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件格式: {file_format}"
            )
            
    except Exception as e:
        logger.error(f"获取数据内容失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取数据失败: {str(e)}"
        )


# ==================== 数据分析接口 ====================

@router.post("/analysis", response_model=ClimateAnalysisResponse, summary="执行气候数据分析")
async def analyze_climate_data(
    request: ClimateAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: DBSession = None,
    redis: RedisClient = None,
    current_user: AuthenticatedUser = None,
    rate_limit: NormalRateLimit = None
):
    """执行气候数据分析"""
    try:
        # 获取数据记录
        data_storage = DataStorage()
        record = await data_storage.get_data_record(request.data_record_id)
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="数据记录不存在"
            )
        
        # 加载数据
        file_path = record["file_path"]
        file_format = record["file_format"]
        
        if file_format == "dataframe":
            data = await data_storage.load_dataframe(file_path)
        elif file_format == "xarray":
            data = await data_storage.load_xarray(file_path)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的数据格式: {file_format}"
            )
        
        # 过滤时间范围
        if request.time_range:
            if isinstance(data, pd.DataFrame):
                if 'time' in data.columns:
                    mask = (
                        (data['time'] >= pd.to_datetime(request.time_range.start_date)) &
                        (data['time'] <= pd.to_datetime(request.time_range.end_date))
                    )
                    data = data[mask]
            # xarray时间过滤逻辑类似
        
        # 过滤变量
        if request.variables:
            if isinstance(data, pd.DataFrame):
                available_vars = [var for var in request.variables if var in data.columns]
                if available_vars:
                    # 保留时间列
                    cols_to_keep = ['time'] if 'time' in data.columns else []
                    cols_to_keep.extend(available_vars)
                    data = data[cols_to_keep]
        
        # 初始化分析模型
        analysis_model = ClimateAnalysisModel()
        
        # 执行分析
        start_time = datetime.utcnow()
        
        result_data = {
            "analysis_type": request.analysis_type,
            "data_record_id": request.data_record_id,
            "execution_time": 0.0
        }
        
        if request.analysis_type == AnalysisType.TREND:
            # 趋势分析
            trends = await analysis_model.analyze_trends(
                data, 
                variables=request.variables,
                **request.parameters or {}
            )
            result_data["trends"] = trends
            
        elif request.analysis_type == AnalysisType.ANOMALY:
            # 异常检测
            anomalies = await analysis_model.detect_anomalies(
                data,
                variables=request.variables,
                **request.parameters or {}
            )
            result_data["anomalies"] = anomalies
            
        elif request.analysis_type == AnalysisType.PATTERN:
            # 模式识别
            patterns = await analysis_model.identify_patterns(
                data,
                variables=request.variables,
                **request.parameters or {}
            )
            result_data["patterns"] = patterns
            
        elif request.analysis_type == AnalysisType.STATISTICS:
            # 统计分析
            statistics = await analysis_model.calculate_statistics(
                data,
                variables=request.variables,
                **request.parameters or {}
            )
            result_data["statistics"] = statistics
            
        elif request.analysis_type == AnalysisType.CORRELATION:
            # 相关性分析
            correlations = await analysis_model.calculate_correlations(
                data,
                variables=request.variables,
                **request.parameters or {}
            )
            result_data["correlations"] = correlations
        
        # 计算执行时间
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        result_data["execution_time"] = execution_time
        
        # 保存分析结果
        background_tasks.add_task(
            save_analysis_result,
            result_data,
            current_user.user_id,
            data_storage
        )
        
        # 构建响应
        analysis_result = ClimateAnalysisResult(**result_data)
        
        return ClimateAnalysisResponse(
            status=ResponseStatus.SUCCESS,
            message="分析完成",
            result=analysis_result
        )
        
    except Exception as e:
        logger.error(f"气候数据分析失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分析失败: {str(e)}"
        )


async def save_analysis_result(result_data: dict, user_id: str, data_storage: DataStorage):
    """保存分析结果（后台任务）"""
    try:
        await data_storage.save_model_result(
            model_type="climate_analysis",
            model_name=f"analysis_{result_data['analysis_type']}",
            result=result_data,
            metadata={
                "user_id": user_id,
                "analysis_type": result_data["analysis_type"],
                "data_record_id": result_data["data_record_id"]
            }
        )
        logger.info(f"分析结果已保存: {result_data['analysis_type']}")
    except Exception as e:
        logger.error(f"保存分析结果失败: {e}")


# ==================== 数据上传接口 ====================

@router.post("/upload", response_model=FileUploadResponse, summary="上传气候数据文件")
async def upload_climate_data(
    file: UploadFile = File(...),
    source: DataSource = Query(..., description="数据源"),
    data_type: DataType = Query(..., description="数据类型"),
    location: Optional[str] = Query(None, description="位置描述"),
    latitude: Optional[float] = Query(None, ge=-90, le=90, description="纬度"),
    longitude: Optional[float] = Query(None, ge=-180, le=180, description="经度"),
    description: Optional[str] = Query(None, description="数据描述"),
    background_tasks: BackgroundTasks = None,
    db: DBSession = None,
    current_user: AuthenticatedUser = None,
    file_validation: bool = validate_file_upload
):
    """上传气候数据文件"""
    try:
        settings = get_settings()
        
        # 验证文件类型
        allowed_types = [
            "text/csv", "application/json", "application/x-netcdf",
            "application/octet-stream"  # for .nc files
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件类型: {file.content_type}"
            )
        
        # 生成文件路径
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(settings.data_storage_path, "uploads", filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存文件
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # 记录文件信息
        file_info = {
            "filename": file.filename,
            "file_path": file_path,
            "file_size": len(content),
            "content_type": file.content_type,
            "source": source.value,
            "data_type": data_type.value,
            "location": location,
            "latitude": latitude,
            "longitude": longitude,
            "description": description,
            "uploaded_by": current_user.user_id
        }
        
        # 后台处理文件
        background_tasks.add_task(
            process_uploaded_file,
            file_info
        )
        
        return FileUploadResponse(
            status=ResponseStatus.SUCCESS,
            message="文件上传成功，正在后台处理",
            file_id=filename,
            filename=file.filename,
            file_path=file_path,
            file_size=len(content),
            content_type=file.content_type,
            upload_time=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传失败: {str(e)}"
        )


async def process_uploaded_file(file_info: dict):
    """处理上传的文件（后台任务）"""
    try:
        # 初始化数据处理组件
        data_cleaner = DataCleaner()
        data_storage = DataStorage()
        
        file_path = file_info["file_path"]
        
        # 根据文件类型读取数据
        if file_info["content_type"] == "text/csv":
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 数据清洗
            cleaned_df = await data_cleaner.clean_dataframe(df)
            
            # 保存清洗后的数据
            cleaned_path = file_path.replace(".csv", "_cleaned.parquet")
            await data_storage.save_dataframe(cleaned_df, cleaned_path)
            
            # 记录数据信息
            await data_storage.save_data_record(
                source=file_info["source"],
                data_type=file_info["data_type"],
                file_path=cleaned_path,
                file_format="dataframe",
                location=file_info.get("location"),
                latitude=file_info.get("latitude"),
                longitude=file_info.get("longitude"),
                variables=list(cleaned_df.columns),
                metadata={
                    "original_file": file_info["filename"],
                    "uploaded_by": file_info["uploaded_by"],
                    "description": file_info.get("description"),
                    "processing_time": datetime.utcnow().isoformat()
                }
            )
            
        elif file_info["content_type"] in ["application/x-netcdf", "application/octet-stream"]:
            # 处理NetCDF文件
            import xarray as xr
            
            ds = xr.open_dataset(file_path)
            
            # 数据清洗
            cleaned_ds = await data_cleaner.clean_xarray(ds)
            
            # 保存清洗后的数据
            cleaned_path = file_path.replace(".nc", "_cleaned.nc")
            await data_storage.save_xarray(cleaned_ds, cleaned_path)
            
            # 记录数据信息
            await data_storage.save_data_record(
                source=file_info["source"],
                data_type=file_info["data_type"],
                file_path=cleaned_path,
                file_format="xarray",
                location=file_info.get("location"),
                latitude=file_info.get("latitude"),
                longitude=file_info.get("longitude"),
                variables=list(cleaned_ds.data_vars.keys()),
                metadata={
                    "original_file": file_info["filename"],
                    "uploaded_by": file_info["uploaded_by"],
                    "description": file_info.get("description"),
                    "processing_time": datetime.utcnow().isoformat(),
                    "dimensions": dict(cleaned_ds.dims),
                    "coordinates": list(cleaned_ds.coords.keys())
                }
            )
        
        logger.info(f"文件处理完成: {file_info['filename']}")
        
    except Exception as e:
        logger.error(f"文件处理失败: {e}")


# ==================== 数据收集接口 ====================

@router.post("/collect", response_model=BaseResponse, summary="启动数据收集任务")
async def start_data_collection(
    sources: List[DataSource] = Query(..., description="数据源列表"),
    data_types: List[DataType] = Query(..., description="数据类型列表"),
    start_date: date = Query(..., description="开始日期"),
    end_date: date = Query(..., description="结束日期"),
    regions: Optional[List[str]] = Query(None, description="区域列表"),
    background_tasks: BackgroundTasks = None,
    current_user: AuthenticatedUser = None,
    rate_limit: NormalRateLimit = None
):
    """启动数据收集任务"""
    try:
        # 验证日期范围
        if end_date < start_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="结束日期不能早于开始日期"
            )
        
        # 创建数据收集任务
        collection_config = {
            "sources": [source.value for source in sources],
            "data_types": [dt.value for dt in data_types],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "regions": regions,
            "user_id": current_user.user_id
        }
        
        # 后台执行数据收集
        background_tasks.add_task(
            execute_data_collection,
            collection_config
        )
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message="数据收集任务已启动，将在后台执行"
        )
        
    except Exception as e:
        logger.error(f"启动数据收集失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动失败: {str(e)}"
        )


async def execute_data_collection(config: dict):
    """执行数据收集任务（后台任务）"""
    try:
        # 初始化数据收集器
        collector = DataCollector()
        data_storage = DataStorage()
        
        # 执行数据收集
        for source in config["sources"]:
            for data_type in config["data_types"]:
                try:
                    # 收集数据
                    if source == "noaa":
                        data = await collector.collect_noaa_data(
                            data_type=data_type,
                            start_date=config["start_date"],
                            end_date=config["end_date"],
                            regions=config.get("regions")
                        )
                    elif source == "ecmwf":
                        data = await collector.collect_ecmwf_data(
                            data_type=data_type,
                            start_date=config["start_date"],
                            end_date=config["end_date"],
                            regions=config.get("regions")
                        )
                    elif source == "satellite":
                        data = await collector.collect_satellite_data(
                            data_type=data_type,
                            start_date=config["start_date"],
                            end_date=config["end_date"],
                            regions=config.get("regions")
                        )
                    
                    # 保存收集的数据
                    if data is not None:
                        await collector.save_collected_data(
                            data=data,
                            source=source,
                            data_type=data_type,
                            metadata={
                                "collection_time": datetime.utcnow().isoformat(),
                                "user_id": config["user_id"],
                                "date_range": f"{config['start_date']} to {config['end_date']}"
                            }
                        )
                        
                        logger.info(f"数据收集完成: {source} - {data_type}")
                    
                except Exception as e:
                    logger.error(f"收集数据失败 {source}-{data_type}: {e}")
                    continue
        
        logger.info("数据收集任务完成")
        
    except Exception as e:
        logger.error(f"数据收集任务失败: {e}")


# ==================== 数据统计接口 ====================

@router.get("/statistics", response_model=Dict[str, Any], summary="获取数据统计信息")
async def get_data_statistics(
    source: Optional[DataSource] = Query(None, description="数据源"),
    data_type: Optional[DataType] = Query(None, description="数据类型"),
    db: DBSession = None,
    redis: RedisClient = None,
    current_user: CurrentUser = None
):
    """获取数据统计信息"""
    try:
        data_storage = DataStorage()
        
        # 构建查询条件
        conditions = {}
        if source:
            conditions["source"] = source.value
        if data_type:
            conditions["data_type"] = data_type.value
        
        # 获取统计信息
        stats = await data_storage.get_data_statistics(conditions)
        
        return {
            "total_records": stats.get("total_records", 0),
            "total_size": stats.get("total_size", 0),
            "sources": stats.get("sources", {}),
            "data_types": stats.get("data_types", {}),
            "date_range": stats.get("date_range", {}),
            "quality_distribution": stats.get("quality_distribution", {}),
            "recent_uploads": stats.get("recent_uploads", [])
        }
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计失败: {str(e)}"
        )