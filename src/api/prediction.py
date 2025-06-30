"""区域气候预测API

提供全球-区域降尺度气候预测、风险评估和热力图生成的API接口。
"""

from typing import List, Optional, Dict, Any, Tuple
from fastapi import APIRouter, HTTPException, status, Query, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
import asyncio
import io
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .models import (
    RegionalPredictionRequest, RegionalPredictionResponse, PredictionResult,
    HeatmapRequest, HeatmapResponse, HeatmapData,
    BaseResponse, ResponseStatus, ErrorResponse,
    ClimateScenario, PredictionModel, RiskLevel,
    GlobalIndicators, RegionalFeatures, Coordinates
)
from .dependencies import (
    DBSession, RedisClient, CurrentUser, AuthenticatedUser,
    NormalRateLimit, StrictRateLimit, PaginationParams
)
from ..models import RegionalPredictionModel
from ..utils.logger import logger
from ..utils.config import get_settings


router = APIRouter(prefix="/prediction", tags=["区域气候预测"])


# ==================== 区域预测接口 ====================

@router.post("/regional", response_model=RegionalPredictionResponse, summary="区域气候预测")
async def predict_regional_climate(
    request: RegionalPredictionRequest,
    background_tasks: BackgroundTasks,
    db: DBSession = None,
    redis: RedisClient = None,
    current_user: AuthenticatedUser = None,
    rate_limit: StrictRateLimit = None
):
    """执行区域气候预测"""
    try:
        settings = get_settings()
        
        # 初始化预测模型
        prediction_model = RegionalPredictionModel()
        
        # 构建预测输入数据
        prediction_input = await build_prediction_input(request)
        
        # 记录预测开始时间
        start_time = datetime.utcnow()
        
        # 执行预测
        prediction_results = []
        
        for target_year in request.target_years:
            try:
                # 单年预测
                if request.model == PredictionModel.ENSEMBLE:
                    result = await prediction_model.ensemble_predict(
                        global_data=prediction_input["global_data"],
                        regional_data=prediction_input["regional_data"],
                        target_year=target_year
                    )
                elif request.model == PredictionModel.TRANSFORMER:
                    result = await prediction_model.predict_with_transformer(
                        global_data=prediction_input["global_data"],
                        regional_data=prediction_input["regional_data"],
                        target_year=target_year
                    )
                elif request.model == PredictionModel.GNN:
                    result = await prediction_model.predict_with_gnn(
                        global_data=prediction_input["global_data"],
                        regional_data=prediction_input["regional_data"],
                        target_year=target_year
                    )
                elif request.model == PredictionModel.XGBOOST:
                    result = await prediction_model.predict_with_xgboost(
                        global_data=prediction_input["global_data"],
                        regional_data=prediction_input["regional_data"],
                        target_year=target_year
                    )
                
                # 风险评估
                risk_assessment = await assess_climate_risk(
                    prediction_result=result,
                    coordinates=request.coordinates,
                    scenario=request.scenario
                )
                
                # 构建预测结果
                prediction_result = PredictionResult(
                    target_year=target_year,
                    coordinates=request.coordinates,
                    scenario=request.scenario,
                    model=request.model,
                    predictions=result.get("predictions", {}),
                    confidence_intervals=result.get("confidence_intervals", {}),
                    risk_assessment=risk_assessment,
                    feature_importance=result.get("feature_importance", {}),
                    uncertainty_metrics=result.get("uncertainty", {})
                )
                
                prediction_results.append(prediction_result)
                
            except Exception as e:
                logger.error(f"预测{target_year}年失败: {e}")
                continue
        
        # 计算预测时间
        end_time = datetime.utcnow()
        prediction_time = (end_time - start_time).total_seconds()
        
        if not prediction_results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="预测失败"
            )
        
        # 后台保存预测记录
        background_tasks.add_task(
            save_prediction_record,
            request.dict(),
            prediction_results,
            current_user.user_id
        )
        
        return RegionalPredictionResponse(
            status=ResponseStatus.SUCCESS,
            message=f"成功完成{len(prediction_results)}年预测",
            results=prediction_results,
            prediction_time=prediction_time
        )
        
    except Exception as e:
        logger.error(f"区域预测失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测失败: {str(e)}"
        )


async def build_prediction_input(request: RegionalPredictionRequest) -> Dict[str, Any]:
    """构建预测输入数据"""
    # 构建全球数据
    global_data = {
        "sst_anomaly": request.global_indicators.sst_anomaly,
        "nao_index": request.global_indicators.nao_index,
        "enso_index": request.global_indicators.enso_index,
        "pdo_index": request.global_indicators.pdo_index,
        "amo_index": request.global_indicators.amo_index,
        "co2_concentration": request.global_indicators.co2_concentration,
        "solar_irradiance": request.global_indicators.solar_irradiance,
        "volcanic_activity": request.global_indicators.volcanic_activity
    }
    
    # 构建区域数据
    regional_data = {
        "latitude": request.coordinates.latitude,
        "longitude": request.coordinates.longitude,
        "elevation": request.regional_features.elevation,
        "distance_to_coast": request.regional_features.distance_to_coast,
        "land_use_type": request.regional_features.land_use_type,
        "vegetation_index": request.regional_features.vegetation_index,
        "soil_type": request.regional_features.soil_type,
        "population_density": request.regional_features.population_density,
        "urban_heat_island": request.regional_features.urban_heat_island
    }
    
    # 添加气候情景信息
    scenario_data = {
        "scenario": request.scenario.value,
        "emission_pathway": get_emission_pathway(request.scenario)
    }
    
    return {
        "global_data": global_data,
        "regional_data": regional_data,
        "scenario_data": scenario_data
    }


def get_emission_pathway(scenario: ClimateScenario) -> Dict[str, float]:
    """获取排放路径参数"""
    pathways = {
        ClimateScenario.RCP26: {
            "co2_growth_rate": 0.5,
            "peak_year": 2020,
            "reduction_rate": 0.03
        },
        ClimateScenario.RCP45: {
            "co2_growth_rate": 1.0,
            "peak_year": 2040,
            "reduction_rate": 0.01
        },
        ClimateScenario.RCP60: {
            "co2_growth_rate": 1.5,
            "peak_year": 2080,
            "reduction_rate": 0.005
        },
        ClimateScenario.RCP85: {
            "co2_growth_rate": 2.0,
            "peak_year": 2100,
            "reduction_rate": 0.0
        }
    }
    
    return pathways.get(scenario, pathways[ClimateScenario.RCP45])


async def assess_climate_risk(
    prediction_result: Dict[str, Any],
    coordinates: Coordinates,
    scenario: ClimateScenario
) -> Dict[str, Any]:
    """评估气候风险"""
    risk_assessment = {
        "overall_risk": RiskLevel.MEDIUM,
        "temperature_risk": RiskLevel.MEDIUM,
        "precipitation_risk": RiskLevel.MEDIUM,
        "extreme_events_risk": RiskLevel.MEDIUM,
        "drought_risk": RiskLevel.MEDIUM,
        "flood_risk": RiskLevel.MEDIUM,
        "risk_factors": [],
        "adaptation_recommendations": []
    }
    
    predictions = prediction_result.get("predictions", {})
    
    # 温度风险评估
    temp_change = predictions.get("temperature_change", 0)
    if temp_change > 3.0:
        risk_assessment["temperature_risk"] = RiskLevel.VERY_HIGH
        risk_assessment["risk_factors"].append("极端高温风险")
    elif temp_change > 2.0:
        risk_assessment["temperature_risk"] = RiskLevel.HIGH
        risk_assessment["risk_factors"].append("高温风险")
    elif temp_change > 1.0:
        risk_assessment["temperature_risk"] = RiskLevel.MEDIUM
    else:
        risk_assessment["temperature_risk"] = RiskLevel.LOW
    
    # 降水风险评估
    precip_change = predictions.get("precipitation_change", 0)
    if abs(precip_change) > 30:
        risk_assessment["precipitation_risk"] = RiskLevel.HIGH
        if precip_change > 0:
            risk_assessment["risk_factors"].append("极端降水风险")
        else:
            risk_assessment["risk_factors"].append("干旱风险")
    elif abs(precip_change) > 15:
        risk_assessment["precipitation_risk"] = RiskLevel.MEDIUM
    else:
        risk_assessment["precipitation_risk"] = RiskLevel.LOW
    
    # 干旱风险评估
    drought_index = predictions.get("drought_index", 0)
    if drought_index > 2.0:
        risk_assessment["drought_risk"] = RiskLevel.VERY_HIGH
        risk_assessment["risk_factors"].append("严重干旱风险")
    elif drought_index > 1.0:
        risk_assessment["drought_risk"] = RiskLevel.HIGH
    elif drought_index > 0.5:
        risk_assessment["drought_risk"] = RiskLevel.MEDIUM
    else:
        risk_assessment["drought_risk"] = RiskLevel.LOW
    
    # 洪水风险评估
    flood_risk_score = predictions.get("flood_risk", 0)
    if flood_risk_score > 0.8:
        risk_assessment["flood_risk"] = RiskLevel.VERY_HIGH
        risk_assessment["risk_factors"].append("洪水风险")
    elif flood_risk_score > 0.6:
        risk_assessment["flood_risk"] = RiskLevel.HIGH
    elif flood_risk_score > 0.4:
        risk_assessment["flood_risk"] = RiskLevel.MEDIUM
    else:
        risk_assessment["flood_risk"] = RiskLevel.LOW
    
    # 综合风险评估
    risk_scores = [
        risk_assessment["temperature_risk"].value,
        risk_assessment["precipitation_risk"].value,
        risk_assessment["drought_risk"].value,
        risk_assessment["flood_risk"].value
    ]
    
    avg_risk = sum(risk_scores) / len(risk_scores)
    
    if avg_risk >= 4:
        risk_assessment["overall_risk"] = RiskLevel.VERY_HIGH
    elif avg_risk >= 3:
        risk_assessment["overall_risk"] = RiskLevel.HIGH
    elif avg_risk >= 2:
        risk_assessment["overall_risk"] = RiskLevel.MEDIUM
    else:
        risk_assessment["overall_risk"] = RiskLevel.LOW
    
    # 生成适应性建议
    risk_assessment["adaptation_recommendations"] = generate_adaptation_recommendations(
        risk_assessment, coordinates, scenario
    )
    
    return risk_assessment


def generate_adaptation_recommendations(
    risk_assessment: Dict[str, Any],
    coordinates: Coordinates,
    scenario: ClimateScenario
) -> List[str]:
    """生成适应性建议"""
    recommendations = []
    
    # 基于温度风险的建议
    if risk_assessment["temperature_risk"] in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
        recommendations.extend([
            "加强城市绿化和遮阴设施建设",
            "改善建筑物隔热性能",
            "建立高温预警系统",
            "推广节能降温技术"
        ])
    
    # 基于降水风险的建议
    if risk_assessment["precipitation_risk"] in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
        recommendations.extend([
            "完善雨水收集和排水系统",
            "建设海绵城市基础设施",
            "加强洪水预警和应急响应"
        ])
    
    # 基于干旱风险的建议
    if risk_assessment["drought_risk"] in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
        recommendations.extend([
            "发展节水农业技术",
            "建设水资源储备设施",
            "推广抗旱作物品种",
            "实施水资源管理政策"
        ])
    
    # 基于洪水风险的建议
    if risk_assessment["flood_risk"] in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
        recommendations.extend([
            "建设防洪堤坝和排水设施",
            "制定洪水应急疏散计划",
            "限制洪泛区开发建设",
            "恢复湿地和自然缓冲区"
        ])
    
    # 基于地理位置的建议
    if coordinates.latitude > 60:  # 高纬度地区
        recommendations.append("应对永久冻土融化影响")
    elif coordinates.latitude < -60:  # 南极地区
        recommendations.append("监测冰盖变化影响")
    elif abs(coordinates.latitude) < 23.5:  # 热带地区
        recommendations.append("加强热带气旋防护")
    
    # 基于气候情景的建议
    if scenario in [ClimateScenario.RCP60, ClimateScenario.RCP85]:
        recommendations.extend([
            "制定长期气候适应战略",
            "投资气候韧性基础设施",
            "建立气候风险保险机制"
        ])
    
    return list(set(recommendations))  # 去重


async def save_prediction_record(
    request_data: dict,
    results: List[PredictionResult],
    user_id: str
):
    """保存预测记录（后台任务）"""
    try:
        from ..data_processing import DataStorage
        
        data_storage = DataStorage()
        
        # 保存预测记录
        await data_storage.save_model_result(
            model_type="regional_prediction",
            model_name=request_data.get("model", "unknown"),
            result={
                "request": request_data,
                "results": [result.dict() for result in results],
                "prediction_time": datetime.utcnow().isoformat()
            },
            metadata={
                "user_id": user_id,
                "num_years": len(results),
                "scenario": request_data.get("scenario"),
                "coordinates": request_data.get("coordinates"),
                "model": request_data.get("model")
            }
        )
        
        logger.info(f"区域预测记录已保存: {len(results)}年预测")
        
    except Exception as e:
        logger.error(f"保存预测记录失败: {e}")


# ==================== 热力图生成接口 ====================

@router.post("/heatmap", response_model=HeatmapResponse, summary="生成风险热力图")
async def generate_risk_heatmap(
    request: HeatmapRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = None,
    rate_limit: NormalRateLimit = None
):
    """生成区域风险热力图"""
    try:
        settings = get_settings()
        
        # 生成网格预测数据
        heatmap_data = await generate_grid_predictions(
            request.bounds,
            request.resolution,
            request.scenario,
            request.target_year,
            request.variable
        )
        
        # 生成热力图图像
        heatmap_image_path = await create_heatmap_image(
            heatmap_data,
            request.variable,
            request.color_scheme,
            request.bounds
        )
        
        # 生成统计信息
        statistics = calculate_heatmap_statistics(heatmap_data)
        
        # 构建响应
        heatmap_response = HeatmapData(
            variable=request.variable,
            scenario=request.scenario,
            target_year=request.target_year,
            bounds=request.bounds,
            resolution=request.resolution,
            data_points=len(heatmap_data),
            image_url=f"/static/heatmaps/{os.path.basename(heatmap_image_path)}",
            statistics=statistics,
            color_scheme=request.color_scheme,
            created_at=datetime.utcnow()
        )
        
        # 后台保存热力图记录
        background_tasks.add_task(
            save_heatmap_record,
            request.dict(),
            heatmap_response.dict(),
            current_user.user_id
        )
        
        return HeatmapResponse(
            status=ResponseStatus.SUCCESS,
            message="热力图生成成功",
            heatmap=heatmap_response
        )
        
    except Exception as e:
        logger.error(f"热力图生成失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成失败: {str(e)}"
        )


async def generate_grid_predictions(
    bounds: Dict[str, float],
    resolution: float,
    scenario: ClimateScenario,
    target_year: int,
    variable: str
) -> List[Dict[str, Any]]:
    """生成网格预测数据"""
    # 计算网格点
    lat_min, lat_max = bounds["south"], bounds["north"]
    lon_min, lon_max = bounds["west"], bounds["east"]
    
    lat_points = np.arange(lat_min, lat_max + resolution, resolution)
    lon_points = np.arange(lon_min, lon_max + resolution, resolution)
    
    grid_data = []
    
    # 初始化预测模型
    prediction_model = RegionalPredictionModel()
    
    for lat in lat_points:
        for lon in lon_points:
            try:
                # 构建预测输入
                prediction_input = {
                    "global_data": get_default_global_indicators(scenario),
                    "regional_data": {
                        "latitude": lat,
                        "longitude": lon,
                        "elevation": estimate_elevation(lat, lon),
                        "distance_to_coast": estimate_distance_to_coast(lat, lon),
                        "land_use_type": 1,  # 默认值
                        "vegetation_index": 0.5,
                        "soil_type": 1,
                        "population_density": 100,
                        "urban_heat_island": 0.0
                    }
                }
                
                # 执行预测（使用快速模型）
                result = await prediction_model.predict_with_xgboost(
                    global_data=prediction_input["global_data"],
                    regional_data=prediction_input["regional_data"],
                    target_year=target_year
                )
                
                # 提取指定变量的值
                value = result.get("predictions", {}).get(variable, 0)
                
                grid_data.append({
                    "latitude": lat,
                    "longitude": lon,
                    "value": value,
                    "variable": variable
                })
                
            except Exception as e:
                logger.warning(f"网格点({lat}, {lon})预测失败: {e}")
                continue
    
    return grid_data


def get_default_global_indicators(scenario: ClimateScenario) -> Dict[str, float]:
    """获取默认全球指标"""
    base_indicators = {
        "sst_anomaly": 0.5,
        "nao_index": 0.0,
        "enso_index": 0.0,
        "pdo_index": 0.0,
        "amo_index": 0.0,
        "co2_concentration": 400.0,
        "solar_irradiance": 1361.0,
        "volcanic_activity": 0.0
    }
    
    # 根据情景调整CO2浓度
    scenario_adjustments = {
        ClimateScenario.RCP26: 450.0,
        ClimateScenario.RCP45: 550.0,
        ClimateScenario.RCP60: 650.0,
        ClimateScenario.RCP85: 850.0
    }
    
    base_indicators["co2_concentration"] = scenario_adjustments.get(
        scenario, base_indicators["co2_concentration"]
    )
    
    return base_indicators


def estimate_elevation(lat: float, lon: float) -> float:
    """估算海拔高度（简化版）"""
    # 简化的海拔估算，实际应用中应使用DEM数据
    if abs(lat) > 60:  # 极地地区
        return 200.0
    elif abs(lat) < 30:  # 热带地区
        return 100.0
    else:  # 温带地区
        return 300.0


def estimate_distance_to_coast(lat: float, lon: float) -> float:
    """估算到海岸的距离（简化版）"""
    # 简化的距离估算，实际应用中应使用海岸线数据
    # 假设内陆地区距离海岸更远
    if abs(lon) > 100:  # 内陆地区
        return 500.0
    else:  # 沿海地区
        return 50.0


async def create_heatmap_image(
    data: List[Dict[str, Any]],
    variable: str,
    color_scheme: str,
    bounds: Dict[str, float]
) -> str:
    """创建热力图图像"""
    settings = get_settings()
    
    # 确保目录存在
    heatmap_dir = os.path.join(settings.data_storage_path, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # 准备数据
    df = pd.DataFrame(data)
    
    if df.empty:
        raise ValueError("没有数据可用于生成热力图")
    
    # 创建透视表
    pivot_table = df.pivot_table(
        values='value',
        index='latitude',
        columns='longitude',
        aggfunc='mean'
    )
    
    # 设置图形大小和样式
    plt.figure(figsize=(12, 8))
    plt.style.use('default')
    
    # 选择颜色方案
    color_maps = {
        "viridis": "viridis",
        "plasma": "plasma",
        "coolwarm": "coolwarm",
        "RdYlBu": "RdYlBu_r",
        "custom_risk": LinearSegmentedColormap.from_list(
            "risk", ["green", "yellow", "orange", "red", "darkred"]
        )
    }
    
    cmap = color_maps.get(color_scheme, "viridis")
    
    # 创建热力图
    sns.heatmap(
        pivot_table,
        cmap=cmap,
        center=pivot_table.values.mean(),
        annot=False,
        fmt='.2f',
        cbar_kws={'label': get_variable_label(variable)}
    )
    
    # 设置标题和标签
    plt.title(f'{get_variable_label(variable)}热力图', fontsize=16, pad=20)
    plt.xlabel('经度', fontsize=12)
    plt.ylabel('纬度', fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"heatmap_{variable}_{timestamp}.png"
    image_path = os.path.join(heatmap_dir, filename)
    
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return image_path


def get_variable_label(variable: str) -> str:
    """获取变量标签"""
    labels = {
        "temperature_change": "温度变化 (°C)",
        "precipitation_change": "降水变化 (%)",
        "drought_index": "干旱指数",
        "flood_risk": "洪水风险",
        "heat_days": "高温日数",
        "extreme_precipitation": "极端降水事件"
    }
    
    return labels.get(variable, variable)


def calculate_heatmap_statistics(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算热力图统计信息"""
    if not data:
        return {}
    
    values = [point["value"] for point in data]
    
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "percentile_25": float(np.percentile(values, 25)),
        "percentile_75": float(np.percentile(values, 75))
    }


async def save_heatmap_record(
    request_data: dict,
    heatmap_data: dict,
    user_id: str
):
    """保存热力图记录（后台任务）"""
    try:
        from ..data_processing import DataStorage
        
        data_storage = DataStorage()
        
        # 保存热力图记录
        await data_storage.save_model_result(
            model_type="heatmap_generation",
            model_name="regional_heatmap",
            result={
                "request": request_data,
                "heatmap": heatmap_data,
                "generation_time": datetime.utcnow().isoformat()
            },
            metadata={
                "user_id": user_id,
                "variable": request_data.get("variable"),
                "scenario": request_data.get("scenario"),
                "target_year": request_data.get("target_year")
            }
        )
        
        logger.info(f"热力图记录已保存: {request_data.get('variable')}")
        
    except Exception as e:
        logger.error(f"保存热力图记录失败: {e}")


# ==================== 预测历史接口 ====================

@router.get("/history", response_model=Dict[str, Any], summary="获取预测历史")
async def get_prediction_history(
    user_id: Optional[str] = Query(None, description="用户ID"),
    model_type: Optional[str] = Query(None, description="模型类型"),
    scenario: Optional[ClimateScenario] = Query(None, description="气候情景"),
    pagination: PaginationParams = None,
    db: DBSession = None,
    current_user: CurrentUser = None
):
    """获取预测历史记录"""
    try:
        from ..data_processing import DataStorage
        
        data_storage = DataStorage()
        
        # 构建查询条件
        conditions = {"model_type": "regional_prediction"}
        
        if user_id:
            conditions["metadata.user_id"] = user_id
        
        if model_type:
            conditions["model_name"] = model_type
        
        if scenario:
            conditions["metadata.scenario"] = scenario.value
        
        # 查询预测记录
        records = await data_storage.search_model_results(
            conditions=conditions,
            limit=pagination.limit,
            offset=pagination.offset
        )
        
        # 格式化记录
        formatted_records = []
        for record in records:
            formatted_record = {
                "id": record.get("id"),
                "model_type": record.get("model_type"),
                "model_name": record.get("model_name"),
                "scenario": record.get("metadata", {}).get("scenario"),
                "coordinates": record.get("metadata", {}).get("coordinates"),
                "num_years": record.get("metadata", {}).get("num_years"),
                "created_at": record.get("created_at"),
                "summary": extract_prediction_summary(record.get("result", {}))
            }
            formatted_records.append(formatted_record)
        
        return {
            "status": "success",
            "records": formatted_records,
            "total": len(formatted_records),
            "pagination": {
                "offset": pagination.offset,
                "limit": pagination.limit
            }
        }
        
    except Exception as e:
        logger.error(f"获取预测历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取历史失败: {str(e)}"
        )


def extract_prediction_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """提取预测摘要信息"""
    results = result.get("results", [])
    
    if not results:
        return {}
    
    # 计算平均风险等级
    risk_levels = []
    for res in results:
        risk_assessment = res.get("risk_assessment", {})
        overall_risk = risk_assessment.get("overall_risk")
        if overall_risk:
            risk_levels.append(overall_risk)
    
    # 提取主要预测变量
    temp_changes = []
    precip_changes = []
    
    for res in results:
        predictions = res.get("predictions", {})
        if "temperature_change" in predictions:
            temp_changes.append(predictions["temperature_change"])
        if "precipitation_change" in predictions:
            precip_changes.append(predictions["precipitation_change"])
    
    summary = {
        "years_predicted": len(results),
        "avg_temperature_change": np.mean(temp_changes) if temp_changes else None,
        "avg_precipitation_change": np.mean(precip_changes) if precip_changes else None,
        "dominant_risk_level": max(set(risk_levels), key=risk_levels.count) if risk_levels else None
    }
    
    return summary


# ==================== 模型比较接口 ====================

@router.post("/compare-models", response_model=Dict[str, Any], summary="比较预测模型")
async def compare_prediction_models(
    coordinates: Coordinates,
    scenario: ClimateScenario,
    target_year: int,
    global_indicators: GlobalIndicators,
    regional_features: RegionalFeatures,
    models: List[PredictionModel] = Query(..., description="要比较的模型列表"),
    current_user: AuthenticatedUser = None,
    rate_limit: StrictRateLimit = None
):
    """比较不同预测模型的性能"""
    try:
        prediction_model = RegionalPredictionModel()
        
        # 构建输入数据
        global_data = global_indicators.dict()
        regional_data = {
            "latitude": coordinates.latitude,
            "longitude": coordinates.longitude,
            **regional_features.dict()
        }
        
        # 执行各模型预测
        model_results = {}
        
        for model in models:
            try:
                if model == PredictionModel.ENSEMBLE:
                    result = await prediction_model.ensemble_predict(
                        global_data=global_data,
                        regional_data=regional_data,
                        target_year=target_year
                    )
                elif model == PredictionModel.TRANSFORMER:
                    result = await prediction_model.predict_with_transformer(
                        global_data=global_data,
                        regional_data=regional_data,
                        target_year=target_year
                    )
                elif model == PredictionModel.GNN:
                    result = await prediction_model.predict_with_gnn(
                        global_data=global_data,
                        regional_data=regional_data,
                        target_year=target_year
                    )
                elif model == PredictionModel.XGBOOST:
                    result = await prediction_model.predict_with_xgboost(
                        global_data=global_data,
                        regional_data=regional_data,
                        target_year=target_year
                    )
                
                model_results[model.value] = result
                
            except Exception as e:
                logger.error(f"模型{model.value}预测失败: {e}")
                model_results[model.value] = {"error": str(e)}
        
        # 计算模型一致性
        consistency_analysis = analyze_model_consistency(model_results)
        
        # 生成比较报告
        comparison_report = generate_comparison_report(
            model_results, consistency_analysis
        )
        
        return {
            "status": "success",
            "model_results": model_results,
            "consistency_analysis": consistency_analysis,
            "comparison_report": comparison_report,
            "metadata": {
                "coordinates": coordinates.dict(),
                "scenario": scenario.value,
                "target_year": target_year,
                "models_compared": [m.value for m in models]
            }
        }
        
    except Exception as e:
        logger.error(f"模型比较失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"比较失败: {str(e)}"
        )


def analyze_model_consistency(model_results: Dict[str, Any]) -> Dict[str, Any]:
    """分析模型一致性"""
    # 提取各模型的预测值
    predictions_by_variable = {}
    
    for model_name, result in model_results.items():
        if "error" in result:
            continue
        
        predictions = result.get("predictions", {})
        for variable, value in predictions.items():
            if variable not in predictions_by_variable:
                predictions_by_variable[variable] = {}
            predictions_by_variable[variable][model_name] = value
    
    # 计算一致性指标
    consistency_metrics = {}
    
    for variable, model_predictions in predictions_by_variable.items():
        values = list(model_predictions.values())
        
        if len(values) > 1:
            consistency_metrics[variable] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "coefficient_of_variation": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0,
                "range": float(np.max(values) - np.min(values)),
                "model_predictions": model_predictions
            }
    
    return consistency_metrics


def generate_comparison_report(model_results: Dict[str, Any], consistency_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """生成比较报告"""
    report = {
        "summary": {
            "total_models": len(model_results),
            "successful_models": len([r for r in model_results.values() if "error" not in r]),
            "failed_models": len([r for r in model_results.values() if "error" in r])
        },
        "model_performance": {},
        "consistency_assessment": {},
        "recommendations": []
    }
    
    # 评估模型性能
    for model_name, result in model_results.items():
        if "error" in result:
            report["model_performance"][model_name] = {
                "status": "failed",
                "error": result["error"]
            }
        else:
            uncertainty = result.get("uncertainty", {})
            report["model_performance"][model_name] = {
                "status": "success",
                "uncertainty_score": uncertainty.get("overall_uncertainty", 0),
                "confidence_level": uncertainty.get("confidence_level", 0)
            }
    
    # 评估一致性
    for variable, metrics in consistency_analysis.items():
        cv = metrics["coefficient_of_variation"]
        
        if cv < 0.1:
            assessment = "高度一致"
        elif cv < 0.3:
            assessment = "中等一致"
        else:
            assessment = "低一致性"
        
        report["consistency_assessment"][variable] = {
            "assessment": assessment,
            "coefficient_of_variation": cv,
            "standard_deviation": metrics["std"]
        }
    
    # 生成建议
    avg_cv = np.mean([m["coefficient_of_variation"] for m in consistency_analysis.values()])
    
    if avg_cv < 0.2:
        report["recommendations"].append("模型预测结果较为一致，可信度较高")
    else:
        report["recommendations"].append("模型预测存在较大差异，建议进一步验证")
    
    successful_models = [name for name, result in model_results.items() if "error" not in result]
    if len(successful_models) > 1:
        report["recommendations"].append("建议使用集成方法结合多个模型的预测结果")
    
    return report


# ==================== 热力图文件接口 ====================

@router.get("/heatmaps/{filename}", summary="获取热力图文件")
async def get_heatmap_file(
    filename: str,
    current_user: CurrentUser = None
):
    """获取热力图图像文件"""
    try:
        settings = get_settings()
        
        # 构建文件路径
        heatmap_dir = os.path.join(settings.data_storage_path, "heatmaps")
        file_path = os.path.join(heatmap_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="热力图文件不存在"
            )
        
        # 返回文件
        return FileResponse(
            file_path,
            media_type="image/png",
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"获取热力图文件失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取文件失败: {str(e)}"
        )