"""图像生成与可视化API

提供生态警示图像生成、模板管理和可视化服务的API接口。
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Query, BackgroundTasks, Form
from fastapi.responses import FileResponse, StreamingResponse
import asyncio
import io
import os
import base64
from datetime import datetime
import tempfile
import json
from PIL import Image
import numpy as np

from .models import (
    ImageGenerationRequest, ImageGenerationResponse, GeneratedImage,
    ImageTemplateRequest, ImageTemplateResponse, ImageTemplate,
    BaseResponse, ResponseStatus, ErrorResponse,
    ImageGenerationModel, EnvironmentScenario, EnvironmentIndicators
)
from .dependencies import (
    DBSession, RedisClient, CurrentUser, AuthenticatedUser,
    NormalRateLimit, StrictRateLimit
)
from ..models import EcologyImageGenerator
from ..utils.logger import logger
from ..utils.config import get_settings


router = APIRouter(prefix="/visualization", tags=["图像生成与可视化"])


# ==================== 图像生成接口 ====================

@router.post("/generate", response_model=ImageGenerationResponse, summary="生成生态警示图像")
async def generate_ecology_image(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    db: DBSession = None,
    redis: RedisClient = None,
    current_user: AuthenticatedUser = None,
    rate_limit: StrictRateLimit = None
):
    """生成生态警示图像"""
    try:
        settings = get_settings()
        
        # 初始化图像生成器
        generator = EcologyImageGenerator()
        
        # 构建生成参数
        generation_params = {
            "model_type": request.model.value,
            "resolution": request.resolution,
            "num_images": request.num_images,
            "seed": request.seed,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps
        }
        
        # 构建提示词
        prompt = await build_generation_prompt(request)
        
        # 记录生成开始时间
        start_time = datetime.utcnow()
        
        # 执行图像生成
        generated_images = []
        total_generation_time = 0.0
        
        for i in range(request.num_images):
            try:
                # 生成单张图像
                if request.model == ImageGenerationModel.GAN:
                    image_array = await generator.generate_with_gan(
                        prompt=prompt,
                        **generation_params
                    )
                elif request.model == ImageGenerationModel.DIFFUSION:
                    image_array = await generator.generate_with_diffusion(
                        prompt=prompt,
                        **generation_params
                    )
                elif request.model == ImageGenerationModel.HYBRID:
                    image_array = await generator.generate_hybrid(
                        prompt=prompt,
                        **generation_params
                    )
                
                # 保存图像
                image_info = await save_generated_image(
                    image_array=image_array,
                    prompt=prompt,
                    model=request.model.value,
                    parameters=generation_params,
                    user_id=current_user.user_id,
                    index=i
                )
                
                generated_images.append(image_info)
                
            except Exception as e:
                logger.error(f"生成第{i+1}张图像失败: {e}")
                continue
        
        # 计算总生成时间
        end_time = datetime.utcnow()
        total_generation_time = (end_time - start_time).total_seconds()
        
        if not generated_images:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="图像生成失败"
            )
        
        # 后台保存生成记录
        background_tasks.add_task(
            save_generation_record,
            request.dict(),
            generated_images,
            current_user.user_id
        )
        
        return ImageGenerationResponse(
            status=ResponseStatus.SUCCESS,
            message=f"成功生成{len(generated_images)}张图像",
            images=generated_images,
            total_generation_time=total_generation_time
        )
        
    except Exception as e:
        logger.error(f"图像生成失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成失败: {str(e)}"
        )


async def build_generation_prompt(request: ImageGenerationRequest) -> str:
    """构建图像生成提示词"""
    prompt_parts = []
    
    # 使用自定义提示词
    if request.custom_prompt:
        prompt_parts.append(request.custom_prompt)
    
    # 使用预设场景
    elif request.scenario:
        scenario_prompts = {
            EnvironmentScenario.GLACIER_MELTING: "melting glaciers, rising sea levels, arctic ice disappearing, climate change impact",
            EnvironmentScenario.FOREST_FIRE: "massive forest fires, burning trees, smoke-filled sky, environmental destruction",
            EnvironmentScenario.FLOOD: "severe flooding, submerged buildings, heavy rainfall, water disaster",
            EnvironmentScenario.DROUGHT: "severe drought, cracked earth, dried rivers, water scarcity",
            EnvironmentScenario.AIR_POLLUTION: "heavy air pollution, smoggy city, industrial emissions, poor air quality",
            EnvironmentScenario.DEFORESTATION: "deforestation, cut down trees, environmental destruction, habitat loss",
            EnvironmentScenario.OCEAN_ACIDIFICATION: "ocean acidification, coral bleaching, marine ecosystem damage",
            EnvironmentScenario.EXTREME_WEATHER: "extreme weather events, storms, hurricanes, climate chaos"
        }
        prompt_parts.append(scenario_prompts.get(request.scenario, ""))
    
    # 根据环境指标调整提示词
    if request.indicators:
        indicator_prompts = await build_indicator_prompts(request.indicators)
        prompt_parts.extend(indicator_prompts)
    
    # 添加风格描述
    if request.style:
        prompt_parts.append(f"in {request.style} style")
    
    # 添加质量和细节描述
    prompt_parts.extend([
        "high quality", "detailed", "realistic", "environmental impact", 
        "climate change visualization", "ecological warning"
    ])
    
    return ", ".join(filter(None, prompt_parts))


async def build_indicator_prompts(indicators: EnvironmentIndicators) -> List[str]:
    """根据环境指标构建提示词"""
    prompts = []
    
    if indicators.co2_emission is not None:
        if indicators.co2_emission > 1000:
            prompts.append("heavy industrial emissions, thick smoke stacks")
        elif indicators.co2_emission > 500:
            prompts.append("moderate pollution, visible emissions")
        else:
            prompts.append("clean environment, minimal emissions")
    
    if indicators.temperature_change is not None:
        if indicators.temperature_change > 2.0:
            prompts.append("extreme heat, scorching sun, heat waves")
        elif indicators.temperature_change > 1.0:
            prompts.append("rising temperatures, warm climate")
        elif indicators.temperature_change < -1.0:
            prompts.append("extreme cold, frozen landscape")
    
    if indicators.air_quality_index is not None:
        if indicators.air_quality_index > 300:
            prompts.append("hazardous air quality, thick smog, visibility reduced")
        elif indicators.air_quality_index > 150:
            prompts.append("unhealthy air, moderate smog")
        elif indicators.air_quality_index < 50:
            prompts.append("clean air, clear sky")
    
    if indicators.forest_coverage is not None:
        if indicators.forest_coverage < 20:
            prompts.append("deforested landscape, bare land, few trees")
        elif indicators.forest_coverage > 80:
            prompts.append("dense forest, lush vegetation, green landscape")
    
    if indicators.sea_level_rise is not None:
        if indicators.sea_level_rise > 1.0:
            prompts.append("significant sea level rise, coastal flooding")
        elif indicators.sea_level_rise > 0.5:
            prompts.append("rising sea levels, coastal erosion")
    
    return prompts


async def save_generated_image(
    image_array: np.ndarray,
    prompt: str,
    model: str,
    parameters: dict,
    user_id: str,
    index: int = 0
) -> GeneratedImage:
    """保存生成的图像"""
    settings = get_settings()
    
    # 生成文件名
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    image_id = f"{timestamp}_{user_id}_{index}"
    filename = f"{image_id}.png"
    
    # 确保目录存在
    image_dir = os.path.join(settings.data_storage_path, "generated_images")
    os.makedirs(image_dir, exist_ok=True)
    
    # 保存图像文件
    image_path = os.path.join(image_dir, filename)
    
    # 转换numpy数组为PIL图像并保存
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    image = Image.fromarray(image_array)
    image.save(image_path, "PNG")
    
    # 生成缩略图
    thumbnail_path = os.path.join(image_dir, f"thumb_{filename}")
    thumbnail = image.copy()
    thumbnail.thumbnail((256, 256), Image.Resampling.LANCZOS)
    thumbnail.save(thumbnail_path, "PNG")
    
    # 计算文件大小
    file_size = os.path.getsize(image_path)
    
    # 生成URL（假设有静态文件服务）
    image_url = f"/static/generated_images/{filename}"
    thumbnail_url = f"/static/generated_images/thumb_{filename}"
    
    return GeneratedImage(
        image_id=image_id,
        image_url=image_url,
        image_path=image_path,
        thumbnail_url=thumbnail_url,
        prompt=prompt,
        model=model,
        parameters=parameters,
        generation_time=parameters.get("generation_time", 0.0),
        file_size=file_size,
        created_at=datetime.utcnow()
    )


async def save_generation_record(request_data: dict, images: List[GeneratedImage], user_id: str):
    """保存生成记录（后台任务）"""
    try:
        from ..data_processing import DataStorage
        
        data_storage = DataStorage()
        
        # 保存生成记录
        await data_storage.save_model_result(
            model_type="image_generation",
            model_name=request_data.get("model", "unknown"),
            result={
                "request": request_data,
                "images": [img.dict() for img in images],
                "generation_time": datetime.utcnow().isoformat()
            },
            metadata={
                "user_id": user_id,
                "num_images": len(images),
                "scenario": request_data.get("scenario"),
                "model": request_data.get("model")
            }
        )
        
        logger.info(f"图像生成记录已保存: {len(images)}张图像")
        
    except Exception as e:
        logger.error(f"保存生成记录失败: {e}")


# ==================== 图像模板接口 ====================

@router.get("/templates", response_model=ImageTemplateResponse, summary="获取图像模板列表")
async def get_image_templates(
    category: Optional[str] = Query(None, description="模板类别"),
    scenario: Optional[EnvironmentScenario] = Query(None, description="环境场景"),
    current_user: CurrentUser = None,
    rate_limit: NormalRateLimit = None
):
    """获取图像模板列表"""
    try:
        # 预定义模板数据
        templates = await get_predefined_templates()
        
        # 过滤模板
        if category:
            templates = [t for t in templates if t.category == category]
        
        if scenario:
            templates = [t for t in templates if t.scenario == scenario]
        
        return ImageTemplateResponse(
            status=ResponseStatus.SUCCESS,
            message="获取模板成功",
            templates=templates
        )
        
    except Exception as e:
        logger.error(f"获取图像模板失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取模板失败: {str(e)}"
        )


async def get_predefined_templates() -> List[ImageTemplate]:
    """获取预定义模板"""
    templates = [
        ImageTemplate(
            template_id="glacier_melting_01",
            name="冰川融化警示",
            description="展示冰川融化对环境的影响",
            category="气候变化",
            scenario=EnvironmentScenario.GLACIER_MELTING,
            prompt_template="melting glaciers, rising sea levels, arctic landscape, climate change impact, dramatic sky, environmental warning, high quality, detailed, realistic",
            default_parameters={
                "model": "diffusion",
                "resolution": "1024x1024",
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            },
            preview_image="/static/templates/glacier_melting_preview.jpg",
            tags=["冰川", "海平面上升", "气候变化", "北极"]
        ),
        ImageTemplate(
            template_id="forest_fire_01",
            name="森林火灾警示",
            description="展示森林火灾的破坏性影响",
            category="自然灾害",
            scenario=EnvironmentScenario.FOREST_FIRE,
            prompt_template="massive forest fires, burning trees, orange flames, smoke-filled sky, environmental destruction, wildlife evacuation, dramatic lighting, high quality, detailed",
            default_parameters={
                "model": "diffusion",
                "resolution": "1024x1024",
                "guidance_scale": 8.0,
                "num_inference_steps": 50
            },
            preview_image="/static/templates/forest_fire_preview.jpg",
            tags=["森林火灾", "环境破坏", "野生动物", "自然灾害"]
        ),
        ImageTemplate(
            template_id="air_pollution_01",
            name="空气污染警示",
            description="展示严重空气污染的城市景象",
            category="环境污染",
            scenario=EnvironmentScenario.AIR_POLLUTION,
            prompt_template="heavy air pollution, smoggy city skyline, industrial smokestacks, poor visibility, people wearing masks, grey atmosphere, environmental health crisis, realistic, detailed",
            default_parameters={
                "model": "diffusion",
                "resolution": "1024x1024",
                "guidance_scale": 7.0,
                "num_inference_steps": 45
            },
            preview_image="/static/templates/air_pollution_preview.jpg",
            tags=["空气污染", "城市", "工业排放", "健康危机"]
        ),
        ImageTemplate(
            template_id="drought_01",
            name="干旱警示",
            description="展示严重干旱对农业和生态的影响",
            category="气候变化",
            scenario=EnvironmentScenario.DROUGHT,
            prompt_template="severe drought, cracked dry earth, withered crops, empty reservoirs, dead trees, scorching sun, water scarcity, environmental crisis, realistic, detailed",
            default_parameters={
                "model": "diffusion",
                "resolution": "1024x1024",
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            },
            preview_image="/static/templates/drought_preview.jpg",
            tags=["干旱", "农业", "水资源", "生态危机"]
        ),
        ImageTemplate(
            template_id="deforestation_01",
            name="森林砍伐警示",
            description="展示大规模森林砍伐的环境影响",
            category="环境破坏",
            scenario=EnvironmentScenario.DEFORESTATION,
            prompt_template="massive deforestation, cut down trees, logging equipment, bare landscape, environmental destruction, habitat loss, before and after contrast, realistic, detailed",
            default_parameters={
                "model": "diffusion",
                "resolution": "1024x1024",
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            },
            preview_image="/static/templates/deforestation_preview.jpg",
            tags=["森林砍伐", "栖息地丧失", "环境破坏", "生物多样性"]
        )
    ]
    
    return templates


# ==================== 图像管理接口 ====================

@router.get("/images/{image_id}", summary="获取生成的图像")
async def get_generated_image(
    image_id: str,
    thumbnail: bool = Query(False, description="是否返回缩略图"),
    current_user: CurrentUser = None
):
    """获取生成的图像文件"""
    try:
        settings = get_settings()
        
        # 构建文件路径
        image_dir = os.path.join(settings.data_storage_path, "generated_images")
        
        if thumbnail:
            filename = f"thumb_{image_id}.png"
        else:
            filename = f"{image_id}.png"
        
        file_path = os.path.join(image_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="图像不存在"
            )
        
        # 返回文件
        return FileResponse(
            file_path,
            media_type="image/png",
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"获取图像失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取图像失败: {str(e)}"
        )


@router.get("/images/{image_id}/info", response_model=Dict[str, Any], summary="获取图像信息")
async def get_image_info(
    image_id: str,
    db: DBSession = None,
    current_user: CurrentUser = None
):
    """获取图像详细信息"""
    try:
        from ..data_processing import DataStorage
        
        data_storage = DataStorage()
        
        # 查询图像生成记录
        records = await data_storage.search_model_results(
            conditions={"model_type": "image_generation"},
            limit=100
        )
        
        # 查找匹配的图像
        for record in records:
            result = record.get("result", {})
            images = result.get("images", [])
            
            for img in images:
                if img.get("image_id") == image_id:
                    return {
                        "image_info": img,
                        "generation_request": result.get("request"),
                        "generation_time": result.get("generation_time"),
                        "metadata": record.get("metadata")
                    }
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="图像信息不存在"
        )
        
    except Exception as e:
        logger.error(f"获取图像信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取信息失败: {str(e)}"
        )


@router.delete("/images/{image_id}", response_model=BaseResponse, summary="删除生成的图像")
async def delete_generated_image(
    image_id: str,
    current_user: AuthenticatedUser = None
):
    """删除生成的图像"""
    try:
        settings = get_settings()
        
        # 构建文件路径
        image_dir = os.path.join(settings.data_storage_path, "generated_images")
        image_path = os.path.join(image_dir, f"{image_id}.png")
        thumbnail_path = os.path.join(image_dir, f"thumb_{image_id}.png")
        
        # 删除文件
        deleted_files = []
        
        if os.path.exists(image_path):
            os.remove(image_path)
            deleted_files.append("原图")
        
        if os.path.exists(thumbnail_path):
            os.remove(thumbnail_path)
            deleted_files.append("缩略图")
        
        if not deleted_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="图像文件不存在"
            )
        
        return BaseResponse(
            status=ResponseStatus.SUCCESS,
            message=f"已删除: {', '.join(deleted_files)}"
        )
        
    except Exception as e:
        logger.error(f"删除图像失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除失败: {str(e)}"
        )


# ==================== 批量操作接口 ====================

@router.post("/batch-generate", response_model=ImageGenerationResponse, summary="批量生成图像")
async def batch_generate_images(
    scenarios: List[EnvironmentScenario] = Form(...),
    model: ImageGenerationModel = Form(ImageGenerationModel.DIFFUSION),
    resolution: str = Form("1024x1024"),
    background_tasks: BackgroundTasks = None,
    current_user: AuthenticatedUser = None,
    rate_limit: StrictRateLimit = None
):
    """批量生成多个场景的图像"""
    try:
        all_generated_images = []
        total_time = 0.0
        
        for scenario in scenarios:
            # 为每个场景创建生成请求
            request = ImageGenerationRequest(
                scenario=scenario,
                model=model,
                resolution=resolution,
                num_images=1
            )
            
            # 生成图像
            try:
                response = await generate_ecology_image(
                    request=request,
                    background_tasks=background_tasks,
                    current_user=current_user,
                    rate_limit=rate_limit
                )
                
                all_generated_images.extend(response.images)
                total_time += response.total_generation_time
                
            except Exception as e:
                logger.error(f"批量生成中场景{scenario}失败: {e}")
                continue
        
        if not all_generated_images:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="批量生成失败"
            )
        
        return ImageGenerationResponse(
            status=ResponseStatus.SUCCESS,
            message=f"批量生成完成，共{len(all_generated_images)}张图像",
            images=all_generated_images,
            total_generation_time=total_time
        )
        
    except Exception as e:
        logger.error(f"批量生成失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量生成失败: {str(e)}"
        )


# ==================== 图像编辑接口 ====================

@router.post("/images/{image_id}/enhance", response_model=GeneratedImage, summary="增强图像质量")
async def enhance_image(
    image_id: str,
    enhancement_type: str = Query("upscale", regex="^(upscale|denoise|sharpen)$"),
    current_user: AuthenticatedUser = None
):
    """增强图像质量"""
    try:
        settings = get_settings()
        
        # 获取原图像
        image_dir = os.path.join(settings.data_storage_path, "generated_images")
        original_path = os.path.join(image_dir, f"{image_id}.png")
        
        if not os.path.exists(original_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="原图像不存在"
            )
        
        # 加载图像
        image = Image.open(original_path)
        image_array = np.array(image)
        
        # 执行增强
        if enhancement_type == "upscale":
            # 简单的双线性插值放大
            enhanced_image = image.resize(
                (image.width * 2, image.height * 2),
                Image.Resampling.LANCZOS
            )
        elif enhancement_type == "denoise":
            # 简单的高斯模糊降噪
            from PIL import ImageFilter
            enhanced_image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        elif enhancement_type == "sharpen":
            # 锐化滤镜
            from PIL import ImageFilter
            enhanced_image = image.filter(ImageFilter.SHARPEN)
        
        # 保存增强后的图像
        enhanced_id = f"{image_id}_enhanced_{enhancement_type}"
        enhanced_filename = f"{enhanced_id}.png"
        enhanced_path = os.path.join(image_dir, enhanced_filename)
        
        enhanced_image.save(enhanced_path, "PNG")
        
        # 生成缩略图
        thumbnail_path = os.path.join(image_dir, f"thumb_{enhanced_filename}")
        thumbnail = enhanced_image.copy()
        thumbnail.thumbnail((256, 256), Image.Resampling.LANCZOS)
        thumbnail.save(thumbnail_path, "PNG")
        
        # 返回增强后的图像信息
        file_size = os.path.getsize(enhanced_path)
        
        return GeneratedImage(
            image_id=enhanced_id,
            image_url=f"/static/generated_images/{enhanced_filename}",
            image_path=enhanced_path,
            thumbnail_url=f"/static/generated_images/thumb_{enhanced_filename}",
            prompt=f"Enhanced ({enhancement_type}) version of {image_id}",
            model="enhancement",
            parameters={"enhancement_type": enhancement_type, "original_id": image_id},
            generation_time=0.0,
            file_size=file_size,
            created_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"图像增强失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"增强失败: {str(e)}"
        )