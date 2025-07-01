"""API数据模型

定义API请求和响应的数据模型，使用Pydantic进行数据验证。
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator, model_validator
import uuid


# ==================== 基础模型 ====================

class ResponseStatus(str, Enum):
    """响应状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


class BaseResponse(BaseModel):
    """基础响应模型"""
    status: ResponseStatus
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class PaginationParams(BaseModel):
    """分页参数"""
    page: int = Field(1, ge=1, description="页码")
    size: int = Field(20, ge=1, le=100, description="每页大小")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.size


class PaginatedResponse(BaseResponse):
    """分页响应模型"""
    total: int = Field(description="总记录数")
    page: int = Field(description="当前页码")
    size: int = Field(description="每页大小")
    pages: int = Field(description="总页数")
    
    @model_validator(mode="after")
    def calculate_pages(self) -> "Self":
        self.pages = (self.total + self.size - 1) // self.size if self.size > 0 else 0
        return self


class CoordinateModel(BaseModel):
    """坐标模型"""
    latitude: float = Field(..., ge=-90, le=90, description="纬度")
    longitude: float = Field(..., ge=-180, le=180, description="经度")


class TimeRangeModel(BaseModel):
    """时间范围模型"""
    start_date: date = Field(..., description="开始日期")
    end_date: date = Field(..., description="结束日期")
    
    @model_validator(mode='after')
    def check_dates(self) -> 'TimeRangeModel':
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError('结束日期必须晚于或等于开始日期')
        return self


# ==================== 气候数据API模型 ====================

class DataSource(str, Enum):
    """数据源枚举"""
    NOAA = "noaa"
    ECMWF = "ecmwf"
    SATELLITE = "satellite"
    LOCAL = "local"


class DataType(str, Enum):
    """数据类型枚举"""
    TEMPERATURE = "temperature"
    PRECIPITATION = "precipitation"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    WIND = "wind"
    RADIATION = "radiation"


class AnalysisType(str, Enum):
    """分析类型枚举"""
    TREND = "trend"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    STATISTICS = "statistics"
    CORRELATION = "correlation"


class ClimateDataQuery(BaseModel):
    """气候数据查询请求"""
    source: Optional[DataSource] = Field(None, description="数据源")
    data_type: Optional[DataType] = Field(None, description="数据类型")
    location: Optional[str] = Field(None, description="位置描述")
    coordinates: Optional[CoordinateModel] = Field(None, description="坐标")
    time_range: Optional[TimeRangeModel] = Field(None, description="时间范围")
    variables: Optional[List[str]] = Field(None, description="变量列表")
    quality_threshold: Optional[float] = Field(None, ge=0, le=1, description="质量阈值")
    format: Literal["json", "csv", "netcdf"] = Field("json", description="返回格式")
    pagination: Optional[PaginationParams] = Field(None, description="分页参数")


class ClimateDataRecord(BaseModel):
    """气候数据记录"""
    id: str = Field(..., description="记录ID")
    source: str = Field(..., description="数据源")
    data_type: str = Field(..., description="数据类型")
    location: Optional[str] = Field(None, description="位置")
    latitude: Optional[float] = Field(None, description="纬度")
    longitude: Optional[float] = Field(None, description="经度")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    file_path: str = Field(..., description="文件路径")
    file_format: str = Field(..., description="文件格式")
    file_size: int = Field(..., description="文件大小（字节）")
    variables: Optional[List[str]] = Field(None, description="变量列表")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    quality_score: Optional[float] = Field(None, description="质量评分")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class ClimateDataResponse(PaginatedResponse):
    """气候数据查询响应"""
    data: List[ClimateDataRecord] = Field(..., description="数据记录列表")


class ClimateAnalysisRequest(BaseModel):
    """气候数据分析请求"""
    data_record_id: str = Field(..., description="数据记录ID")
    analysis_type: AnalysisType = Field(..., description="分析类型")
    parameters: Optional[Dict[str, Any]] = Field(None, description="分析参数")
    variables: Optional[List[str]] = Field(None, description="分析变量")
    time_range: Optional[TimeRangeModel] = Field(None, description="分析时间范围")


class TrendAnalysisResult(BaseModel):
    """趋势分析结果"""
    variable: str = Field(..., description="变量名")
    slope: float = Field(..., description="趋势斜率")
    trend_type: Literal["increasing", "decreasing", "stable"] = Field(..., description="趋势类型")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    r_squared: float = Field(..., ge=0, le=1, description="决定系数")
    p_value: float = Field(..., description="p值")


class AnomalyDetectionResult(BaseModel):
    """异常检测结果"""
    variable: str = Field(..., description="变量名")
    anomaly_count: int = Field(..., description="异常点数量")
    anomaly_percentage: float = Field(..., description="异常点百分比")
    anomaly_indices: List[int] = Field(..., description="异常点索引")
    anomaly_scores: List[float] = Field(..., description="异常分数")
    threshold: float = Field(..., description="异常阈值")


class StatisticsResult(BaseModel):
    """统计分析结果"""
    variable: str = Field(..., description="变量名")
    count: int = Field(..., description="数据点数量")
    mean: float = Field(..., description="均值")
    std: float = Field(..., description="标准差")
    min: float = Field(..., description="最小值")
    max: float = Field(..., description="最大值")
    percentiles: Dict[str, float] = Field(..., description="百分位数")
    missing_count: int = Field(..., description="缺失值数量")
    missing_percentage: float = Field(..., description="缺失值百分比")


class ClimateAnalysisResult(BaseModel):
    """气候分析结果"""
    analysis_type: AnalysisType = Field(..., description="分析类型")
    data_record_id: str = Field(..., description="数据记录ID")
    execution_time: float = Field(..., description="执行时间（秒）")
    trends: Optional[List[TrendAnalysisResult]] = Field(None, description="趋势分析结果")
    anomalies: Optional[List[AnomalyDetectionResult]] = Field(None, description="异常检测结果")
    statistics: Optional[List[StatisticsResult]] = Field(None, description="统计分析结果")
    patterns: Optional[Dict[str, Any]] = Field(None, description="模式识别结果")
    correlations: Optional[Dict[str, float]] = Field(None, description="相关性分析结果")
    metadata: Optional[Dict[str, Any]] = Field(None, description="分析元数据")


class ClimateAnalysisResponse(BaseResponse):
    """气候分析响应"""
    result: ClimateAnalysisResult = Field(..., description="分析结果")


# ==================== 图像生成API模型 ====================

class ImageGenerationModel(str, Enum):
    """图像生成模型枚举"""
    GAN = "gan"
    DIFFUSION = "diffusion"
    HYBRID = "hybrid"


class EnvironmentScenario(str, Enum):
    """环境场景枚举"""
    GLACIER_MELTING = "glacier_melting"
    FOREST_FIRE = "forest_fire"
    FLOOD = "flood"
    DROUGHT = "drought"
    AIR_POLLUTION = "air_pollution"
    DEFORESTATION = "deforestation"
    OCEAN_ACIDIFICATION = "ocean_acidification"
    EXTREME_WEATHER = "extreme_weather"


class EnvironmentIndicators(BaseModel):
    """环境指标"""
    co2_emission: Optional[float] = Field(None, ge=0, description="CO2排放量（吨）")
    temperature_change: Optional[float] = Field(None, description="温度变化（°C）")
    precipitation_change: Optional[float] = Field(None, description="降水变化（%）")
    air_quality_index: Optional[float] = Field(None, ge=0, le=500, description="空气质量指数")
    forest_coverage: Optional[float] = Field(None, ge=0, le=100, description="森林覆盖率（%）")
    sea_level_rise: Optional[float] = Field(None, ge=0, description="海平面上升（米）")
    biodiversity_index: Optional[float] = Field(None, ge=0, le=1, description="生物多样性指数")


class ImageGenerationRequest(BaseModel):
    """图像生成请求"""
    scenario: Optional[EnvironmentScenario] = Field(None, description="预设环境场景")
    indicators: Optional[EnvironmentIndicators] = Field(None, description="环境指标")
    custom_prompt: Optional[str] = Field(None, description="自定义提示词")
    model: ImageGenerationModel = Field(ImageGenerationModel.DIFFUSION, description="生成模型")
    style: Optional[str] = Field(None, description="图像风格")
    resolution: Literal["512x512", "1024x1024", "1536x1536"] = Field("1024x1024", description="图像分辨率")
    num_images: int = Field(1, ge=1, le=4, description="生成图像数量")
    seed: Optional[int] = Field(None, description="随机种子")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="引导强度")
    num_inference_steps: int = Field(50, ge=10, le=100, description="推理步数")


class GeneratedImage(BaseModel):
    """生成的图像"""
    image_id: str = Field(..., description="图像ID")
    image_url: str = Field(..., description="图像URL")
    image_path: str = Field(..., description="图像文件路径")
    thumbnail_url: Optional[str] = Field(None, description="缩略图URL")
    prompt: str = Field(..., description="生成提示词")
    model: str = Field(..., description="使用的模型")
    parameters: Dict[str, Any] = Field(..., description="生成参数")
    generation_time: float = Field(..., description="生成时间（秒）")
    file_size: int = Field(..., description="文件大小（字节）")
    created_at: datetime = Field(..., description="创建时间")


class ImageGenerationResponse(BaseResponse):
    """图像生成响应"""
    images: List[GeneratedImage] = Field(..., description="生成的图像列表")
    total_generation_time: float = Field(..., description="总生成时间（秒）")


class ImageTemplateRequest(BaseModel):
    """图像模板请求"""
    category: Optional[str] = Field(None, description="模板类别")
    scenario: Optional[EnvironmentScenario] = Field(None, description="环境场景")


class ImageTemplate(BaseModel):
    """图像模板"""
    template_id: str = Field(..., description="模板ID")
    name: str = Field(..., description="模板名称")
    description: str = Field(..., description="模板描述")
    category: str = Field(..., description="模板类别")
    scenario: EnvironmentScenario = Field(..., description="环境场景")
    prompt_template: str = Field(..., description="提示词模板")
    default_parameters: Dict[str, Any] = Field(..., description="默认参数")
    preview_image: Optional[str] = Field(None, description="预览图像URL")
    tags: List[str] = Field(..., description="标签")


class ImageTemplateResponse(BaseResponse):
    """图像模板响应"""
    templates: List[ImageTemplate] = Field(..., description="模板列表")


# ==================== 区域预测API模型 ====================

class ClimateScenario(str, Enum):
    """气候情景枚举"""
    RCP26 = "rcp26"
    RCP45 = "rcp45"
    RCP60 = "rcp60"
    RCP85 = "rcp85"
    SSP119 = "ssp119"
    SSP126 = "ssp126"
    SSP245 = "ssp245"
    SSP370 = "ssp370"
    SSP585 = "ssp585"


class PredictionModel(str, Enum):
    """预测模型枚举"""
    TRANSFORMER = "transformer"
    GNN = "gnn"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"


class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class GlobalIndicators(BaseModel):
    """全球指标"""
    global_temperature_anomaly: Optional[float] = Field(None, description="全球温度异常（°C）")
    sea_surface_temperature: Optional[float] = Field(None, description="海表温度（°C）")
    atmospheric_co2: Optional[float] = Field(None, description="大气CO2浓度（ppm）")
    enso_index: Optional[float] = Field(None, description="ENSO指数")
    pdo_index: Optional[float] = Field(None, description="PDO指数")
    arctic_ice_extent: Optional[float] = Field(None, description="北极海冰范围（百万平方公里）")


class RegionalFeatures(BaseModel):
    """区域特征"""
    elevation: Optional[float] = Field(None, description="海拔高度（米）")
    land_use_type: Optional[str] = Field(None, description="土地利用类型")
    distance_to_coast: Optional[float] = Field(None, description="距海岸距离（公里）")
    population_density: Optional[float] = Field(None, description="人口密度（人/平方公里）")
    vegetation_index: Optional[float] = Field(None, description="植被指数")
    soil_type: Optional[str] = Field(None, description="土壤类型")


class RegionalPredictionRequest(BaseModel):
    """区域预测请求"""
    coordinates: CoordinateModel = Field(..., description="预测坐标")
    scenario: ClimateScenario = Field(..., description="气候情景")
    prediction_years: List[int] = Field(..., description="预测年份")
    global_indicators: Optional[GlobalIndicators] = Field(None, description="全球指标")
    regional_features: Optional[RegionalFeatures] = Field(None, description="区域特征")
    model: PredictionModel = Field(PredictionModel.ENSEMBLE, description="预测模型")
    variables: List[str] = Field(["temperature", "precipitation"], description="预测变量")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="置信水平")


class PredictionResult(BaseModel):
    """预测结果"""
    variable: str = Field(..., description="变量名")
    year: int = Field(..., description="预测年份")
    mean_value: float = Field(..., description="预测均值")
    std_value: float = Field(..., description="预测标准差")
    confidence_interval: List[float] = Field(..., description="置信区间 [下界, 上界]")
    risk_level: RiskLevel = Field(..., description="风险等级")
    risk_probability: float = Field(..., ge=0, le=1, description="风险概率")


class ModelPerformance(BaseModel):
    """模型性能"""
    model_name: str = Field(..., description="模型名称")
    mae: float = Field(..., description="平均绝对误差")
    rmse: float = Field(..., description="均方根误差")
    r2_score: float = Field(..., description="R²分数")
    confidence_score: float = Field(..., ge=0, le=1, description="置信度分数")


class RegionalPredictionResult(BaseModel):
    """区域预测结果"""
    coordinates: CoordinateModel = Field(..., description="预测坐标")
    scenario: ClimateScenario = Field(..., description="气候情景")
    model: PredictionModel = Field(..., description="使用的模型")
    predictions: List[PredictionResult] = Field(..., description="预测结果列表")
    model_performance: List[ModelPerformance] = Field(..., description="模型性能")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="特征重要性")
    execution_time: float = Field(..., description="执行时间（秒）")
    metadata: Optional[Dict[str, Any]] = Field(None, description="预测元数据")


class RegionalPredictionResponse(BaseResponse):
    """区域预测响应"""
    result: RegionalPredictionResult = Field(..., description="预测结果")


class HeatmapRequest(BaseModel):
    """热力图请求"""
    scenario: ClimateScenario = Field(..., description="气候情景")
    year: int = Field(..., description="预测年份")
    variable: str = Field(..., description="预测变量")
    region: Optional[str] = Field(None, description="区域名称")
    bbox: Optional[List[float]] = Field(None, description="边界框 [min_lon, min_lat, max_lon, max_lat]")
    resolution: float = Field(0.5, ge=0.1, le=2.0, description="空间分辨率（度）")
    risk_threshold: Optional[float] = Field(None, description="风险阈值")


class HeatmapData(BaseModel):
    """热力图数据"""
    coordinates: List[List[float]] = Field(..., description="坐标网格 [[lat, lon], ...]")
    values: List[float] = Field(..., description="预测值")
    risk_levels: List[RiskLevel] = Field(..., description="风险等级")
    metadata: Dict[str, Any] = Field(..., description="元数据")


class HeatmapResponse(BaseResponse):
    """热力图响应"""
    data: HeatmapData = Field(..., description="热力图数据")
    image_url: Optional[str] = Field(None, description="热力图图像URL")


# ==================== 任务管理API模型 ====================

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """任务类型枚举"""
    DATA_COLLECTION = "data_collection"
    DATA_CLEANING = "data_cleaning"
    DATA_ANALYSIS = "data_analysis"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    IMAGE_GENERATION = "image_generation"


class TaskInfo(BaseModel):
    """任务信息"""
    task_id: str = Field(..., description="任务ID")
    task_type: TaskType = Field(..., description="任务类型")
    name: str = Field(..., description="任务名称")
    description: Optional[str] = Field(None, description="任务描述")
    status: TaskStatus = Field(..., description="任务状态")
    progress: float = Field(..., ge=0, le=100, description="任务进度（%）")
    created_at: datetime = Field(..., description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    execution_time: Optional[float] = Field(None, description="执行时间（秒）")
    error_message: Optional[str] = Field(None, description="错误信息")
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")


class TaskListResponse(PaginatedResponse):
    """任务列表响应"""
    tasks: List[TaskInfo] = Field(..., description="任务列表")


class TaskDetailResponse(BaseResponse):
    """任务详情响应"""
    task: TaskInfo = Field(..., description="任务详情")


# ==================== 系统状态API模型 ====================

class SystemHealth(BaseModel):
    """系统健康状态"""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="系统状态")
    uptime: float = Field(..., description="运行时间（秒）")
    version: str = Field(..., description="系统版本")
    timestamp: datetime = Field(..., description="检查时间")


class DatabaseStatus(BaseModel):
    """数据库状态"""
    postgresql: bool = Field(..., description="PostgreSQL连接状态")
    influxdb: bool = Field(..., description="InfluxDB连接状态")
    redis: bool = Field(..., description="Redis连接状态")


class ServiceStatus(BaseModel):
    """服务状态"""
    data_pipeline: bool = Field(..., description="数据管道状态")
    model_service: bool = Field(..., description="模型服务状态")
    image_generator: bool = Field(..., description="图像生成服务状态")


class SystemMetrics(BaseModel):
    """系统指标"""
    cpu_usage: float = Field(..., description="CPU使用率（%）")
    memory_usage: float = Field(..., description="内存使用率（%）")
    disk_usage: float = Field(..., description="磁盘使用率（%）")
    active_tasks: int = Field(..., description="活跃任务数")
    total_requests: int = Field(..., description="总请求数")
    error_rate: float = Field(..., description="错误率（%）")


class SystemStatusResponse(BaseResponse):
    """系统状态响应"""
    health: SystemHealth = Field(..., description="系统健康状态")
    databases: DatabaseStatus = Field(..., description="数据库状态")
    services: ServiceStatus = Field(..., description="服务状态")
    metrics: SystemMetrics = Field(..., description="系统指标")


# ==================== 错误模型 ====================

class ErrorDetail(BaseModel):
    """错误详情"""
    code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误消息")
    field: Optional[str] = Field(None, description="相关字段")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")


class ErrorResponse(BaseResponse):
    """错误响应"""
    status: Literal[ResponseStatus.ERROR] = ResponseStatus.ERROR
    error: ErrorDetail = Field(..., description="错误信息")
    trace_id: Optional[str] = Field(None, description="追踪ID")


class ValidationErrorResponse(ErrorResponse):
    """验证错误响应"""
    errors: List[ErrorDetail] = Field(..., description="验证错误列表")


# ==================== 文件上传模型 ====================

class FileUploadResponse(BaseResponse):
    """文件上传响应"""
    file_id: str = Field(..., description="文件ID")
    filename: str = Field(..., description="文件名")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(..., description="文件大小（字节）")
    content_type: str = Field(..., description="文件类型")
    upload_time: datetime = Field(..., description="上传时间")
    checksum: Optional[str] = Field(None, description="文件校验和")


class FileInfo(BaseModel):
    """文件信息"""
    file_id: str = Field(..., description="文件ID")
    filename: str = Field(..., description="文件名")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(..., description="文件大小（字节）")
    content_type: str = Field(..., description="文件类型")
    created_at: datetime = Field(..., description="创建时间")
    accessed_at: Optional[datetime] = Field(None, description="最后访问时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="文件元数据")


class FileListResponse(PaginatedResponse):
    """文件列表响应"""
    files: List[FileInfo] = Field(..., description="文件列表")