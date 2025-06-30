"""数据管道模块

实现数据收集、清洗、存储的完整流程管理，支持异步处理和任务调度。
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

try:
    import xarray as xr
except ImportError:
    warnings.warn("xarray未安装，NetCDF数据处理功能受限")
    xr = None

try:
    from kafka import KafkaProducer, KafkaConsumer
except ImportError:
    warnings.warn("kafka-python未安装，消息队列功能受限")
    KafkaProducer = None
    KafkaConsumer = None

from .data_collector import DataCollector
from .data_cleaner import DataCleaner
from .data_storage import DataStorage
from ..utils.config import get_settings
from ..utils.logger import get_logger

logger = get_logger("data_pipeline")
settings = get_settings()


class PipelineStatus(Enum):
    """管道状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """任务类型枚举"""
    DATA_COLLECTION = "data_collection"
    DATA_CLEANING = "data_cleaning"
    DATA_STORAGE = "data_storage"
    DATA_ANALYSIS = "data_analysis"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"


@dataclass
class PipelineTask:
    """管道任务定义"""
    task_id: str
    task_type: TaskType
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # 依赖的任务ID
    status: PipelineStatus = PipelineStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    progress: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None  # 超时时间（秒）
    priority: int = 0  # 优先级（数字越大优先级越高）


@dataclass
class PipelineConfig:
    """管道配置"""
    name: str
    description: str = ""
    max_concurrent_tasks: int = 5
    enable_kafka: bool = False
    kafka_topic: str = "climate_data_pipeline"
    retry_delay: int = 60  # 重试延迟（秒）
    checkpoint_interval: int = 300  # 检查点间隔（秒）
    enable_monitoring: bool = True
    data_retention_days: int = 30


class DataPipeline:
    """数据管道管理器
    
    协调数据收集、清洗、存储等各个环节，支持任务调度和监控。
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tasks: Dict[str, PipelineTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        
        # 组件初始化
        self.data_collector = DataCollector()
        self.data_cleaner = DataCleaner()
        self.data_storage = DataStorage()
        
        # 执行器
        self.thread_executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # Kafka相关
        self.kafka_producer = None
        self.kafka_consumer = None
        
        # 状态管理
        self.is_running = False
        self.start_time = None
        self.last_checkpoint = None
        
        # 监控数据
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "average_task_time": 0.0,
            "data_processed_mb": 0.0
        }
    
    async def initialize(self) -> None:
        """初始化管道"""
        logger.info(f"初始化数据管道: {self.config.name}")
        
        try:
            # 初始化存储
            await self.data_storage.initialize()
            
            # 初始化Kafka（如果启用）
            if self.config.enable_kafka and KafkaProducer:
                await self._initialize_kafka()
            
            logger.info("数据管道初始化完成")
            
        except Exception as e:
            logger.error(f"数据管道初始化失败: {e}")
            raise
    
    async def _initialize_kafka(self) -> None:
        """初始化Kafka连接"""
        try:
            # 创建生产者
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            # 创建消费者
            self.kafka_consumer = KafkaConsumer(
                self.config.kafka_topic,
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                group_id=f"pipeline_{self.config.name}"
            )
            
            logger.info("Kafka连接已建立")
            
        except Exception as e:
            logger.error(f"Kafka初始化失败: {e}")
            self.config.enable_kafka = False
    
    async def close(self) -> None:
        """关闭管道"""
        logger.info("关闭数据管道...")
        
        # 停止运行
        self.is_running = False
        
        # 等待运行中的任务完成
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        # 关闭执行器
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        # 关闭Kafka连接
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.kafka_consumer:
            self.kafka_consumer.close()
        
        # 关闭存储
        await self.data_storage.close()
        
        logger.info("数据管道已关闭")
    
    # ==================== 任务管理 ====================
    
    def add_task(
        self,
        task_id: str,
        task_type: TaskType,
        name: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        priority: int = 0,
        timeout: Optional[int] = None,
        max_retries: int = 3
    ) -> PipelineTask:
        """添加任务到管道
        
        Args:
            task_id: 任务ID
            task_type: 任务类型
            name: 任务名称
            description: 任务描述
            parameters: 任务参数
            dependencies: 依赖任务列表
            priority: 优先级
            timeout: 超时时间
            max_retries: 最大重试次数
            
        Returns:
            创建的任务对象
        """
        if task_id in self.tasks:
            raise ValueError(f"任务ID已存在: {task_id}")
        
        task = PipelineTask(
            task_id=task_id,
            task_type=task_type,
            name=name,
            description=description,
            parameters=parameters or {},
            dependencies=dependencies or [],
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )
        
        self.tasks[task_id] = task
        logger.info(f"任务已添加: {task_id} ({task_type.value})")
        
        return task
    
    def remove_task(self, task_id: str) -> bool:
        """移除任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功移除
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # 检查任务状态
        if task.status == PipelineStatus.RUNNING:
            logger.warning(f"无法移除正在运行的任务: {task_id}")
            return False
        
        # 检查依赖关系
        dependent_tasks = [t for t in self.tasks.values() if task_id in t.dependencies]
        if dependent_tasks:
            logger.warning(f"无法移除任务 {task_id}，存在依赖任务: {[t.task_id for t in dependent_tasks]}")
            return False
        
        del self.tasks[task_id]
        logger.info(f"任务已移除: {task_id}")
        
        return True
    
    def get_task(self, task_id: str) -> Optional[PipelineTask]:
        """获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务对象
        """
        return self.tasks.get(task_id)
    
    def get_tasks_by_status(self, status: PipelineStatus) -> List[PipelineTask]:
        """根据状态获取任务列表
        
        Args:
            status: 任务状态
            
        Returns:
            任务列表
        """
        return [task for task in self.tasks.values() if task.status == status]
    
    def get_ready_tasks(self) -> List[PipelineTask]:
        """获取准备执行的任务
        
        Returns:
            准备执行的任务列表
        """
        ready_tasks = []
        
        for task in self.tasks.values():
            if task.status != PipelineStatus.PENDING:
                continue
            
            # 检查依赖是否完成
            dependencies_completed = all(
                self.tasks.get(dep_id, {}).status == PipelineStatus.COMPLETED
                for dep_id in task.dependencies
            )
            
            if dependencies_completed:
                ready_tasks.append(task)
        
        # 按优先级排序
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        return ready_tasks
    
    # ==================== 管道执行 ====================
    
    async def run(self) -> None:
        """运行管道"""
        if self.is_running:
            logger.warning("管道已在运行中")
            return
        
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        self.last_checkpoint = self.start_time
        
        logger.info(f"开始运行数据管道: {self.config.name}")
        
        try:
            # 启动任务调度器
            scheduler_task = asyncio.create_task(self._task_scheduler())
            
            # 启动监控器（如果启用）
            monitor_task = None
            if self.config.enable_monitoring:
                monitor_task = asyncio.create_task(self._monitor())
            
            # 启动Kafka消费者（如果启用）
            kafka_task = None
            if self.config.enable_kafka and self.kafka_consumer:
                kafka_task = asyncio.create_task(self._kafka_consumer_loop())
            
            # 等待所有任务完成或管道停止
            tasks_to_wait = [scheduler_task]
            if monitor_task:
                tasks_to_wait.append(monitor_task)
            if kafka_task:
                tasks_to_wait.append(kafka_task)
            
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"管道运行异常: {e}")
            raise
        finally:
            self.is_running = False
            logger.info("数据管道已停止")
    
    async def _task_scheduler(self) -> None:
        """任务调度器"""
        logger.info("任务调度器已启动")
        
        while self.is_running:
            try:
                # 获取准备执行的任务
                ready_tasks = self.get_ready_tasks()
                
                # 检查并发限制
                available_slots = self.config.max_concurrent_tasks - len(self.running_tasks)
                
                if ready_tasks and available_slots > 0:
                    # 选择要执行的任务
                    tasks_to_run = ready_tasks[:available_slots]
                    
                    for task in tasks_to_run:
                        await self._start_task(task)
                
                # 清理已完成的任务
                await self._cleanup_completed_tasks()
                
                # 检查是否所有任务都已完成
                if self._all_tasks_completed():
                    logger.info("所有任务已完成，管道即将停止")
                    break
                
                # 等待一段时间再检查
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"任务调度器异常: {e}")
                await asyncio.sleep(5)
    
    async def _start_task(self, task: PipelineTask) -> None:
        """启动任务
        
        Args:
            task: 要启动的任务
        """
        task.status = PipelineStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        
        logger.info(f"启动任务: {task.task_id} ({task.task_type.value})")
        
        # 创建任务协程
        task_coroutine = self._execute_task(task)
        
        # 添加超时控制
        if task.timeout:
            task_coroutine = asyncio.wait_for(task_coroutine, timeout=task.timeout)
        
        # 启动任务
        async_task = asyncio.create_task(task_coroutine)
        self.running_tasks[task.task_id] = async_task
        
        # 发送Kafka消息（如果启用）
        if self.config.enable_kafka and self.kafka_producer:
            await self._send_kafka_message("task_started", {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "started_at": task.started_at.isoformat()
            })
    
    async def _execute_task(self, task: PipelineTask) -> None:
        """执行任务
        
        Args:
            task: 要执行的任务
        """
        try:
            # 根据任务类型执行相应的处理
            if task.task_type == TaskType.DATA_COLLECTION:
                result = await self._execute_data_collection(task)
            elif task.task_type == TaskType.DATA_CLEANING:
                result = await self._execute_data_cleaning(task)
            elif task.task_type == TaskType.DATA_STORAGE:
                result = await self._execute_data_storage(task)
            elif task.task_type == TaskType.DATA_ANALYSIS:
                result = await self._execute_data_analysis(task)
            else:
                raise ValueError(f"不支持的任务类型: {task.task_type}")
            
            # 任务成功完成
            task.status = PipelineStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            task.result = result
            task.progress = 100.0
            
            # 更新指标
            self.metrics["tasks_completed"] += 1
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self.metrics["total_processing_time"] += execution_time
            self.metrics["average_task_time"] = (
                self.metrics["total_processing_time"] / self.metrics["tasks_completed"]
            )
            
            logger.info(f"任务完成: {task.task_id}, 耗时: {execution_time:.2f}秒")
            
            # 发送Kafka消息
            if self.config.enable_kafka and self.kafka_producer:
                await self._send_kafka_message("task_completed", {
                    "task_id": task.task_id,
                    "completed_at": task.completed_at.isoformat(),
                    "execution_time": execution_time,
                    "result": result
                })
            
        except asyncio.TimeoutError:
            # 任务超时
            task.status = PipelineStatus.FAILED
            task.completed_at = datetime.now(timezone.utc)
            task.error_message = "任务执行超时"
            
            logger.error(f"任务超时: {task.task_id}")
            
            self.metrics["tasks_failed"] += 1
            
        except Exception as e:
            # 任务执行失败
            task.status = PipelineStatus.FAILED
            task.completed_at = datetime.now(timezone.utc)
            task.error_message = str(e)
            
            logger.error(f"任务执行失败: {task.task_id}, 错误: {e}")
            
            self.metrics["tasks_failed"] += 1
            
            # 检查是否需要重试
            if task.retry_count < task.max_retries:
                await self._schedule_retry(task)
            
            # 发送Kafka消息
            if self.config.enable_kafka and self.kafka_producer:
                await self._send_kafka_message("task_failed", {
                    "task_id": task.task_id,
                    "error_message": task.error_message,
                    "retry_count": task.retry_count
                })
    
    async def _schedule_retry(self, task: PipelineTask) -> None:
        """安排任务重试
        
        Args:
            task: 要重试的任务
        """
        task.retry_count += 1
        task.status = PipelineStatus.PENDING
        task.started_at = None
        task.completed_at = None
        task.error_message = None
        task.progress = 0.0
        
        logger.info(f"安排任务重试: {task.task_id}, 重试次数: {task.retry_count}")
        
        # 延迟重试
        await asyncio.sleep(self.config.retry_delay)
    
    async def _cleanup_completed_tasks(self) -> None:
        """清理已完成的任务"""
        completed_task_ids = []
        
        for task_id, async_task in self.running_tasks.items():
            if async_task.done():
                completed_task_ids.append(task_id)
        
        for task_id in completed_task_ids:
            del self.running_tasks[task_id]
    
    def _all_tasks_completed(self) -> bool:
        """检查是否所有任务都已完成
        
        Returns:
            是否所有任务都已完成
        """
        if not self.tasks:
            return True
        
        for task in self.tasks.values():
            if task.status in [PipelineStatus.PENDING, PipelineStatus.RUNNING]:
                return False
        
        return True
    
    # ==================== 任务执行器 ====================
    
    async def _execute_data_collection(self, task: PipelineTask) -> Dict[str, Any]:
        """执行数据收集任务
        
        Args:
            task: 数据收集任务
            
        Returns:
            执行结果
        """
        params = task.parameters
        source = params.get("source")
        data_type = params.get("data_type")
        location = params.get("location")
        start_date = params.get("start_date")
        end_date = params.get("end_date")
        
        logger.info(f"开始数据收集: {source} - {data_type}")
        
        # 执行数据收集
        if source == "noaa":
            data = await self.data_collector.collect_noaa_data(
                data_type=data_type,
                location=location,
                start_date=start_date,
                end_date=end_date
            )
        elif source == "ecmwf":
            data = await self.data_collector.collect_ecmwf_data(
                data_type=data_type,
                location=location,
                start_date=start_date,
                end_date=end_date
            )
        elif source == "satellite":
            data = await self.data_collector.collect_satellite_data(
                data_type=data_type,
                location=location,
                start_date=start_date,
                end_date=end_date
            )
        else:
            raise ValueError(f"不支持的数据源: {source}")
        
        # 保存原始数据
        if data is not None and not data.empty:
            filename = f"{source}_{data_type}_{task.task_id}"
            file_path = self.data_storage.save_dataframe(
                data=data,
                filename=filename,
                data_category="raw"
            )
            
            # 保存数据记录
            record_id = self.data_storage.save_data_record(
                source=source,
                data_type=data_type,
                file_path=file_path,
                location=location,
                variables=list(data.columns),
                metadata={
                    "task_id": task.task_id,
                    "collection_time": datetime.now(timezone.utc).isoformat(),
                    "row_count": len(data),
                    "column_count": len(data.columns)
                }
            )
            
            self.metrics["data_processed_mb"] += data.memory_usage(deep=True).sum() / (1024 * 1024)
            
            return {
                "status": "success",
                "record_id": record_id,
                "file_path": file_path,
                "data_shape": data.shape,
                "data_size_mb": data.memory_usage(deep=True).sum() / (1024 * 1024)
            }
        else:
            return {
                "status": "no_data",
                "message": "未收集到数据"
            }
    
    async def _execute_data_cleaning(self, task: PipelineTask) -> Dict[str, Any]:
        """执行数据清洗任务
        
        Args:
            task: 数据清洗任务
            
        Returns:
            执行结果
        """
        params = task.parameters
        input_record_id = params.get("input_record_id")
        cleaning_config = params.get("cleaning_config", {})
        
        logger.info(f"开始数据清洗: {input_record_id}")
        
        # 获取输入数据
        record = self.data_storage.get_data_record(input_record_id)
        if not record:
            raise ValueError(f"未找到数据记录: {input_record_id}")
        
        # 加载数据
        data = self.data_storage.load_dataframe(record["file_path"])
        if data is None:
            raise ValueError(f"无法加载数据文件: {record['file_path']}")
        
        # 执行数据清洗
        original_shape = data.shape
        
        # 基础清洗
        if cleaning_config.get("remove_duplicates", True):
            data = self.data_cleaner.remove_duplicates(data)
        
        if cleaning_config.get("handle_missing", True):
            missing_strategy = cleaning_config.get("missing_strategy", "interpolate")
            data = self.data_cleaner.handle_missing_values(
                data, strategy=missing_strategy
            )
        
        if cleaning_config.get("detect_outliers", True):
            outlier_method = cleaning_config.get("outlier_method", "iqr")
            data = self.data_cleaner.detect_and_handle_outliers(
                data, method=outlier_method
            )
        
        # 数据验证
        if cleaning_config.get("validate_ranges", True):
            range_config = cleaning_config.get("range_config", {})
            data = self.data_cleaner.validate_data_ranges(data, range_config)
        
        # 时间序列规整
        if cleaning_config.get("regularize_time", True):
            freq = cleaning_config.get("time_frequency", "1H")
            data = self.data_cleaner.regularize_time_series(data, freq=freq)
        
        # 保存清洗后的数据
        filename = f"cleaned_{record['source']}_{record['data_type']}_{task.task_id}"
        file_path = self.data_storage.save_dataframe(
            data=data,
            filename=filename,
            data_category="processed"
        )
        
        # 生成质量报告
        quality_report = self.data_cleaner.generate_quality_report(data)
        
        # 保存清洗后的数据记录
        cleaned_record_id = self.data_storage.save_data_record(
            source=record["source"],
            data_type=f"cleaned_{record['data_type']}",
            file_path=file_path,
            location=record["location"],
            variables=list(data.columns),
            metadata={
                "task_id": task.task_id,
                "original_record_id": input_record_id,
                "cleaning_time": datetime.now(timezone.utc).isoformat(),
                "original_shape": original_shape,
                "cleaned_shape": data.shape,
                "cleaning_config": cleaning_config,
                "quality_report": quality_report
            },
            quality_score=quality_report.get("overall_score", 0.0)
        )
        
        return {
            "status": "success",
            "cleaned_record_id": cleaned_record_id,
            "file_path": file_path,
            "original_shape": original_shape,
            "cleaned_shape": data.shape,
            "quality_report": quality_report
        }
    
    async def _execute_data_storage(self, task: PipelineTask) -> Dict[str, Any]:
        """执行数据存储任务
        
        Args:
            task: 数据存储任务
            
        Returns:
            执行结果
        """
        params = task.parameters
        input_record_id = params.get("input_record_id")
        storage_type = params.get("storage_type", "time_series")
        
        logger.info(f"开始数据存储: {input_record_id} -> {storage_type}")
        
        # 获取输入数据
        record = self.data_storage.get_data_record(input_record_id)
        if not record:
            raise ValueError(f"未找到数据记录: {input_record_id}")
        
        # 加载数据
        data = self.data_storage.load_dataframe(record["file_path"])
        if data is None:
            raise ValueError(f"无法加载数据文件: {record['file_path']}")
        
        result = {"status": "success"}
        
        # 根据存储类型执行相应操作
        if storage_type == "time_series":
            # 存储到InfluxDB
            measurement = params.get("measurement", record["data_type"])
            tags = params.get("tags", {
                "source": record["source"],
                "location": record["location"]
            })
            
            success = self.data_storage.save_time_series_data(
                measurement=measurement,
                data=data,
                tags=tags
            )
            
            result["influxdb_stored"] = success
        
        elif storage_type == "cache":
            # 存储到缓存
            cache_key = params.get("cache_key", f"data_{input_record_id}")
            expire = params.get("cache_expire", 3600)
            
            success = self.data_storage.cache_set(
                key=cache_key,
                value=data.to_dict(),
                expire=expire
            )
            
            result["cache_stored"] = success
            result["cache_key"] = cache_key
        
        elif storage_type == "archive":
            # 归档存储
            archive_format = params.get("archive_format", "parquet")
            
            filename = f"archive_{record['source']}_{record['data_type']}_{task.task_id}"
            archive_path = self.data_storage.save_dataframe(
                data=data,
                filename=filename,
                data_category="processed",
                format=archive_format
            )
            
            result["archive_path"] = archive_path
        
        return result
    
    async def _execute_data_analysis(self, task: PipelineTask) -> Dict[str, Any]:
        """执行数据分析任务
        
        Args:
            task: 数据分析任务
            
        Returns:
            执行结果
        """
        params = task.parameters
        input_record_id = params.get("input_record_id")
        analysis_type = params.get("analysis_type")
        
        logger.info(f"开始数据分析: {input_record_id} -> {analysis_type}")
        
        # 获取输入数据
        record = self.data_storage.get_data_record(input_record_id)
        if not record:
            raise ValueError(f"未找到数据记录: {input_record_id}")
        
        # 加载数据
        data = self.data_storage.load_dataframe(record["file_path"])
        if data is None:
            raise ValueError(f"无法加载数据文件: {record['file_path']}")
        
        result = {"status": "success", "analysis_type": analysis_type}
        
        # 根据分析类型执行相应操作
        if analysis_type == "statistics":
            # 统计分析
            stats = {
                "shape": data.shape,
                "dtypes": data.dtypes.to_dict(),
                "describe": data.describe().to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "correlation": data.corr().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 1 else {}
            }
            result["statistics"] = stats
        
        elif analysis_type == "trend":
            # 趋势分析
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            trends = {}
            
            for column in numeric_columns:
                if data[column].notna().sum() > 10:  # 至少需要10个有效数据点
                    # 简单线性趋势
                    x = np.arange(len(data))
                    y = data[column].fillna(method='ffill').fillna(method='bfill')
                    
                    if len(y) > 0:
                        slope = np.polyfit(x, y, 1)[0]
                        trends[column] = {
                            "slope": float(slope),
                            "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                        }
            
            result["trends"] = trends
        
        elif analysis_type == "quality":
            # 数据质量分析
            quality_report = self.data_cleaner.generate_quality_report(data)
            result["quality_report"] = quality_report
        
        return result
    
    # ==================== Kafka集成 ====================
    
    async def _send_kafka_message(self, event_type: str, data: Dict[str, Any]) -> None:
        """发送Kafka消息
        
        Args:
            event_type: 事件类型
            data: 消息数据
        """
        if not self.kafka_producer:
            return
        
        try:
            message = {
                "event_type": event_type,
                "pipeline_name": self.config.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data
            }
            
            self.kafka_producer.send(
                self.config.kafka_topic,
                key=event_type,
                value=message
            )
            
        except Exception as e:
            logger.error(f"发送Kafka消息失败: {e}")
    
    async def _kafka_consumer_loop(self) -> None:
        """Kafka消费者循环"""
        logger.info("Kafka消费者已启动")
        
        try:
            while self.is_running:
                # 获取消息
                message_pack = self.kafka_consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        await self._process_kafka_message(message.value)
                
        except Exception as e:
            logger.error(f"Kafka消费者异常: {e}")
    
    async def _process_kafka_message(self, message: Dict[str, Any]) -> None:
        """处理Kafka消息
        
        Args:
            message: 消息内容
        """
        try:
            event_type = message.get("event_type")
            data = message.get("data", {})
            
            logger.info(f"收到Kafka消息: {event_type}")
            
            # 根据事件类型处理消息
            if event_type == "add_task":
                # 动态添加任务
                self.add_task(**data)
            elif event_type == "cancel_task":
                # 取消任务
                task_id = data.get("task_id")
                if task_id in self.tasks:
                    self.tasks[task_id].status = PipelineStatus.CANCELLED
            
        except Exception as e:
            logger.error(f"处理Kafka消息失败: {e}")
    
    # ==================== 监控和指标 ====================
    
    async def _monitor(self) -> None:
        """监控管道状态"""
        logger.info("管道监控器已启动")
        
        while self.is_running:
            try:
                # 检查点操作
                now = datetime.now(timezone.utc)
                if (now - self.last_checkpoint).total_seconds() >= self.config.checkpoint_interval:
                    await self._create_checkpoint()
                    self.last_checkpoint = now
                
                # 清理过期数据
                await self._cleanup_expired_data()
                
                # 等待下次检查
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"监控器异常: {e}")
                await asyncio.sleep(60)
    
    async def _create_checkpoint(self) -> None:
        """创建检查点"""
        try:
            checkpoint_data = {
                "pipeline_name": self.config.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tasks": {
                    task_id: {
                        "status": task.status.value,
                        "progress": task.progress,
                        "retry_count": task.retry_count
                    }
                    for task_id, task in self.tasks.items()
                },
                "metrics": self.metrics.copy(),
                "running_tasks": list(self.running_tasks.keys())
            }
            
            # 保存检查点到缓存
            checkpoint_key = f"checkpoint_{self.config.name}_{int(datetime.now(timezone.utc).timestamp())}"
            self.data_storage.cache_set(
                key=checkpoint_key,
                value=checkpoint_data,
                expire=86400  # 24小时
            )
            
            logger.info(f"检查点已创建: {checkpoint_key}")
            
        except Exception as e:
            logger.error(f"创建检查点失败: {e}")
    
    async def _cleanup_expired_data(self) -> None:
        """清理过期数据"""
        try:
            # 清理过期的缓存数据
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.data_retention_days)
            
            # 这里可以添加具体的清理逻辑
            # 例如删除过期的临时文件、缓存记录等
            
            logger.debug("过期数据清理完成")
            
        except Exception as e:
            logger.error(f"清理过期数据失败: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """获取管道状态
        
        Returns:
            管道状态字典
        """
        status = {
            "name": self.config.name,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (
                (datetime.now(timezone.utc) - self.start_time).total_seconds()
                if self.start_time else 0
            ),
            "task_counts": {
                "total": len(self.tasks),
                "pending": len(self.get_tasks_by_status(PipelineStatus.PENDING)),
                "running": len(self.get_tasks_by_status(PipelineStatus.RUNNING)),
                "completed": len(self.get_tasks_by_status(PipelineStatus.COMPLETED)),
                "failed": len(self.get_tasks_by_status(PipelineStatus.FAILED)),
                "cancelled": len(self.get_tasks_by_status(PipelineStatus.CANCELLED))
            },
            "running_tasks": list(self.running_tasks.keys()),
            "metrics": self.metrics.copy(),
            "config": {
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "enable_kafka": self.config.enable_kafka,
                "enable_monitoring": self.config.enable_monitoring
            }
        }
        
        return status
    
    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务详细信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务详细信息
        """
        task = self.get_task(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "name": task.name,
            "description": task.description,
            "status": task.status.value,
            "progress": task.progress,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "execution_time": (
                (task.completed_at - task.started_at).total_seconds()
                if task.started_at and task.completed_at else None
            ),
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "dependencies": task.dependencies,
            "parameters": task.parameters,
            "result": task.result,
            "error_message": task.error_message
        }


# ==================== 预定义管道配置 ====================

def create_climate_data_pipeline() -> DataPipeline:
    """创建气候数据处理管道
    
    Returns:
        配置好的数据管道
    """
    config = PipelineConfig(
        name="climate_data_pipeline",
        description="气候数据收集、清洗和存储管道",
        max_concurrent_tasks=3,
        enable_kafka=True,
        enable_monitoring=True
    )
    
    pipeline = DataPipeline(config)
    
    # 添加数据收集任务
    pipeline.add_task(
        task_id="collect_noaa_temperature",
        task_type=TaskType.DATA_COLLECTION,
        name="收集NOAA温度数据",
        description="从NOAA收集历史温度数据",
        parameters={
            "source": "noaa",
            "data_type": "temperature",
            "location": "global",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31"
        },
        priority=10
    )
    
    # 添加数据清洗任务
    pipeline.add_task(
        task_id="clean_temperature_data",
        task_type=TaskType.DATA_CLEANING,
        name="清洗温度数据",
        description="清洗和质量控制温度数据",
        parameters={
            "input_record_id": "collect_noaa_temperature",  # 将在运行时动态设置
            "cleaning_config": {
                "remove_duplicates": True,
                "handle_missing": True,
                "missing_strategy": "interpolate",
                "detect_outliers": True,
                "outlier_method": "iqr",
                "validate_ranges": True,
                "range_config": {
                    "temperature": {"min": -50, "max": 60}
                }
            }
        },
        dependencies=["collect_noaa_temperature"],
        priority=8
    )
    
    # 添加数据存储任务
    pipeline.add_task(
        task_id="store_temperature_data",
        task_type=TaskType.DATA_STORAGE,
        name="存储温度数据",
        description="将清洗后的温度数据存储到时序数据库",
        parameters={
            "input_record_id": "clean_temperature_data",  # 将在运行时动态设置
            "storage_type": "time_series",
            "measurement": "temperature",
            "tags": {
                "source": "noaa",
                "data_type": "temperature"
            }
        },
        dependencies=["clean_temperature_data"],
        priority=5
    )
    
    return pipeline


def create_model_training_pipeline() -> DataPipeline:
    """创建模型训练管道
    
    Returns:
        配置好的模型训练管道
    """
    config = PipelineConfig(
        name="model_training_pipeline",
        description="AI模型训练和评估管道",
        max_concurrent_tasks=2,
        enable_kafka=True,
        enable_monitoring=True
    )
    
    pipeline = DataPipeline(config)
    
    # 这里可以添加模型训练相关的任务
    # 例如数据预处理、特征工程、模型训练、模型评估等
    
    return pipeline