# 代码质量和可维护性改进建议

基于对当前 Hugging Face 图像生成功能的分析，以下是一些建议来进一步提升代码质量和可维护性。

## 🏗️ 架构改进建议

### 1. 配置管理优化

**当前状态**: 模型配置硬编码在类中
**建议改进**: 使用配置文件管理模型设置

```python
# 创建 config/models.yaml
huggingface_models:
  stable_diffusion_v1_5:
    model_id: "runwayml/stable-diffusion-v1-5"
    description: "Stable Diffusion v1.5 - 通用模型，适合各种场景"
    memory_requirements:
      min_vram: "4GB"
      recommended_vram: "8GB"
    default_params:
      height: 512
      width: 512
      steps: 50
      guidance_scale: 7.5

# 在代码中使用
class ModelConfig:
    @classmethod
    def load_model_configs(cls) -> Dict[str, Any]:
        with open("config/models.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
```

### 2. 依赖注入模式

**当前状态**: 硬编码依赖关系
**建议改进**: 使用依赖注入提高可测试性

```python
from abc import ABC, abstractmethod

class ModelLoader(ABC):
    @abstractmethod
    def load_model(self, model_id: str) -> Any:
        pass

class HuggingFaceModelLoader(ModelLoader):
    def load_model(self, model_id: str) -> Any:
        return StableDiffusionPipeline.from_pretrained(model_id)

class MockModelLoader(ModelLoader):
    def load_model(self, model_id: str) -> Any:
        return MockPipeline()

class EnvironmentalImageGenerator:
    def __init__(self, model_loader: ModelLoader = None):
        self.model_loader = model_loader or HuggingFaceModelLoader()
```

### 3. 策略模式用于生成方法

**当前状态**: 生成逻辑混合在一个方法中
**建议改进**: 使用策略模式分离不同生成策略

```python
class GenerationStrategy(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        pass

class HuggingFaceStrategy(GenerationStrategy):
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Hugging Face 生成逻辑
        pass

class APIStrategy(GenerationStrategy):
    def __init__(self, api_url: str, api_type: str):
        self.api_url = api_url
        self.api_type = api_type
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # API 调用逻辑
        pass

class ExampleStrategy(GenerationStrategy):
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # 示例图像生成逻辑
        pass
```

## 🧪 测试改进建议

### 1. 单元测试覆盖率

**建议**: 为每个核心方法添加单元测试

```python
# tests/test_environmental_image_generator.py
import pytest
from unittest.mock import Mock, patch
from src.models.environmental_image_generator import EnvironmentalImageGenerator

class TestEnvironmentalImageGenerator:
    @pytest.fixture
    def generator(self):
        return EnvironmentalImageGenerator()
    
    def test_list_supported_models(self, generator):
        models = generator.list_supported_models()
        assert isinstance(models, dict)
        assert len(models) > 0
        assert "runwayml/stable-diffusion-v1-5" in models
    
    def test_set_model(self, generator):
        original_model = generator.model_id
        new_model = "stabilityai/stable-diffusion-2-1"
        generator.set_model(new_model)
        assert generator.model_id == new_model
    
    @patch('src.models.environmental_image_generator.requests')
    def test_api_connection_success(self, mock_requests, generator):
        mock_requests.get.return_value.status_code = 200
        result = generator.test_api_connection()
        assert result['success'] is True
        assert result['status_code'] == 200
    
    def test_enhance_environmental_prompt(self, generator):
        original = "城市污染"
        enhanced = generator.enhance_prompt(original)
        assert len(enhanced) > len(original)
        assert "environmental" in enhanced.lower()
```

### 2. 集成测试

```python
# tests/test_integration.py
class TestIntegration:
    def test_full_generation_pipeline(self):
        """测试完整的图像生成流程"""
        generator = EnvironmentalImageGenerator()
        
        # 测试数据
        input_data = {"prompt": "工业污染警示"}
        
        # 执行生成
        result = generator.generate_image(input_data["prompt"])
        
        # 验证结果
        assert "success" in result
        assert "images" in result
        assert result["success"] is True
```

### 3. 性能测试

```python
# tests/test_performance.py
import time
import psutil

class TestPerformance:
    def test_memory_usage(self):
        """测试内存使用情况"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        generator = EnvironmentalImageGenerator()
        # API版本不需要本地模型构建
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # 确保内存增长在合理范围内（例如 < 2GB）
        assert memory_increase < 2 * 1024 * 1024 * 1024
    
    def test_generation_speed(self):
        """测试生成速度"""
        generator = EnvironmentalImageGenerator()
        
        start_time = time.time()
        result = generator.generate_image("测试提示")
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # 确保生成时间在合理范围内
        assert generation_time < 60  # 60秒内完成
```

## 🔧 错误处理改进

### 1. 自定义异常类

```python
# src/exceptions.py
class EnvironmentalImageGeneratorError(Exception):
    """环境图像生成器基础异常"""
    pass

class APIConnectionError(EnvironmentalImageGeneratorError):
    """API连接异常"""
    pass

class GenerationError(EnvironmentalImageGeneratorError):
    """图像生成异常"""
    pass

class NetworkError(EnvironmentalImageGeneratorError):
    """网络连接异常"""
    pass
```

### 2. 重试机制

```python
from functools import wraps
import time

def retry(max_attempts=3, delay=1.0, backoff=2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except (NetworkError, ConnectionError) as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise e
                    
                    logger.warning(f"第 {attempts} 次尝试失败，{current_delay}秒后重试")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator

class EnvironmentalImageGenerator:
    @retry(max_attempts=3, delay=2.0)
    def test_api_connection(self) -> Dict[str, Any]:
        # 现有的API连接逻辑
        pass
```

### 3. 优雅降级

```python
class FallbackManager:
    def __init__(self):
        self.strategies = [
            HuggingFaceStrategy(),
            APIStrategy(),
            ExampleStrategy()
        ]
    
    def generate_with_fallback(self, prompt: str, **kwargs) -> Dict[str, Any]:
        last_error = None
        
        for strategy in self.strategies:
            try:
                result = strategy.generate(prompt, **kwargs)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"策略 {strategy.__class__.__name__} 失败: {e}")
                last_error = e
                continue
        
        # 所有策略都失败
        raise GenerationError(f"所有生成策略都失败，最后错误: {last_error}")
```

## 📊 监控和日志改进

### 1. 结构化日志

```python
import structlog

logger = structlog.get_logger()

class EnvironmentalImageGenerator:
    def generate_image(self, user_input: str, **kwargs) -> Dict[str, Any]:
        logger.info(
            "开始生成图像",
            prompt_length=len(user_input),
            model_id=self.model_id,
            api_endpoint=self.api_url
        )
        
        try:
            # 加载逻辑
            logger.info(
                "模型加载成功",
                model_id=self.model_id,
                load_time=load_time,
                memory_used=self._get_memory_usage()
            )
            return True
        except Exception as e:
            logger.error(
                "模型加载失败",
                model_id=self.model_id,
                error=str(e),
                error_type=type(e).__name__
            )
            return False
```

### 2. 性能指标收集

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class GenerationMetrics:
    model_id: str
    prompt_length: int
    generation_time: float
    memory_peak: int
    success: bool
    error_type: Optional[str] = None

class MetricsCollector:
    def __init__(self):
        self.metrics = []
    
    def record_generation(self, metrics: GenerationMetrics):
        self.metrics.append(metrics)
    
    def get_average_generation_time(self) -> float:
        successful_metrics = [m for m in self.metrics if m.success]
        if not successful_metrics:
            return 0.0
        return sum(m.generation_time for m in successful_metrics) / len(successful_metrics)
    
    def get_success_rate(self) -> float:
        if not self.metrics:
            return 0.0
        successful = sum(1 for m in self.metrics if m.success)
        return successful / len(self.metrics)
```

## 🔒 安全性改进

### 1. 输入验证

```python
from typing import Union
import re

class InputValidator:
    @staticmethod
    def validate_prompt(prompt: str) -> bool:
        """验证提示词安全性"""
        if not isinstance(prompt, str):
            return False
        
        # 长度限制
        if len(prompt) > 1000:
            return False
        
        # 禁止的内容模式
        forbidden_patterns = [
            r'<script.*?>',
            r'javascript:',
            r'data:text/html'
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """清理提示词"""
        # 移除HTML标签
        prompt = re.sub(r'<[^>]+>', '', prompt)
        
        # 移除特殊字符
        prompt = re.sub(r'[<>"\']', '', prompt)
        
        # 限制长度
        return prompt[:500]
```

### 2. 资源限制

```python
class ResourceManager:
    def __init__(self, max_memory_gb: float = 8.0, max_generation_time: int = 300):
        self.max_memory_gb = max_memory_gb
        self.max_generation_time = max_generation_time
    
    def check_memory_usage(self) -> bool:
        """检查内存使用是否超限"""
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024 ** 3)
        return memory_gb < self.max_memory_gb
    
    def with_timeout(self, func, *args, **kwargs):
        """为函数添加超时限制"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"操作超时 ({self.max_generation_time}秒)")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.max_generation_time)
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # 取消超时
            return result
        except TimeoutError:
            signal.alarm(0)
            raise
```

## 📈 可扩展性改进

### 1. 插件系统

```python
from abc import ABC, abstractmethod

class GenerationPlugin(ABC):
    @abstractmethod
    def process_prompt(self, prompt: str) -> str:
        """处理提示词"""
        pass
    
    @abstractmethod
    def process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """处理生成结果"""
        pass

class EnvironmentalEnhancementPlugin(GenerationPlugin):
    def process_prompt(self, prompt: str) -> str:
        return self._enhance_environmental_prompt(prompt)
    
    def process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        result["environmental_score"] = self._calculate_environmental_score(result)
        return result

class PluginManager:
    def __init__(self):
        self.plugins = []
    
    def register_plugin(self, plugin: GenerationPlugin):
        self.plugins.append(plugin)
    
    def process_prompt(self, prompt: str) -> str:
        for plugin in self.plugins:
            prompt = plugin.process_prompt(prompt)
        return prompt
    
    def process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        for plugin in self.plugins:
            result = plugin.process_result(result)
        return result
```

### 2. 异步支持

```python
import asyncio
from typing import AsyncGenerator

class AsyncEnvironmentalImageGenerator:
    async def load_model_async(self, model_id: str) -> bool:
        """异步加载模型"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load_model_from_huggingface, model_id)
    
    async def generate_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """异步生成图像"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, {"prompt": prompt}, **kwargs)
    
    async def batch_generate(self, prompts: List[str], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """批量异步生成"""
        tasks = [self.generate_async(prompt, **kwargs) for prompt in prompts]
        
        for task in asyncio.as_completed(tasks):
            result = await task
            yield result
```

## 🎯 总结

这些改进建议旨在提高代码的：

1. **可维护性** - 通过模块化设计和清晰的架构
2. **可测试性** - 通过依赖注入和全面的测试覆盖
3. **可扩展性** - 通过插件系统和策略模式
4. **可靠性** - 通过错误处理和重试机制
5. **性能** - 通过异步支持和资源管理
6. **安全性** - 通过输入验证和资源限制

建议按优先级逐步实施这些改进，从核心架构开始，然后添加测试，最后实现高级功能。

## 📋 实施计划

### 第一阶段（核心架构）
- [ ] 实现配置管理系统
- [ ] 添加自定义异常类
- [ ] 实现基本的单元测试

### 第二阶段（稳定性）
- [ ] 添加重试机制
- [ ] 实现优雅降级
- [ ] 完善错误处理

### 第三阶段（高级功能）
- [ ] 实现插件系统
- [ ] 添加异步支持
- [ ] 完善监控和日志

### 第四阶段（优化）
- [ ] 性能优化
- [ ] 安全性加固
- [ ] 文档完善

通过这些改进，您的 Hugging Face 图像生成系统将更加健壮、可维护和可扩展。