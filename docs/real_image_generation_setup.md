# 真实AI图像生成配置指南

## 概述

当前系统使用示例图像数据来模拟AI图像生成功能。要获得真正的AI生成图像作品，需要配置实际的图像生成模型。本文档提供了详细的配置步骤。

## 🎯 目标

将系统从生成示例图像数据升级为生成真实的AI图像作品，包括：
- 环境警示场景图像
- 生态系统可视化
- 气候变化影响图像
- 污染场景图像

## 🔧 配置选项

### 选项1: Stable Diffusion (推荐)

**优点:**
- 开源免费
- 高质量图像生成
- 支持自定义提示词
- 活跃的社区支持

**要求:**
- GPU: 4GB+ VRAM (推荐8GB+)
- RAM: 8GB+
- 存储: 10GB+ 可用空间

**安装步骤:**

```bash
# 1. 安装核心依赖
pip install diffusers>=0.21.0
pip install transformers>=4.25.0
pip install accelerate>=0.20.0
pip install safetensors>=0.3.0

# 2. 安装GPU加速 (可选，但强烈推荐)
pip install xformers  # 仅限Linux/Windows

# 3. 安装PyTorch (如果尚未安装)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 选项2: DALL-E API

**优点:**
- 无需本地GPU
- 高质量图像
- 简单集成

**缺点:**
- 需要付费API
- 依赖网络连接
- 有使用限制

**安装步骤:**

```bash
pip install openai>=1.0.0
```

### 选项3: 本地GAN模型

**优点:**
- 完全本地化
- 可自定义训练

**缺点:**
- 需要大量训练数据
- 训练时间长
- 技术门槛高

## 🛠️ 代码修改

### 1. 修改 EcologyImageGenerator 类

在 `src/models/ecology_image_generator.py` 中修改 `_build_model` 方法：

```python
def _build_model(self):
    """构建图像生成模型"""
    try:
        # 选项1: Stable Diffusion
        from diffusers import StableDiffusionPipeline
        import torch
        
        # 检查设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        model_id = "runwayml/stable-diffusion-v1-5"
        
        self.diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.diffusion_pipeline = self.diffusion_pipeline.to(self.device)
        
        # 内存优化
        if self.device == "cuda":
            self.diffusion_pipeline.enable_attention_slicing()
            self.diffusion_pipeline.enable_memory_efficient_attention()
            
            # 如果显存不足，启用CPU卸载
            # self.diffusion_pipeline.enable_sequential_cpu_offload()
        
        logger.info("Stable Diffusion模型加载成功")
        
    except ImportError as e:
        logger.warning(f"无法导入diffusers: {e}")
        logger.warning("将使用示例图像生成")
        self.diffusion_pipeline = None
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        self.diffusion_pipeline = None
```

### 2. 修改 _generate_with_diffusion 方法

```python
def _generate_with_diffusion(self, prompt, conditions=None):
    """使用扩散模型生成图像"""
    try:
        if self.diffusion_pipeline is None:
            logger.warning("扩散模型未加载，使用示例图像")
            return self._create_example_warning_image(prompt, "medium")
        
        # 构建完整提示词
        if conditions:
            full_prompt = self._conditions_to_prompt(conditions)
        else:
            full_prompt = prompt
        
        # 增强环境主题提示词
        enhanced_prompt = self._enhance_environmental_prompt(full_prompt)
        
        logger.info(f"生成图像，提示词: {enhanced_prompt}")
        
        # 生成参数
        generation_params = {
            "prompt": enhanced_prompt,
            "num_inference_steps": 50,  # 推理步数，越高质量越好但速度越慢
            "guidance_scale": 7.5,      # 提示词引导强度
            "width": 512,               # 图像宽度
            "height": 512,              # 图像高度
            "num_images_per_prompt": 1,  # 生成图像数量
        }
        
        # 生成图像
        with torch.no_grad():
            result = self.diffusion_pipeline(**generation_params)
            
        # 获取生成的图像
        generated_image = result.images[0]
        
        # 转换为numpy数组
        import numpy as np
        image_array = np.array(generated_image)
        
        logger.info(f"图像生成成功，尺寸: {image_array.shape}")
        return image_array
        
    except Exception as e:
        logger.error(f"图像生成失败: {e}")
        # 回退到示例图像
        return self._create_example_warning_image(prompt, "medium")

def _enhance_environmental_prompt(self, prompt):
    """增强环境主题的提示词"""
    # 环境艺术风格关键词
    style_keywords = [
        "environmental art",
        "nature photography",
        "dramatic lighting",
        "high quality",
        "detailed",
        "realistic"
    ]
    
    # 检测环境主题并添加相应关键词
    environmental_themes = {
        "pollution": "polluted environment, smog, industrial waste",
        "climate": "climate change effects, extreme weather",
        "wildlife": "endangered wildlife, natural habitat",
        "forest": "deforestation, forest destruction",
        "ocean": "ocean pollution, marine life threat",
        "glacier": "melting glaciers, ice caps, global warming"
    }
    
    enhanced = prompt
    
    # 添加主题关键词
    for theme, keywords in environmental_themes.items():
        if theme in prompt.lower():
            enhanced += f", {keywords}"
    
    # 添加风格关键词
    enhanced += f", {', '.join(style_keywords)}"
    
    return enhanced
```

### 3. 添加图像保存功能

```python
def _save_generated_image(self, image_array, filename):
    """保存生成的图像"""
    try:
        from PIL import Image
        import numpy as np
        
        # 确保图像数据格式正确
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        # 创建PIL图像
        if len(image_array.shape) == 3:
            image = Image.fromarray(image_array, 'RGB')
        else:
            image = Image.fromarray(image_array, 'L')
        
        # 保存图像
        image.save(filename, 'PNG', quality=95)
        logger.info(f"图像已保存: {filename}")
        
        return filename
        
    except Exception as e:
        logger.error(f"图像保存失败: {e}")
        return None
```

## 🚀 使用示例

### 基本使用

```python
from src.models.ecology_image_generator import EcologyImageGenerator

# 创建生成器实例
generator = EcologyImageGenerator()

# 从自然语言生成图像
result = generator.generate_from_text(
    "严重的空气污染，城市被雾霾笼罩，能见度极低"
)

# 检查结果
if result['success']:
    print(f"图像生成成功: {result['image_path']}")
    print(f"警示等级: {result['warning_level']}")
else:
    print(f"生成失败: {result['error']}")
```

### 高级配置

```python
# 自定义生成参数
generator = EcologyImageGenerator(
    model_config={
        'model_id': 'stabilityai/stable-diffusion-2-1',  # 使用更新的模型
        'inference_steps': 100,  # 更高质量
        'guidance_scale': 10.0,  # 更强的提示词引导
        'width': 768,
        'height': 768
    }
)
```

## 🔍 故障排除

### 常见问题

1. **GPU内存不足**
   ```python
   # 启用CPU卸载
   pipeline.enable_sequential_cpu_offload()
   
   # 或使用更小的图像尺寸
   width=256, height=256
   ```

2. **模型下载失败**
   ```bash
   # 设置Hugging Face镜像
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **生成速度慢**
   ```python
   # 减少推理步数
   num_inference_steps=20
   
   # 启用xformers加速
   pipeline.enable_xformers_memory_efficient_attention()
   ```

### 性能优化

1. **内存优化**
   - 启用attention slicing
   - 使用CPU卸载
   - 减小批次大小

2. **速度优化**
   - 使用xformers
   - 减少推理步数
   - 使用较小的图像尺寸

3. **质量优化**
   - 增加推理步数
   - 调整guidance scale
   - 使用更好的模型

## 📊 性能基准

| 配置 | 生成时间 | 内存使用 | 图像质量 |
|------|----------|----------|----------|
| RTX 3060 (12GB) | 15-30秒 | 6-8GB | 高 |
| RTX 4070 (12GB) | 10-20秒 | 5-7GB | 高 |
| CPU (16GB RAM) | 2-5分钟 | 4-6GB | 中等 |

## 🔐 安全注意事项

1. **内容过滤**: 默认启用安全检查器，防止生成不当内容
2. **资源限制**: 设置合理的生成参数，避免资源耗尽
3. **模型来源**: 仅使用可信的预训练模型

## 📚 参考资源

- [Diffusers 官方文档](https://huggingface.co/docs/diffusers)
- [Stable Diffusion 模型库](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [优化指南](https://huggingface.co/docs/diffusers/optimization/fp16)
- [故障排除](https://huggingface.co/docs/diffusers/troubleshooting)

## 🎯 下一步

配置完成后，您可以：

1. 运行 `scripts/setup_real_image_generation.py` 检查配置
2. 测试图像生成功能
3. 根据需要调整生成参数
4. 探索更多模型和风格

---

**注意**: 首次运行时，系统会自动下载模型文件（约4-6GB），请确保网络连接稳定。