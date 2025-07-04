# 环境保护图像生成器

基于 Stable Diffusion 3.5 Large Turbo 模型的环境保护警示图像生成工具

## 功能特点

- 🌍 **环境主题专用**: 专门针对环境保护主题优化的图像生成
- 🗣️ **自然语言输入**: 支持中文和英文自然语言描述
- 📋 **内置提示词模板**: 10+ 种环境主题的预设模板
- 🎨 **多种风格预设**: 纪实、艺术、科学等多种视觉风格
- ⚙️ **可调节参数**: 支持质量、速度、尺寸等参数调整
- 📁 **批量生成**: 支持批量生成多个主题的图像

## 环境主题

| 主题 | 描述 | 关键词 |
|------|------|--------|
| 空气污染 | 工业污染、雾霾、空气质量 | 污染、雾霾、工厂烟囱 |
| 水污染 | 河流污染、工业废水、有毒物质 | 水污染、工业废水、河流污染 |
| 森林砍伐 | 大规模砍伐、生态破坏、栖息地丧失 | 森林砍伐、毁林、生态破坏 |
| 气候变化 | 全球变暖、冰川融化、海平面上升 | 气候变化、全球变暖、冰川融化 |
| 塑料污染 | 海洋垃圾、微塑料、海洋生物影响 | 塑料污染、海洋垃圾、微塑料 |
| 可再生能源 | 太阳能、风能、清洁技术 | 可再生能源、太阳能、风能 |
| 废物管理 | 垃圾填埋、回收、废物处理 | 垃圾处理、回收、废物管理 |
| 生物多样性 | 濒危物种、栖息地破坏、生态系统 | 生物多样性、濒危物种、生态保护 |
| 城市污染 | 交通污染、城市热岛、空气质量 | 城市污染、交通污染、城市环境 |
| 土壤退化 | 荒漠化、土地侵蚀、农业污染 | 土壤退化、荒漠化、土地侵蚀 |

## 安装要求

```bash
# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate pillow

# 确保有足够的 GPU 内存 (推荐 8GB+)
```

## 快速开始

### 1. 基本使用

```python
from environmental_image_generator import EnvironmentalImageGenerator

# 初始化生成器
generator = EnvironmentalImageGenerator(
    model_id="stabilityai/stable-diffusion-3.5-large-turbo",
    device="auto"
)

# 生成图像
results = generator.generate_image(
    user_input="工厂排放黑烟污染空气的场景",
    guidance_scale=7.5,
    num_inference_steps=28,
    height=1024,
    width=1024
)

if results['success']:
    print(f"图像已保存到: {results['image_paths'][0]}")
else:
    print(f"生成失败: {results['error']}")
```

### 2. 使用演示脚本

```bash
# 运行交互式演示
python demo_environmental_generator.py
```

演示脚本提供以下功能：
- 预设模板生成
- 自然语言输入生成
- 批量生成演示
- 配置信息查看

### 3. 运行测试

```bash
# 运行功能测试
python test_environmental_generator.py
```

## 配置文件

配置文件位于 `config/environmental_prompts.json`，包含：

- **environmental_prompts**: 环境主题提示词模板
- **generation_settings**: 生成质量设置（默认、高质量、快速、批量）
- **style_presets**: 风格预设（写实、纪实、艺术、科学、戏剧性）
- **quality_enhancers**: 质量增强词
- **negative_prompts**: 负面提示词

### 自定义配置

您可以编辑配置文件来：
- 添加新的环境主题
- 修改现有提示词模板
- 调整生成参数
- 添加新的风格预设

## 生成参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| guidance_scale | 提示词遵循度 | 7.5 (标准), 9.0 (高质量) |
| num_inference_steps | 推理步数 | 28 (标准), 35 (高质量), 20 (快速) |
| height/width | 图像尺寸 | 1024x1024 (高质量), 768x768 (快速) |
| num_images | 生成数量 | 1 (单张), 4 (批量) |

## 使用示例

### 中文自然语言输入

```python
# 空气污染主题
results = generator.generate_image("工厂烟囱冒出黑烟，城市被雾霾笼罩")

# 海洋污染主题
results = generator.generate_image("海洋中漂浮着大量塑料垃圾，海龟被塑料袋缠绕")

# 森林破坏主题
results = generator.generate_image("大片森林被砍伐，只剩下光秃秃的树桩")
```

### 英文自然语言输入

```python
# 气候变化主题
results = generator.generate_image("melting glaciers due to global warming, polar bears on shrinking ice")

# 可再生能源主题
results = generator.generate_image("solar panels and wind turbines in a clean energy landscape")
```

### 使用预设模板

```python
import json

# 加载配置
with open('config/environmental_prompts.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 使用空气污染模板
air_pollution = config['environmental_prompts']['air_pollution']
prompt = air_pollution['base_prompt']
style = config['style_presets']['documentary']

results = generator.generate_image(f"{prompt}, {style}")
```

## 输出文件

生成的图像保存在 `outputs/environmental_images/` 目录下：

```
outputs/environmental_images/
├── 20241201_143022_air_pollution/
│   ├── image_001.png
│   └── generation_report.json
├── 20241201_143156_water_pollution/
│   ├── image_001.png
│   └── generation_report.json
└── ...
```

每次生成都会创建一个带时间戳的文件夹，包含：
- 生成的图像文件 (PNG 格式)
- 生成报告 (JSON 格式，包含参数和元数据)

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```python
   # 使用较小的图像尺寸
   generator.generate_image(prompt, height=512, width=512)
   ```

2. **模型下载失败**
   ```python
   # 检查网络连接和 Hugging Face 访问
   # 确保有足够的磁盘空间 (模型约 8GB)
   ```

3. **生成速度慢**
   ```python
   # 使用快速设置
   generator.generate_image(
       prompt, 
       num_inference_steps=20,
       height=768, 
       width=768
   )
   ```

### 性能优化

- 使用 GPU 加速 (CUDA)
- 调整推理步数和图像尺寸
- 批量生成多张图像
- 使用混合精度 (自动启用)

## 许可证

本项目遵循 MIT 许可证。使用的 Stable Diffusion 模型请遵循相应的许可证条款。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

---

**注意**: 生成的图像仅用于教育和环保宣传目的，请遵守相关法律法规。