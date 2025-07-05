# 环境保护警示图像生成器 (API版本)

基于 Hugging Face Inference API 的环境保护主题图像生成工具，通过云端 Stable Diffusion 3.5 模型生成具有环境警示意义的图像。

## 功能特点

- 🎨 **云端图像生成**: 基于 Hugging Face Inference API 调用 Stable Diffusion 3.5 Large Turbo 模型
- 🌍 **环境主题优化**: 专门针对环境保护场景优化的提示词模板
- 🔧 **自动提示词增强**: 根据用户输入自动生成专业的环境警示提示词
- 📊 **多类别支持**: 支持海洋污染、森林破坏、空气污染等多种环境问题
- 💾 **完整记录**: 自动保存生成图像和详细的生成报告
- ⚡ **快速启动**: 无需本地模型下载，即开即用
- 💰 **资源节省**: 无需本地GPU，节省硬件成本

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

## 环境要求

### 硬件要求
- **网络**: 稳定的互联网连接
- **内存**: 4GB+ 系统内存
- **存储**: 1GB+ 可用空间（用于输出图像）

### 软件要求
- Python 3.8+
- Hugging Face Token (免费注册获取)

## 安装步骤

### 1. 克隆项目
```bash
git clone <repository-url>
cd env-platform
```

### 2. 安装依赖
```bash
# 安装基本依赖
pip install requests pillow

# 或安装完整依赖
pip install -r requirements.txt
```

### 3. 获取 Hugging Face Token

1. 访问 [Hugging Face](https://huggingface.co/)
2. 注册/登录账户
3. 进入 Settings → Access Tokens
4. 创建新的 Token (选择 "Read" 权限)
5. 复制生成的 Token

### 4. 配置环境变量

**Windows:**
```cmd
set HF_TOKEN=your_token_here
```

**Linux/macOS:**
```bash
export HF_TOKEN="your_token_here"
```

## 快速开始

### 1. 基本使用

```python
from environmental_image_generator import EnvironmentalImageGenerator

# 初始化生成器 (API版本)
generator = EnvironmentalImageGenerator()

# 测试API连接
if generator.test_api_connection():
    print("API连接成功！")
    
    # 生成图像
    results = generator.generate_image(
        user_input="工厂排放黑烟污染空气的场景"
    )
    
    if results['success']:
        print(f"图像已保存到: {results['image_paths'][0]}")
        print(f"生成时间: {results['generation_time']}秒")
    else:
        print(f"生成失败: {results['error']}")
else:
    print("API连接失败，请检查Token设置")
```

### 2. 使用演示脚本

```bash
# 运行交互式演示 (API版本)
python environmental_image_generator.py
```

```bash
# 运行功能演示 (无需Token)
python demo_without_token.py
```

演示脚本提供以下功能：
- API连接测试
- 自然语言输入生成
- 环境类别检测
- 提示词增强演示

### 3. 运行测试

```bash
# 运行API版本测试 (需要HF_TOKEN)
python test_api_generator.py
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

1. **API连接失败**
   ```bash
   # 检查Token设置
   echo $HF_TOKEN  # Linux/macOS
   echo %HF_TOKEN%  # Windows
   
   # 重新设置Token
   export HF_TOKEN="your_valid_token"  # Linux/macOS
   set HF_TOKEN=your_valid_token  # Windows
   ```

2. **生成失败 (401错误)**
   - 检查Token是否有效
   - 确认Token有模型访问权限
   - 重新生成Token

3. **生成速度慢 (503错误)**
   - 模型正在冷启动，请稍后重试
   - 避免频繁调用API
   - 检查网络连接稳定性

4. **请求频率限制 (429错误)**
   - 降低API调用频率
   - 等待一段时间后重试
   - 考虑升级Hugging Face账户

### 性能优化

- 稳定的网络连接
- 合理的API调用频率
- 优化提示词长度
- 批量处理多个请求

## 许可证

本项目遵循 MIT 许可证。使用的 Stable Diffusion 模型请遵循相应的许可证条款。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

---

**注意**: 生成的图像仅用于教育和环保宣传目的，请遵守相关法律法规。