# Hugging Face 库安装指南

本指南将帮助您安装 Hugging Face 相关库，以启用生态警示图像生成系统的完整功能。

## 🎯 为什么需要 Hugging Face 库？

生态警示图像生成系统支持多种生成模式：

- **GAN模式**: 基础功能，无需额外库
- **扩散模式**: 需要 Hugging Face 库支持
- **混合模式**: 需要 Hugging Face 库支持

安装 Hugging Face 库后，您将获得：
- 更高质量的图像生成
- 更多样化的生成风格
- 更好的文本到图像转换能力

## 📦 安装步骤

### 1. 基础安装

```bash
# 安装核心库
pip install transformers diffusers accelerate

# 如果使用 CUDA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 如果只使用 CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. 可选依赖（推荐）

```bash
# 图像处理增强
pip install pillow opencv-python

# 性能优化
pip install xformers  # 仅支持 CUDA

# 模型缓存和下载
pip install huggingface_hub
```

### 3. 验证安装

运行以下命令验证安装：

```bash
python -c "import transformers, diffusers; print('Hugging Face 库安装成功!')"
```

## 🔧 配置说明

### 环境变量设置

```bash
# 设置 Hugging Face 缓存目录（可选）
export HF_HOME=/path/to/your/cache

# 设置代理（如果需要）
export HF_ENDPOINT=https://hf-mirror.com
```

### GPU 支持

如果您有 NVIDIA GPU，系统会自动使用 GPU 加速：

```python
# 检查 GPU 可用性
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
```

## 🚀 使用示例

安装完成后，您可以使用所有生成模式：

```python
from src.models.ecology_image_generator import EcologyImageGenerator

# 初始化生成器
generator = EcologyImageGenerator()

# 使用扩散模式
generator.set_generation_mode('diffusion')

# 生成高质量图像
result = generator.generate_warning_image(
    environmental_indicators={
        "co2_level": 450.0,
        "pm25_level": 100.0,
        "temperature": 35.0,
        "forest_coverage": 30.0,
        "water_quality": 4.0,
        "air_quality": 3.0
    },
    style='realistic',
    num_images=1
)
```

## 📊 性能对比

| 生成模式 | 质量 | 速度 | 内存使用 | 依赖要求 |
|---------|------|------|----------|----------|
| GAN | 中等 | 快 | 低 | 基础 |
| 扩散 | 高 | 慢 | 高 | Hugging Face |
| 混合 | 高 | 中等 | 中等 | Hugging Face |

## 🛠️ 故障排除

### 常见问题

**1. 安装失败**
```bash
# 升级 pip
pip install --upgrade pip

# 清理缓存
pip cache purge

# 重新安装
pip install --no-cache-dir transformers diffusers
```

**2. 内存不足**
```python
# 使用较小的模型
from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,  # 使用半精度
    low_cpu_mem_usage=True
)
```

**3. 网络问题**
```bash
# 使用镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers diffusers

# 或设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
```

**4. CUDA 版本不匹配**
```bash
# 检查 CUDA 版本
nvcc --version

# 安装对应版本的 PyTorch
# 访问 https://pytorch.org/get-started/locally/ 获取正确命令
```

## 📋 系统要求

### 最低要求
- Python 3.8+
- 8GB RAM
- 10GB 可用存储空间

### 推荐配置
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU (6GB+ VRAM)
- 20GB+ 可用存储空间
- 稳定的网络连接

## 🔍 验证完整安装

运行改进版交互式系统来验证所有功能：

```bash
cd /path/to/env-platform
python scripts/improved_interactive_ecology_demo.py
```

如果看到以下输出，说明安装成功：

```
📋 可用的生成模式:
• GAN模式: ✅ 可用 (快速生成)
• 扩散模式: ✅ 可用 (高质量生成)
• 混合模式: ✅ 可用 (平衡质量与速度)
```

## 📚 相关资源

- [Hugging Face 官方文档](https://huggingface.co/docs)
- [Diffusers 库文档](https://huggingface.co/docs/diffusers)
- [Transformers 库文档](https://huggingface.co/docs/transformers)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)

## 💡 使用建议

1. **首次使用**: 建议先使用 GAN 模式熟悉系统
2. **生产环境**: 推荐使用混合模式平衡质量和速度
3. **展示用途**: 使用扩散模式获得最佳视觉效果
4. **资源有限**: 在内存或GPU资源有限时使用 GAN 模式

---

如果您在安装过程中遇到问题，请参考故障排除部分或联系技术支持。