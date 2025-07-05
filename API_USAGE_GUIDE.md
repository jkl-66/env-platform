# 环境保护警示图像生成器 - API版本使用指南

## 概述

本项目已升级为基于 Hugging Face Inference API 的版本，无需本地下载大型模型文件，直接通过云端API调用 Stable Diffusion 3.5 模型生成环境保护警示图像。

## 主要优势

✅ **无需本地存储**: 不再需要下载几GB的模型文件  
✅ **快速启动**: 无需等待模型加载时间  
✅ **稳定可靠**: 使用Hugging Face官方API服务  
✅ **自动更新**: 始终使用最新版本的模型  
✅ **节省资源**: 不占用本地GPU和内存资源  

## 环境配置

### 1. 获取 Hugging Face Token

1. 访问 [Hugging Face](https://huggingface.co/)
2. 注册/登录账户
3. 进入 Settings → Access Tokens
4. 创建新的 Token (建议选择 "Read" 权限)
5. 复制生成的 Token

### 2. 设置环境变量

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="your_token_here"
```

**Windows (命令提示符):**
```cmd
set HF_TOKEN=your_token_here
```

**Linux/macOS:**
```bash
export HF_TOKEN="your_token_here"
```

### 3. 安装依赖

```bash
pip install requests pillow
```

## 使用方法

### 基本使用

```python
from environmental_image_generator import EnvironmentalImageGenerator

# 初始化生成器
generator = EnvironmentalImageGenerator()

# 测试API连接
if generator.test_api_connection():
    print("API连接成功！")
    
    # 生成图像
    result = generator.generate_and_save(
        user_input="森林砍伐导致的环境破坏"
    )
    
    if result["success"]:
        print(f"图像已保存: {result['saved_files'][0]}")
else:
    print("API连接失败，请检查Token设置")
```

### 运行测试脚本

```bash
python test_api_generator.py
```

### 交互式使用

```bash
python environmental_image_generator.py
```

## API参数说明

### EnvironmentalImageGenerator 初始化参数

- `model_id`: 模型ID (默认: "stabilityai/stable-diffusion-3.5-large-turbo")
- `hf_token`: Hugging Face Token (可选，优先使用环境变量)

### generate_image 方法参数

- `user_input`: 用户自然语言输入描述
- `category`: 环境类别 (可选)
- `output_dir`: 输出目录 (默认: "outputs/environmental_images")

## 支持的环境类别

- 🌊 **海洋污染**: 海洋塑料、石油泄漏等
- 🌳 **森林破坏**: 砍伐、火灾、荒漠化等  
- 🏭 **空气污染**: 工业排放、雾霾、温室气体等
- 🗑️ **废物污染**: 垃圾堆积、有害废物等
- 🐾 **生物多样性**: 物种灭绝、栖息地破坏等
- ⚡ **能源环境**: 化石燃料、可再生能源对比等
- 🌡️ **气候变化**: 全球变暖、极端天气等

## 输出文件说明

生成的文件将保存在指定目录中，文件名格式：
```
{描述}_{时间戳}_{序号}_api.png
```

同时会生成详细的报告文件：
```
generation_report_{时间戳}.json
```

## 故障排除

### 常见问题

**1. API连接失败**
- 检查网络连接
- 确认HF_TOKEN环境变量已设置
- 验证Token是否有效

**2. 生成失败**
- 检查提示词是否合适
- 确认API服务状态
- 查看错误日志信息

**3. 响应缓慢**
- API调用需要网络传输时间
- 模型可能需要冷启动时间
- 可以稍后重试

### 错误代码说明

- `401`: Token无效或未授权
- `429`: API调用频率过高
- `503`: 模型正在加载中

## 注意事项

1. **网络要求**: 需要稳定的网络连接
2. **Token安全**: 请妥善保管您的HF_TOKEN
3. **使用限制**: 遵守Hugging Face的使用条款
4. **内容政策**: 生成内容需符合相关政策规定

## 技术支持

如遇到问题，请检查：
1. 网络连接状态
2. Token设置是否正确
3. 依赖库是否正确安装
4. 查看详细错误日志