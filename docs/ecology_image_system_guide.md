# 基于生成式AI的生态警示图像系统使用指南

## 系统概述

本系统是一个基于生成式AI技术的生态警示图像生成平台，旨在通过直观的视觉图像提升公众的环保意识。系统支持多种生成模式，包括条件GAN和扩散模型，能够根据用户输入的环境指标生成具有警示意义的图像。

## 核心功能

### 1. 智能环境指标分析
- **多维度评估**: 支持CO₂浓度、PM2.5、温度、森林覆盖率等多种环境指标
- **智能警示等级**: 自动评估环境风险等级（1-5级）
- **个性化建议**: 根据分析结果提供针对性的环保建议

### 2. 多模式图像生成
- **GAN模式**: 快速生成（2-5秒/图），适合实时应用
- **扩散模式**: 高质量生成（8-15秒/图），适合精细化展示
- **混合模式**: 平衡速度与质量（5-10秒/图）

### 3. 教育场景定制
- **目标受众适配**: 支持小学生、中学生、公众等不同群体
- **内容复杂度调节**: 根据受众自动调整展示复杂度
- **对比式教育**: 展示良好实践与不良后果的对比

## 使用方法

### 基础使用

```python
from src.models.ecology_image_generator import EcologyImageGenerator

# 初始化生成器
generator = EcologyImageGenerator()

# 设置环境指标
environmental_data = {
    "co2_level": 450,        # CO₂浓度 (ppm)
    "pm25_level": 120,       # PM2.5浓度 (μg/m³)
    "temperature": 35,       # 温度 (°C)
    "forest_coverage": 25,   # 森林覆盖率 (%)
    "water_quality": 4,      # 水质等级 (1-10)
    "air_quality": 3         # 空气质量 (1-10)
}

# 生成警示图像
result = generator.generate_warning_image(
    environmental_indicators=environmental_data,
    style="realistic",
    num_images=2
)

print(f"警示等级: {result['warning_level']}/5")
print(f"使用模板: {result['template_used']}")
```

### 高级功能

#### 1. 使用预设场景模板

```python
# 获取预设模板
templates = generator.get_condition_templates()

# 使用空气污染模板
air_pollution_data = templates["空气污染"]
result = generator.generate_warning_image(
    environmental_indicators=air_pollution_data,
    style="dramatic"
)
```

#### 2. 教育对比生成

```python
# 好的实践
good_practice = {
    "forest_coverage": 80,
    "co2_level": 350,
    "air_quality": 9
}

# 不良后果
bad_consequence = {
    "forest_coverage": 10,
    "co2_level": 500,
    "air_quality": 2
}

# 生成对比图像
good_result = generator.generate_warning_image(good_practice, style="educational")
bad_result = generator.generate_warning_image(bad_consequence, style="educational")
```

#### 3. 批量生成

```python
# 批量生成多个场景
scenarios = [
    {"name": "工业污染", "data": {"pm25_level": 150, "air_quality": 2}},
    {"name": "森林砍伐", "data": {"forest_coverage": 15, "biodiversity": 3}},
    {"name": "气候变化", "data": {"temperature": 40, "co2_level": 480}}
]

results = []
for scenario in scenarios:
    result = generator.generate_warning_image(
        environmental_indicators=scenario["data"],
        style="photorealistic"
    )
    results.append({"scenario": scenario["name"], "result": result})
```

## API接口使用

### 图像生成API

```bash
# POST /api/visualization/generate
curl -X POST "http://localhost:8000/api/visualization/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "environmental_indicators": {
      "co2_level": 450,
      "pm25_level": 120,
      "temperature": 35
    },
    "generation_mode": "gan",
    "style": "realistic",
    "num_images": 2,
    "custom_prompt": "工业污染导致的环境恶化"
  }'
```

### 获取预设模板

```bash
# GET /api/visualization/templates
curl "http://localhost:8000/api/visualization/templates"
```

### 批量生成

```bash
# POST /api/visualization/batch_generate
curl -X POST "http://localhost:8000/api/visualization/batch_generate" \
  -H "Content-Type: application/json" \
  -d '{
    "scenarios": [
      {
        "name": "空气污染",
        "indicators": {"pm25_level": 150, "air_quality": 2}
      },
      {
        "name": "水质污染",
        "indicators": {"water_quality": 2, "pollution_level": 8}
      }
    ],
    "generation_mode": "diffusion",
    "style": "educational"
  }'
```

## 环境指标说明

### 主要指标

| 指标名称 | 单位 | 正常范围 | 警示阈值 | 说明 |
|---------|------|----------|----------|------|
| co2_level | ppm | 350-400 | >450 | 大气CO₂浓度 |
| pm25_level | μg/m³ | 0-35 | >75 | PM2.5颗粒物浓度 |
| temperature | °C | 15-25 | >30 | 平均气温 |
| humidity | % | 40-70 | <30 或 >80 | 相对湿度 |
| forest_coverage | % | 50-80 | <30 | 森林覆盖率 |
| water_quality | 1-10 | 7-10 | <5 | 水质等级 |
| air_quality | 1-10 | 7-10 | <5 | 空气质量等级 |
| biodiversity | 1-10 | 7-10 | <5 | 生物多样性指数 |
| pollution_level | 1-10 | 1-3 | >6 | 综合污染等级 |

### 警示等级说明

- **等级1 (绿色)**: 环境状况良好，无需特别关注
- **等级2 (浅绿)**: 环境状况较好，建议保持现状
- **等级3 (黄色)**: 环境状况一般，需要关注和改善
- **等级4 (橙色)**: 环境状况较差，需要采取措施
- **等级5 (红色)**: 环境状况严重，需要紧急行动

## 应用场景

### 1. 学校环保教育
- **小学阶段**: 使用简单直观的图像和文字
- **中学阶段**: 结合科学数据和深入分析
- **大学阶段**: 提供详细的技术原理和研究数据

### 2. 公众意识提升
- **社区宣传**: 生成本地环境问题的可视化图像
- **媒体报道**: 为新闻报道提供直观的视觉素材
- **社交媒体**: 创建易于分享的环保宣传图像

### 3. 政策制定支持
- **影响评估**: 可视化政策实施前后的环境变化
- **公众沟通**: 帮助政府向公众解释环保政策
- **决策支持**: 为政策制定者提供直观的数据展示

### 4. 企业环保培训
- **员工教育**: 提升员工的环保意识
- **合规培训**: 展示不合规行为的环境后果
- **CSR报告**: 为企业社会责任报告提供视觉素材

## 技术架构

### 前端技术栈
- **框架**: React.js / Vue.js
- **UI组件**: Ant Design / Element UI
- **图表库**: ECharts / D3.js
- **图像处理**: Canvas API / WebGL

### 后端技术栈
- **框架**: Python FastAPI
- **AI模型**: PyTorch / TensorFlow
- **图像处理**: PIL / OpenCV
- **数据库**: PostgreSQL / MongoDB

### AI模型
- **GAN模型**: StyleGAN3 / Progressive GAN
- **扩散模型**: Stable Diffusion / DALL-E
- **条件生成**: Conditional GAN / ControlNet
- **图像增强**: ESRGAN / Real-ESRGAN

## 性能优化

### 1. 生成速度优化
- **模型量化**: 减少模型大小和推理时间
- **批量处理**: 支持多图像并行生成
- **缓存机制**: 缓存常用模板和结果
- **GPU加速**: 利用CUDA进行并行计算

### 2. 质量提升
- **多模型融合**: 结合不同模型的优势
- **后处理优化**: 图像增强和质量提升
- **用户反馈**: 基于用户评价持续改进
- **A/B测试**: 对比不同生成策略的效果

### 3. 系统扩展
- **微服务架构**: 支持水平扩展
- **负载均衡**: 分布式处理请求
- **容器化部署**: Docker + Kubernetes
- **监控告警**: 实时监控系统状态

## 部署指南

### 开发环境

```bash
# 克隆项目
git clone <repository-url>
cd env-platform

# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 生产环境

```bash
# 构建Docker镜像
docker build -t ecology-image-system .

# 运行容器
docker run -d \
  --name ecology-system \
  -p 8000:8000 \
  -v /data:/app/data \
  ecology-image-system

# 使用Docker Compose
docker-compose up -d
```

### Kubernetes部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ecology-image-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ecology-system
  template:
    metadata:
      labels:
        app: ecology-system
    spec:
      containers:
      - name: ecology-system
        image: ecology-image-system:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件是否存在
   - 确认GPU内存是否充足
   - 验证依赖库版本兼容性

2. **生成速度慢**
   - 检查GPU是否正常工作
   - 考虑使用更小的模型
   - 启用批量处理模式

3. **图像质量差**
   - 调整生成参数
   - 尝试不同的模型
   - 检查输入数据质量

4. **内存不足**
   - 减少批量大小
   - 使用模型量化
   - 增加系统内存

### 日志分析

```bash
# 查看应用日志
docker logs ecology-system

# 实时监控日志
docker logs -f ecology-system

# 查看特定时间段的日志
docker logs --since="2024-01-01T00:00:00" ecology-system
```

## 贡献指南

### 开发流程

1. **Fork项目**: 在GitHub上fork项目到个人账户
2. **创建分支**: `git checkout -b feature/new-feature`
3. **开发功能**: 编写代码并添加测试
4. **提交代码**: `git commit -m "Add new feature"`
5. **推送分支**: `git push origin feature/new-feature`
6. **创建PR**: 在GitHub上创建Pull Request

### 代码规范

- **Python**: 遵循PEP 8规范
- **JavaScript**: 使用ESLint和Prettier
- **文档**: 使用Markdown格式
- **测试**: 保持测试覆盖率>80%

### 提交规范

```
type(scope): description

[optional body]

[optional footer]
```

类型说明：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

## 许可证

本项目采用MIT许可证，详见[LICENSE](../LICENSE)文件。

## 联系我们

- **项目主页**: https://github.com/your-org/ecology-image-system
- **问题反馈**: https://github.com/your-org/ecology-image-system/issues
- **邮箱**: ecology-system@example.com
- **文档**: https://ecology-system.readthedocs.io

---

*最后更新: 2024年12月*