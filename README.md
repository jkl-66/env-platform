# 气候数据分析与生态警示系统

## 项目概述

本项目是一个基于AI的气候数据分析与生态警示图像生成系统，集成了历史气候数据分析、生成式AI图像创建和区域气候预测功能。

## 核心功能模块

### 1. 数据处理模块 (Data Processing Module)
- 多源数据获取与整合
- 数据清洗与标准化
- 时序数据库存储

### 2. AI模型训练与推理模块 (Model Module)
- 历史气候数据分析模型
- 生态警示图像生成模型
- 区域气候预测模型

### 3. 服务接口模块 (API Service)
- RESTful API设计
- 容器化部署架构

## 技术栈

### 后端框架
- **Web框架**: FastAPI
- **数据处理**: Pandas, Xarray, Dask
- **机器学习**: PyTorch, TensorFlow, Scikit-learn
- **数据库**: InfluxDB (时序), PostgreSQL (元数据), Redis (缓存)
- **消息队列**: Apache Kafka
- **容器化**: Docker, Kubernetes

### AI模型
- **时间序列分析**: Prophet, LSTM
- **异常检测**: Isolation Forest, Autoencoder
- **图像生成**: Hugging Face Inference API (Stable Diffusion 3.5)
- **预测模型**: XGBoost, Transformer, GNN

## 项目结构

```
research_fair/
├── src/
│   ├── data_processing/     # 数据处理模块
│   ├── models/             # AI模型模块
│   ├── api/                # API服务模块
│   └── utils/              # 工具函数
├── config/                 # 配置文件
├── data/                   # 数据存储
├── models/                 # 训练好的模型
├── tests/                  # 测试文件
├── docker/                 # Docker配置
└── docs/                   # 文档
```

## 快速开始

### 环境要求
- Python 3.9+
- Docker & Docker Compose
- 网络连接 (用于API调用)
- Hugging Face Token (用于图像生成)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动服务
```bash
docker-compose up -d
python src/main.py
```

## API文档

启动服务后访问: http://localhost:8000/docs

## 开发计划

- [x] 项目架构设计
- [ ] 数据处理模块开发 (第1-2周)
- [ ] 历史气候分析模型 (第3-4周)
- [ ] 生成式AI图像模型 (第5-6周)
- [ ] 区域气候预测模型 (第7-8周)
- [ ] API接口与部署 (第9-10周)

## 许可证

MIT License