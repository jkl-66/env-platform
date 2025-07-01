# NOAA数据下载和存储设置指南

本指南将帮助您快速设置和使用NOAA气候数据下载功能。

## 前置要求

### 1. 获取NOAA API密钥

1. 访问 [NOAA Climate Data Online Token Request](https://www.ncdc.noaa.gov/cdo-web/token)
2. 填写邮箱地址申请API密钥
3. 检查邮箱获取API密钥

### 2. 配置环境变量

在项目根目录的 `.env` 文件中设置您的API密钥：

```bash
# NOAA API配置
NOAA_API_KEY="您的NOAA_API密钥"
```

### 3. 安装依赖

确保已安装所有必要的Python包：

```bash
pip install -r requirements.txt
```

## 快速开始

### 方法1: 使用快速设置脚本（推荐）

运行快速设置脚本，它会自动下载最近30天的示例数据：

```bash
cd env-platform
python scripts/setup_noaa.py
```

这个脚本会：
- 创建必要的数据目录
- 初始化数据存储系统
- 下载最近30天的NOAA数据
- 将数据存储到文件系统和数据库

### 方法2: 使用完整下载脚本

下载更大范围的历史数据：

```bash
cd env-platform
python scripts/download_climate_data.py
```

这会下载最近3年的数据（可能需要较长时间）。

### 方法3: 测试功能

运行测试脚本验证功能：

```bash
cd env-platform
python test_noaa_download.py
```

## 数据存储结构

下载的数据会存储在以下位置：

```
data/
├── raw/                    # 原始数据文件
│   ├── noaa_daily_*.csv   # NOAA日数据
│   └── noaa_monthly_*.csv # NOAA月度数据
├── processed/             # 处理后的数据
└── cache/                 # 缓存数据
```

## 数据库存储

数据同时存储在以下数据库中：

1. **PostgreSQL**: 元数据和数据记录信息
2. **InfluxDB**: 时序数据（如果配置）
3. **Redis**: 缓存（如果配置）

## 可用的数据类型

### NOAA日数据包含：
- TMAX: 最高温度
- TMIN: 最低温度
- TAVG: 平均温度
- PRCP: 降水量
- SNOW: 降雪量

### NOAA月度数据包含：
- 月度汇总统计
- 区域气候数据

## 编程接口使用

### 基本用法

```python
import asyncio
from scripts.download_climate_data import ClimateDataManager

async def download_data():
    # 创建管理器
    manager = ClimateDataManager()
    await manager.initialize()
    
    # 下载数据
    await manager.download_all_data(
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # 清理资源
    await manager.storage.close()

# 运行
asyncio.run(download_data())
```

### 查询存储的数据

```python
from src.data_processing.data_storage import DataStorage

# 创建存储实例
storage = DataStorage()
await storage.initialize()

# 搜索数据记录
records = storage.search_data_records(
    source="noaa_daily",
    data_type="daily",
    limit=10
)

# 加载数据文件
for record in records:
    data = storage.load_dataframe(record['file_path'])
    print(f"数据形状: {data.shape}")
```

## 故障排除

### 常见问题

1. **API密钥错误**
   - 确保在.env文件中正确设置了NOAA_API_KEY
   - 检查API密钥是否有效

2. **网络连接问题**
   - 确保网络连接正常
   - NOAA服务器可能有时响应较慢

3. **数据库连接问题**
   - 检查PostgreSQL/InfluxDB/Redis配置
   - 如果数据库未配置，数据仍会保存到文件系统

4. **权限问题**
   - 确保对data目录有写权限

### 日志查看

运行脚本时会显示详细的日志信息，包括：
- 下载进度
- 存储状态
- 错误信息

## 性能优化

1. **限制下载范围**: 首次使用时建议下载较小的时间范围
2. **气象站选择**: 可以修改脚本中的气象站数量限制
3. **并发控制**: NOAA API有速率限制，避免过于频繁的请求

## 下一步

成功下载数据后，您可以：

1. 使用数据进行气候分析
2. 训练机器学习模型
3. 创建可视化图表
4. 集成到Web应用中

更多功能请参考项目的其他文档和示例代码。