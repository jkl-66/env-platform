# GRIB文件处理指南

本指南介绍如何使用环境数据平台处理GRIB格式的气象数据文件。

## 概述

GRIB（GRIdded Binary）是世界气象组织（WMO）标准的二进制数据格式，广泛用于存储和交换气象数据。本平台提供了完整的GRIB文件处理功能，包括：

- 文件信息查看
- 格式转换（GRIB → NetCDF）
- 数据处理和质量控制
- 数据分析和可视化
- 批量处理

## 安装依赖

系统已包含处理GRIB文件所需的依赖：
- `cfgrib==0.9.10.4` - GRIB文件读取
- `xarray` - 数据处理
- `matplotlib` - 可视化

## 使用方法

### 1. 命令行工具（CLI）

#### 查看GRIB文件信息
```bash
python cli.py grib info <grib_file> [--output output_dir]
```

示例：
```bash
python cli.py grib info data/weather.grib2 --output output/grib_info
```

#### 转换GRIB文件为NetCDF
```bash
python cli.py grib convert <input_grib> [--output output_file] [--variables var1,var2] [--compression]
```

示例：
```bash
python cli.py grib convert data/weather.grib2 --output data/weather.nc --variables t2m,tp --compression
```

#### 处理GRIB文件（质量控制）
```bash
python cli.py grib process <grib_file> [--variables var1,var2] [--output-format netcdf] [--output output_dir] [--no-process]
```

示例：
```bash
python cli.py grib process data/weather.grib2 --variables t2m,tp --output output/processed
```

#### 分析GRIB文件
```bash
python cli.py grib analyze <grib_file> [--variables var1,var2] [--output output_dir]
```

示例：
```bash
python cli.py grib analyze data/weather.grib2 --variables t2m,tp --output output/analysis
```

#### 批量处理GRIB文件
```bash
python cli.py grib batch <input_dir> [--output output_dir] [--pattern *.grib*] [--output-format netcdf] [--no-process]
```

示例：
```bash
python cli.py grib batch data/grib_files/ --output output/batch --pattern "*.grib2"
```

### 2. Python API

#### 使用GRIBProcessor类

```python
from src.data_processing.grib_processor import GRIBProcessor
from src.data_processing.data_processor import ProcessingConfig

# 初始化处理器
processor = GRIBProcessor()

# 加载GRIB文件
data = processor.load_grib_file('data/weather.grib2')

# 查看文件信息
info = processor.get_grib_info('data/weather.grib2')
print(f"变量: {info['variables']}")
print(f"时间范围: {info['time_range']}")

# 提取特定变量
temp_data = processor.extract_variables(data, ['t2m'])

# 转换为NetCDF
processor.convert_to_netcdf(
    'data/weather.grib2',
    'output/weather.nc',
    variables=['t2m', 'tp']
)

# 数据处理
config = ProcessingConfig(
    remove_outliers=True,
    fill_missing=True,
    quality_control=True
)

processed_data = processor.process_grib_data(
    'data/weather.grib2',
    config=config,
    variables=['t2m']
)

# 保存处理后的数据
processor.save_processed_data(
    processed_data,
    'output/processed_weather.nc'
)
```

#### 使用DataStorage类

```python
from src.data_processing.data_storage import DataStorage

# 初始化存储管理器
storage = DataStorage()

# 自动检测并加载GRIB文件
data = storage.load_xarray('data/weather.grib2')

# 或者显式加载GRIB文件
data = storage.load_grib_file(
    'data/weather.grib2',
    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}}
)
```

### 3. 常见GRIB变量

| 变量代码 | 描述 | 单位 |
|---------|------|------|
| t2m | 2米气温 | K |
| tp | 总降水量 | m |
| u10 | 10米U风分量 | m/s |
| v10 | 10米V风分量 | m/s |
| msl | 海平面气压 | Pa |
| sp | 地面气压 | Pa |
| d2m | 2米露点温度 | K |
| tcc | 总云量 | 0-1 |
| sst | 海表温度 | K |
| swvl1 | 土壤湿度第1层 | m³/m³ |

### 4. 输出文件说明

#### 信息查看输出
- `grib_info.json` - 文件基本信息
- `variable_statistics.json` - 变量统计信息
- `grib_overview.png` - 数据概览图

#### 分析输出
- `analysis_report.json` - 详细分析报告
- `variable_maps/` - 变量空间分布图
- `time_series/` - 时间序列图
- `statistics_summary.png` - 统计摘要图

#### 处理输出
- 处理后的数据文件（NetCDF或Zarr格式）
- `processing_summary.json` - 处理摘要
- `quality_report.json` - 质量控制报告

## 配置选项

### ProcessingConfig参数

```python
config = ProcessingConfig(
    # 质量控制
    quality_control=True,
    
    # 异常值处理
    remove_outliers=True,
    outlier_method='iqr',  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold=3.0,
    
    # 缺失值处理
    fill_missing=True,
    missing_method='interpolate',  # 'interpolate', 'forward_fill', 'backward_fill'
    
    # 空间聚合
    spatial_aggregation='mean',  # 'mean', 'sum', 'min', 'max'
    
    # 时间聚合
    temporal_aggregation='daily',  # 'hourly', 'daily', 'monthly'
    
    # 数据范围
    time_range=('2020-01-01', '2020-12-31'),
    spatial_bounds=(-180, -90, 180, 90),  # (west, south, east, north)
    
    # 输出选项
    compression=True,
    chunk_size={'time': 100, 'latitude': 50, 'longitude': 50}
)
```

## 故障排除

### 常见问题

1. **无法读取GRIB文件**
   - 确保文件路径正确
   - 检查文件是否损坏
   - 验证GRIB文件版本（支持GRIB1和GRIB2）

2. **内存不足**
   - 使用变量过滤：`variables=['t2m']`
   - 设置时间范围：`time_range=('2020-01-01', '2020-01-31')`
   - 使用空间裁剪：`spatial_bounds=(100, 20, 120, 40)`

3. **处理速度慢**
   - 启用并行处理
   - 使用SSD存储
   - 减少数据精度

### 错误代码

- `GRIB_001`: 文件不存在或无法访问
- `GRIB_002`: 不支持的GRIB格式
- `GRIB_003`: 变量不存在
- `GRIB_004`: 内存不足
- `GRIB_005`: 磁盘空间不足

## 性能优化

1. **使用Dask进行大文件处理**
2. **启用数据压缩**
3. **合理设置chunk大小**
4. **使用变量和时间过滤**
5. **并行批量处理**

## 示例数据

可以从以下来源获取GRIB测试数据：
- ECMWF ERA5再分析数据
- NOAA GFS预报数据
- 中国气象局数值预报产品

## 技术支持

如需技术支持，请提供：
1. GRIB文件信息（使用`grib info`命令）
2. 错误信息和日志
3. 系统环境信息
4. 具体的使用场景