# 气候模型快速开始指南

本指南将帮助您快速开始使用已训练的气候模型进行温度预测。

## 1. 基本使用方法

### 1.1 简单预测示例

```python
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from src.ml.model_manager import ModelManager

def create_time_features(base_time):
    """为给定的时间创建时间特征"""
    dt = pd.to_datetime(base_time)
    return {
        'year': dt.year,
        'month': dt.month,
        'day': dt.day,
        'hour': dt.hour,
        'day_of_year': dt.dayofyear,
        'season': {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}[dt.month],
        'month_sin': np.sin(2 * np.pi * dt.month / 12),
        'month_cos': np.cos(2 * np.pi * dt.month / 12),
        'day_sin': np.sin(2 * np.pi * dt.dayofyear / 365.25),
        'day_cos': np.cos(2 * np.pi * dt.dayofyear / 365.25),
        'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dt.hour / 24)
    }

async def simple_prediction():
    # 创建模型管理器
    model_manager = ModelManager()
    
    # 获取可用模型
    models = model_manager.list_models()
    if not models:
        print("没有找到可用的模型，请先训练模型")
        return
    
    model_id = models[0].id
    print(f"使用模型: {model_id}")
    
    # 设置预测时间
    prediction_time = datetime.now()
    time_features = create_time_features(prediction_time)
    
    # 准备预测数据（包含时间特征）
    predict_data = pd.DataFrame({
        'latitude': [39.9042],      # 纬度
        'longitude': [116.4074],    # 经度
        'number': [0],              # GRIB数据编号
        'step': [0],                # 时间步长
        'surface': [1],             # 地表层标识
        'msl': [101325.0],          # 海平面压力（Pa）
        'sst': [15.5],              # 海表温度（°C）
        'sp': [101000.0],           # 地面压力（Pa）
        'quality_score': [1.0],     # 数据质量分数
        **{k: [v] for k, v in time_features.items()}  # 添加时间特征
    })
    
    # 执行预测
    predictions = await model_manager.predict(model_id, predict_data)
    print(f"预测时间: {prediction_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"预测月份: {time_features['month']}月")
    print(f"预测温度: {predictions[0]:.2f}K ({predictions[0]-273.15:.2f}°C)")

# 运行预测
asyncio.run(simple_prediction())
```

### 1.2 批量预测多个地点

```python
import asyncio
import pandas as pd
from src.ml.model_manager import ModelManager

async def batch_prediction():
    model_manager = ModelManager()
    models = model_manager.list_models()
    
    if not models:
        print("没有找到可用的模型")
        return
    
    model_id = models[0].id
    
    # 设置预测时间
    prediction_time = datetime.now()
    time_features = create_time_features(prediction_time)
    
    # 准备多个城市的数据
    cities_data = pd.DataFrame({
        'city': ['北京', '上海', '广州'],
        'latitude': [39.9042, 31.2304, 23.1291],
        'longitude': [116.4074, 121.4737, 113.2644],
        'number': [0, 0, 0],
        'step': [0, 0, 0],
        'surface': [1, 1, 1],
        'msl': [101325.0, 101325.0, 101325.0],
        'sst': [15.5, 18.2, 22.8],
        'sp': [101000.0, 101000.0, 101000.0],
        'quality_score': [1.0, 1.0, 1.0]
    })
    
    # 为每个城市添加时间特征
    for key, value in time_features.items():
        cities_data[key] = value
    
    # 执行批量预测
    prediction_data = cities_data.drop('city', axis=1)
    predictions = await model_manager.predict(model_id, prediction_data)
    
    # 显示结果
    print(f"预测时间: {prediction_time.strftime('%Y-%m-%d %H:%M:%S')} ({time_features['month']}月)")
    for i, city in enumerate(cities_data['city']):
        temp_celsius = predictions[i] - 273.15
        print(f"{city}: {temp_celsius:.2f}°C")

# 运行批量预测
asyncio.run(batch_prediction())
```

## 2. 必需的输入特征

模型需要以下21个特征列（包括时间特征）：

### 基础气象特征（9个）
| 特征名 | 描述 | 单位 | 示例值 |
|--------|------|------|--------|
| `latitude` | 纬度 | 度 | 39.9042 |
| `longitude` | 经度 | 度 | 116.4074 |
| `number` | GRIB数据编号 | - | 0 |
| `step` | 时间步长 | - | 0 |
| `surface` | 地表层标识 | - | 1 |
| `msl` | 海平面压力 | Pa | 101325.0 |
| `sst` | 海表温度 | °C | 15.5 |
| `sp` | 地面压力 | Pa | 101000.0 |
| `quality_score` | 数据质量分数 | 0-1 | 1.0 |

### 时间特征（12个）
这些特征会根据预测时间自动生成：
- `year`, `month`, `day`, `hour` - 基础时间信息
- `day_of_year`, `season` - 季节性特征
- `month_sin`, `month_cos` - 月份的周期性编码
- `day_sin`, `day_cos` - 日期的周期性编码
- `hour_sin`, `hour_cos` - 小时的周期性编码

**重要：** 时间特征对预测精度至关重要，因为气温具有明显的季节性和日变化规律。

## 3. 预测结果说明

- **输出单位**: 模型输出的温度单位是开尔文（K）
- **转换为摄氏度**: `温度(°C) = 预测值(K) - 273.15`
- **预测目标**: 2米高度处的气温（t2m）

## 4. 常见问题

### Q: 如何查看可用的模型？
```python
from src.ml.model_manager import ModelManager

model_manager = ModelManager()
models = model_manager.list_models()

for model in models:
    print(f"模型ID: {model.id}")
    print(f"算法: {model.algorithm}")
    print(f"R²得分: {model.metrics.r2:.4f}")
    print(f"RMSE: {model.metrics.rmse:.4f}")
    print("---")
```

### Q: 预测失败怎么办？
1. 检查输入数据是否包含所有必需的特征列
2. 确保特征列名完全匹配（区分大小写）
3. 检查数据类型是否为数值型
4. 确保没有缺失值（NaN）

### Q: 如何提高预测精度？
1. 使用更准确的输入数据（特别是海表温度sst）
2. 确保地理坐标准确
3. 根据实际情况调整气压值

## 5. 完整示例脚本

运行完整的示例脚本：
```bash
python examples/use_climate_model.py
```

该脚本包含了4个不同的使用示例：
1. 简单的单点温度预测
2. 批量预测多个城市的温度
3. 区域温度分析
4. 使用预测引擎进行高级预测

## 6. 进阶使用

更多高级功能请参考：
- [完整使用指南](MODEL_USAGE_GUIDE.md)
- [API文档](../src/api/)
- [模型训练指南](../scripts/train_climate_model.py)

---

**注意**: 确保在运行预测之前已经训练了模型。如果没有可用的模型，请先运行 `python scripts/train_climate_model.py` 进行模型训练。