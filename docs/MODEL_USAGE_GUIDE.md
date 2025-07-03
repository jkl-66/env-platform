# 气候模型使用指南

本指南将详细介绍如何使用已训练的气候模型进行预测任务。

## 1. 模型概览

### 已训练的模型
- **模型ID**: `temperature_prediction_rf_20250703_105653`
- **模型类型**: 回归模型（Regression）
- **算法**: 随机森林（Random Forest）
- **目标变量**: 2米温度（t2m）
- **特征变量**: 纬度、经度、海平面压力、海表温度、地面压力等
- **模型性能**: R²=0.851，RMSE=0.497

## 2. 使用方式

### 2.1 直接使用模型管理器进行预测

```python
import pandas as pd
import asyncio
from src.ml.model_manager import ModelManager

# 创建模型管理器
model_manager = ModelManager()

# 准备预测数据
predict_data = pd.DataFrame({
    'latitude': [40.0, 41.0, 42.0],
    'longitude': [116.0, 117.0, 118.0],
    'msl': [101325.0, 101300.0, 101350.0],  # 海平面压力
    'sst': [15.5, 16.0, 15.8],              # 海表温度
    'sp': [101000.0, 100980.0, 101020.0],   # 地面压力
    'quality_score': [1.0, 1.0, 1.0]
})

# 使用模型进行预测
async def predict_temperature():
    model_id = "temperature_prediction_rf_20250703_105653"
    predictions = await model_manager.predict(model_id, predict_data)
    print(f"预测的温度值: {predictions}")
    return predictions

# 运行预测
result = asyncio.run(predict_temperature())
```

### 2.2 使用预测引擎进行复杂预测任务

```python
import pandas as pd
import asyncio
from src.ml.prediction_engine import PredictionEngine, PredictionConfig, PredictionType
from src.ml.model_manager import ModelManager

# 创建预测引擎
model_manager = ModelManager()
prediction_engine = PredictionEngine(model_manager=model_manager)

# 配置预测任务
config = PredictionConfig(
    prediction_type=PredictionType.REAL_TIME.value,
    target_variable="t2m",
    prediction_horizon=1,  # 预测1步
    temporal_resolution="daily",
    confidence_interval=0.95
)

# 准备输入数据
input_data = pd.DataFrame({
    'latitude': [39.9, 40.1, 40.3],
    'longitude': [116.3, 116.4, 116.5],
    'msl': [101325.0, 101320.0, 101330.0],
    'sst': [15.2, 15.4, 15.6],
    'sp': [101000.0, 100995.0, 101005.0],
    'quality_score': [1.0, 1.0, 1.0]
})

async def run_prediction_task():
    # 创建预测任务
    model_ids = ["temperature_prediction_rf_20250703_105653"]
    task_id = await prediction_engine.create_prediction_task(
        config=config,
        model_ids=model_ids,
        input_data=input_data
    )
    
    print(f"创建预测任务: {task_id}")
    
    # 执行预测任务
    result = await prediction_engine.run_prediction_task(task_id)
    
    print(f"预测结果: {result.predictions}")
    if result.confidence_lower is not None:
        print(f"置信区间下界: {result.confidence_lower}")
        print(f"置信区间上界: {result.confidence_upper}")
    
    return result

# 运行预测任务
result = asyncio.run(run_prediction_task())
```

### 2.3 批量预测

```python
import pandas as pd
import asyncio
from src.ml.model_manager import ModelManager

# 创建大量预测数据
def create_batch_data(num_points=1000):
    import numpy as np
    
    # 生成随机的地理坐标和气象数据
    np.random.seed(42)
    
    data = pd.DataFrame({
        'latitude': np.random.uniform(20, 60, num_points),
        'longitude': np.random.uniform(70, 140, num_points),
        'msl': np.random.uniform(99000, 103000, num_points),
        'sst': np.random.uniform(10, 25, num_points),
        'sp': np.random.uniform(98000, 102000, num_points),
        'quality_score': np.ones(num_points)
    })
    
    return data

async def batch_predict():
    model_manager = ModelManager()
    model_id = "temperature_prediction_rf_20250703_105653"
    
    # 创建批量数据
    batch_data = create_batch_data(1000)
    
    print(f"开始批量预测 {len(batch_data)} 个数据点...")
    
    # 执行批量预测
    predictions = await model_manager.predict(model_id, batch_data)
    
    # 将预测结果添加到原数据中
    batch_data['predicted_temperature'] = predictions
    
    print(f"批量预测完成！")
    print(f"预测温度范围: {predictions.min():.2f}°C 到 {predictions.max():.2f}°C")
    print(f"平均预测温度: {predictions.mean():.2f}°C")
    
    # 保存结果
    output_path = "data/predictions/batch_temperature_predictions.csv"
    batch_data.to_csv(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")
    
    return batch_data

# 运行批量预测
result = asyncio.run(batch_predict())
```

## 3. 输入数据要求

### 3.1 必需的特征列
模型需要以下21个输入特征（包括时间特征）：

#### 基础气象特征（9个）
| 特征名 | 描述 | 单位 | 示例值 |
|--------|------|------|--------|
| `latitude` | 纬度 | 度 | 39.9042 |
| `longitude` | 经度 | 度 | 116.4074 |
| `number` | 模型集合成员编号 | - | 0 |
| `step` | 预报时效 | 小时 | 0 |
| `surface` | 地面气压 | hPa | 1013.25 |
| `msl` | 海平面气压 | Pa | 1013.25 |
| `sst` | 海表温度 | K | 285.15 |
| `sp` | 地面气压 | Pa | 101325.0 |
| `quality_score` | 数据质量评分 | - | 0.95 |

#### 时间特征（12个）
| 特征名 | 描述 | 单位 | 示例值 |
|--------|------|------|--------|
| `year` | 年份 | - | 2025 |
| `month` | 月份 | - | 7 |
| `day` | 日期 | - | 3 |
| `hour` | 小时 | - | 14 |
| `day_of_year` | 一年中的第几天 | - | 184 |
| `season` | 季节编码 | - | 3 (夏季) |
| `month_sin` | 月份正弦编码 | - | 0.866 |
| `month_cos` | 月份余弦编码 | - | 0.5 |
| `day_sin` | 日期正弦编码 | - | 0.707 |
| `day_cos` | 日期余弦编码 | - | 0.707 |
| `hour_sin` | 小时正弦编码 | - | -0.866 |
| `hour_cos` | 小时余弦编码 | - | -0.5 |

**注意：** 时间特征对于准确预测至关重要，因为气温具有明显的季节性和日变化特征。

### 3.2 数据格式要求
- 所有数值列必须是数值类型
- 不能包含NaN值（会自动用均值填充）
- 数据范围应在训练数据的合理范围内

### 3.3 数据示例
```python
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

# 设置预测时间
prediction_time = datetime.now()
time_features = create_time_features(prediction_time)

# 正确的输入数据格式（包含时间特征）
data = pd.DataFrame({
    'latitude': [39.9042],      # 北京纬度
    'longitude': [116.4074],    # 北京经度
    'number': [0],              # 模型集合成员编号
    'step': [0],                # 预报时效
    'surface': [1013.25],       # 地面气压
    'msl': [1013.25],           # 海平面气压
    'sst': [285.15],            # 海表温度（K）
    'sp': [101325.0],           # 地面压力
    'quality_score': [0.95],    # 高质量数据
    **{k: [v] for k, v in time_features.items()}  # 添加时间特征
})
```

## 4. 预测结果解释

### 4.1 输出格式
- 预测结果是numpy数组，包含预测的温度值（°C）
- 对于单点预测，返回单个数值
- 对于批量预测，返回与输入数据行数相同的数组

### 4.2 模型性能指标
- **R²得分**: 0.851（模型能解释85.1%的温度变异）
- **RMSE**: 0.497°C（均方根误差）
- **MAE**: 0.384°C（平均绝对误差）

### 4.3 特征重要性
根据训练结果，各特征的重要性为：
- 海表温度（sst）: 98.4%（最重要）
- 纬度（latitude）: 1.5%
- 经度（longitude）: 0.07%
- 其他特征: <0.1%

## 5. 实际应用场景

### 5.1 天气预报
```python
# 批量预测多个城市
prediction_time = datetime.now()
time_features = create_time_features(prediction_time)

cities_data = pd.DataFrame({
    'city': ['北京', '上海', '广州'],
    'latitude': [39.9042, 31.2304, 23.1291],
    'longitude': [116.4074, 121.4737, 113.2644],
    'number': [0, 0, 0],
    'step': [0, 0, 0],
    'surface': [1013.25, 1013.25, 1013.25],
    'msl': [1013.25, 1013.25, 1013.25],
    'sst': [285.15, 288.15, 295.15],
    'sp': [101325.0, 101325.0, 101325.0],
    'quality_score': [0.95, 0.95, 0.95]
})

# 为每个城市添加时间特征
for key, value in time_features.items():
    cities_data[key] = value

# 预测各城市温度
predictions = await model_manager.predict(model_id, cities_data)
cities_data['predicted_temp_c'] = predictions - 273.15
print(f"预测时间: {prediction_time.strftime('%Y-%m-%d %H:%M:%S')} ({time_features['month']}月)")
print(cities_data[['city', 'predicted_temp_c']])
```

### 5.2 气候监测
```python
# 监测某区域的温度变化
region_data = pd.DataFrame({
    'latitude': np.arange(35, 45, 0.5),  # 35°N到45°N
    'longitude': [110] * 20,             # 固定经度110°E
    'msl': [101325.0] * 20,
    'sst': np.linspace(12, 20, 20),      # 海表温度梯度
    'sp': [101000.0] * 20,
    'quality_score': [1.0] * 20
})

predictions = await model_manager.predict(model_id, region_data)
# 分析温度随纬度的变化
```

## 6. 注意事项

### 6.1 模型限制
- 模型基于历史GRIB数据训练，适用于类似的气象条件
- 预测精度在训练数据范围内最高
- 极端天气条件下预测可能不准确

### 6.2 数据质量
- 输入数据质量直接影响预测精度
- 建议使用高质量的气象观测数据
- 避免使用明显异常的数据值

### 6.3 性能优化
- 批量预测比单点预测更高效
- 大数据量预测时考虑分批处理
- 可以缓存模型以避免重复加载

## 7. 故障排除

### 7.1 常见错误
```python
# 错误：缺少必需特征
# ValueError: 缺少特征列: ['msl', 'sst']

# 解决：确保包含所有必需特征
data = data[['latitude', 'longitude', 'msl', 'sst', 'sp', 'quality_score']]
```

### 7.2 模型加载问题
```python
# 检查模型是否存在
model_info = model_manager.get_model_info(model_id)
if model_info is None:
    print(f"模型不存在: {model_id}")
    # 列出可用模型
    models = model_manager.list_models()
    print(f"可用模型: {[m.id for m in models]}")
```

### 7.3 数据格式问题
```python
# 检查数据类型
print(data.dtypes)
# 转换数据类型
data = data.astype(float)
```

## 8. 扩展功能

### 8.1 模型集成
可以使用多个模型进行集成预测以提高精度：

```python
# 使用多个模型进行集成预测
config = PredictionConfig(
    prediction_type=PredictionType.ENSEMBLE.value,
    target_variable="t2m",
    prediction_horizon=1,
    ensemble_methods=["average", "weighted"]
)

model_ids = [
    "temperature_prediction_rf_20250703_105653",
    # 可以添加其他模型ID
]
```

### 8.2 时间序列预测
对于时间序列数据，可以进行多步预测：

```python
config = PredictionConfig(
    prediction_type=PredictionType.TIME_SERIES.value,
    target_variable="t2m",
    prediction_horizon=7,  # 预测未来7天
    temporal_resolution="daily"
)
```

这个指南涵盖了使用已训练气候模型的各种场景和方法。根据具体需求选择合适的使用方式。