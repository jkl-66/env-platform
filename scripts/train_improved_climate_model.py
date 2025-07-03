#!/usr/bin/env python3
"""
改进的气候模型训练脚本
解决数据变异性和时间特征利用问题
"""

import asyncio
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from src.ml.model_manager import ModelManager, ModelConfig, ModelType
from src.data_processing.grib_processor import GRIBProcessor

async def main():
    """改进的模型训练主函数"""
    try:
        logger.info("开始改进的气候模型训练...")
        
        # 1. 初始化组件
        model_manager = ModelManager()
        processor = GRIBProcessor()
        
        # 2. 加载更多样化的数据
        logger.info("加载GRIB数据...")
        df = processor.process_grib_to_dataframe(
            'data/raw/6cd7cc57755a5204a65bc7db615cd36b.grib', 
            sample_size=200  # 增加样本数量
        )
        
        if df.empty:
            logger.error("无法加载数据")
            return
            
        logger.info(f"加载了 {len(df)} 条数据")
        logger.info(f"原始列: {list(df.columns)}")
        
        # 3. 数据预处理和特征工程
        logger.info("进行特征工程...")
        
        # 处理时间特征
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            
            # 提取时间特征
            df['year'] = df['time'].dt.year
            df['month'] = df['time'].dt.month
            df['day'] = df['time'].dt.day
            df['hour'] = df['time'].dt.hour
            df['day_of_year'] = df['time'].dt.dayofyear
            df['season'] = df['month'].map({
                12: 1, 1: 1, 2: 1,  # 冬季
                3: 2, 4: 2, 5: 2,   # 春季
                6: 3, 7: 3, 8: 3,   # 夏季
                9: 4, 10: 4, 11: 4  # 秋季
            })
            
            # 周期性时间特征
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            logger.info(f"时间范围: {df['time'].min()} 到 {df['time'].max()}")
        
        # 4. 特征选择 - 只使用有变异性的特征
        logger.info("分析特征变异性...")
        
        # 获取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除时间列
        exclude_cols = ['time', 'valid_time']
        for col in exclude_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        # 分析每个特征的变异性
        useful_features = []
        for col in numeric_cols:
            if col in df.columns:
                try:
                    # 确保是数值类型
                    if df[col].dtype in ['object', 'datetime64[ns]', 'timedelta64[ns]']:
                        continue
                        
                    std_dev = df[col].std()
                    unique_count = df[col].nunique()
                    
                    # 只保留有变异性的特征（标准差>0且唯一值>1）
                    if pd.notna(std_dev) and std_dev > 0 and unique_count > 1:
                        useful_features.append(col)
                        logger.info(f"保留特征 {col}: 标准差={std_dev:.6f}, 唯一值={unique_count}")
                    else:
                        logger.info(f"排除特征 {col}: 标准差={std_dev:.6f}, 唯一值={unique_count}")
                except Exception as e:
                    logger.warning(f"分析特征 {col} 时出错: {e}")
                    continue
        
        logger.info(f"有用的特征: {useful_features}")
        
        # 5. 确定目标变量和特征
        target_candidates = ['t2m', 'temperature', 'temp']
        target = None
        
        for candidate in target_candidates:
            if candidate in useful_features:
                target = candidate
                break
        
        if target is None:
            logger.error("未找到合适的目标变量")
            return
        
        # 特征列表（排除目标变量）
        features = [col for col in useful_features if col != target]
        
        if len(features) < 2:
            logger.error(f"特征数量不足: {len(features)}")
            return
        
        logger.info(f"目标变量: {target}")
        logger.info(f"特征变量: {features}")
        
        # 6. 数据增强 - 创建更多样化的训练样本
        logger.info("进行数据增强...")
        
        # 添加一些噪声来增加数据多样性
        enhanced_data = []
        
        for _, row in df.iterrows():
            # 原始数据
            enhanced_data.append(row.to_dict())
            
            # 添加轻微噪声的数据（模拟测量误差）
            for i in range(2):  # 为每个原始样本生成2个噪声版本
                noisy_row = row.copy()
                
                # 为气候变量添加小量噪声
                for feature in ['latitude', 'longitude', 't2m', 'msl', 'sst', 'sp']:
                    if feature in noisy_row and pd.notna(noisy_row[feature]):
                        noise_factor = 0.01  # 1%的噪声
                        noise = np.random.normal(0, abs(noisy_row[feature]) * noise_factor)
                        noisy_row[feature] += noise
                
                enhanced_data.append(noisy_row.to_dict())
        
        enhanced_df = pd.DataFrame(enhanced_data)
        logger.info(f"数据增强后: {len(enhanced_df)} 条数据")
        
        # 7. 清理数据
        required_columns = features + [target]
        enhanced_df = enhanced_df.dropna(subset=required_columns)
        
        if len(enhanced_df) < 10:
            logger.error(f"清理后数据不足: {len(enhanced_df)}")
            return
        
        logger.info(f"最终训练数据: {len(enhanced_df)} 条")
        
        # 8. 训练模型
        logger.info("训练改进的模型...")
        
        # 根据数据量调整模型参数
        n_estimators = min(100, max(20, len(enhanced_df) // 5))
        max_depth = min(10, max(3, len(enhanced_df) // 20))
        
        model_config = ModelConfig(
            name="improved_temperature_prediction_rf",
            model_type=ModelType.REGRESSION.value,
            algorithm="random_forest",
            parameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            },
            features=features,
            target=target
        )
        
        # 创建训练数据
        training_data = enhanced_df[features + [target]].copy()
        
        # 训练模型
        model_id = await model_manager.create_model(config=model_config, data=training_data)
        
        if model_id:
            logger.info(f"改进模型训练成功！模型ID: {model_id}")
            
            # 获取模型信息
            model_info = model_manager.get_model_info(model_id)
            if model_info and model_info.metrics:
                logger.info(f"模型评估指标: {model_info.metrics}")
                
                # 显示特征重要性
                if hasattr(model_info.metrics, 'feature_importance'):
                    logger.info("特征重要性:")
                    for feature, importance in model_info.metrics.feature_importance.items():
                        logger.info(f"  {feature}: {importance:.6f}")
        else:
            logger.error("改进模型训练失败")
            
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())