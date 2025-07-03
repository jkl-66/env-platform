#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气候模型训练脚本

该脚本演示了如何使用项目中的模块来训练、评估和保存气候模型。
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_processing.data_storage import DataStorage
from src.data_processing.data_processor import DataProcessor, ProcessingConfig
from src.data_processing.grib_processor import GRIBProcessor
from src.ml.model_manager import ModelManager, ModelConfig, ModelType
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def main():
    """主函数，执行模型训练流程"""
    logger.info("开始执行气候模型训练脚本")

    try:
        # 1. 初始化组件
        data_storage = DataStorage()
        await data_storage.initialize()  # 初始化数据库连接
        data_processor = DataProcessor(storage=data_storage)
        model_manager = ModelManager(storage=data_storage)

        # 2. 加载数据 - 直接从GRIB文件中提取实际气候数据
        logger.info("正在从GRIB文件加载实际气候数据...")
        
        # 获取GRIB文件路径
        metadata_query = "SELECT file_path FROM climate_data_records WHERE file_format = '.grib' ORDER BY created_at DESC"
        try:
            metadata_df = await data_storage.fetch_data_as_dataframe(metadata_query)
            if metadata_df.empty:
                logger.error("数据库中没有GRIB文件记录。脚本将退出。")
                return

            grib_processor = GRIBProcessor()
            all_data_list = []
            
            for index, row in metadata_df.iterrows():
                file_path = project_root / row['file_path']
                if file_path.exists():
                    logger.info(f"正在从GRIB文件提取气候数据: {file_path}")
                    # 提取适量数据进行训练（100万个数据点应该足够训练）
                    df = grib_processor.process_grib_to_dataframe(str(file_path), sample_size=1000000)
                    if df is not None and not df.empty:
                        logger.info(f"从文件 {file_path.name} 提取了 {len(df)} 条气候数据记录")
                        logger.info(f"GRIB数据列: {list(df.columns)}")
                        
                        # 检查原始数据中的NaN值
                        nan_counts = df.isnull().sum()
                        logger.info(f"原始数据中的NaN值统计: {nan_counts.to_dict()}")
                        
                        all_data_list.append(df)
                else:
                    logger.warning(f"GRIB文件未找到: {file_path}")
            
            if not all_data_list:
                logger.error("没有从GRIB文件中提取到任何气候数据。")
                return

            raw_df = pd.concat(all_data_list, ignore_index=True)
            logger.info(f"成功从GRIB文件提取了 {len(raw_df)} 条气候数据记录")
            
            # 数据处理步骤
            logger.info("开始数据处理以保证训练结果的正确性...")
            
            # 1. 检查并处理NaN值
            nan_counts_before = raw_df.isnull().sum()
            logger.info(f"处理前的NaN值统计: {nan_counts_before.to_dict()}")
            
            # 2. 数据质量检查 - 移除完全为NaN的列
            cols_before = len(raw_df.columns)
            raw_df = raw_df.dropna(axis=1, how='all')
            cols_after = len(raw_df.columns)
            if cols_before != cols_after:
                logger.info(f"移除了 {cols_before - cols_after} 个完全为NaN的列")
            
            # 3. 移除包含过多NaN值的行（超过50%的列为NaN）
            rows_before = len(raw_df)
            threshold = len(raw_df.columns) * 0.5
            raw_df = raw_df.dropna(thresh=threshold)
            rows_after = len(raw_df)
            if rows_before != rows_after:
                logger.info(f"移除了 {rows_before - rows_after} 行包含过多NaN值的数据")
            
            # 4. 对于数值列，使用前向填充和后向填充处理剩余的NaN值
            numeric_columns = raw_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if raw_df[col].isnull().any():
                    # 先尝试前向填充，再后向填充
                    raw_df[col] = raw_df[col].fillna(method='ffill').fillna(method='bfill')
                    # 如果还有NaN，用该列的中位数填充
                    if raw_df[col].isnull().any():
                        median_val = raw_df[col].median()
                        raw_df[col] = raw_df[col].fillna(median_val)
            
            # 5. 异常值检测和处理（使用IQR方法）
            for col in numeric_columns:
                if col not in ['latitude', 'longitude', 'time']:  # 保留坐标和时间列
                    Q1 = raw_df[col].quantile(0.25)
                    Q3 = raw_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_before = len(raw_df[(raw_df[col] < lower_bound) | (raw_df[col] > upper_bound)])
                    if outliers_before > 0:
                        # 将异常值替换为边界值
                        raw_df[col] = raw_df[col].clip(lower=lower_bound, upper=upper_bound)
                        logger.info(f"列 {col}: 处理了 {outliers_before} 个异常值")
            
            # 6. 最终数据质量检查
            nan_counts_after = raw_df.isnull().sum()
            logger.info(f"处理后的NaN值统计: {nan_counts_after.to_dict()}")
            logger.info(f"最终数据集大小: {raw_df.shape}")

        except Exception as e:
            logger.error(f"从GRIB文件加载数据时发生错误: {e}")
            return

        if raw_df is None or raw_df.empty:
            logger.error("未能加载数据，或加载的数据为空。脚本将退出。")
            return

        logger.info(f"成功加载 {len(raw_df)} 条数据")
        logger.info("原始数据 (raw_df) head:")
        logger.info(raw_df.head())
        logger.info("原始数据 (raw_df) info:")
        raw_df.info(buf=sys.stdout) # Redirect info() output to logger

        # 3. 数据预处理 - 跳过可能产生NaN的处理步骤
        logger.info("跳过复杂的数据预处理，直接使用原始GRIB数据进行训练...")
        
        # 直接使用原始数据，只进行基本的数据类型检查
        processed_df = raw_df.copy()
        
        # 检查原始数据中的NaN情况
        logger.info("原始数据NaN统计:")
        for col in processed_df.columns:
            nan_count = processed_df[col].isna().sum()
            total_count = len(processed_df)
            logger.info(f"  {col}: {nan_count}/{total_count} NaN值")
        
        logger.info("使用原始GRIB数据进行训练")
        logger.info(f"数据列: {processed_df.columns.tolist()}")
        logger.info(f"数据形状: {processed_df.shape}")

        # 4. 特征工程 - 使用实际气候数据并正确处理时间信息
        logger.info("正在进行特征工程...")
        
        # 检查从GRIB文件提取的气候数据列
        logger.info(f"GRIB数据列: {list(processed_df.columns)}")
        logger.info(f"数据形状: {processed_df.shape}")
        logger.info(f"数据类型: {processed_df.dtypes.to_dict()}")
        
        # 处理时间信息 - 从时间列中提取有用的时间特征
        if 'time' in processed_df.columns:
            logger.info("提取时间特征...")
            processed_df['time'] = pd.to_datetime(processed_df['time'])
            
            # 提取时间特征
            processed_df['year'] = processed_df['time'].dt.year
            processed_df['month'] = processed_df['time'].dt.month
            processed_df['day'] = processed_df['time'].dt.day
            processed_df['hour'] = processed_df['time'].dt.hour
            processed_df['day_of_year'] = processed_df['time'].dt.dayofyear
            processed_df['season'] = processed_df['month'].map({
                12: 1, 1: 1, 2: 1,  # 冬季
                3: 2, 4: 2, 5: 2,   # 春季
                6: 3, 7: 3, 8: 3,   # 夏季
                9: 4, 10: 4, 11: 4  # 秋季
            })
            
            # 添加周期性时间特征（使用三角函数编码）
            processed_df['month_sin'] = np.sin(2 * np.pi * processed_df['month'] / 12)
            processed_df['month_cos'] = np.cos(2 * np.pi * processed_df['month'] / 12)
            processed_df['day_sin'] = np.sin(2 * np.pi * processed_df['day_of_year'] / 365.25)
            processed_df['day_cos'] = np.cos(2 * np.pi * processed_df['day_of_year'] / 365.25)
            processed_df['hour_sin'] = np.sin(2 * np.pi * processed_df['hour'] / 24)
            processed_df['hour_cos'] = np.cos(2 * np.pi * processed_df['hour'] / 24)
            
            logger.info(f"时间范围: {processed_df['time'].min()} 到 {processed_df['time'].max()}")
            logger.info(f"包含的年份: {sorted(processed_df['year'].unique())}")
            logger.info(f"包含的月份: {sorted(processed_df['month'].unique())}")
            
        # 处理valid_time列（如果存在且与time不同）
        if 'valid_time' in processed_df.columns and 'time' in processed_df.columns:
            if not processed_df['valid_time'].equals(processed_df['time']):
                logger.info("处理valid_time列...")
                processed_df['valid_time'] = pd.to_datetime(processed_df['valid_time'])
                processed_df['forecast_lead_hours'] = (processed_df['valid_time'] - processed_df['time']).dt.total_seconds() / 3600
        
        # 选择数值型气候变量作为特征
        numeric_columns = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除原始时间列，但保留提取的时间特征
        exclude_columns = ['time', 'valid_time'] if 'time' in processed_df.columns else []
        for col in exclude_columns:
            if col in numeric_columns:
                numeric_columns.remove(col)
        
        logger.info(f"可用的数值气候变量（包含时间特征）: {numeric_columns}")
        
        if not numeric_columns:
            logger.error("没有找到数值气候变量用于训练。脚本将退出。")
            return
        
        # 检查每列的非空值数量
        logger.info("各气候变量非空值统计:")
        for col in numeric_columns:
            non_null_count = processed_df[col].notna().sum()
            total_count = len(processed_df)
            logger.info(f"  {col}: {non_null_count}/{total_count}")
        
        if len(numeric_columns) < 2:
            logger.error(f"气候数值变量不足，无法进行训练。当前变量: {numeric_columns}")
            return
        
        # 选择目标变量和特征 - 使用实际的气候变量
        # 通常选择温度作为目标变量，其他变量作为特征
        target_candidates = ['t2m', 'temperature', 'temp', 'T']
        target = None
        
        for candidate in target_candidates:
            if candidate in numeric_columns:
                target = candidate
                break
        
        if target is None:
            # 如果没有找到温度变量，使用第一个数值列
            target = numeric_columns[0]
        
        features = [col for col in numeric_columns if col != target]
        
        logger.info(f"目标变量（预测目标）: {target}")
        logger.info(f"特征变量: {features}")
        logger.info(f"时间特征: {[col for col in features if any(time_word in col for time_word in ['year', 'month', 'day', 'hour', 'season', 'sin', 'cos'])]}")
        
        # 确保有足够的特征
        if len(features) < 1:
            logger.error("没有足够的特征变量用于训练。")
            return
        
        # 确保目标变量和特征都存在
        required_columns = features + [target]
        missing_columns = [col for col in required_columns if col not in processed_df.columns]
        if missing_columns:
            logger.error(f"缺少必要的列: {missing_columns}")
            return
            
        logger.info(f"执行dropna前的数据行数: {len(processed_df)}")
        # 只对必要的列执行dropna
        processed_df = processed_df.dropna(subset=required_columns)
        logger.info(f"执行dropna后的数据行数: {len(processed_df)}")
        
        # 准备训练数据
        X = processed_df[features]
        y = processed_df[target]
        
        # 删除包含NaN的行
        logger.info(f"删除NaN前的数据行数: {len(X)}")
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        logger.info(f"删除NaN后的数据行数: {len(X)}")

        if processed_df.empty:
            logger.error("经过特征工程和dropna后，没有数据可用于训练。脚本将退出。")
            return

        logger.info("特征工程完成")

        # 5. 定义和训练模型 (以文件大小均值预测为例)
        logger.info("正在定义和训练模型...")

        # 确保所有特征都是数值类型
        for col in features:
            if not pd.api.types.is_numeric_dtype(processed_df[col]):
                logger.warning(f"特征 '{col}' 不是数值类型，将尝试转换。")
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        processed_df.dropna(subset=features + [target], inplace=True)

        # 检查数据量是否足够 - 降低最小要求
        if len(processed_df) < 2:
            logger.error(f"数据量不足，无法进行模型训练。当前数据行数: {len(processed_df)}")
            return
        
        logger.info(f"使用 {len(processed_df)} 行数据进行模型训练")

        # 根据数据量调整模型参数
        n_estimators = min(50, max(10, len(processed_df) // 2))
        max_depth = min(5, max(3, len(processed_df) // 10))
        logger.info(f"根据数据量调整模型参数: n_estimators={n_estimators}, max_depth={max_depth}")

        model_config = ModelConfig(
            name="temperature_prediction_rf",
            model_type=ModelType.REGRESSION.value,
            algorithm="random_forest",
            parameters={"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42},
            features=features,
            target=target
        )

        # 使用准备好的训练数据创建模型
        training_data = processed_df[features + [target]].copy()
        model_id = await model_manager.create_model(config=model_config, data=training_data)

        if model_id:
            logger.info(f"模型训练成功！模型ID: {model_id}")
            # 打印模型评估结果
            model_info = model_manager.get_model_info(model_id)
            if model_info and model_info.metrics:
                logger.info(f"模型评估指标: {model_info.metrics}")
        else:
            logger.error("模型训练失败。")

    except Exception as e:
        logger.error(f"脚本执行过程中发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())