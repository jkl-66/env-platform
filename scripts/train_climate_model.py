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

        # 2. 加载数据
        logger.info("正在加载数据...")
        query = "SELECT * FROM climate_data_records ORDER BY created_at DESC"
        try:
            metadata_df = await data_storage.fetch_data_as_dataframe(query)
            if metadata_df.empty:
                logger.error("未能加载元数据，或加载的元数据为空。脚本将退出。")
                return

            grib_processor = GRIBProcessor()
            all_data_list = []
            for index, row in metadata_df.iterrows():
                file_path = project_root / row['file_path']
                if file_path.exists():
                    logger.info(f"正在处理文件: {file_path}")
                    df = grib_processor.process_grib_to_dataframe(str(file_path))
                    if df is not None and not df.empty:
                        all_data_list.append(df)
                else:
                    logger.warning(f"文件未找到: {file_path}")
            
            if not all_data_list:
                logger.error("没有从GRIB文件中加载到任何数据。")
                return

            raw_df = pd.concat(all_data_list, ignore_index=True)

        except Exception as e:
            logger.error(f"加载和处理GRIB数据时发生错误: {e}")
            raw_df = pd.DataFrame() # 创建一个空的DataFrame

        if raw_df is None or raw_df.empty:
            logger.error("未能加载数据，或加载的数据为空。脚本将退出。")
            return

        logger.info(f"成功加载 {len(raw_df)} 条数据")
        logger.info("原始数据 (raw_df) head:")
        logger.info(raw_df.head())
        logger.info("原始数据 (raw_df) info:")
        raw_df.info(buf=sys.stdout) # Redirect info() output to logger

        # 3. 数据预处理
        logger.info("正在进行数据预处理...")
        processing_config = ProcessingConfig(
            outlier_method='iqr',
            interpolation_method='linear',
            normalization_method='zscore'
        )
        processed_df = await data_processor.process_climate_data(raw_df, config=processing_config, save_result=False)
        logger.info("数据预处理完成")
        logger.info("处理后数据 (processed_df) head:")
        logger.info(processed_df.head())
        logger.info("处理后数据 (processed_df) info:")
        processed_df.info(buf=sys.stdout) # Redirect info() output to logger
        logger.info(f"处理后的数据列: {processed_df.columns}")

        # 4. 特征工程 (示例)
        logger.info("正在进行特征工程...")
        # The index is already a DatetimeIndex set by the DataProcessor.
        processed_df['month'] = processed_df.index.month
        processed_df['day_of_year'] = processed_df.index.dayofyear

        features = ['month', 'day_of_year']
        target = 'file_size_mean'

        # Only add lag feature if there is enough data
        if len(processed_df) > 1:
            processed_df['temp_lag_1'] = processed_df[target].shift(1)
            features.append('temp_lag_1')

        logger.info(f"执行dropna前的数据行数: {len(processed_df)}")
        processed_df.dropna(inplace=True)
        logger.info(f"执行dropna后的数据行数: {len(processed_df)}")

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

        model_config = ModelConfig(
            name="temperature_prediction_rf",
            model_type=ModelType.REGRESSION.value,
            algorithm="random_forest",
            parameters={"n_estimators": 100, "max_depth": 10},
            features=features,
            target=target
        )

        model_id = await model_manager.create_model(config=model_config, data=processed_df)

        if model_id:
            logger.info(f"模型训练成功！模型ID: {model_id}")
            # 打印模型评估结果
            model_info = await model_manager.get_model_info(model_id)
            if model_info and model_info.metrics:
                logger.info(f"模型评估指标: {model_info.metrics}")
        else:
            logger.error("模型训练失败。")

    except Exception as e:
        logger.error(f"脚本执行过程中发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())