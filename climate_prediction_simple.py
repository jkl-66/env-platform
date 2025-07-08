#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气候预测系统简化示例
基于XGBoost和Transformer模型的气候预测演示

作者: AI Assistant
日期: 2024-12-19
"""

import os
import sys
import logging
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from climate_prediction_models import ClimatePredictionSystem
from grib_to_mysql import GRIBToMySQLProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'climate_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_grib_data_to_mysql():
    """
    加载GRIB数据到MySQL数据库
    """
    logger.info("开始加载GRIB数据到MySQL数据库...")
    
    try:
        # 初始化GRIB处理器
        processor = GRIBToMySQLProcessor()
        
        # 加载GRIB文件（如果文件不存在，会自动生成模拟数据）
        grib_files = [
            'data/temperature_data.grib',
            'data/pressure_data.grib',
            'data/humidity_data.grib'
        ]
        
        for grib_file in grib_files:
            logger.info(f"处理文件: {grib_file}")
            
            # 加载GRIB数据
            data = processor.load_grib_data(grib_file)
            
            if data is not None:
                # 处理数据
                processed_data = processor.process_grib_data(data, grib_file)
                
                # 保存到数据库
                processor.save_to_database(processed_data)
                logger.info(f"{grib_file} 数据已保存到数据库")
            else:
                logger.warning(f"无法加载 {grib_file}，跳过")
        
        # 显示数据库统计信息
        stats = processor.get_statistics()
        logger.info("数据库统计信息:")
        logger.info(f"总记录数: {stats['total_records']}")
        logger.info(f"变量统计: {stats['variable_stats']}")
        logger.info(f"时间范围: {stats['time_range']}")
        logger.info(f"空间范围: {stats['spatial_range']}")
        
        processor.close()
        logger.info("GRIB数据加载完成")
        
    except Exception as e:
        logger.error(f"GRIB数据加载失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def run_climate_prediction():
    """
    运行气候预测系统
    """
    logger.info("开始运行气候预测系统...")
    
    try:
        # 1. 初始化预测系统
        prediction_system = ClimatePredictionSystem()
        
        # 2. 加载数据
        logger.info("正在从MySQL数据库加载气象数据...")
        prediction_system.load_data_from_mysql(
            lat_range=(30.67, 31.88),  # 北纬30°40′至31°53′
            lon_range=(120.87, 122.20),  # 东经120°52′至122°12′
            start_date='2020-01-01',
            end_date='2024-12-31'
        )
        
        # 3. 准备训练数据
        logger.info("正在准备训练数据...")
        prediction_system.prepare_features()
        X, y = prediction_system.create_sequences(sequence_length=30, target_days=1)
        
        # 4. 训练XGBoost模型
        logger.info("开始训练XGBoost模型...")
        xgb_metrics = prediction_system.train_xgboost_model(test_size=0.2)
        
        # 5. 训练Transformer模型
        logger.info("开始训练Transformer模型...")
        transformer_metrics = prediction_system.train_transformer_model(
            test_size=0.2,
            batch_size=32,
            epochs=20,  # 减少训练轮数以加快演示
            learning_rate=0.001
        )
        
        # 6. 模型性能对比
        logger.info("模型性能对比:")
        logger.info(f"XGBoost模型 - MAE: {xgb_metrics['mae']:.4f}, RMSE: {xgb_metrics['rmse']:.4f}, R²: {xgb_metrics['r2']:.4f}")
        logger.info(f"Transformer模型 - MAE: {transformer_metrics['mae']:.4f}, RMSE: {transformer_metrics['rmse']:.4f}, R²: {transformer_metrics['r2']:.4f}")
        
        # 7. 预测未来气温
        logger.info("正在预测未来30天气温...")
        future_predictions = prediction_system.predict_future(days=30)
        
        # 8. 创建可视化
        logger.info("正在创建可视化图表...")
        prediction_system.create_visualizations(future_predictions)
        
        # 9. 保存模型
        logger.info("正在保存模型...")
        prediction_system.save_models()
        
        # 10. 保存预测结果
        output_path = f"outputs/future_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs('outputs', exist_ok=True)
        future_predictions.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"预测结果已保存至: {output_path}")
        
        # 11. 显示预测结果摘要
        logger.info("预测结果摘要:")
        logger.info(f"预测期间: {future_predictions['date'].min()} 至 {future_predictions['date'].max()}")
        logger.info(f"平均预测温度: {future_predictions['predicted_temperature'].mean():.2f}°C")
        logger.info(f"最高预测温度: {future_predictions['predicted_temperature'].max():.2f}°C")
        logger.info(f"最低预测温度: {future_predictions['predicted_temperature'].min():.2f}°C")
        
        # 12. 显示前10天的详细预测
        logger.info("未来10天详细预测:")
        for i, row in future_predictions.head(10).iterrows():
            logger.info(f"{row['date'].strftime('%Y-%m-%d')}: {row['predicted_temperature']:.2f}°C")
        
        logger.info("气候预测系统运行完成!")
        
    except Exception as e:
        logger.error(f"气候预测系统运行出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """
    主函数
    """
    logger.info("气候预测系统演示开始")
    logger.info("目标区域: 东经120°52′至122°12′，北纬30°40′至31°53′")
    logger.info("使用模型: XGBoost + Transformer")
    
    # 步骤1: 加载GRIB数据到MySQL
    logger.info("\n" + "="*50)
    logger.info("步骤1: 加载GRIB数据到MySQL数据库")
    logger.info("="*50)
    load_grib_data_to_mysql()
    
    # 步骤2: 运行气候预测
    logger.info("\n" + "="*50)
    logger.info("步骤2: 运行气候预测系统")
    logger.info("="*50)
    run_climate_prediction()
    
    logger.info("\n" + "="*50)
    logger.info("气候预测系统演示完成!")
    logger.info("="*50)
    
    # 显示输出文件位置
    logger.info("\n输出文件位置:")
    logger.info("- 模型文件: models/climate_prediction/")
    logger.info("- 预测结果: outputs/")
    logger.info("- 可视化图表: outputs/")
    logger.info("- 日志文件: 当前目录")


if __name__ == "__main__":
    main()