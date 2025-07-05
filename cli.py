#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 驱动的气候洞察与生态意识提升平台 - 命令行界面

整合历史气候数据分析、生态警示图像生成和区域气候预测功能。
提供完整的命令行操作界面。
"""

import sys
import os
import argparse
import logging
import asyncio
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入核心模块
try:
    from src.data_processing.data_collector import DataCollector as ClimateDataCollector
    from src.data_processing.data_storage import DataStorage as ClimateDataStore
    from src.data_processing.data_processor import DataProcessor
    from src.data_processing.grib_processor import GRIBProcessor
    from src.models.historical_climate_analyzer import HistoricalClimateAnalyzer, analyze_climate_data
    from environmental_image_generator import EnvironmentalImageGenerator
    from src.models.regional_climate_predictor import RegionalClimatePredictor, predict_regional_climate_risk
    from src.ml.model_manager import ModelManager
    from src.ml.prediction_engine import PredictionEngine
    from src.visualization.charts import ChartGenerator
    from src.utils.logger import setup_logger, get_logger
    from src.utils.config import load_config, get_config
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有依赖模块都已正确安装")
    sys.exit(1)


class ClimateInsightCLI:
    """气候洞察平台命令行接口"""
    
    def __init__(self, config_path: str = None):
        """初始化CLI"""
        # 设置日志
        setup_logger()
        self.logger = get_logger(__name__)
        
        # 加载配置
        if config_path:
            load_config(config_path)
        self.config = get_config()
        
        # 初始化核心组件
        self.data_collector = ClimateDataCollector()
        self.data_store = ClimateDataStore()
        self.data_processor = DataProcessor()
        self.grib_processor = GRIBProcessor(self.data_store)
        self.climate_analyzer = HistoricalClimateAnalyzer()
        self.image_generator = EnvironmentalImageGenerator()
        self.climate_predictor = RegionalClimatePredictor()
        self.model_manager = ModelManager()
        self.prediction_engine = PredictionEngine()
        self.chart_generator = ChartGenerator()
        
        self.logger.info("气候洞察平台CLI初始化完成")
    
    def analyze_historical_data(
        self,
        data_source: str,
        variables: list = None,
        start_date: str = None,
        end_date: str = None,
        output_dir: str = "output/analysis"
    ):
        """分析历史气候数据"""
        self.logger.info(f"开始分析历史数据: {data_source}")
        
        try:
            # 收集数据
            if data_source.endswith('.csv'):
                data = pd.read_csv(data_source)
                if 'date' in data.columns:
                    data['time'] = pd.to_datetime(data['date'])
                elif 'time' not in data.columns:
                    # 创建时间索引
                    data['time'] = pd.date_range(start='2000-01-01', periods=len(data), freq='D')
            elif data_source.endswith('.nc'):
                try:
                    import xarray as xr
                    data = xr.open_dataset(data_source)
                except ImportError:
                    self.logger.error("需要安装xarray库来处理NetCDF文件")
                    return None
            else:
                # 生成示例数据用于演示
                self.logger.info("生成示例气候数据用于分析")
                dates = pd.date_range(start='1990-01-01', end='2023-12-31', freq='D')
                np.random.seed(42)
                data = pd.DataFrame({
                    'time': dates,
                    'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + 
                                 0.01 * np.arange(len(dates)) + np.random.normal(0, 2, len(dates)),
                    'precipitation': np.maximum(0, 50 + 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25 + np.pi/2) + 
                                              np.random.normal(0, 20, len(dates))),
                    'humidity': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + 
                              np.random.normal(0, 5, len(dates)),
                    'pressure': 1013 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + 
                              np.random.normal(0, 3, len(dates))
                })
            
            # 数据预处理
            if isinstance(data, pd.DataFrame):
                if variables is None:
                    variables = [col for col in data.columns if col not in ['time', 'date']]
                
                processed_data = self.data_processor.process_dataframe(data, variables)
            else:
                processed_data = data
                if variables is None:
                    variables = ['temperature', 'precipitation']
            
            # 执行分析
            analysis_report = self.climate_analyzer.generate_comprehensive_report(
                processed_data, f"历史气候数据分析 - {data_source}", variables
            )
            
            # 保存结果
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存分析报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = output_path / f"analysis_report_{timestamp}.json"
            self._save_analysis_report(analysis_report, report_file)
            
            # 生成可视化图表
            self._generate_analysis_charts(analysis_report, output_path, timestamp)
            
            self.logger.info(f"历史数据分析完成，结果保存至: {output_path}")
            return analysis_report
            
        except Exception as e:
            self.logger.error(f"历史数据分析失败: {e}")
            raise
    
    def generate_ecology_warning_image(
        self,
        carbon_emission: float = 100.0,
        pollution_index: float = 50.0,
        deforestation_rate: float = 10.0,
        output_path: str = "output/warning_images"
    ):
        """生成生态警示图像"""
        self.logger.info("开始生成生态警示图像")
        
        try:
            # 准备环境输入参数
            environmental_conditions = {
                'carbon_emission': carbon_emission,
                'pollution_index': pollution_index,
                'deforestation_rate': deforestation_rate,
                'temperature_increase': carbon_emission / 100.0,
                'biodiversity_loss': deforestation_rate * 2
            }
            
            # 生成警示图像
            # 构建环境保护相关的提示词
            prompt = f"Environmental warning scene: carbon emission {carbon_emission} ppm, pollution index {pollution_index}, deforestation rate {deforestation_rate}%, environmental degradation, climate change impact"
            
            result = self.image_generator.generate_image(
                prompt=prompt,
                category="pollution",
                output_dir=output_path,
                filename_prefix="ecology_warning"
            )
            
            # 保存图像
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_file = output_dir / f"ecology_warning_{timestamp}.png"
            
            if result and result.get('success', False):
                image_paths = result.get('image_paths', [])
                if image_paths:
                    # 使用API生成的图像
                    image_file = Path(image_paths[0])  # 使用第一张生成的图像
                    self.logger.info(f"生态警示图像已保存至: {image_file}")
                else:
                    # 生成示例警示图像
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as patches
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # 创建警示场景
                    hazard_level = result.get('hazard_level', 'medium')
                    
                    if hazard_level == 'high':
                        bg_color = '#8B0000'  # 深红色
                        warning_text = "严重生态危机"
                    elif hazard_level == 'medium':
                        bg_color = '#FF4500'  # 橙红色
                        warning_text = "生态环境恶化"
                    else:
                        bg_color = '#FFD700'  # 金色
                        warning_text = "生态环境警告"
                    
                    ax.set_facecolor(bg_color)
                    
                    # 添加警示元素
                    # 烟囱和烟雾
                    chimney = patches.Rectangle((0.2, 0.1), 0.1, 0.4, facecolor='gray')
                    ax.add_patch(chimney)
                    
                    # 烟雾效果
                    for i in range(5):
                        smoke = patches.Circle((0.25 + i*0.05, 0.5 + i*0.1), 
                                             0.03 + i*0.01, facecolor='black', alpha=0.6)
                        ax.add_patch(smoke)
                    
                    # 枯树
                    tree_x = [0.7, 0.7, 0.65, 0.75, 0.68, 0.72]
                    tree_y = [0.1, 0.6, 0.4, 0.4, 0.5, 0.5]
                    ax.plot(tree_x, tree_y, 'brown', linewidth=3)
                    
                    # 污染水体
                    water = patches.Rectangle((0.1, 0.05), 0.8, 0.1, facecolor='brown', alpha=0.7)
                    ax.add_patch(water)
                    
                    # 添加文字警告
                    ax.text(0.5, 0.8, warning_text, fontsize=20, fontweight='bold', 
                           ha='center', va='center', color='white')
                    
                    ax.text(0.5, 0.7, f"碳排放: {carbon_emission} ppm", fontsize=12, 
                           ha='center', va='center', color='white')
                    ax.text(0.5, 0.65, f"污染指数: {pollution_index}", fontsize=12, 
                           ha='center', va='center', color='white')
                    ax.text(0.5, 0.6, f"森林砍伐率: {deforestation_rate}%", fontsize=12, 
                           ha='center', va='center', color='white')
                    
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_aspect('equal')
                    ax.axis('off')
                    
                    plt.title("生态环境警示图像", fontsize=16, fontweight='bold', color='white', pad=20)
                    plt.savefig(image_file, dpi=300, bbox_inches='tight', facecolor=bg_color)
                    plt.close()
                
                self.logger.info(f"生态警示图像已保存至: {image_file}")
                
                # 保存元数据
                hazard_level = 'high' if pollution_index > 70 else 'medium' if pollution_index > 40 else 'low'
                metadata = {
                    'generation_time': datetime.now().isoformat(),
                    'environmental_conditions': environmental_conditions,
                    'hazard_level': hazard_level,
                    'warning_message': f'Environmental warning: {hazard_level} risk level',
                    'image_path': str(image_file),
                    'api_result': result
                }
                
                metadata_file = output_dir / f"metadata_{timestamp}.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                return {
                    'image_path': str(image_file),
                    'metadata_path': str(metadata_file),
                    'result': result
                }
            else:
                self.logger.warning("图像生成失败或返回空结果")
                return None
                
        except Exception as e:
            self.logger.error(f"生态警示图像生成失败: {e}")
            raise
    
    def predict_regional_climate(
        self,
        region_name: str,
        scenario: str = "RCP4.5",
        global_temp_increase: float = 2.0,
        co2_increase: float = 100.0,
        output_path: str = "output/predictions"
    ):
        """预测区域气候风险"""
        self.logger.info(f"开始预测区域气候风险: {region_name}, 情景: {scenario}")
        
        try:
            # 执行预测
            prediction_result = predict_regional_climate_risk(
                region_name=region_name,
                scenario_name=scenario,
                global_temp_increase=global_temp_increase,
                co2_increase=co2_increase
            )
            
            # 保存结果
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = output_dir / f"climate_prediction_{region_name}_{scenario}_{timestamp}.json"
            
            # 准备保存数据
            save_data = {
                'prediction_result': {
                    'region_name': prediction_result.region_name,
                    'scenario': {
                        'name': prediction_result.scenario.name,
                        'description': prediction_result.scenario.description,
                        'global_warming': prediction_result.scenario.global_warming,
                        'co2_increase': prediction_result.scenario.co2_increase
                    },
                    'risk_predictions': {k.value: v for k, v in prediction_result.risk_predictions.items()},
                    'risk_levels': {k.value: v.value for k, v in prediction_result.risk_levels.items()},
                    'confidence_scores': {k.value: v for k, v in prediction_result.confidence_scores.items()},
                    'time_horizon': prediction_result.time_horizon,
                    'prediction_date': prediction_result.prediction_date.isoformat(),
                    'recommendations': prediction_result.recommendations
                }
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            # 生成可视化图表
            self._generate_prediction_charts(prediction_result, output_dir, timestamp)
            
            self.logger.info(f"区域气候预测完成，结果保存至: {result_file}")
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"区域气候预测失败: {e}")
            raise
    
    def _save_analysis_report(self, report, file_path):
        """保存分析报告"""
        # 转换报告为可序列化格式
        serializable_report = {
            'dataset_name': report.dataset_name,
            'analysis_date': report.analysis_date.isoformat(),
            'time_period': [report.time_period[0].isoformat(), report.time_period[1].isoformat()],
            'variables_analyzed': report.variables_analyzed,
            'summary': report.summary,
            'trend_results': {},
            'seasonality_results': {},
            'anomaly_results': {},
            'pattern_results': {},
            'extreme_events': {},
            'correlations': report.correlations
        }
        
        # 转换趋势结果
        for var, result in report.trend_results.items():
            serializable_report['trend_results'][var] = {
                'variable': result.variable,
                'method': result.method.value,
                'slope': result.slope,
                'intercept': result.intercept,
                'r_squared': result.r_squared,
                'p_value': result.p_value,
                'confidence_interval': result.confidence_interval,
                'trend_direction': result.trend_direction,
                'significance': result.significance,
                'annual_change': result.annual_change,
                'decadal_change': result.decadal_change
            }
        
        # 转换异常结果
        for var, result in report.anomaly_results.items():
            serializable_report['anomaly_results'][var] = {
                'variable': result.variable,
                'method': result.method.value,
                'anomaly_indices': result.anomaly_indices,
                'anomaly_scores': result.anomaly_scores,
                'anomaly_dates': [d.isoformat() for d in result.anomaly_dates],
                'anomaly_values': result.anomaly_values,
                'threshold': result.threshold,
                'total_anomalies': result.total_anomalies,
                'anomaly_rate': result.anomaly_rate
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2)
    
    def _generate_analysis_charts(self, report, output_dir, timestamp):
        """生成分析图表"""
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 趋势图表
            if report.trend_results:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('历史气候数据趋势分析', fontsize=16)
                
                for i, (var, result) in enumerate(list(report.trend_results.items())[:4]):
                    ax = axes[i//2, i%2]
                    
                    # 生成示例数据点
                    x = np.arange(100)
                    y = result.slope * x + result.intercept + np.random.normal(0, 1, 100)
                    
                    ax.plot(x, y, alpha=0.7, label='数据', color='blue')
                    ax.plot(x, result.slope * x + result.intercept, 'r-', linewidth=2, label='趋势线')
                    ax.set_title(f'{var} - {result.trend_direction}')
                    ax.set_xlabel('时间')
                    ax.set_ylabel('值')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / f'trend_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 异常检测图表
            if report.anomaly_results:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                anomaly_counts = [result.total_anomalies for result in report.anomaly_results.values()]
                variables = list(report.anomaly_results.keys())
                
                bars = ax.bar(variables, anomaly_counts, color='red', alpha=0.7)
                ax.set_title('各变量异常点数量', fontsize=14)
                ax.set_xlabel('变量')
                ax.set_ylabel('异常点数量')
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, count in zip(bars, anomaly_counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / f'anomaly_detection_{timestamp}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.warning(f"生成分析图表时出错: {e}")
    
    def _generate_prediction_charts(self, prediction_result, output_dir, timestamp):
        """生成预测图表"""
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 风险等级图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 风险概率柱状图
            risks = list(prediction_result.risk_predictions.keys())
            probabilities = list(prediction_result.risk_predictions.values())
            risk_names = [risk.value for risk in risks]
            
            bars = ax1.bar(risk_names, probabilities, color='orange', alpha=0.7)
            ax1.set_title(f'{prediction_result.region_name} - {prediction_result.scenario.name} 气候风险概率')
            ax1.set_xlabel('风险类型')
            ax1.set_ylabel('概率')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, prob in zip(bars, probabilities):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{prob:.2f}', ha='center', va='bottom')
            
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 风险等级饼图
            risk_levels = list(prediction_result.risk_levels.values())
            level_counts = {}
            for level in risk_levels:
                level_name = level.name
                level_counts[level_name] = level_counts.get(level_name, 0) + 1
            
            if level_counts:
                colors = ['green', 'yellow', 'orange', 'red', 'darkred']
                ax2.pie(level_counts.values(), labels=level_counts.keys(), autopct='%1.1f%%', 
                       colors=colors[:len(level_counts)])
                ax2.set_title('风险等级分布')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'climate_prediction_{prediction_result.region_name}_{timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"生成预测图表时出错: {e}")
    
    def process_grib_file(
        self,
        grib_file: str,
        variables: list = None,
        output_format: str = 'netcdf',
        output_dir: str = 'output/grib_processed',
        process_data: bool = True
    ):
        """处理GRIB文件"""
        self.logger.info(f"开始处理GRIB文件: {grib_file}")
        
        try:
            # 检查文件是否存在
            grib_path = Path(grib_file)
            if not grib_path.exists():
                raise FileNotFoundError(f"GRIB文件不存在: {grib_file}")
            
            # 获取文件信息
            file_info = self.grib_processor.get_grib_info(grib_file)
            self.logger.info(f"GRIB文件信息: {file_info['variables']}")
            
            # 处理数据
            if process_data:
                from src.data_processing.data_processor import ProcessingConfig
                config = ProcessingConfig(
                    remove_outliers=True,
                    fill_missing=True,
                    smooth_data=False,
                    normalize=False
                )
                processed_data = self.grib_processor.process_grib_data(
                    grib_file, config, variables
                )
            else:
                if variables:
                    processed_data = self.grib_processor.extract_variables(
                        grib_file, variables
                    )
                else:
                    processed_data = self.grib_processor.load_grib_file(grib_file)
            
            # 保存处理后的数据
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_path / f"{grib_path.stem}_processed_{timestamp}"
            
            saved_file = self.grib_processor.save_processed_data(
                processed_data, output_file, output_format
            )
            
            # 保存文件信息
            info_file = output_path / f"{grib_path.stem}_info_{timestamp}.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(file_info, f, indent=2, ensure_ascii=False, default=str)
            
            result = {
                'input_file': str(grib_file),
                'output_file': saved_file,
                'info_file': str(info_file),
                'variables': list(processed_data.data_vars.keys()),
                'dimensions': dict(processed_data.dims),
                'file_info': file_info
            }
            
            processed_data.close()
            
            self.logger.info(f"GRIB文件处理完成: {saved_file}")
            return result
            
        except Exception as e:
            self.logger.error(f"处理GRIB文件失败: {e}")
            raise
    
    def convert_grib_to_netcdf(
        self,
        grib_file: str,
        output_file: str = None,
        variables: list = None,
        compression: bool = True
    ):
        """转换GRIB文件为NetCDF格式"""
        self.logger.info(f"开始转换GRIB文件: {grib_file}")
        
        try:
            if output_file is None:
                grib_path = Path(grib_file)
                output_file = f"output/converted/{grib_path.stem}.nc"
            
            # 确保输出目录存在
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换文件
            converted_file = self.grib_processor.convert_to_netcdf(
                grib_file, output_file, variables, compression
            )
            
            # 验证转换结果
            converted_data = self.data_store.load_xarray(converted_file)
            if converted_data:
                result = {
                    'input_file': grib_file,
                    'output_file': converted_file,
                    'variables': list(converted_data.data_vars.keys()),
                    'dimensions': dict(converted_data.dims),
                    'file_size': os.path.getsize(converted_file)
                }
                converted_data.close()
            else:
                raise ValueError("转换后的文件无法读取")
            
            self.logger.info(f"GRIB转换完成: {converted_file}")
            return result
            
        except Exception as e:
            self.logger.error(f"转换GRIB文件失败: {e}")
            raise
    
    def batch_process_grib_files(
        self,
        input_dir: str,
        output_dir: str = 'output/batch_grib',
        pattern: str = '*.grib*',
        output_format: str = 'netcdf',
        process_data: bool = True
    ):
        """批量处理GRIB文件"""
        self.logger.info(f"开始批量处理GRIB文件: {input_dir}")
        
        try:
            from src.data_processing.data_processor import ProcessingConfig
            
            config = ProcessingConfig(
                remove_outliers=True,
                fill_missing=True,
                smooth_data=False,
                normalize=False
            ) if process_data else None
            
            processed_files = self.grib_processor.batch_process_grib_files(
                input_dir, output_dir, pattern, config, output_format
            )
            
            # 生成批量处理报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report = {
                'input_directory': input_dir,
                'output_directory': output_dir,
                'pattern': pattern,
                'output_format': output_format,
                'processed_files': processed_files,
                'total_files': len(processed_files),
                'timestamp': timestamp
            }
            
            # 保存报告
            report_file = Path(output_dir) / f"batch_report_{timestamp}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"批量处理完成，共处理 {len(processed_files)} 个文件")
            return report
            
        except Exception as e:
            self.logger.error(f"批量处理GRIB文件失败: {e}")
            raise
    
    def analyze_grib_file(
        self,
        grib_file: str,
        variables: list = None,
        output_dir: str = 'output/grib_analysis'
    ):
        """分析GRIB文件内容"""
        self.logger.info(f"开始分析GRIB文件: {grib_file}")
        
        try:
            # 获取文件详细信息
            file_info = self.grib_processor.get_grib_info(grib_file)
            
            # 加载数据进行分析
            if variables:
                dataset = self.grib_processor.extract_variables(grib_file, variables)
            else:
                dataset = self.grib_processor.load_grib_file(grib_file)
            
            # 生成统计信息
            analysis_result = {
                'file_info': file_info,
                'variable_statistics': {},
                'data_quality': {},
                'spatial_coverage': {},
                'temporal_coverage': {}
            }
            
            # 分析每个变量
            for var_name in dataset.data_vars:
                var_data = dataset[var_name]
                
                # 基本统计
                stats = {
                    'shape': var_data.shape,
                    'dtype': str(var_data.dtype),
                    'min': float(var_data.min()),
                    'max': float(var_data.max()),
                    'mean': float(var_data.mean()),
                    'std': float(var_data.std()),
                    'missing_values': int(var_data.isnull().sum())
                }
                analysis_result['variable_statistics'][var_name] = stats
                
                # 数据质量评估
                total_points = var_data.size
                valid_points = total_points - stats['missing_values']
                quality_score = valid_points / total_points if total_points > 0 else 0
                
                analysis_result['data_quality'][var_name] = {
                    'total_points': total_points,
                    'valid_points': valid_points,
                    'missing_points': stats['missing_values'],
                    'quality_score': quality_score
                }
            
            # 空间覆盖分析
            if 'latitude' in dataset.coords and 'longitude' in dataset.coords:
                lat_range = (float(dataset.latitude.min()), float(dataset.latitude.max()))
                lon_range = (float(dataset.longitude.min()), float(dataset.longitude.max()))
                
                analysis_result['spatial_coverage'] = {
                    'latitude_range': lat_range,
                    'longitude_range': lon_range,
                    'spatial_resolution': {
                        'latitude': float(dataset.latitude.diff('latitude').mean()) if len(dataset.latitude) > 1 else None,
                        'longitude': float(dataset.longitude.diff('longitude').mean()) if len(dataset.longitude) > 1 else None
                    }
                }
            
            # 时间覆盖分析
            if 'time' in dataset.coords:
                time_range = (str(dataset.time.min().values), str(dataset.time.max().values))
                time_count = len(dataset.time)
                
                analysis_result['temporal_coverage'] = {
                    'time_range': time_range,
                    'time_points': time_count,
                    'temporal_resolution': str(dataset.time.diff('time').mean().values) if time_count > 1 else None
                }
            
            # 保存分析结果
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            grib_name = Path(grib_file).stem
            
            # 保存JSON报告
            report_file = output_path / f"grib_analysis_{grib_name}_{timestamp}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False, default=str)
            
            # 生成可视化图表
            self._generate_grib_analysis_charts(dataset, analysis_result, output_path, timestamp, grib_name)
            
            dataset.close()
            
            self.logger.info(f"GRIB文件分析完成: {report_file}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"分析GRIB文件失败: {e}")
            raise
    
    def _generate_grib_analysis_charts(self, dataset, analysis_result, output_dir, timestamp, grib_name):
        """生成GRIB分析图表"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 变量统计图表
            if analysis_result['variable_statistics']:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'GRIB文件分析 - {grib_name}', fontsize=16)
                
                variables = list(analysis_result['variable_statistics'].keys())
                
                # 数据质量得分
                quality_scores = [analysis_result['data_quality'][var]['quality_score'] 
                                for var in variables]
                
                axes[0, 0].bar(variables, quality_scores, color='green', alpha=0.7)
                axes[0, 0].set_title('数据质量得分')
                axes[0, 0].set_ylabel('质量得分')
                axes[0, 0].set_ylim(0, 1)
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # 缺失值数量
                missing_counts = [analysis_result['variable_statistics'][var]['missing_values'] 
                                for var in variables]
                
                axes[0, 1].bar(variables, missing_counts, color='red', alpha=0.7)
                axes[0, 1].set_title('缺失值数量')
                axes[0, 1].set_ylabel('缺失值数量')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # 数据范围（最小值-最大值）
                if len(variables) > 0:
                    var_ranges = []
                    var_names = []
                    for var in variables[:4]:  # 最多显示4个变量
                        stats = analysis_result['variable_statistics'][var]
                        var_ranges.append([stats['min'], stats['max']])
                        var_names.append(var)
                    
                    if var_ranges:
                        x_pos = np.arange(len(var_names))
                        mins = [r[0] for r in var_ranges]
                        maxs = [r[1] for r in var_ranges]
                        
                        axes[1, 0].bar(x_pos - 0.2, mins, 0.4, label='最小值', alpha=0.7)
                        axes[1, 0].bar(x_pos + 0.2, maxs, 0.4, label='最大值', alpha=0.7)
                        axes[1, 0].set_title('变量数值范围')
                        axes[1, 0].set_ylabel('数值')
                        axes[1, 0].set_xticks(x_pos)
                        axes[1, 0].set_xticklabels(var_names, rotation=45)
                        axes[1, 0].legend()
                
                # 空间覆盖可视化
                if analysis_result['spatial_coverage']:
                    spatial = analysis_result['spatial_coverage']
                    if 'latitude_range' in spatial and 'longitude_range' in spatial:
                        lat_range = spatial['latitude_range']
                        lon_range = spatial['longitude_range']
                        
                        # 创建简单的覆盖范围图
                        axes[1, 1].add_patch(plt.Rectangle(
                            (lon_range[0], lat_range[0]),
                            lon_range[1] - lon_range[0],
                            lat_range[1] - lat_range[0],
                            fill=False, edgecolor='blue', linewidth=2
                        ))
                        axes[1, 1].set_xlim(lon_range[0] - 5, lon_range[1] + 5)
                        axes[1, 1].set_ylim(lat_range[0] - 5, lat_range[1] + 5)
                        axes[1, 1].set_xlabel('经度')
                        axes[1, 1].set_ylabel('纬度')
                        axes[1, 1].set_title('空间覆盖范围')
                        axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / f'grib_analysis_{grib_name}_{timestamp}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            # 如果有时间序列数据，生成时间序列图
            if 'time' in dataset.coords and len(dataset.time) > 1:
                self._generate_grib_timeseries_charts(dataset, output_dir, timestamp, grib_name)
                
        except Exception as e:
            self.logger.warning(f"生成GRIB分析图表时出错: {e}")
    
    def _generate_grib_timeseries_charts(self, dataset, output_dir, timestamp, grib_name):
        """生成GRIB时间序列图表"""
        try:
            import matplotlib.pyplot as plt
            
            # 选择前几个变量进行时间序列可视化
            variables = list(dataset.data_vars.keys())[:4]  # 最多4个变量
            
            if variables:
                fig, axes = plt.subplots(len(variables), 1, figsize=(12, 3 * len(variables)))
                if len(variables) == 1:
                    axes = [axes]
                
                fig.suptitle(f'GRIB时间序列 - {grib_name}', fontsize=16)
                
                for i, var in enumerate(variables):
                    var_data = dataset[var]
                    
                    # 计算空间平均值
                    if len(var_data.dims) > 1:
                        # 对空间维度求平均
                        spatial_dims = [dim for dim in var_data.dims if dim != 'time']
                        if spatial_dims:
                            time_series = var_data.mean(dim=spatial_dims)
                        else:
                            time_series = var_data
                    else:
                        time_series = var_data
                    
                    # 绘制时间序列
                    axes[i].plot(dataset.time, time_series, linewidth=1.5)
                    axes[i].set_title(f'{var} 时间序列（空间平均）')
                    axes[i].set_ylabel(var)
                    axes[i].grid(True, alpha=0.3)
                    
                    # 设置x轴标签
                    if i == len(variables) - 1:
                        axes[i].set_xlabel('时间')
                    
                    # 旋转x轴标签
                    axes[i].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(output_dir / f'grib_timeseries_{grib_name}_{timestamp}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.warning(f"生成GRIB时间序列图表时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AI 驱动的气候洞察与生态意识提升平台 - 命令行界面",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析历史数据
  python cli.py analyze --data data/climate_data.csv --variables temperature,precipitation
  
  # 生成生态警示图像
  python cli.py generate-image --carbon 150 --pollution 80
  
  # 预测区域气候风险
  python cli.py predict --region 北京 --scenario RCP4.5 --temp-increase 2.5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 历史数据分析命令
    analyze_parser = subparsers.add_parser('analyze', help='分析历史气候数据')
    analyze_parser.add_argument('--data', default='demo', help='数据源路径或demo生成示例数据')
    analyze_parser.add_argument('--variables', help='要分析的变量，用逗号分隔')
    analyze_parser.add_argument('--start-date', help='开始日期 (YYYY-MM-DD)')
    analyze_parser.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    analyze_parser.add_argument('--output', default='output/analysis', help='输出目录')
    
    # 生态警示图像生成命令
    image_parser = subparsers.add_parser('generate-image', help='生成生态警示图像')
    image_parser.add_argument('--carbon', type=float, default=100.0, help='碳排放量')
    image_parser.add_argument('--pollution', type=float, default=50.0, help='污染指数')
    image_parser.add_argument('--deforestation', type=float, default=10.0, help='森林砍伐率')
    image_parser.add_argument('--output', default='output/warning_images', help='输出目录')
    
    # 区域气候预测命令
    predict_parser = subparsers.add_parser('predict', help='预测区域气候风险')
    predict_parser.add_argument('--region', default='北京', help='区域名称')
    predict_parser.add_argument('--scenario', default='RCP4.5', help='气候情景')
    predict_parser.add_argument('--temp-increase', type=float, default=2.0, help='全球升温幅度')
    predict_parser.add_argument('--co2-increase', type=float, default=100.0, help='CO2浓度增加')
    predict_parser.add_argument('--output', default='output/predictions', help='输出目录')
    
    # GRIB文件处理命令
    grib_parser = subparsers.add_parser('grib', help='处理GRIB格式文件')
    grib_subparsers = grib_parser.add_subparsers(dest='grib_command', help='GRIB处理子命令')
    
    # GRIB文件信息查看
    grib_info_parser = grib_subparsers.add_parser('info', help='查看GRIB文件信息')
    grib_info_parser.add_argument('file', help='GRIB文件路径')
    grib_info_parser.add_argument('--output', default='output/grib_info', help='输出目录')
    
    # GRIB文件转换
    grib_convert_parser = grib_subparsers.add_parser('convert', help='转换GRIB文件为其他格式')
    grib_convert_parser.add_argument('input', help='输入GRIB文件路径')
    grib_convert_parser.add_argument('--output', help='输出文件路径')
    grib_convert_parser.add_argument('--variables', help='要转换的变量，用逗号分隔')
    grib_convert_parser.add_argument('--compression', action='store_true', default=True, help='启用压缩')
    
    # GRIB文件处理
    grib_process_parser = grib_subparsers.add_parser('process', help='处理GRIB文件（质量控制、清洗等）')
    grib_process_parser.add_argument('file', help='GRIB文件路径')
    grib_process_parser.add_argument('--variables', help='要处理的变量，用逗号分隔')
    grib_process_parser.add_argument('--output-format', default='netcdf', choices=['netcdf', 'zarr'], help='输出格式')
    grib_process_parser.add_argument('--output', default='output/grib_processed', help='输出目录')
    grib_process_parser.add_argument('--no-process', action='store_true', help='不进行数据处理，仅提取')
    
    # GRIB文件分析
    grib_analyze_parser = grib_subparsers.add_parser('analyze', help='分析GRIB文件内容')
    grib_analyze_parser.add_argument('file', help='GRIB文件路径')
    grib_analyze_parser.add_argument('--variables', help='要分析的变量，用逗号分隔')
    grib_analyze_parser.add_argument('--output', default='output/grib_analysis', help='输出目录')
    
    # GRIB批量处理
    grib_batch_parser = grib_subparsers.add_parser('batch', help='批量处理GRIB文件')
    grib_batch_parser.add_argument('input_dir', help='输入目录')
    grib_batch_parser.add_argument('--output', default='output/batch_grib', help='输出目录')
    grib_batch_parser.add_argument('--pattern', default='*.grib*', help='文件匹配模式')
    grib_batch_parser.add_argument('--output-format', default='netcdf', choices=['netcdf', 'zarr'], help='输出格式')
    grib_batch_parser.add_argument('--no-process', action='store_true', help='不进行数据处理，仅转换格式')
    
    # 通用参数
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 初始化CLI
        cli = ClimateInsightCLI(args.config)
        
        if args.command == 'analyze':
            variables = args.variables.split(',') if args.variables else None
            result = cli.analyze_historical_data(
                args.data, variables, args.start_date, args.end_date, args.output
            )
            print(f"\n=== 分析完成 ===")
            print(f"数据集: {result.dataset_name}")
            print(f"分析变量: {len(result.variables_analyzed)}")
            print(f"显著趋势: {result.summary['significant_trends']}")
            print(f"异常点总数: {result.summary['total_anomalies']}")
            
        elif args.command == 'generate-image':
            result = cli.generate_ecology_warning_image(
                args.carbon, args.pollution, args.deforestation, args.output
            )
            if result:
                print(f"\n=== 图像生成完成 ===")
                print(f"图像路径: {result['image_path']}")
                print(f"元数据: {result['metadata_path']}")
            else:
                print("图像生成失败")
                
        elif args.command == 'predict':
            result = cli.predict_regional_climate(
                args.region, args.scenario, args.temp_increase, 
                args.co2_increase, args.output
            )
            print(f"\n=== 预测完成 ===")
            print(f"区域: {result.region_name}")
            print(f"情景: {result.scenario.name}")
            print(f"主要风险: {[k.value for k, v in result.risk_levels.items() if v.value >= 3]}")
            print(f"建议措施: {len(result.recommendations)}")
            
        elif args.command == 'grib':
            if args.grib_command == 'info':
                result = cli.analyze_grib_file(
                    args.file,
                    args.variables.split(',') if hasattr(args, 'variables') and args.variables else None,
                    args.output
                )
                print(f"\n=== GRIB文件分析完成 ===")
                print(f"文件: {args.file}")
                print(f"变量数量: {len(result['variable_statistics'])}")
                
            elif args.grib_command == 'convert':
                result = cli.convert_grib_to_netcdf(
                    args.input,
                    args.output,
                    args.variables.split(',') if args.variables else None,
                    args.compression
                )
                print(f"\n=== GRIB转换完成 ===")
                print(f"输出文件: {result['output_file']}")
                
            elif args.grib_command == 'process':
                result = cli.process_grib_file(
                    args.file,
                    args.variables.split(',') if args.variables else None,
                    args.output_format,
                    args.output,
                    not args.no_process
                )
                print(f"\n=== GRIB处理完成 ===")
                print(f"输出文件: {result['output_file']}")
                
            elif args.grib_command == 'analyze':
                result = cli.analyze_grib_file(
                    args.file,
                    args.variables.split(',') if args.variables else None,
                    args.output
                )
                print(f"\n=== GRIB分析完成 ===")
                print(f"变量数量: {len(result['variable_statistics'])}")
                
            elif args.grib_command == 'batch':
                result = cli.batch_process_grib_files(
                    args.input_dir,
                    args.output,
                    args.pattern,
                    args.output_format,
                    not args.no_process
                )
                print(f"\n=== 批量处理完成 ===")
                print(f"处理文件数: {result['total_files']}")
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()