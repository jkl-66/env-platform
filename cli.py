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
    from src.models.historical_climate_analyzer import HistoricalClimateAnalyzer, analyze_climate_data
    from src.models.ecology_image_generator import EcologyImageGenerator
    from src.models.regional_climate_predictor import RegionalClimatePredictor, predict_regional_climate_risk
    from src.ml.model_manager import ModelManager
    from src.ml.prediction_engine import PredictionEngine
    from src.visualization.charts import ChartGenerator
    from src.utils.logger import setup_logging, get_logger
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
        setup_logging()
        self.logger = get_logger(__name__)
        
        # 加载配置
        if config_path:
            load_config(config_path)
        self.config = get_config()
        
        # 初始化核心组件
        self.data_collector = ClimateDataCollector()
        self.data_store = ClimateDataStore()
        self.data_processor = DataProcessor()
        self.climate_analyzer = HistoricalClimateAnalyzer()
        self.image_generator = EcologyImageGenerator()
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
            result = self.image_generator.generate_warning_image(
                environmental_conditions
            )
            
            # 保存图像
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_file = output_dir / f"ecology_warning_{timestamp}.png"
            
            if result and 'image' in result:
                # 保存图像
                if hasattr(result['image'], 'save'):
                    result['image'].save(image_file)
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
                metadata = {
                    'generation_time': datetime.now().isoformat(),
                    'environmental_conditions': environmental_conditions,
                    'hazard_level': result.get('hazard_level', 'unknown'),
                    'warning_message': result.get('warning_message', ''),
                    'image_path': str(image_file)
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