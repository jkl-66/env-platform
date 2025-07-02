#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 驱动的气候洞察与生态意识提升平台 - 功能演示脚本

本脚本演示平台的三大核心功能：
1. 历史气候数据分析
2. 生态警示图像生成
3. 区域气候风险预测
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入核心模块
try:
    from src.models.historical_climate_analyzer import (
        HistoricalClimateAnalyzer, 
        analyze_climate_data,
        ClimateVariable,
        TrendMethod,
        AnomalyMethod
    )
    from src.models.ecology_image_generator import EcologyImageGenerator
    from src.models.regional_climate_predictor import (
        RegionalClimatePredictor,
        predict_regional_climate_risk,
        create_global_climate_data,
        ClimateRisk
    )
    from src.utils.logger import setup_logger, get_logger
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有依赖模块都已正确安装")
    sys.exit(1)


class ClimateInsightDemo:
    """气候洞察平台演示类"""
    
    def __init__(self):
        """初始化演示"""
        setup_logger()
        self.logger = get_logger(__name__)
        
        # 创建输出目录
        self.output_dir = Path("demo_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.climate_analyzer = HistoricalClimateAnalyzer()
        self.image_generator = EcologyImageGenerator()
        self.climate_predictor = RegionalClimatePredictor()
        
        self.logger.info("气候洞察平台演示初始化完成")
    
    def generate_sample_climate_data(self, years: int = 30) -> pd.DataFrame:
        """生成示例气候数据"""
        self.logger.info(f"生成 {years} 年的示例气候数据")
        
        # 生成日期范围
        start_date = datetime.now() - timedelta(days=years*365)
        end_date = datetime.now()
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 设置随机种子以确保可重现性
        np.random.seed(42)
        
        # 生成温度数据（包含长期趋势、季节性和随机噪声）
        days = np.arange(len(dates))
        
        # 基础温度（15°C）+ 季节性变化 + 长期趋势 + 随机噪声
        temperature = (
            15.0 +  # 基础温度
            10.0 * np.sin(2 * np.pi * days / 365.25) +  # 季节性
            0.02 * days / 365.25 +  # 长期升温趋势（每年0.02°C）
            np.random.normal(0, 2, len(dates))  # 随机噪声
        )
        
        # 生成降水数据
        precipitation = np.maximum(0, 
            50 +  # 基础降水量
            30 * np.sin(2 * np.pi * days / 365.25 + np.pi/2) +  # 季节性（夏季多雨）
            np.random.normal(0, 20, len(dates))  # 随机变化
        )
        
        # 生成湿度数据
        humidity = np.clip(
            60 +  # 基础湿度
            20 * np.sin(2 * np.pi * days / 365.25) +  # 季节性
            np.random.normal(0, 5, len(dates)),  # 随机噪声
            0, 100  # 湿度范围限制
        )
        
        # 生成气压数据
        pressure = (
            1013 +  # 标准大气压
            10 * np.sin(2 * np.pi * days / 365.25) +  # 季节性变化
            np.random.normal(0, 3, len(dates))  # 随机噪声
        )
        
        # 添加一些极端事件
        # 热浪事件
        heatwave_indices = np.random.choice(len(dates), size=10, replace=False)
        for idx in heatwave_indices:
            if idx < len(temperature) - 7:  # 确保不超出范围
                temperature[idx:idx+7] += np.random.uniform(8, 15)  # 持续一周的高温
        
        # 干旱事件
        drought_indices = np.random.choice(len(dates), size=5, replace=False)
        for idx in drought_indices:
            if idx < len(precipitation) - 30:  # 确保不超出范围
                precipitation[idx:idx+30] *= 0.1  # 持续一个月的少雨
        
        # 创建DataFrame
        data = pd.DataFrame({
            'time': dates,
            'temperature': temperature,
            'precipitation': precipitation,
            'humidity': humidity,
            'pressure': pressure
        })
        
        return data
    
    def demo_historical_analysis(self):
        """演示历史气候数据分析功能"""
        print("\n" + "="*60)
        print("🌡️  演示1: 历史气候数据分析")
        print("="*60)
        
        try:
            # 生成示例数据
            climate_data = self.generate_sample_climate_data(30)
            
            # 保存示例数据
            data_file = self.output_dir / "sample_climate_data.csv"
            climate_data.to_csv(data_file, index=False)
            print(f"📊 生成示例数据: {len(climate_data)} 条记录")
            print(f"📁 数据已保存至: {data_file}")
            
            # 执行分析
            print("\n🔍 开始分析历史气候数据...")
            
            variables = ['temperature', 'precipitation', 'humidity', 'pressure']
            analysis_report = self.climate_analyzer.generate_comprehensive_report(
                climate_data, 
                "30年历史气候数据演示分析", 
                variables
            )
            
            # 显示分析结果摘要
            print(f"\n📈 分析结果摘要:")
            print(f"   • 数据集: {analysis_report.dataset_name}")
            print(f"   • 分析时间: {analysis_report.analysis_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   • 时间范围: {analysis_report.time_period[0].strftime('%Y-%m-%d')} 至 {analysis_report.time_period[1].strftime('%Y-%m-%d')}")
            print(f"   • 分析变量: {len(analysis_report.variables_analyzed)} 个")
            
            # 显示趋势分析结果
            if analysis_report.trend_results:
                print(f"\n📊 趋势分析结果:")
                for var, result in analysis_report.trend_results.items():
                    direction = "📈 上升" if result.slope > 0 else "📉 下降" if result.slope < 0 else "➡️ 平稳"
                    significance = "显著" if result.p_value < 0.05 else "不显著"
                    print(f"   • {var}: {direction} (斜率: {result.slope:.4f}, R²: {result.r_squared:.3f}, {significance})")
            
            # 显示异常检测结果
            if analysis_report.anomaly_results:
                print(f"\n🚨 异常检测结果:")
                total_anomalies = 0
                for var, result in analysis_report.anomaly_results.items():
                    total_anomalies += result.total_anomalies
                    print(f"   • {var}: 检测到 {result.total_anomalies} 个异常点 (异常率: {result.anomaly_rate:.2%})")
                print(f"   • 总异常点数: {total_anomalies}")
            
            # 保存详细报告
            report_file = self.output_dir / "historical_analysis_report.json"
            self._save_analysis_report(analysis_report, report_file)
            print(f"\n💾 详细分析报告已保存至: {report_file}")
            
            # 生成可视化图表
            self._create_analysis_charts(climate_data, analysis_report)
            print(f"📊 可视化图表已保存至: {self.output_dir}")
            
            return analysis_report
            
        except Exception as e:
            self.logger.error(f"历史数据分析演示失败: {e}")
            print(f"❌ 演示失败: {e}")
            return None
    
    def demo_ecology_image_generation(self):
        """演示生态警示图像生成功能"""
        print("\n" + "="*60)
        print("🖼️  演示2: 生态警示图像生成")
        print("="*60)
        
        try:
            # 定义不同的环境危害场景
            scenarios = [
                {
                    "name": "轻度污染场景",
                    "carbon_emission": 80.0,
                    "pollution_index": 30.0,
                    "deforestation_rate": 5.0
                },
                {
                    "name": "中度污染场景",
                    "carbon_emission": 150.0,
                    "pollution_index": 60.0,
                    "deforestation_rate": 15.0
                },
                {
                    "name": "重度污染场景",
                    "carbon_emission": 250.0,
                    "pollution_index": 90.0,
                    "deforestation_rate": 30.0
                }
            ]
            
            generated_images = []
            
            for i, scenario in enumerate(scenarios, 1):
                print(f"\n🎨 生成场景 {i}: {scenario['name']}")
                print(f"   • 碳排放量: {scenario['carbon_emission']} ppm")
                print(f"   • 污染指数: {scenario['pollution_index']}")
                print(f"   • 森林砍伐率: {scenario['deforestation_rate']}%")
                
                # 准备环境条件
                environmental_conditions = {
                    'carbon_emission': scenario['carbon_emission'],
                    'pollution_index': scenario['pollution_index'],
                    'deforestation_rate': scenario['deforestation_rate'],
                    'temperature_increase': scenario['carbon_emission'] / 100.0,
                    'biodiversity_loss': scenario['deforestation_rate'] * 2
                }
                
                # 生成警示图像
                result = self.image_generator.generate_warning_image(environmental_conditions)
                
                if result:
                    # 创建可视化图像
                    image_path = self._create_warning_visualization(
                        scenario, environmental_conditions, i
                    )
                    
                    generated_images.append({
                        'scenario': scenario['name'],
                        'image_path': image_path,
                        'hazard_level': result.get('hazard_level', 'unknown'),
                        'warning_message': result.get('warning_message', '')
                    })
                    
                    print(f"   ✅ 图像生成成功: {image_path}")
                    print(f"   ⚠️  危害等级: {result.get('hazard_level', 'unknown')}")
                else:
                    print(f"   ❌ 图像生成失败")
            
            # 保存生成结果摘要
            summary_file = self.output_dir / "ecology_images_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'generation_time': datetime.now().isoformat(),
                    'total_images': len(generated_images),
                    'scenarios': generated_images
                }, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 生成摘要已保存至: {summary_file}")
            print(f"🖼️  共生成 {len(generated_images)} 张警示图像")
            
            return generated_images
            
        except Exception as e:
            self.logger.error(f"生态图像生成演示失败: {e}")
            print(f"❌ 演示失败: {e}")
            return None
    
    def demo_regional_climate_prediction(self):
        """演示区域气候风险预测功能"""
        print("\n" + "="*60)
        print("🌍 演示3: 区域气候风险预测")
        print("="*60)
        
        try:
            # 定义预测场景
            regions = ["北京", "上海", "广州", "成都"]
            scenarios = [
                {"name": "RCP2.6", "temp_increase": 1.5, "co2_increase": 50},
                {"name": "RCP4.5", "temp_increase": 2.5, "co2_increase": 100},
                {"name": "RCP8.5", "temp_increase": 4.0, "co2_increase": 200}
            ]
            
            prediction_results = []
            
            for region in regions:
                print(f"\n🏙️  预测区域: {region}")
                
                region_predictions = []
                
                for scenario in scenarios:
                    print(f"   📊 情景: {scenario['name']} (升温 {scenario['temp_increase']}°C)")
                    
                    # 执行预测
                    prediction_result = predict_regional_climate_risk(
                        region_name=region,
                        scenario_name=scenario['name'],
                        global_temp_increase=scenario['temp_increase'],
                        co2_increase=scenario['co2_increase']
                    )
                    
                    # 显示预测结果
                    high_risks = []
                    medium_risks = []
                    
                    for risk_type, risk_level in prediction_result.risk_levels.items():
                        if risk_level.value >= 4:  # 高风险
                            high_risks.append(risk_type.value)
                        elif risk_level.value >= 3:  # 中等风险
                            medium_risks.append(risk_type.value)
                    
                    print(f"      🔴 高风险: {', '.join(high_risks) if high_risks else '无'}")
                    print(f"      🟡 中等风险: {', '.join(medium_risks) if medium_risks else '无'}")
                    
                    region_predictions.append({
                        'scenario': scenario['name'],
                        'prediction_result': prediction_result,
                        'high_risks': high_risks,
                        'medium_risks': medium_risks
                    })
                
                prediction_results.append({
                    'region': region,
                    'predictions': region_predictions
                })
            
            # 生成预测对比图表
            self._create_prediction_charts(prediction_results)
            
            # 保存预测结果
            results_file = self.output_dir / "regional_climate_predictions.json"
            self._save_prediction_results(prediction_results, results_file)
            
            print(f"\n💾 预测结果已保存至: {results_file}")
            print(f"📊 预测对比图表已保存至: {self.output_dir}")
            
            # 显示总结
            print(f"\n📋 预测总结:")
            print(f"   • 预测区域: {len(regions)} 个")
            print(f"   • 气候情景: {len(scenarios)} 个")
            print(f"   • 总预测数: {len(regions) * len(scenarios)} 个")
            
            return prediction_results
            
        except Exception as e:
            self.logger.error(f"区域气候预测演示失败: {e}")
            print(f"❌ 演示失败: {e}")
            return None
    
    def _save_analysis_report(self, report, file_path):
        """保存分析报告"""
        serializable_report = {
            'dataset_name': report.dataset_name,
            'analysis_date': report.analysis_date.isoformat(),
            'time_period': [report.time_period[0].isoformat(), report.time_period[1].isoformat()],
            'variables_analyzed': report.variables_analyzed,
            'summary': report.summary,
            'trend_results': {},
            'anomaly_results': {},
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
                'total_anomalies': result.total_anomalies,
                'anomaly_rate': result.anomaly_rate,
                'threshold': result.threshold
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2)
    
    def _create_analysis_charts(self, data, report):
        """创建分析图表"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 时间序列图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('历史气候数据时间序列分析', fontsize=16, fontweight='bold')
        
        variables = ['temperature', 'precipitation', 'humidity', 'pressure']
        titles = ['温度 (°C)', '降水量 (mm)', '湿度 (%)', '气压 (hPa)']
        colors = ['red', 'blue', 'green', 'purple']
        
        for i, (var, title, color) in enumerate(zip(variables, titles, colors)):
            ax = axes[i//2, i%2]
            
            # 绘制原始数据
            ax.plot(data['time'], data[var], alpha=0.6, color=color, linewidth=0.5, label='原始数据')
            
            # 绘制趋势线
            if var in report.trend_results:
                result = report.trend_results[var]
                x_numeric = np.arange(len(data))
                trend_line = result.slope * x_numeric + result.intercept
                ax.plot(data['time'], trend_line, 'r-', linewidth=2, label=f'趋势线 (斜率: {result.slope:.4f})')
            
            # 标记异常点
            if var in report.anomaly_results:
                anomaly_result = report.anomaly_results[var]
                if anomaly_result.anomaly_indices:
                    anomaly_times = data['time'].iloc[anomaly_result.anomaly_indices]
                    anomaly_values = data[var].iloc[anomaly_result.anomaly_indices]
                    ax.scatter(anomaly_times, anomaly_values, color='red', s=20, alpha=0.8, label='异常点')
            
            ax.set_title(title)
            ax.set_xlabel('时间')
            ax.set_ylabel('值')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'climate_analysis_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 异常检测统计图
        if report.anomaly_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            vars_list = list(report.anomaly_results.keys())
            anomaly_counts = [report.anomaly_results[var].total_anomalies for var in vars_list]
            anomaly_rates = [report.anomaly_results[var].anomaly_rate * 100 for var in vars_list]
            
            x = np.arange(len(vars_list))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, anomaly_counts, width, label='异常点数量', color='red', alpha=0.7)
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, anomaly_rates, width, label='异常率 (%)', color='orange', alpha=0.7)
            
            ax.set_xlabel('变量')
            ax.set_ylabel('异常点数量', color='red')
            ax2.set_ylabel('异常率 (%)', color='orange')
            ax.set_title('各变量异常检测结果')
            ax.set_xticks(x)
            ax.set_xticklabels(vars_list)
            
            # 添加数值标签
            for bar, count in zip(bars1, anomaly_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(count), ha='center', va='bottom')
            
            for bar, rate in zip(bars2, anomaly_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{rate:.1f}%', ha='center', va='bottom')
            
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'anomaly_detection_stats.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_warning_visualization(self, scenario, conditions, index):
        """创建警示图像可视化"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 根据危害程度确定背景颜色
        total_hazard = (conditions['carbon_emission'] + 
                       conditions['pollution_index'] + 
                       conditions['deforestation_rate']) / 3
        
        if total_hazard > 150:
            bg_color = '#8B0000'  # 深红色
            warning_level = "严重危机"
        elif total_hazard > 100:
            bg_color = '#FF4500'  # 橙红色
            warning_level = "环境恶化"
        else:
            bg_color = '#FFD700'  # 金色
            warning_level = "环境警告"
        
        ax.set_facecolor(bg_color)
        
        # 创建警示场景元素
        import matplotlib.patches as patches
        
        # 工厂烟囱
        chimney = patches.Rectangle((0.1, 0.1), 0.08, 0.4, facecolor='gray', edgecolor='black')
        ax.add_patch(chimney)
        
        # 烟雾
        smoke_intensity = conditions['carbon_emission'] / 300.0
        for i in range(int(5 * smoke_intensity) + 1):
            smoke = patches.Circle((0.14 + i*0.04, 0.5 + i*0.08), 
                                 0.02 + i*0.008, facecolor='black', alpha=0.6)
            ax.add_patch(smoke)
        
        # 污染水体
        pollution_width = 0.6 + conditions['pollution_index'] / 200.0
        water = patches.Rectangle((0.2, 0.05), pollution_width, 0.08, 
                                facecolor='brown', alpha=0.8)
        ax.add_patch(water)
        
        # 枯萎的树木
        deforestation_factor = conditions['deforestation_rate'] / 50.0
        tree_color = 'brown' if deforestation_factor > 0.3 else 'darkgreen'
        
        for i in range(3):
            tree_x = 0.7 + i * 0.1
            tree_height = 0.3 * (1 - deforestation_factor * 0.5)
            tree = patches.Rectangle((tree_x, 0.1), 0.02, tree_height, 
                                   facecolor=tree_color)
            ax.add_patch(tree)
        
        # 添加文字信息
        ax.text(0.5, 0.85, f"{scenario['name']}", fontsize=20, fontweight='bold',
               ha='center', va='center', color='white')
        
        ax.text(0.5, 0.75, warning_level, fontsize=16, fontweight='bold',
               ha='center', va='center', color='yellow')
        
        # 环境指标
        indicators = [
            f"碳排放: {conditions['carbon_emission']:.0f} ppm",
            f"污染指数: {conditions['pollution_index']:.0f}",
            f"森林砍伐: {conditions['deforestation_rate']:.0f}%",
            f"温度上升: {conditions['temperature_increase']:.1f}°C"
        ]
        
        for i, indicator in enumerate(indicators):
            ax.text(0.5, 0.65 - i*0.05, indicator, fontsize=12,
                   ha='center', va='center', color='white')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title("生态环境警示图像", fontsize=16, fontweight='bold', color='white', pad=20)
        
        image_path = self.output_dir / f"ecology_warning_{index}_{scenario['name'].replace(' ', '_')}.png"
        plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
        plt.close()
        
        return str(image_path)
    
    def _create_prediction_charts(self, prediction_results):
        """创建预测对比图表"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 风险等级热力图
        regions = [result['region'] for result in prediction_results]
        scenarios = ['RCP2.6', 'RCP4.5', 'RCP8.5']
        risk_types = ['drought', 'flood', 'heatwave', 'extreme_precipitation']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('区域气候风险预测对比', fontsize=16, fontweight='bold')
        
        for i, risk_type in enumerate(risk_types):
            ax = axes[i//2, i%2]
            
            # 创建风险等级矩阵
            risk_matrix = np.zeros((len(regions), len(scenarios)))
            
            for r, region_result in enumerate(prediction_results):
                for s, scenario in enumerate(scenarios):
                    # 查找对应情景的预测结果
                    scenario_prediction = None
                    for pred in region_result['predictions']:
                        if pred['scenario'] == scenario:
                            scenario_prediction = pred['prediction_result']
                            break
                    
                    if scenario_prediction:
                        # 获取风险等级
                        for risk, level in scenario_prediction.risk_levels.items():
                            if risk.value == risk_type:
                                risk_matrix[r, s] = level.value
                                break
            
            # 绘制热力图
            im = ax.imshow(risk_matrix, cmap='YlOrRd', aspect='auto', vmin=1, vmax=5)
            
            # 设置标签
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels(scenarios)
            ax.set_yticks(range(len(regions)))
            ax.set_yticklabels(regions)
            
            # 添加数值标签
            for r in range(len(regions)):
                for s in range(len(scenarios)):
                    text = ax.text(s, r, f'{risk_matrix[r, s]:.0f}',
                                 ha="center", va="center", color="black", fontweight='bold')
            
            ax.set_title(f'{risk_type.replace("_", " ").title()} 风险等级')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('风险等级 (1-5)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'regional_climate_risk_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 区域风险对比柱状图
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(regions))
        width = 0.25
        
        colors = ['green', 'orange', 'red']
        
        for s, (scenario, color) in enumerate(zip(scenarios, colors)):
            high_risk_counts = []
            
            for region_result in prediction_results:
                high_risk_count = 0
                for pred in region_result['predictions']:
                    if pred['scenario'] == scenario:
                        high_risk_count = len(pred['high_risks'])
                        break
                high_risk_counts.append(high_risk_count)
            
            bars = ax.bar(x + s*width, high_risk_counts, width, label=scenario, color=color, alpha=0.7)
            
            # 添加数值标签
            for bar, count in zip(bars, high_risk_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       str(count), ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('区域')
        ax.set_ylabel('高风险类型数量')
        ax.set_title('各区域高风险类型数量对比')
        ax.set_xticks(x + width)
        ax.set_xticklabels(regions)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'regional_high_risk_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_prediction_results(self, prediction_results, file_path):
        """保存预测结果"""
        serializable_results = []
        
        for region_result in prediction_results:
            region_data = {
                'region': region_result['region'],
                'predictions': []
            }
            
            for pred in region_result['predictions']:
                prediction_data = {
                    'scenario': pred['scenario'],
                    'high_risks': pred['high_risks'],
                    'medium_risks': pred['medium_risks'],
                    'risk_predictions': {k.value: v for k, v in pred['prediction_result'].risk_predictions.items()},
                    'risk_levels': {k.value: v.value for k, v in pred['prediction_result'].risk_levels.items()},
                    'confidence_scores': {k.value: v for k, v in pred['prediction_result'].confidence_scores.items()},
                    'recommendations': pred['prediction_result'].recommendations
                }
                region_data['predictions'].append(prediction_data)
            
            serializable_results.append(region_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'generation_time': datetime.now().isoformat(),
                'total_regions': len(prediction_results),
                'results': serializable_results
            }, f, ensure_ascii=False, indent=2)
    
    def run_full_demo(self):
        """运行完整演示"""
        print("🌍 AI 驱动的气候洞察与生态意识提升平台 - 功能演示")
        print("="*80)
        print("本演示将展示平台的三大核心功能：")
        print("1. 历史气候数据分析")
        print("2. 生态警示图像生成")
        print("3. 区域气候风险预测")
        print("="*80)
        
        start_time = datetime.now()
        
        # 演示1: 历史气候数据分析
        analysis_result = self.demo_historical_analysis()
        
        # 演示2: 生态警示图像生成
        image_results = self.demo_ecology_image_generation()
        
        # 演示3: 区域气候风险预测
        prediction_results = self.demo_regional_climate_prediction()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # 演示总结
        print("\n" + "="*60)
        print("🎉 演示完成总结")
        print("="*60)
        print(f"⏱️  总耗时: {duration.total_seconds():.1f} 秒")
        print(f"📁 输出目录: {self.output_dir}")
        
        if analysis_result:
            print(f"✅ 历史数据分析: 成功 (分析了 {len(analysis_result.variables_analyzed)} 个变量)")
        else:
            print(f"❌ 历史数据分析: 失败")
        
        if image_results:
            print(f"✅ 生态图像生成: 成功 (生成了 {len(image_results)} 张图像)")
        else:
            print(f"❌ 生态图像生成: 失败")
        
        if prediction_results:
            total_predictions = sum(len(r['predictions']) for r in prediction_results)
            print(f"✅ 气候风险预测: 成功 (完成了 {total_predictions} 个预测)")
        else:
            print(f"❌ 气候风险预测: 失败")
        
        print(f"\n📊 生成的文件:")
        output_files = list(self.output_dir.glob('*'))
        for file_path in sorted(output_files):
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"   • {file_path.name} ({file_size:.1f} KB)")
        
        print(f"\n🎯 演示亮点:")
        print(f"   • 自动生成30年历史气候数据并进行深度分析")
        print(f"   • 基于环境指标生成直观的生态警示图像")
        print(f"   • 预测多个城市在不同气候情景下的风险")
        print(f"   • 生成专业的可视化图表和详细报告")
        
        print(f"\n💡 使用建议:")
        print(f"   • 查看生成的图表了解分析结果")
        print(f"   • 阅读JSON报告获取详细数据")
        print(f"   • 尝试修改参数运行不同场景")
        print(f"   • 集成到实际应用中进行扩展")
        
        return {
            'duration': duration.total_seconds(),
            'analysis_success': analysis_result is not None,
            'image_generation_success': image_results is not None,
            'prediction_success': prediction_results is not None,
            'output_files': len(output_files)
        }


def main():
    """主函数"""
    try:
        demo = ClimateInsightDemo()
        result = demo.run_full_demo()
        
        if all([result['analysis_success'], result['image_generation_success'], result['prediction_success']]):
            print("\n🎊 所有演示功能运行成功！")
            return 0
        else:
            print("\n⚠️  部分演示功能运行失败，请检查日志")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  演示被用户中断")
        return 1
    except Exception as e:
        print(f"\n💥 演示运行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())