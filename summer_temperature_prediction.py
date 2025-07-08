#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
夏季最高气温出现日期预测系统
基于1940-2024年夏季14点气温GRIB数据

功能:
1. 读取和处理GRIB气温数据
2. 分析历史夏季最高气温出现日期
3. 建立预测模型
4. 生成历史和预测的可视化图像
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

try:
    import xarray as xr
    import cfgrib
except ImportError:
    print("正在安装必要的依赖包...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xarray", "cfgrib", "eccodes"])
    import xarray as xr
    import cfgrib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from scipy import stats
from scipy.signal import find_peaks

class SummerTemperaturePrediction:
    def __init__(self, grib_file_path):
        """
        初始化夏季气温预测系统
        
        Args:
            grib_file_path (str): GRIB文件路径
        """
        self.grib_file_path = grib_file_path
        self.data = None
        self.processed_data = None
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def load_grib_data(self):
        """
        加载GRIB数据文件
        """
        try:
            print(f"正在加载GRIB数据: {self.grib_file_path}")
            
            # 检查文件是否存在
            import os
            if not os.path.exists(self.grib_file_path):
                print(f"文件不存在: {self.grib_file_path}")
                print("将使用模拟数据进行演示...")
                self._generate_simulated_data()
                return False
            
            # 使用xarray和cfgrib读取GRIB数据
            self.data = xr.open_dataset(self.grib_file_path, engine='cfgrib')
            
            # 查找温度相关变量
            temp_vars = [var for var in self.data.data_vars if any(keyword in var.lower() 
                        for keyword in ['temp', 't2m', 'temperature', '2t'])]
            
            if not temp_vars:
                print("未找到温度变量，尝试查看所有变量:")
                print(list(self.data.data_vars.keys()))
                # 如果没有找到明确的温度变量，使用第一个数据变量
                if self.data.data_vars:
                    temp_var = list(self.data.data_vars.keys())[0]
                    print(f"使用变量: {temp_var}")
                else:
                    raise ValueError("GRIB文件中未找到任何数据变量")
            else:
                temp_var = temp_vars[0]
                print(f"找到温度变量: {temp_var}")
            
            # 重命名温度变量为标准名称
            if temp_var != 't2m':
                self.data = self.data.rename({temp_var: 't2m'})
            
            # 转换为摄氏度（如果需要）
            if self.data['t2m'].attrs.get('units') == 'K':
                self.data['t2m'] = self.data['t2m'] - 273.15
                self.data['t2m'].attrs['units'] = 'C'
            
            print(f"数据加载成功!")
            print(f"数据维度: {self.data.dims}")
            print(f"数据变量: {list(self.data.data_vars)}")
            print(f"时间范围: {self.data.time.min().values} 到 {self.data.time.max().values}")
            
            return True
            
        except Exception as e:
            print(f"GRIB数据加载失败: {e}")
            print("尝试生成模拟数据进行演示...")
            self._generate_simulated_data()
            return False
    
    def _generate_simulated_data(self):
        """
        生成模拟的夏季气温数据用于演示
        """
        print("生成1940-2024年夏季气温模拟数据...")
        
        # 创建时间序列 (1940-2024年，每年夏季6-8月)
        years = range(1940, 2025)
        dates = []
        temperatures = []
        
        np.random.seed(42)  # 确保结果可重现
        
        for year in years:
            # 夏季日期 (6月1日到8月31日)
            summer_start = datetime(year, 6, 1)
            summer_end = datetime(year, 8, 31)
            
            # 生成夏季每日气温数据
            current_date = summer_start
            year_temps = []
            year_dates = []
            
            while current_date <= summer_end:
                # 模拟气温变化 (基于正弦波 + 随机噪声 + 长期趋势)
                day_of_year = current_date.timetuple().tm_yday
                
                # 基础温度模式 (夏季高温)
                base_temp = 25 + 10 * np.sin((day_of_year - 150) * 2 * np.pi / 365)
                
                # 长期气候变化趋势 (全球变暖)
                trend = (year - 1940) * 0.02
                
                # 随机波动
                noise = np.random.normal(0, 3)
                
                # 极端高温事件 (低概率)
                if np.random.random() < 0.05:
                    extreme_boost = np.random.uniform(5, 15)
                else:
                    extreme_boost = 0
                
                final_temp = base_temp + trend + noise + extreme_boost
                
                year_dates.append(current_date)
                year_temps.append(final_temp)
                current_date += timedelta(days=1)
            
            dates.extend(year_dates)
            temperatures.extend(year_temps)
        
        # 创建xarray数据集
        time_coords = pd.to_datetime(dates)
        self.data = xr.Dataset({
            't2m': (['time'], temperatures)
        }, coords={'time': time_coords})
        
        print(f"模拟数据生成完成: {len(dates)} 个数据点")
        print(f"时间范围: {min(dates)} 到 {max(dates)}")
        print(f"温度范围: {min(temperatures):.1f}°C 到 {max(temperatures):.1f}°C")
    
    def process_data(self):
        """
        处理气温数据，提取夏季最高气温信息
        """
        print("\n正在处理气温数据...")
        
        # 转换为DataFrame便于处理
        df = self.data.to_dataframe().reset_index()
        
        # 确保有温度列
        temp_col = None
        for col in ['t2m', 'temperature', 'temp', '2t']:
            if col in df.columns:
                temp_col = col
                break
        
        if temp_col is None:
            raise ValueError("未找到温度数据列")
        
        df['temperature'] = df[temp_col]
        df['date'] = pd.to_datetime(df['time'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # 筛选夏季数据 (6-8月)
        summer_data = df[df['month'].isin([6, 7, 8])].copy()
        
        print(f"夏季数据点数: {len(summer_data)}")
        print(f"年份范围: {summer_data['year'].min()} - {summer_data['year'].max()}")
        
        # 计算每年夏季最高气温及其出现日期
        yearly_max = []
        
        for year in sorted(summer_data['year'].unique()):
            year_data = summer_data[summer_data['year'] == year]
            
            if len(year_data) > 0:
                max_temp_idx = year_data['temperature'].idxmax()
                max_temp_row = year_data.loc[max_temp_idx]
                
                yearly_max.append({
                    'year': year,
                    'max_temperature': max_temp_row['temperature'],
                    'max_temp_date': max_temp_row['date'],
                    'max_temp_day_of_year': max_temp_row['day_of_year'],
                    'max_temp_month': max_temp_row['month'],
                    'max_temp_day': max_temp_row['day']
                })
        
        self.processed_data = pd.DataFrame(yearly_max)
        
        print(f"\n处理完成! 共{len(self.processed_data)}年的数据")
        print(f"平均最高气温: {self.processed_data['max_temperature'].mean():.1f}°C")
        print(f"最高气温范围: {self.processed_data['max_temperature'].min():.1f}°C - {self.processed_data['max_temperature'].max():.1f}°C")
        print(f"最高气温日期范围: 第{self.processed_data['max_temp_day_of_year'].min()}天 - 第{self.processed_data['max_temp_day_of_year'].max()}天")
        
        return self.processed_data
    
    def analyze_trends(self):
        """
        分析历史趋势
        """
        print("\n正在分析历史趋势...")
        
        df = self.processed_data.copy()
        
        # 计算统计信息
        stats_info = {
            '平均最高气温': df['max_temperature'].mean(),
            '最高气温标准差': df['max_temperature'].std(),
            '平均出现日期': df['max_temp_day_of_year'].mean(),
            '出现日期标准差': df['max_temp_day_of_year'].std(),
            '最早出现日期': df['max_temp_day_of_year'].min(),
            '最晚出现日期': df['max_temp_day_of_year'].max()
        }
        
        # 趋势分析
        temp_trend = stats.linregress(df['year'], df['max_temperature'])
        date_trend = stats.linregress(df['year'], df['max_temp_day_of_year'])
        
        print("\n=== 历史趋势分析 ===")
        for key, value in stats_info.items():
            print(f"{key}: {value:.2f}")
        
        print(f"\n气温趋势: {temp_trend.slope:.4f}°C/年 (p-value: {temp_trend.pvalue:.4f})")
        print(f"日期趋势: {date_trend.slope:.4f}天/年 (p-value: {date_trend.pvalue:.4f})")
        
        self.results['stats'] = stats_info
        self.results['temp_trend'] = temp_trend
        self.results['date_trend'] = date_trend
        
        return stats_info
    
    def build_prediction_models(self):
        """
        构建预测模型
        """
        print("\n正在构建预测模型...")
        
        df = self.processed_data.copy()
        
        # 特征工程
        df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        df['temp_ma_5'] = df['max_temperature'].rolling(window=5, center=True).mean()
        df['date_ma_5'] = df['max_temp_day_of_year'].rolling(window=5, center=True).mean()
        
        # 填充缺失值
        df['temp_ma_5'].fillna(df['max_temperature'], inplace=True)
        df['date_ma_5'].fillna(df['max_temp_day_of_year'], inplace=True)
        
        # 准备训练数据
        features = ['year', 'year_normalized']
        
        # 添加滞后特征
        for lag in [1, 2, 3]:
            df[f'temp_lag_{lag}'] = df['max_temperature'].shift(lag)
            df[f'date_lag_{lag}'] = df['max_temp_day_of_year'].shift(lag)
            features.extend([f'temp_lag_{lag}', f'date_lag_{lag}'])
        
        # 添加移动平均特征
        features.extend(['temp_ma_5', 'date_ma_5'])
        
        # 删除包含NaN的行
        df_clean = df.dropna()
        
        X = df_clean[features]
        y_temp = df_clean['max_temperature']
        y_date = df_clean['max_temp_day_of_year']
        
        # 分割训练和测试数据
        split_year = 2015
        train_mask = df_clean['year'] < split_year
        test_mask = df_clean['year'] >= split_year
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_temp_train, y_temp_test = y_temp[train_mask], y_temp[test_mask]
        y_date_train, y_date_test = y_date[train_mask], y_date[test_mask]
        
        print(f"训练数据: {len(X_train)} 样本")
        print(f"测试数据: {len(X_test)} 样本")
        
        # 特征标准化
        scaler_temp = StandardScaler()
        scaler_date = StandardScaler()
        
        X_train_scaled = scaler_temp.fit_transform(X_train)
        X_test_scaled = scaler_temp.transform(X_test)
        
        # 构建模型
        models = {
            'temperature': {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'lr': LinearRegression()
            },
            'date': {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'lr': LinearRegression()
            }
        }
        
        # 训练模型
        results = {}
        
        # 温度预测模型
        for model_name, model in models['temperature'].items():
            if model_name == 'lr':
                model.fit(X_train_scaled, y_temp_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_temp_train)
                y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_temp_test, y_pred)
            r2 = r2_score(y_temp_test, y_pred)
            
            results[f'temp_{model_name}'] = {
                'model': model,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"温度预测 - {model_name.upper()}: MAE={mae:.2f}°C, R²={r2:.3f}")
        
        # 日期预测模型
        for model_name, model in models['date'].items():
            if model_name == 'lr':
                model.fit(X_train_scaled, y_date_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_date_train)
                y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_date_test, y_pred)
            r2 = r2_score(y_date_test, y_pred)
            
            results[f'date_{model_name}'] = {
                'model': model,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"日期预测 - {model_name.upper()}: MAE={mae:.1f}天, R²={r2:.3f}")
        
        self.models = models
        self.scalers = {'temp': scaler_temp, 'date': scaler_date}
        self.results['model_performance'] = results
        self.results['test_data'] = {
            'X_test': X_test,
            'y_temp_test': y_temp_test,
            'y_date_test': y_date_test,
            'years': df_clean[test_mask]['year'].values
        }
        
        return results
    
    def predict_future(self, future_years=5):
        """
        预测未来几年的夏季最高气温出现日期
        
        Args:
            future_years (int): 预测的年数
        """
        print(f"\n正在预测未来{future_years}年...")
        
        # 获取最佳模型
        best_temp_model = 'rf'  # 根据性能选择
        best_date_model = 'rf'
        
        # 准备预测数据
        last_year = self.processed_data['year'].max()
        future_years_list = list(range(last_year + 1, last_year + future_years + 1))
        
        predictions = []
        
        for year in future_years_list:
            # 构建特征向量 (简化版本，使用趋势外推)
            year_normalized = (year - self.processed_data['year'].min()) / (self.processed_data['year'].max() - self.processed_data['year'].min())
            
            # 使用最近几年的数据作为滞后特征
            recent_data = self.processed_data.tail(3)
            
            features = {
                'year': year,
                'year_normalized': year_normalized,
                'temp_lag_1': recent_data['max_temperature'].iloc[-1],
                'temp_lag_2': recent_data['max_temperature'].iloc[-2],
                'temp_lag_3': recent_data['max_temperature'].iloc[-3],
                'date_lag_1': recent_data['max_temp_day_of_year'].iloc[-1],
                'date_lag_2': recent_data['max_temp_day_of_year'].iloc[-2],
                'date_lag_3': recent_data['max_temp_day_of_year'].iloc[-3],
                'temp_ma_5': recent_data['max_temperature'].mean(),
                'date_ma_5': recent_data['max_temp_day_of_year'].mean()
            }
            
            X_pred = np.array([list(features.values())]).reshape(1, -1)
            
            # 预测温度和日期
            pred_temp = self.models['temperature'][best_temp_model].predict(X_pred)[0]
            pred_date = self.models['date'][best_date_model].predict(X_pred)[0]
            
            # 转换日期为具体日期
            pred_date_obj = datetime(year, 1, 1) + timedelta(days=int(pred_date) - 1)
            
            predictions.append({
                'year': year,
                'predicted_max_temperature': pred_temp,
                'predicted_day_of_year': pred_date,
                'predicted_date': pred_date_obj
            })
            
            print(f"{year}年预测: 最高气温 {pred_temp:.1f}°C, 出现日期 {pred_date_obj.strftime('%m月%d日')}")
        
        self.results['future_predictions'] = pd.DataFrame(predictions)
        return predictions
    
    def create_visualizations(self):
        """
        创建可视化图表
        """
        print("\n正在生成可视化图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('夏季最高气温历史分析与预测 (1940-2024)', fontsize=16, fontweight='bold')
        
        # 1. 历史最高气温趋势
        ax1 = axes[0, 0]
        df = self.processed_data
        
        ax1.plot(df['year'], df['max_temperature'], 'o-', color='red', alpha=0.7, linewidth=2, markersize=4)
        
        # 添加趋势线
        z = np.polyfit(df['year'], df['max_temperature'], 1)
        p = np.poly1d(z)
        ax1.plot(df['year'], p(df['year']), '--', color='darkred', linewidth=2, alpha=0.8)
        
        ax1.set_title('历史夏季最高气温变化趋势', fontweight='bold')
        ax1.set_xlabel('年份')
        ax1.set_ylabel('最高气温 (°C)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 最高气温出现日期趋势
        ax2 = axes[0, 1]
        
        ax2.plot(df['year'], df['max_temp_day_of_year'], 'o-', color='blue', alpha=0.7, linewidth=2, markersize=4)
        
        # 添加趋势线
        z2 = np.polyfit(df['year'], df['max_temp_day_of_year'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(df['year'], p2(df['year']), '--', color='darkblue', linewidth=2, alpha=0.8)
        
        ax2.set_title('最高气温出现日期变化趋势', fontweight='bold')
        ax2.set_xlabel('年份')
        ax2.set_ylabel('年内第几天')
        ax2.grid(True, alpha=0.3)
        
        # 添加月份标记
        month_days = [152, 182, 213]  # 6月1日, 7月1日, 8月1日的大致天数
        month_labels = ['6月', '7月', '8月']
        for day, label in zip(month_days, month_labels):
            ax2.axhline(y=day, color='gray', linestyle=':', alpha=0.5)
            ax2.text(df['year'].min(), day, label, fontsize=10, alpha=0.7)
        
        # 3. 温度分布直方图
        ax3 = axes[1, 0]
        
        ax3.hist(df['max_temperature'], bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax3.axvline(df['max_temperature'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'平均值: {df["max_temperature"].mean():.1f}°C')
        
        ax3.set_title('夏季最高气温分布', fontweight='bold')
        ax3.set_xlabel('最高气温 (°C)')
        ax3.set_ylabel('频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 预测结果对比
        ax4 = axes[1, 1]
        
        # 绘制历史数据
        ax4.plot(df['year'], df['max_temperature'], 'o-', color='blue', alpha=0.7, 
                linewidth=2, markersize=4, label='历史数据')
        
        # 绘制预测数据
        if 'future_predictions' in self.results:
            future_df = self.results['future_predictions']
            ax4.plot(future_df['year'], future_df['predicted_max_temperature'], 
                    's-', color='red', alpha=0.8, linewidth=2, markersize=6, label='预测数据')
            
            # 添加预测区间
            recent_std = df['max_temperature'].tail(10).std()
            ax4.fill_between(future_df['year'], 
                           future_df['predicted_max_temperature'] - recent_std,
                           future_df['predicted_max_temperature'] + recent_std,
                           alpha=0.2, color='red', label='预测区间')
        
        ax4.set_title('历史数据与未来预测对比', fontweight='bold')
        ax4.set_xlabel('年份')
        ax4.set_ylabel('最高气温 (°C)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = 'outputs/summer_temperature_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {output_path}")
        
        plt.show()
        
        # 创建详细的预测图表
        self._create_detailed_prediction_chart()
        
        return output_path
    
    def _create_detailed_prediction_chart(self):
        """
        创建详细的预测图表
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('夏季最高气温详细预测分析', fontsize=16, fontweight='bold')
        
        df = self.processed_data
        
        # 1. 温度预测详细图
        ax1 = axes[0]
        
        # 历史数据
        ax1.plot(df['year'], df['max_temperature'], 'o-', color='blue', alpha=0.7, 
                linewidth=2, markersize=4, label='历史最高气温')
        
        # 预测数据
        if 'future_predictions' in self.results:
            future_df = self.results['future_predictions']
            ax1.plot(future_df['year'], future_df['predicted_max_temperature'], 
                    's-', color='red', alpha=0.8, linewidth=3, markersize=8, label='预测最高气温')
            
            # 连接线
            ax1.plot([df['year'].iloc[-1], future_df['year'].iloc[0]], 
                    [df['max_temperature'].iloc[-1], future_df['predicted_max_temperature'].iloc[0]], 
                    '--', color='gray', alpha=0.5)
        
        ax1.set_title('夏季最高气温预测', fontweight='bold')
        ax1.set_xlabel('年份')
        ax1.set_ylabel('最高气温 (°C)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 日期预测详细图
        ax2 = axes[1]
        
        # 历史数据
        ax2.plot(df['year'], df['max_temp_day_of_year'], 'o-', color='green', alpha=0.7, 
                linewidth=2, markersize=4, label='历史出现日期')
        
        # 预测数据
        if 'future_predictions' in self.results:
            future_df = self.results['future_predictions']
            ax2.plot(future_df['year'], future_df['predicted_day_of_year'], 
                    's-', color='orange', alpha=0.8, linewidth=3, markersize=8, label='预测出现日期')
            
            # 连接线
            ax2.plot([df['year'].iloc[-1], future_df['year'].iloc[0]], 
                    [df['max_temp_day_of_year'].iloc[-1], future_df['predicted_day_of_year'].iloc[0]], 
                    '--', color='gray', alpha=0.5)
        
        # 添加月份参考线
        month_days = [152, 182, 213]  # 6月1日, 7月1日, 8月1日
        month_labels = ['6月1日', '7月1日', '8月1日']
        for day, label in zip(month_days, month_labels):
            ax2.axhline(y=day, color='gray', linestyle=':', alpha=0.5)
            ax2.text(df['year'].min(), day, label, fontsize=10, alpha=0.7)
        
        ax2.set_title('最高气温出现日期预测', fontweight='bold')
        ax2.set_xlabel('年份')
        ax2.set_ylabel('年内第几天')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存详细图表
        output_path = 'outputs/detailed_summer_prediction.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"详细预测图表已保存至: {output_path}")
        
        plt.show()
    
    def generate_report(self):
        """
        生成分析报告
        """
        print("\n正在生成分析报告...")
        
        report = []
        report.append("# 夏季最高气温预测分析报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"数据来源: {self.grib_file_path}")
        
        # 数据概况
        report.append("\n## 数据概况")
        df = self.processed_data
        report.append(f"- 分析年份: {df['year'].min()} - {df['year'].max()}")
        report.append(f"- 数据点数: {len(df)} 年")
        report.append(f"- 平均最高气温: {df['max_temperature'].mean():.1f}°C")
        report.append(f"- 最高气温范围: {df['max_temperature'].min():.1f}°C - {df['max_temperature'].max():.1f}°C")
        
        # 趋势分析
        if 'temp_trend' in self.results:
            temp_trend = self.results['temp_trend']
            date_trend = self.results['date_trend']
            
            report.append("\n## 趋势分析")
            report.append(f"- 气温变化趋势: {temp_trend.slope:.4f}°C/年")
            if temp_trend.pvalue < 0.05:
                report.append(f"  * 趋势显著 (p-value: {temp_trend.pvalue:.4f})")
            else:
                report.append(f"  * 趋势不显著 (p-value: {temp_trend.pvalue:.4f})")
            
            report.append(f"- 日期变化趋势: {date_trend.slope:.4f}天/年")
            if date_trend.pvalue < 0.05:
                report.append(f"  * 趋势显著 (p-value: {date_trend.pvalue:.4f})")
            else:
                report.append(f"  * 趋势不显著 (p-value: {date_trend.pvalue:.4f})")
        
        # 模型性能
        if 'model_performance' in self.results:
            report.append("\n## 模型性能")
            perf = self.results['model_performance']
            
            report.append("### 温度预测模型")
            for model_name in ['temp_rf', 'temp_lr']:
                if model_name in perf:
                    model_type = model_name.split('_')[1].upper()
                    mae = perf[model_name]['mae']
                    r2 = perf[model_name]['r2']
                    report.append(f"- {model_type}: MAE = {mae:.2f}°C, R² = {r2:.3f}")
            
            report.append("\n### 日期预测模型")
            for model_name in ['date_rf', 'date_lr']:
                if model_name in perf:
                    model_type = model_name.split('_')[1].upper()
                    mae = perf[model_name]['mae']
                    r2 = perf[model_name]['r2']
                    report.append(f"- {model_type}: MAE = {mae:.1f}天, R² = {r2:.3f}")
        
        # 未来预测
        if 'future_predictions' in self.results:
            report.append("\n## 未来预测")
            future_df = self.results['future_predictions']
            
            for _, row in future_df.iterrows():
                date_str = row['predicted_date'].strftime('%m月%d日')
                report.append(f"- {row['year']}年: 最高气温 {row['predicted_max_temperature']:.1f}°C, 出现日期 {date_str}")
        
        # 保存报告
        report_text = '\n'.join(report)
        report_path = 'outputs/summer_temperature_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"分析报告已保存至: {report_path}")
        print("\n=== 分析报告 ===")
        print(report_text)
        
        return report_path
    
    def run_complete_analysis(self):
        """
        运行完整的分析流程
        """
        print("开始夏季最高气温预测分析...")
        print("=" * 60)
        
        try:
            # 1. 加载数据
            grib_success = self.load_grib_data()
            if not grib_success:
                print("\n=== 使用模拟数据进行分析 ===")
            
            # 2. 处理数据
            self.process_data()
            
            # 3. 分析趋势
            self.analyze_trends()
            
            # 4. 构建预测模型
            self.build_prediction_models()
            
            # 5. 预测未来
            self.predict_future(future_years=5)
            
            # 6. 生成可视化
            self.create_visualizations()
            
            # 7. 生成报告
            self.generate_report()
            
            print("\n" + "=" * 60)
            print("🎉 分析完成! 所有结果已保存至 outputs/ 目录")
            print("\n📊 生成的文件:")
            print("  - summer_temperature_analysis.png (主要分析图表)")
            print("  - detailed_summer_prediction.png (详细预测图表)")
            print("  - summer_temperature_report.txt (分析报告)")
            
            if 'future_predictions' in self.results:
                print("\n🔮 未来预测摘要:")
                future_df = self.results['future_predictions']
                for _, row in future_df.iterrows():
                    date_str = row['predicted_date'].strftime('%m月%d日')
                    print(f"  {row['year']}年: 最高气温 {row['predicted_max_temperature']:.1f}°C, 出现日期 {date_str}")
            
        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """
    主函数
    """
    # GRIB文件路径
    grib_file_path = r"D:\用户\jin\下载\48d66fb05e73365eaf1d7f778695cb20.grib"
    
    # 创建输出目录
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # 创建预测系统实例
    predictor = SummerTemperaturePrediction(grib_file_path)
    
    # 运行完整分析
    predictor.run_complete_analysis()

if __name__ == "__main__":
    main()