#!/usr/bin/env python3
"""
分析训练数据的脚本
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / 'src'))

# 设置环境变量
os.environ['PYTHONPATH'] = str(current_dir / 'src')

try:
    from src.data_processing.grib_processor import GRIBProcessor
except ImportError:
    print("无法导入GRIBProcessor，尝试直接分析已有的模型数据...")
    GRIBProcessor = None

def analyze_training_data():
    """分析训练数据的统计信息"""
    print("正在分析训练数据...")
    
    try:
        if GRIBProcessor is None:
            print("无法使用GRIBProcessor，尝试分析已有的CSV数据...")
            # 尝试读取已有的预测数据来了解数据结构
            csv_files = list(Path('examples').glob('*.csv'))
            if csv_files:
                print(f"找到CSV文件: {[f.name for f in csv_files]}")
                df = pd.read_csv(csv_files[0])
                print(f"从 {csv_files[0].name} 加载数据")
            else:
                print("没有找到可分析的数据文件")
                return None
        else:
            # 使用GRIB处理器加载数据
             processor = GRIBProcessor()
             df = processor.process_grib_to_dataframe('data/raw/6cd7cc57755a5204a65bc7db615cd36b.grib', sample_size=50)
        
        print(f"\n数据形状: {df.shape}")
        print(f"数据列: {list(df.columns)}")
        
        print("\n=== 数据统计信息 ===")
        print(df.describe())
        
        print("\n=== 各列的唯一值数量 ===")
        for col in df.columns:
            unique_count = df[col].nunique()
            total_count = len(df)
            print(f"{col}: {unique_count} 个唯一值 (总共 {total_count} 行)")
            
            # 如果唯一值很少，显示这些值
            if unique_count <= 10:
                unique_values = df[col].unique()
                print(f"  唯一值: {unique_values}")
        
        print("\n=== 数据变异性分析 ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                std_dev = df[col].std()
                mean_val = df[col].mean()
                if pd.isna(std_dev) or pd.isna(mean_val):
                    print(f"{col}: 包含NaN值，跳过分析")
                    continue
                cv = std_dev / abs(mean_val) if mean_val != 0 else float('inf')
                print(f"{col}: 标准差={std_dev:.6f}, 均值={mean_val:.6f}, 变异系数={cv:.6f}")
            except Exception as e:
                print(f"{col}: 分析时出错 - {e}")
        
        # 检查时间信息
        if 'time' in df.columns:
            print("\n=== 时间信息分析 ===")
            df['time'] = pd.to_datetime(df['time'])
            print(f"时间范围: {df['time'].min()} 到 {df['time'].max()}")
            print(f"时间跨度: {df['time'].max() - df['time'].min()}")
            print(f"唯一时间点数量: {df['time'].nunique()}")
            
            # 检查时间分布
            time_counts = df['time'].value_counts().head(10)
            print(f"\n最常见的时间点:")
            for time_val, count in time_counts.items():
                print(f"  {time_val}: {count} 次")
        
        return df
        
    except Exception as e:
        print(f"分析数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyze_training_data()