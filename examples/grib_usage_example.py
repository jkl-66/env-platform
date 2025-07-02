"""GRIB文件处理使用示例

本示例展示如何使用GRIBProcessor类来处理GRIB格式的气象数据文件。
包括：
- 加载GRIB文件
- 提取特定变量
- 数据处理和质量控制
- 格式转换
- 批量处理
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.grib_processor import GRIBProcessor
from src.data_processing.data_processor import ProcessingConfig
from src.data_processing.data_storage import DataStorage
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

def example_basic_grib_loading():
    """示例1：基本GRIB文件加载"""
    print("\n=== 示例1：基本GRIB文件加载 ===")
    
    # 创建GRIB处理器
    grib_processor = GRIBProcessor()
    
    # 假设你有一个GRIB文件
    grib_file_path = "data/raw/sample_weather_data.grib"
    
    # 检查文件是否存在
    if not Path(grib_file_path).exists():
        print(f"GRIB文件不存在: {grib_file_path}")
        print("请将你的GRIB文件放在指定路径，或修改路径")
        return
    
    try:
        # 加载GRIB文件
        dataset = grib_processor.load_grib_file(grib_file_path)
        
        print(f"成功加载GRIB文件: {grib_file_path}")
        print(f"数据维度: {dataset.dims}")
        print(f"包含变量: {list(dataset.data_vars.keys())}")
        print(f"坐标系: {list(dataset.coords.keys())}")
        
        # 显示数据集基本信息
        print("\n数据集信息:")
        print(dataset)
        
        dataset.close()
        
    except Exception as e:
        print(f"加载GRIB文件时出错: {e}")

def example_grib_file_info():
    """示例2：获取GRIB文件详细信息"""
    print("\n=== 示例2：获取GRIB文件信息 ===")
    
    grib_processor = GRIBProcessor()
    grib_file_path = "data/raw/sample_weather_data.grib"
    
    if not Path(grib_file_path).exists():
        print(f"GRIB文件不存在: {grib_file_path}")
        return
    
    try:
        # 获取文件信息
        info = grib_processor.get_grib_info(grib_file_path)
        
        print("GRIB文件详细信息:")
        print(f"文件路径: {info['file_path']}")
        print(f"文件大小: {info['file_size'] / 1024 / 1024:.2f} MB")
        print(f"变量列表: {info['variables']}")
        print(f"坐标系: {info['coordinates']}")
        print(f"维度信息: {info['dimensions']}")
        
        if info['time_range']:
            print(f"时间范围: {info['time_range']['start']} 到 {info['time_range']['end']}")
            print(f"时间点数量: {info['time_range']['count']}")
        
        if info['spatial_extent']:
            print("空间范围:")
            for coord, extent in info['spatial_extent'].items():
                print(f"  {coord}: {extent['min']:.2f} 到 {extent['max']:.2f} ({extent['count']} 点)")
        
    except Exception as e:
        print(f"获取GRIB文件信息时出错: {e}")

def example_extract_variables():
    """示例3：提取特定变量和时间范围"""
    print("\n=== 示例3：提取特定变量 ===")
    
    grib_processor = GRIBProcessor()
    grib_file_path = "data/raw/sample_weather_data.grib"
    
    if not Path(grib_file_path).exists():
        print(f"GRIB文件不存在: {grib_file_path}")
        return
    
    try:
        # 显示常见GRIB变量
        common_vars = GRIBProcessor.get_common_grib_variables()
        print("常见GRIB变量:")
        for var, desc in list(common_vars.items())[:5]:  # 显示前5个
            print(f"  {var}: {desc}")
        
        # 提取特定变量（根据你的GRIB文件调整变量名）
        variables_to_extract = ['t2m', 'tp']  # 2米气温和总降水量
        
        # 定义时间范围（可选）
        start_time = datetime.now() - timedelta(days=7)
        end_time = datetime.now()
        
        # 定义空间范围（可选）
        spatial_bounds = {
            'latitude': (30, 50),   # 纬度范围
            'longitude': (100, 130)  # 经度范围
        }
        
        extracted_data = grib_processor.extract_variables(
            grib_file_path,
            variables=variables_to_extract,
            time_range=(start_time, end_time),
            spatial_bounds=spatial_bounds
        )
        
        print(f"\n成功提取变量: {list(extracted_data.data_vars.keys())}")
        print(f"提取后的数据维度: {extracted_data.dims}")
        
        # 显示每个变量的统计信息
        for var in extracted_data.data_vars:
            data_var = extracted_data[var]
            print(f"\n变量 {var} 统计信息:")
            print(f"  形状: {data_var.shape}")
            print(f"  最小值: {float(data_var.min()):.2f}")
            print(f"  最大值: {float(data_var.max()):.2f}")
            print(f"  平均值: {float(data_var.mean()):.2f}")
        
        extracted_data.close()
        
    except Exception as e:
        print(f"提取变量时出错: {e}")

def example_convert_to_netcdf():
    """示例4：转换GRIB文件为NetCDF格式"""
    print("\n=== 示例4：转换为NetCDF格式 ===")
    
    grib_processor = GRIBProcessor()
    grib_file_path = "data/raw/sample_weather_data.grib"
    output_path = "data/processed/converted_weather_data.nc"
    
    if not Path(grib_file_path).exists():
        print(f"GRIB文件不存在: {grib_file_path}")
        return
    
    try:
        # 转换为NetCDF格式
        converted_file = grib_processor.convert_to_netcdf(
            grib_path=grib_file_path,
            output_path=output_path,
            variables=['t2m', 'tp'],  # 只转换特定变量
            compression=True  # 启用压缩
        )
        
        print(f"成功转换GRIB文件为NetCDF: {converted_file}")
        
        # 验证转换后的文件
        data_storage = DataStorage()
        converted_data = data_storage.load_xarray(converted_file)
        
        if converted_data:
            print(f"转换后文件包含变量: {list(converted_data.data_vars.keys())}")
            converted_data.close()
        
    except Exception as e:
        print(f"转换文件时出错: {e}")

def example_process_grib_data():
    """示例5：处理GRIB数据（质量控制、清洗等）"""
    print("\n=== 示例5：处理GRIB数据 ===")
    
    grib_processor = GRIBProcessor()
    grib_file_path = "data/raw/sample_weather_data.grib"
    
    if not Path(grib_file_path).exists():
        print(f"GRIB文件不存在: {grib_file_path}")
        return
    
    try:
        # 配置数据处理参数
        processing_config = ProcessingConfig(
            remove_outliers=True,
            outlier_method='iqr',
            fill_missing=True,
            interpolation_method='linear',
            smooth_data=True,
            smoothing_method='rolling_mean',
            smoothing_window=3,
            normalize=False,
            temporal_aggregation='daily'
        )
        
        # 处理GRIB数据
        processed_data = grib_processor.process_grib_data(
            grib_file_path,
            config=processing_config,
            variables=['t2m', 'tp']
        )
        
        print(f"数据处理完成")
        print(f"处理后的变量: {list(processed_data.data_vars.keys())}")
        
        # 保存处理后的数据
        output_path = "data/processed/processed_weather_data.nc"
        saved_file = grib_processor.save_processed_data(
            processed_data,
            output_path,
            format='netcdf'
        )
        
        print(f"处理后的数据已保存: {saved_file}")
        
        processed_data.close()
        
    except Exception as e:
        print(f"处理数据时出错: {e}")

def example_batch_processing():
    """示例6：批量处理多个GRIB文件"""
    print("\n=== 示例6：批量处理GRIB文件 ===")
    
    grib_processor = GRIBProcessor()
    input_dir = "data/raw/grib_files"  # 包含多个GRIB文件的目录
    output_dir = "data/processed/batch_output"
    
    # 检查输入目录
    if not Path(input_dir).exists():
        print(f"输入目录不存在: {input_dir}")
        print("请创建目录并放入GRIB文件")
        return
    
    try:
        # 配置处理参数
        processing_config = ProcessingConfig(
            remove_outliers=True,
            fill_missing=True,
            normalize=False
        )
        
        # 批量处理
        processed_files = grib_processor.batch_process_grib_files(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern="*.grib*",  # 匹配所有GRIB文件
            config=processing_config,
            output_format='netcdf'
        )
        
        print(f"批量处理完成，共处理 {len(processed_files)} 个文件:")
        for file_path in processed_files:
            print(f"  - {file_path}")
        
    except Exception as e:
        print(f"批量处理时出错: {e}")

def example_multiple_grib_files():
    """示例7：加载和合并多个GRIB文件"""
    print("\n=== 示例7：合并多个GRIB文件 ===")
    
    grib_processor = GRIBProcessor()
    
    # 多个GRIB文件路径
    grib_files = [
        "data/raw/weather_20240101.grib",
        "data/raw/weather_20240102.grib",
        "data/raw/weather_20240103.grib"
    ]
    
    # 检查文件是否存在
    existing_files = [f for f in grib_files if Path(f).exists()]
    
    if not existing_files:
        print("没有找到GRIB文件")
        return
    
    try:
        # 加载并合并多个文件
        merged_data = grib_processor.load_multiple_grib_files(
            existing_files,
            concat_dim='time'  # 按时间维度合并
        )
        
        print(f"成功合并 {len(existing_files)} 个GRIB文件")
        print(f"合并后的维度: {merged_data.dims}")
        print(f"时间范围: {merged_data.time.min().values} 到 {merged_data.time.max().values}")
        
        # 保存合并后的数据
        output_path = "data/processed/merged_weather_data.nc"
        saved_file = grib_processor.save_processed_data(
            merged_data,
            output_path,
            format='netcdf'
        )
        
        print(f"合并后的数据已保存: {saved_file}")
        
        merged_data.close()
        
    except Exception as e:
        print(f"合并文件时出错: {e}")

def main():
    """主函数：运行所有示例"""
    print("GRIB文件处理示例")
    print("=" * 50)
    
    # 创建必要的目录
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 运行示例
    try:
        example_basic_grib_loading()
        example_grib_file_info()
        example_extract_variables()
        example_convert_to_netcdf()
        example_process_grib_data()
        example_batch_processing()
        example_multiple_grib_files()
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行示例时出错: {e}")
    
    print("\n=== 使用说明 ===")
    print("1. 将你的GRIB文件放在 data/raw/ 目录下")
    print("2. 根据你的GRIB文件调整变量名称")
    print("3. 根据需要修改时间和空间范围")
    print("4. 处理后的文件将保存在 data/processed/ 目录下")
    
    print("\n=== 常见GRIB变量 ===")
    common_vars = GRIBProcessor.get_common_grib_variables()
    for var, desc in common_vars.items():
        print(f"  {var}: {desc}")

if __name__ == "__main__":
    main()