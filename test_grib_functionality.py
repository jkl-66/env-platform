#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRIB处理功能测试脚本

此脚本用于测试GRIB文件处理功能是否正常工作。
包括创建模拟GRIB数据、测试各种处理功能等。
"""

import os
import sys
import tempfile
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.grib_processor import GRIBProcessor
from src.data_processing.data_storage import DataStorage
from src.data_processing.data_processor import ProcessingConfig

def create_mock_grib_data(output_path):
    """
    创建模拟的GRIB格式数据用于测试
    """
    print("创建模拟GRIB数据...")
    
    # 创建时间序列
    times = [datetime(2023, 1, 1) + timedelta(hours=i*6) for i in range(8)]  # 2天，每6小时
    
    # 创建空间网格
    lats = np.linspace(20, 50, 31)  # 31个纬度点
    lons = np.linspace(100, 130, 31)  # 31个经度点
    
    # 创建模拟数据
    np.random.seed(42)  # 确保可重复性
    
    # 2米温度 (t2m)
    t2m_data = np.random.normal(288.15, 10, (len(times), len(lats), len(lons)))  # 约15°C ± 10K
    
    # 总降水量 (tp)
    tp_data = np.random.exponential(0.001, (len(times), len(lats), len(lons)))  # 指数分布，单位m
    
    # 10米风速分量
    u10_data = np.random.normal(0, 5, (len(times), len(lats), len(lons)))  # m/s
    v10_data = np.random.normal(0, 5, (len(times), len(lats), len(lons)))  # m/s
    
    # 海平面气压
    msl_data = np.random.normal(101325, 1000, (len(times), len(lats), len(lons)))  # Pa
    
    # 创建xarray Dataset
    ds = xr.Dataset({
        't2m': (['time', 'latitude', 'longitude'], t2m_data, {
            'long_name': '2 metre temperature',
            'units': 'K',
            'standard_name': 'air_temperature'
        }),
        'tp': (['time', 'latitude', 'longitude'], tp_data, {
            'long_name': 'Total precipitation',
            'units': 'm',
            'standard_name': 'precipitation_amount'
        }),
        'u10': (['time', 'latitude', 'longitude'], u10_data, {
            'long_name': '10 metre U wind component',
            'units': 'm s**-1',
            'standard_name': 'eastward_wind'
        }),
        'v10': (['time', 'latitude', 'longitude'], v10_data, {
            'long_name': '10 metre V wind component',
            'units': 'm s**-1',
            'standard_name': 'northward_wind'
        }),
        'msl': (['time', 'latitude', 'longitude'], msl_data, {
            'long_name': 'Mean sea level pressure',
            'units': 'Pa',
            'standard_name': 'air_pressure_at_mean_sea_level'
        })
    }, coords={
        'time': times,
        'latitude': lats,
        'longitude': lons
    })
    
    # 添加全局属性
    ds.attrs.update({
        'title': 'Mock GRIB data for testing',
        'institution': 'Environmental Data Platform',
        'source': 'Test data generator',
        'history': f'Created on {datetime.now().isoformat()}',
        'Conventions': 'CF-1.6'
    })
    
    # 保存为NetCDF（模拟GRIB数据）
    ds.to_netcdf(output_path, engine='netcdf4')
    print(f"模拟数据已保存到: {output_path}")
    
    return output_path

def test_grib_processor():
    """
    测试GRIBProcessor类的功能
    """
    print("\n=== 测试GRIBProcessor类 ===")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建模拟数据
        mock_file = os.path.join(temp_dir, 'test_data.nc')
        create_mock_grib_data(mock_file)
        
        # 初始化处理器
        processor = GRIBProcessor()
        
        try:
            # 测试1: 加载文件
            print("\n1. 测试文件加载...")
            data = processor.load_grib_file(mock_file)
            print(f"   ✓ 成功加载数据，形状: {data.dims}")
            print(f"   ✓ 变量: {list(data.data_vars.keys())}")
            
            # 测试2: 获取文件信息
            print("\n2. 测试文件信息获取...")
            info = processor.get_grib_info(mock_file)
            print(f"   ✓ 文件大小: {info['file_size_mb']:.2f} MB")
            print(f"   ✓ 变量数量: {len(info['variables'])}")
            print(f"   ✓ 时间范围: {info['time_range']}")
            
            # 测试3: 提取变量
            print("\n3. 测试变量提取...")
            temp_data = processor.extract_variables(data, ['t2m'])
            print(f"   ✓ 成功提取温度数据，形状: {temp_data.dims}")
            
            # 测试4: 转换为NetCDF
            print("\n4. 测试格式转换...")
            output_nc = os.path.join(temp_dir, 'converted.nc')
            result = processor.convert_to_netcdf(
                mock_file, 
                output_nc, 
                variables=['t2m', 'tp']
            )
            print(f"   ✓ 成功转换为NetCDF: {result['output_file']}")
            print(f"   ✓ 文件大小: {result['file_size_mb']:.2f} MB")
            
            # 测试5: 数据处理
            print("\n5. 测试数据处理...")
            config = ProcessingConfig(
                quality_control=True,
                remove_outliers=True,
                fill_missing=True
            )
            
            processed_data = processor.process_grib_data(
                mock_file,
                config=config,
                variables=['t2m']
            )
            print(f"   ✓ 成功处理数据，形状: {processed_data.dims}")
            
            # 测试6: 保存处理后的数据
            print("\n6. 测试数据保存...")
            output_processed = os.path.join(temp_dir, 'processed.nc')
            save_result = processor.save_processed_data(
                processed_data,
                output_processed
            )
            print(f"   ✓ 成功保存处理后的数据: {save_result['output_file']}")
            
            print("\n✅ GRIBProcessor所有测试通过！")
            
        except Exception as e:
            print(f"\n❌ GRIBProcessor测试失败: {str(e)}")
            return False
    
    return True

def test_data_storage():
    """
    测试DataStorage类的GRIB支持
    """
    print("\n=== 测试DataStorage类 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建模拟数据
        mock_file = os.path.join(temp_dir, 'test_data.nc')
        create_mock_grib_data(mock_file)
        
        # 初始化存储管理器
        storage = DataStorage()
        
        try:
            # 测试1: 自动检测加载
            print("\n1. 测试自动检测加载...")
            data = storage.load_xarray(mock_file)
            print(f"   ✓ 成功加载数据，变量: {list(data.data_vars.keys())}")
            
            # 测试2: 显式GRIB加载（使用NetCDF作为替代）
            print("\n2. 测试显式文件加载...")
            data2 = storage.load_xarray(mock_file)
            print(f"   ✓ 成功加载数据，时间维度: {len(data2.time)}")
            
            print("\n✅ DataStorage所有测试通过！")
            
        except Exception as e:
            print(f"\n❌ DataStorage测试失败: {str(e)}")
            return False
    
    return True

def test_common_variables():
    """
    测试常见GRIB变量信息
    """
    print("\n=== 测试常见GRIB变量 ===")
    
    try:
        variables = GRIBProcessor.get_common_grib_variables()
        print(f"\n支持的常见GRIB变量数量: {len(variables)}")
        
        # 显示前5个变量
        print("\n前5个变量:")
        for i, (code, info) in enumerate(list(variables.items())[:5]):
            print(f"   {code}: {info['description']} ({info['units']})")
        
        print("\n✅ 变量信息测试通过！")
        
    except Exception as e:
        print(f"\n❌ 变量信息测试失败: {str(e)}")
        return False
    
    return True

def main():
    """
    主测试函数
    """
    print("开始GRIB处理功能测试...")
    print("=" * 50)
    
    # 检查依赖
    try:
        import cfgrib
        print(f"✓ cfgrib版本: {cfgrib.__version__}")
    except ImportError:
        print("⚠️  cfgrib未安装，某些功能可能不可用")
    
    # 运行测试
    tests = [
        ("GRIBProcessor功能", test_grib_processor),
        ("DataStorage GRIB支持", test_data_storage),
        ("常见GRIB变量", test_common_variables)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试出现异常: {str(e)}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("测试总结:")
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("\n🎉 所有GRIB处理功能测试通过！")
        print("\n可以开始使用以下功能:")
        print("  - 命令行工具: python cli.py grib --help")
        print("  - Python API: from src.data_processing.grib_processor import GRIBProcessor")
        print("  - 文档: docs/GRIB_PROCESSING_GUIDE.md")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)