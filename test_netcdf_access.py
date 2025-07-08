#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试NetCDF文件访问和复制
"""

import os
import shutil
from pathlib import Path
import sys

def test_file_access(file_path):
    """测试文件访问"""
    print(f"🔍 测试文件访问: {file_path}")
    
    # 测试路径是否存在
    if os.path.exists(file_path):
        print("✅ 文件存在")
    else:
        print("❌ 文件不存在")
        return False
    
    # 测试是否为文件
    if os.path.isfile(file_path):
        print("✅ 是文件")
    else:
        print("❌ 不是文件")
        return False
    
    # 测试文件大小
    try:
        size = os.path.getsize(file_path)
        print(f"✅ 文件大小: {size:,} 字节 ({size/1024/1024:.2f} MB)")
    except Exception as e:
        print(f"❌ 获取文件大小失败: {e}")
        return False
    
    # 测试读取权限
    try:
        with open(file_path, 'rb') as f:
            # 读取前100字节
            data = f.read(100)
            print(f"✅ 文件可读，前100字节: {len(data)} 字节")
    except Exception as e:
        print(f"❌ 文件读取失败: {e}")
        return False
    
    return True

def copy_file_to_local(source_path, local_dir="data"):
    """复制文件到本地目录"""
    print(f"\n📋 复制文件到本地目录...")
    
    # 创建本地目录
    local_path = Path(local_dir)
    local_path.mkdir(exist_ok=True)
    
    # 生成本地文件名
    source_file = Path(source_path)
    local_file = local_path / source_file.name
    
    try:
        print(f"📂 源文件: {source_path}")
        print(f"📂 目标文件: {local_file}")
        
        # 复制文件
        shutil.copy2(source_path, local_file)
        
        print(f"✅ 文件复制成功: {local_file}")
        return str(local_file)
    except Exception as e:
        print(f"❌ 文件复制失败: {e}")
        return None

def test_netcdf_loading(file_path):
    """测试NetCDF文件加载"""
    print(f"\n🧪 测试NetCDF文件加载: {file_path}")
    
    try:
        import xarray as xr
        
        # 打开NetCDF文件
        print("📖 正在打开NetCDF文件...")
        ds = xr.open_dataset(file_path)
        
        print("✅ NetCDF文件打开成功!")
        print(f"📊 维度: {dict(ds.dims)}")
        print(f"📊 变量: {list(ds.data_vars)}")
        print(f"📊 坐标: {list(ds.coords)}")
        
        # 转换为DataFrame
        print("🔄 转换为DataFrame...")
        df = ds.to_dataframe().reset_index()
        print(f"✅ DataFrame创建成功，形状: {df.shape}")
        
        # 移除NaN值
        df_clean = df.dropna()
        print(f"🧹 清理后数据形状: {df_clean.shape}")
        
        if not df_clean.empty:
            print(f"📋 列名: {list(df_clean.columns)}")
            print(f"📋 数据样本:")
            print(df_clean.head())
        
        ds.close()
        return df_clean
        
    except ImportError:
        print("❌ 需要安装xarray库")
        return None
    except Exception as e:
        print(f"❌ NetCDF文件加载失败: {e}")
        return None

def main():
    """主函数"""
    # 测试文件路径
    test_files = [
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]_1_1~12\global_hot-dry_1951_1~12.nc",
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]_1_1~12\global_hot-dry_2020_1~12.nc"
    ]
    
    print("🚀 开始NetCDF文件访问测试...")
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\n{'='*80}")
        print(f"📁 测试文件 {i}: {os.path.basename(test_file)}")
        
        # 测试文件访问
        if test_file_access(test_file):
            print("\n✅ 文件访问测试通过")
            
            # 复制文件到本地
            local_file = copy_file_to_local(test_file)
            
            if local_file:
                # 测试本地文件的NetCDF加载
                df = test_netcdf_loading(local_file)
                
                if df is not None and not df.empty:
                    print(f"\n🎉 成功! 文件 {i} 可以正常加载和处理")
                    print(f"📊 最终数据形状: {df.shape}")
                    
                    # 保存处理结果
                    output_file = f"test_result_{i}.parquet"
                    df.to_parquet(output_file)
                    print(f"💾 测试结果已保存: {output_file}")
                    
                    break  # 找到一个可用文件就停止
                else:
                    print(f"❌ 文件 {i} NetCDF加载失败")
            else:
                print(f"❌ 文件 {i} 复制失败")
        else:
            print(f"❌ 文件 {i} 访问失败")
    
    print(f"\n{'='*80}")
    print("🏁 测试完成")

if __name__ == "__main__":
    main()