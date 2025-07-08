#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查找实际存在的NetCDF文件
"""

import os
from pathlib import Path

def find_netcdf_files(base_path):
    """查找指定路径下的NetCDF文件"""
    print(f"🔍 搜索路径: {base_path}")
    
    if not os.path.exists(base_path):
        print(f"❌ 路径不存在: {base_path}")
        return []
    
    if not os.path.isdir(base_path):
        print(f"❌ 不是目录: {base_path}")
        return []
    
    netcdf_files = []
    
    try:
        # 列出目录内容
        items = os.listdir(base_path)
        print(f"📁 目录包含 {len(items)} 个项目")
        
        for item in items:
            item_path = os.path.join(base_path, item)
            if os.path.isfile(item_path) and item.endswith('.nc'):
                netcdf_files.append(item_path)
                print(f"✅ 找到NetCDF文件: {item}")
            elif os.path.isfile(item_path):
                print(f"📄 其他文件: {item}")
            else:
                print(f"📁 子目录: {item}")
    
    except PermissionError:
        print(f"❌ 权限不足，无法访问: {base_path}")
    except Exception as e:
        print(f"❌ 访问目录时发生错误: {e}")
    
    return netcdf_files

def main():
    """主函数"""
    # 测试路径
    test_paths = [
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]_1_1~12",
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry-windy_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]-[sfcwind_p_95_+0.5_0_1]_1_1~12",
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-wet_[tasmax_p_95_all_7_1]-[pr_p_95_+1_0_1]_1_1~12",
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_wet-windy_[pr_p_95_+1_0_1]-[sfcwind_p_95_+0.5_0_1]_1_1~12"
    ]
    
    all_netcdf_files = []
    
    for path in test_paths:
        print(f"\n{'='*80}")
        netcdf_files = find_netcdf_files(path)
        all_netcdf_files.extend(netcdf_files)
        
        if netcdf_files:
            print(f"\n📊 在此目录找到 {len(netcdf_files)} 个NetCDF文件:")
            for i, file_path in enumerate(netcdf_files[:5], 1):  # 只显示前5个
                print(f"  {i}. {os.path.basename(file_path)}")
            if len(netcdf_files) > 5:
                print(f"  ... 还有 {len(netcdf_files) - 5} 个文件")
        else:
            print("❌ 未找到NetCDF文件")
    
    print(f"\n{'='*80}")
    print(f"🎯 总结:")
    print(f"  总共找到 {len(all_netcdf_files)} 个NetCDF文件")
    
    if all_netcdf_files:
        print(f"\n🔧 可用于测试的文件路径:")
        for i, file_path in enumerate(all_netcdf_files[:3], 1):  # 显示前3个用于测试
            print(f"  {i}. {file_path}")
        
        # 保存文件列表
        output_file = "found_netcdf_files.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for file_path in all_netcdf_files:
                f.write(f"{file_path}\n")
        print(f"\n💾 完整文件列表已保存到: {output_file}")
    else:
        print("❌ 未找到任何NetCDF文件")

if __name__ == "__main__":
    main()