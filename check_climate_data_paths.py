#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查气候数据路径脚本

该脚本用于检查用户指定的气候数据路径，并提供数据导入的指导建议。
"""

import os
from pathlib import Path
from typing import List, Dict

def check_path_access(path_str: str) -> Dict:
    """检查路径访问权限和内容"""
    result = {
        'path': path_str,
        'exists': False,
        'accessible': False,
        'is_directory': False,
        'files_count': 0,
        'file_types': [],
        'sample_files': [],
        'error': None
    }
    
    try:
        path = Path(path_str)
        result['exists'] = path.exists()
        
        if result['exists']:
            result['is_directory'] = path.is_dir()
            
            if result['is_directory']:
                try:
                    # 尝试列出目录内容
                    items = list(path.iterdir())
                    result['accessible'] = True
                    result['files_count'] = len([item for item in items if item.is_file()])
                    
                    # 收集文件类型
                    file_extensions = set()
                    sample_files = []
                    
                    for item in items:
                        if item.is_file():
                            ext = item.suffix.lower()
                            if ext:
                                file_extensions.add(ext)
                            
                            # 收集前5个文件作为样本
                            if len(sample_files) < 5:
                                sample_files.append({
                                    'name': item.name,
                                    'size': item.stat().st_size,
                                    'extension': ext
                                })
                    
                    result['file_types'] = list(file_extensions)
                    result['sample_files'] = sample_files
                    
                except PermissionError:
                    result['error'] = "权限不足，无法访问目录内容"
                except Exception as e:
                    result['error'] = f"访问目录时发生错误: {str(e)}"
            else:
                # 如果是文件
                try:
                    stat = path.stat()
                    result['accessible'] = True
                    result['file_types'] = [path.suffix.lower()]
                    result['sample_files'] = [{
                        'name': path.name,
                        'size': stat.st_size,
                        'extension': path.suffix.lower()
                    }]
                except Exception as e:
                    result['error'] = f"访问文件时发生错误: {str(e)}"
        
    except Exception as e:
        result['error'] = f"检查路径时发生错误: {str(e)}"
    
    return result

def suggest_solutions(check_results: List[Dict]) -> List[str]:
    """基于检查结果提供解决方案建议"""
    suggestions = []
    
    # 检查是否有路径不存在
    missing_paths = [r for r in check_results if not r['exists']]
    if missing_paths:
        suggestions.append("❌ 部分路径不存在，请检查路径是否正确")
        for result in missing_paths:
            suggestions.append(f"   - {result['path']}")
    
    # 检查权限问题
    permission_issues = [r for r in check_results if r['exists'] and not r['accessible']]
    if permission_issues:
        suggestions.append("🔒 部分路径存在权限问题，建议解决方案:")
        suggestions.append("   1. 将数据文件复制到项目目录下的 'data' 文件夹")
        suggestions.append("   2. 或者以管理员权限运行脚本")
        suggestions.append("   3. 或者修改文件夹权限")
    
    # 检查文件格式
    accessible_results = [r for r in check_results if r['accessible']]
    if accessible_results:
        all_file_types = set()
        for result in accessible_results:
            all_file_types.update(result['file_types'])
        
        supported_types = {'.csv', '.xlsx', '.xls', '.nc', '.txt', '.dat', '.json', '.tif', '.tiff', '.hdf', '.h5'}
        unsupported_types = all_file_types - supported_types
        
        if unsupported_types:
            suggestions.append(f"⚠️ 发现不支持的文件格式: {', '.join(unsupported_types)}")
            suggestions.append("   建议将数据转换为支持的格式 (CSV, Excel, NetCDF等)")
        
        if all_file_types & supported_types:
            suggestions.append(f"✅ 发现支持的文件格式: {', '.join(all_file_types & supported_types)}")
    
    # 提供数据复制建议
    if any(r['exists'] for r in check_results):
        suggestions.append("\n💡 推荐解决方案:")
        suggestions.append("1. 创建项目数据目录:")
        suggestions.append("   mkdir -p data/climate_datasets")
        suggestions.append("2. 将数据文件复制到项目目录:")
        suggestions.append("   cp -r '原始路径/*' data/climate_datasets/")
        suggestions.append("3. 修改脚本中的路径指向项目目录")
    
    return suggestions

def main():
    """主函数"""
    print("🔍 检查气候数据路径...")
    print("="*80)
    
    # 用户指定的4个数据集路径
    dataset_paths = [
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]_1_1~12",
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry-windy_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]-[sfcwind_p_95_+0.5_0_1]_1_1~12",
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-wet_[tasmax_p_95_all_7_1]-[pr_p_95_+1_0_1]_1_1~12",
        r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_wet-windy_[pr_p_95_+1_0_1]-[sfcwind_p_95_+0.5_0_1]_1_1~12"
    ]
    
    check_results = []
    
    for i, path in enumerate(dataset_paths, 1):
        print(f"\n📂 检查路径 {i}/4:")
        print(f"   {path}")
        
        result = check_path_access(path)
        check_results.append(result)
        
        # 显示检查结果
        if result['exists']:
            print(f"   ✅ 路径存在")
            if result['accessible']:
                print(f"   ✅ 可访问")
                if result['is_directory']:
                    print(f"   📁 目录，包含 {result['files_count']} 个文件")
                    if result['file_types']:
                        print(f"   📄 文件类型: {', '.join(result['file_types'])}")
                    if result['sample_files']:
                        print(f"   📋 样本文件:")
                        for file_info in result['sample_files']:
                            size_mb = file_info['size'] / (1024 * 1024)
                            print(f"      - {file_info['name']} ({size_mb:.2f} MB)")
                else:
                    print(f"   📄 文件")
            else:
                print(f"   ❌ 无法访问: {result['error']}")
        else:
            print(f"   ❌ 路径不存在")
    
    # 生成建议
    print("\n" + "="*80)
    print("📋 检查结果摘要")
    print("="*80)
    
    existing_count = sum(1 for r in check_results if r['exists'])
    accessible_count = sum(1 for r in check_results if r['accessible'])
    
    print(f"总路径数: {len(check_results)}")
    print(f"存在的路径: {existing_count}")
    print(f"可访问的路径: {accessible_count}")
    
    # 显示建议
    suggestions = suggest_solutions(check_results)
    if suggestions:
        print("\n💡 建议解决方案:")
        print("-" * 40)
        for suggestion in suggestions:
            print(suggestion)
    
    # 生成数据复制脚本
    print("\n🔧 自动生成数据复制脚本:")
    print("-" * 40)
    
    # 创建data目录的命令
    print("# 1. 创建项目数据目录")
    print("mkdir -p data\\climate_datasets")
    print("")
    
    # 为每个存在的路径生成复制命令
    print("# 2. 复制数据文件 (请根据实际情况调整)")
    for i, result in enumerate(check_results):
        if result['exists']:
            dataset_type = ['hot-dry', 'hot-dry-windy', 'hot-wet', 'wet-windy'][i]
            print(f"# 复制 {dataset_type} 数据")
            print(f"xcopy \"{result['path']}\\*\" \"data\\climate_datasets\\{dataset_type}\\\" /E /I")
            print("")
    
    print("# 3. 修改脚本路径 (在 import_multiple_climate_datasets.py 中)")
    print("# 将 dataset_paths 修改为:")
    for i, result in enumerate(check_results):
        dataset_type = ['hot-dry', 'hot-dry-windy', 'hot-wet', 'wet-windy'][i]
        print(f"#     r\"data\\climate_datasets\\{dataset_type}\",")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()