#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试路径访问脚本
"""

import os
from pathlib import Path

def test_single_path():
    """测试单个路径访问"""
    test_path = r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]_1_1~12"
    
    print(f"测试路径: {test_path}")
    print(f"路径长度: {len(test_path)}")
    
    path = Path(test_path)
    
    print(f"路径存在: {path.exists()}")
    print(f"是目录: {path.is_dir()}")
    print(f"是文件: {path.is_file()}")
    
    if path.exists():
        try:
            print("尝试列出目录内容...")
            items = list(path.iterdir())
            print(f"找到 {len(items)} 个项目")
            
            for i, item in enumerate(items[:10]):
                if item.is_file():
                    size = item.stat().st_size
                    print(f"  文件 {i+1}: {item.name} ({size} 字节, {item.suffix})")
                elif item.is_dir():
                    print(f"  目录 {i+1}: {item.name}/")
                    
            if len(items) > 10:
                print(f"  ... 还有 {len(items) - 10} 个项目")
                
        except PermissionError as e:
            print(f"权限错误: {e}")
        except Exception as e:
            print(f"其他错误: {e}")
    else:
        print("路径不存在")
        
        # 尝试检查父目录
        parent = path.parent
        print(f"\n检查父目录: {parent}")
        print(f"父目录存在: {parent.exists()}")
        
        if parent.exists():
            try:
                parent_items = list(parent.iterdir())
                print(f"父目录包含 {len(parent_items)} 个项目:")
                for item in parent_items[:10]:
                    print(f"  - {item.name}")
            except Exception as e:
                print(f"无法访问父目录: {e}")

if __name__ == "__main__":
    test_single_path()