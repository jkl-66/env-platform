#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test path access script
"""

import os
from pathlib import Path

def test_single_path():
    """测试单个路径访问"""
    test_path = r"D:\用户\jin\桌面\全球0.5°逐年复合极端气候事件数据集（1951-2022年）-数据实体\global_hot-dry_[tasmax_p_95_all_7_1]-[pr_p_5_+0_0_1]_1_1~12"
    
    print(f"Test path: {test_path}")
print(f"Path length: {len(test_path)}")
    
    path = Path(test_path)
    
    print(f"Path exists: {path.exists()}")
print(f"Is directory: {path.is_dir()}")
print(f"Is file: {path.is_file()}")
    
    if path.exists():
        try:
            print("Attempting to list directory contents...")
        items = list(path.iterdir())
        print(f"Found {len(items)} items")
            
            for i, item in enumerate(items[:10]):
                if item.is_file():
                    size = item.stat().st_size
                    print(f"  File {i+1}: {item.name} ({size} bytes, {item.suffix})")
                elif item.is_dir():
                    print(f"  Directory {i+1}: {item.name}/")
                    
            if len(items) > 10:
                print(f"  ... {len(items) - 10} more items")
                
        except PermissionError as e:
            print(f"Permission error: {e}")
    except Exception as e:
        print(f"Other error: {e}")
else:
    print("Path does not exist")
        
        # Try to check parent directory
        parent = path.parent
        print(f"\nChecking parent directory: {parent}")
        print(f"Parent directory exists: {parent.exists()}")
        
        if parent.exists():
            try:
                parent_items = list(parent.iterdir())
                print(f"Parent directory contains {len(parent_items)} items:")
                for item in parent_items[:10]:
                    print(f"  - {item.name}")
            except Exception as e:
                print(f"Cannot access parent directory: {e}")

if __name__ == "__main__":
    test_single_path()