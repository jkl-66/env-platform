#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语法检查测试脚本
"""

import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """检查Python文件的语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # 尝试解析AST
        ast.parse(source)
        print(f"✓ {file_path}: 语法正确")
        return True
        
    except SyntaxError as e:
        print(f"✗ {file_path}: 语法错误")
        print(f"  行 {e.lineno}: {e.text.strip() if e.text else ''}")
        print(f"  错误: {e.msg}")
        return False
        
    except Exception as e:
        print(f"✗ {file_path}: 检查失败 - {e}")
        return False

def main():
    """主函数"""
    # 检查 app.py 文件
    app_file = Path("src/web/app.py")
    
    if not app_file.exists():
        print(f"文件不存在: {app_file}")
        sys.exit(1)
    
    print("检查 app.py 语法...")
    success = check_syntax(app_file)
    
    if success:
        print("\n🎉 所有语法检查通过!")
        sys.exit(0)
    else:
        print("\n❌ 发现语法错误")
        sys.exit(1)

if __name__ == "__main__":
    main()