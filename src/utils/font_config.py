#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文字体配置工具模块

提供统一的中文字体配置功能，确保matplotlib图表能正确显示中文字符。
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import List, Optional
import platform
import warnings

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_available_chinese_fonts() -> List[str]:
    """
    获取系统中可用的中文字体列表
    
    Returns:
        List[str]: 可用的中文字体名称列表
    """
    chinese_fonts = []
    
    # 常见的中文字体名称
    common_chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'SimSun',           # 宋体
        'KaiTi',            # 楷体
        'FangSong',         # 仿宋
        'Arial Unicode MS', # Arial Unicode MS
        'PingFang SC',      # 苹方 (macOS)
        'Hiragino Sans GB', # 冬青黑体 (macOS)
        'WenQuanYi Micro Hei', # 文泉驿微米黑 (Linux)
        'Noto Sans CJK SC', # 思源黑体 (Linux)
        'DejaVu Sans'       # DejaVu Sans (通用)
    ]
    
    # 获取系统所有字体
    system_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 检查哪些中文字体可用
    for font in common_chinese_fonts:
        if font in system_fonts:
            chinese_fonts.append(font)
            logger.debug(f"找到可用中文字体: {font}")
    
    # 如果没有找到任何中文字体，添加通用字体
    if not chinese_fonts:
        chinese_fonts = ['DejaVu Sans', 'sans-serif']
        logger.warning("未找到中文字体，使用默认字体")
    
    return chinese_fonts


def configure_chinese_fonts(font_size: int = 10) -> None:
    """
    配置matplotlib的中文字体支持
    
    Args:
        font_size (int): 默认字体大小
    """
    try:
        # 获取可用的中文字体
        chinese_fonts = get_available_chinese_fonts()
        
        # 配置matplotlib参数
        plt.rcParams['font.sans-serif'] = chinese_fonts
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 设置字体大小
        plt.rcParams['font.size'] = font_size
        plt.rcParams['figure.titlesize'] = font_size + 4
        plt.rcParams['axes.titlesize'] = font_size + 2
        plt.rcParams['axes.labelsize'] = font_size
        plt.rcParams['xtick.labelsize'] = font_size - 1
        plt.rcParams['ytick.labelsize'] = font_size - 1
        plt.rcParams['legend.fontsize'] = font_size - 1
        
        # 清除字体缓存
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['font.sans-serif'] = chinese_fonts
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = font_size
        plt.rcParams['figure.titlesize'] = font_size + 4
        plt.rcParams['axes.titlesize'] = font_size + 2
        plt.rcParams['axes.labelsize'] = font_size
        plt.rcParams['xtick.labelsize'] = font_size - 1
        plt.rcParams['ytick.labelsize'] = font_size - 1
        plt.rcParams['legend.fontsize'] = font_size - 1
        
        logger.info(f"中文字体配置完成，使用字体: {chinese_fonts[0]}")
        
    except Exception as e:
        logger.error(f"配置中文字体失败: {e}")
        # 使用备用配置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        logger.warning("使用备用字体配置")


def test_chinese_font_display() -> bool:
    """
    测试中文字体显示是否正常
    
    Returns:
        bool: 如果中文字体显示正常返回True，否则返回False
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from io import BytesIO
        
        # 配置中文字体
        configure_chinese_fonts()
        
        # 创建测试图表
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 测试数据
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        # 绘制图表
        ax.plot(x, y, label='正弦波')
        ax.set_title('中文字体测试图表')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('幅度')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加中文注释
        ax.text(5, 0.5, '这是中文测试文本\n包含：数字123、符号-+*/、英文ABC', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                ha='center', va='center')
        
        # 保存到内存中测试
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        plt.close(fig)
        
        # 检查是否成功生成图像
        if buffer.getvalue():
            logger.info("中文字体测试通过")
            return True
        else:
            logger.error("中文字体测试失败：无法生成图像")
            return False
            
    except Exception as e:
        logger.error(f"中文字体测试失败: {e}")
        return False


def format_number(value, decimal_places=None):
    """
    格式化数字，确保在中文环境下正确显示
    
    Args:
        value: 要格式化的数值
        decimal_places: 小数位数，None表示自动判断
    
    Returns:
        str: 格式化后的数字字符串
    """
    try:
        if decimal_places is not None:
            formatted = f"{float(value):.{decimal_places}f}"
        else:
            # 自动判断小数位数
            if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
                formatted = f"{int(value)}"
            else:
                formatted = f"{float(value):.4f}".rstrip('0').rstrip('.')
        
        # 确保小数点正确显示
        return formatted.replace('.', '.')
    except (ValueError, TypeError):
        return str(value)


def format_percentage(value, decimal_places=1):
    """
    格式化百分比，确保在中文环境下正确显示
    
    Args:
        value: 要格式化的数值（0-100）
        decimal_places: 小数位数
    
    Returns:
        str: 格式化后的百分比字符串
    """
    try:
        formatted = f"{float(value):.{decimal_places}f}"
        return formatted.replace('.', '.') + '%'
    except (ValueError, TypeError):
        return str(value) + '%'


def get_font_info() -> dict:
    """
    获取当前字体配置信息
    
    Returns:
        dict: 字体配置信息
    """
    import matplotlib
    return {
        'current_font': plt.rcParams['font.sans-serif'],
        'font_size': plt.rcParams['font.size'],
        'unicode_minus': plt.rcParams['axes.unicode_minus'],
        'available_fonts': get_available_chinese_fonts(),
        'system': platform.system(),
        'matplotlib_version': matplotlib.__version__
    }


def print_font_info() -> None:
    """
    打印字体配置信息
    """
    info = get_font_info()
    
    print("\n=== 字体配置信息 ===")
    print(f"当前字体: {info['current_font']}")
    print(f"字体大小: {info['font_size']}")
    print(f"负号显示: {'正常' if not info['unicode_minus'] else '异常'}")
    print(f"操作系统: {info['system']}")
    print(f"Matplotlib版本: {info['matplotlib_version']}")
    print(f"\n可用中文字体:")
    for i, font in enumerate(info['available_fonts'], 1):
        print(f"  {i}. {font}")
    print("=" * 25)


# 自动配置中文字体（模块导入时执行）
if __name__ != '__main__':
    try:
        configure_chinese_fonts()
    except Exception as e:
        logger.warning(f"自动配置中文字体失败: {e}")


if __name__ == '__main__':
    # 测试脚本
    print("🔤 中文字体配置测试")
    print("=" * 30)
    
    # 打印字体信息
    print_font_info()
    
    # 测试中文字体显示
    print("\n🧪 测试中文字体显示...")
    if test_chinese_font_display():
        print("✅ 中文字体显示测试通过")
    else:
        print("❌ 中文字体显示测试失败")
    
    print("\n测试完成！")