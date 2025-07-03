#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生态警示图像查看器
用于查看和展示生成的生态警示图像
"""

import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

class EcologyImageViewer:
    """生态图像查看器"""
    
    def __init__(self, images_dir="outputs/ecology_images"):
        self.images_dir = Path(images_dir)
        self.scenario_names = {
            "forest_protection": "森林保护",
            "air_pollution": "空气污染警示", 
            "water_conservation": "水资源保护",
            "climate_change": "气候变化影响",
            "renewable_energy": "可再生能源",
            "wildlife_protection": "野生动物保护",
            "custom_城市绿化": "城市绿化",
            "custom_海洋保护": "海洋保护"
        }
    
    def get_latest_report(self):
        """获取最新的生成报告"""
        report_files = list(self.images_dir.glob("generation_report_*.json"))
        if not report_files:
            return None
        
        # 按修改时间排序，获取最新的
        latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_report, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_all_images(self):
        """获取所有图像文件"""
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(self.images_dir.glob(ext))
        
        # 排除报告文件
        image_files = [f for f in image_files if not f.name.startswith('generation_report')]
        
        return sorted(image_files)
    
    def display_image_grid(self, max_images=8):
        """以网格形式显示图像"""
        image_files = self.get_all_images()
        
        if not image_files:
            print("未找到生成的图像文件")
            return
        
        # 限制显示数量
        image_files = image_files[:max_images]
        
        # 计算网格布局
        n_images = len(image_files)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        fig.suptitle('生态警示图像生成结果', fontsize=16, fontweight='bold')
        
        # 确保axes是二维数组
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, img_file in enumerate(image_files):
            row = i // cols
            col = i % cols
            
            try:
                # 加载并显示图像
                img = Image.open(img_file)
                axes[row, col].imshow(img)
                
                # 从文件名提取场景信息
                filename = img_file.stem
                scenario_key = '_'.join(filename.split('_')[:-2])  # 移除时间戳和序号
                
                # 获取中文名称
                chinese_name = self.scenario_names.get(scenario_key, scenario_key)
                
                axes[row, col].set_title(chinese_name, fontsize=12, fontweight='bold')
                axes[row, col].axis('off')
                
                # 添加边框
                rect = patches.Rectangle((0, 0), img.width-1, img.height-1, 
                                       linewidth=2, edgecolor='green', facecolor='none')
                axes[row, col].add_patch(rect)
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'加载失败\n{str(e)}', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'错误: {img_file.name}', fontsize=10)
                axes[row, col].axis('off')
        
        # 隐藏多余的子图
        for i in range(n_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def display_single_image(self, image_path):
        """显示单个图像"""
        try:
            img = Image.open(image_path)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            
            # 从文件名提取场景信息
            filename = Path(image_path).stem
            scenario_key = '_'.join(filename.split('_')[:-2])
            chinese_name = self.scenario_names.get(scenario_key, scenario_key)
            
            plt.title(f'{chinese_name}\n{Path(image_path).name}', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # 添加图像信息
            info_text = f'尺寸: {img.size[0]}x{img.size[1]}\n模式: {img.mode}'
            plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"显示图像失败: {e}")
    
    def print_generation_summary(self):
        """打印生成摘要"""
        report = self.get_latest_report()
        image_files = self.get_all_images()
        
        print("=" * 60)
        print("生态警示图像生成摘要")
        print("=" * 60)
        
        if report:
            print(f"生成时间: {report['timestamp']}")
            print(f"使用设备: {report['device_used']}")
            print(f"生成模式: {report['generation_mode']}")
            print(f"成功场景: {report['successful_scenarios']}/{report['total_scenarios']}")
            print(f"总图像数: {report['total_images_generated']}")
            print()
            
            print("场景详情:")
            for scenario_key, result in report['results'].items():
                chinese_name = self.scenario_names.get(scenario_key, scenario_key)
                status = "✅" if result['success'] else "❌"
                print(f"  {status} {chinese_name}: {result['num_images']} 张图像")
        
        print(f"\n当前图像文件: {len(image_files)} 个")
        print(f"保存目录: {self.images_dir.absolute()}")
        
        print("\n图像用途:")
        print("- 环保教育宣传材料")
        print("- 生态警示展示")
        print("- 气候变化科普")
        print("- 可持续发展教育")
        print("- 环境保护意识提升")
        
        print("=" * 60)
    
    def create_image_catalog(self):
        """创建图像目录HTML文件"""
        image_files = self.get_all_images()
        report = self.get_latest_report()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>生态警示图像目录</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            background: linear-gradient(135deg, #4CAF50, #2E7D32);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .image-card {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .image-card:hover {{
            transform: translateY(-5px);
        }}
        .image-card img {{
            width: 100%;
            height: 200px;
            object-fit: cover;
        }}
        .image-info {{
            padding: 15px;
        }}
        .image-title {{
            font-size: 18px;
            font-weight: bold;
            color: #2E7D32;
            margin-bottom: 10px;
        }}
        .image-details {{
            color: #666;
            font-size: 14px;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-number {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .stat-label {{
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌱 生态警示图像目录</h1>
        <p>AI生成的环保教育和生态警示图像集合</p>
    </div>
    
    <div class="summary">
        <h2>生成摘要</h2>
        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{len(image_files)}</div>
                <div class="stat-label">总图像数</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{report['successful_scenarios'] if report else 'N/A'}</div>
                <div class="stat-label">成功场景</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{report['generation_mode'].upper() if report else 'N/A'}</div>
                <div class="stat-label">生成模式</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{report['device_used'].upper() if report else 'N/A'}</div>
                <div class="stat-label">使用设备</div>
            </div>
        </div>
        <p><strong>生成时间:</strong> {report['timestamp'] if report else 'N/A'}</p>
    </div>
    
    <div class="image-grid">
"""
        
        for img_file in image_files:
            filename = img_file.stem
            scenario_key = '_'.join(filename.split('_')[:-2])
            chinese_name = self.scenario_names.get(scenario_key, scenario_key)
            
            # 获取图像信息
            try:
                img = Image.open(img_file)
                img_info = f"{img.size[0]}x{img.size[1]}, {img.mode}"
            except:
                img_info = "信息获取失败"
            
            html_content += f"""
        <div class="image-card">
            <img src="{img_file.name}" alt="{chinese_name}">
            <div class="image-info">
                <div class="image-title">{chinese_name}</div>
                <div class="image-details">
                    <p><strong>文件名:</strong> {img_file.name}</p>
                    <p><strong>图像信息:</strong> {img_info}</p>
                </div>
            </div>
        </div>
"""
        
        html_content += """
    </div>
    
    <div style="text-align: center; margin-top: 40px; color: #666;">
        <p>这些图像可用于环保教育、生态警示展示、气候变化科普等用途</p>
        <p>生成时间: {}</p>
    </div>
</body>
</html>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # 保存HTML文件
        html_file = self.images_dir / "image_catalog.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"图像目录已创建: {html_file}")
        return html_file

def main():
    """主函数"""
    viewer = EcologyImageViewer()
    
    # 打印生成摘要
    viewer.print_generation_summary()
    
    # 创建HTML目录
    html_file = viewer.create_image_catalog()
    
    # 显示图像网格
    print("\n正在显示图像网格...")
    try:
        viewer.display_image_grid()
    except Exception as e:
        print(f"显示图像网格失败: {e}")
        print("请确保已安装matplotlib和PIL库")
    
    print(f"\n✅ 图像查看完成")
    print(f"📁 图像目录: {viewer.images_dir.absolute()}")
    print(f"🌐 HTML目录: {html_file}")

if __name__ == "__main__":
    main()