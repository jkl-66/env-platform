#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生态警示图像生成脚本
使用CPU生成各种生态警示和环保教育图像
"""

import sys
import torch
import numpy as np
from pathlib import Path
import json
import datetime
from typing import List, Dict, Any

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.models.ecology_image_generator import EcologyImageGenerator

class EcologyImageGenerationSystem:
    """生态图像生成系统"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.generator = EcologyImageGenerator(device=device)
        self.output_dir = Path("outputs/ecology_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 预定义的生态场景
        self.ecology_scenarios = {
            "forest_protection": {
                "name": "森林保护",
                "conditions": [0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.8, 0.9],
                "description": "展示健康森林生态系统的重要性"
            },
            "air_pollution": {
                "name": "空气污染警示",
                "conditions": [0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.1, 0.2],
                "description": "警示空气污染对环境的危害"
            },
            "water_conservation": {
                "name": "水资源保护",
                "conditions": [0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.9, 0.8],
                "description": "强调水资源保护的重要性"
            },
            "climate_change": {
                "name": "气候变化影响",
                "conditions": [0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.3, 0.7, 0.2, 0.3],
                "description": "展示气候变化对生态的影响"
            },
            "renewable_energy": {
                "name": "可再生能源",
                "conditions": [0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2, 0.9, 0.9],
                "description": "推广清洁能源的使用"
            },
            "wildlife_protection": {
                "name": "野生动物保护",
                "conditions": [0.8, 0.2, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.8, 0.8],
                "description": "保护野生动物栖息地"
            }
        }
    
    def initialize_system(self):
        """初始化生成系统"""
        print("=== 初始化生态图像生成系统 ===")
        print(f"使用设备: {self.device}")
        
        # 构建模型
        print("构建生成模型...")
        self.generator.build_model()
        print("模型构建完成")
        
        # 设置为GAN模式（因为扩散模型需要网络连接）
        self.generator.generation_mode = "gan"
        print(f"生成模式: {self.generator.generation_mode}")
    
    def generate_scenario_images(self, scenario_key: str, num_images: int = 3) -> Dict[str, Any]:
        """生成特定场景的图像"""
        if scenario_key not in self.ecology_scenarios:
            raise ValueError(f"未知场景: {scenario_key}")
        
        scenario = self.ecology_scenarios[scenario_key]
        print(f"\n=== 生成场景: {scenario['name']} ===")
        print(f"描述: {scenario['description']}")
        print(f"条件: {scenario['conditions']}")
        
        try:
            # 准备输入数据
            input_data = {
                "conditions": scenario['conditions']
            }
            
            # 生成图像
            result = self.generator.predict(input_data, num_images=num_images)
            
            if "error" in result:
                print(f"生成失败: {result['error']}")
                return {"success": False, "error": result['error']}
            
            # 保存图像
            saved_files = self.save_scenario_images(
                result['generated_images'], 
                scenario_key, 
                scenario['name']
            )
            
            print(f"成功生成 {len(result['generated_images'])} 张图像")
            
            return {
                "success": True,
                "scenario": scenario['name'],
                "num_images": len(result['generated_images']),
                "saved_files": saved_files,
                "generation_mode": result.get('generation_mode', 'unknown')
            }
            
        except Exception as e:
            print(f"生成场景图像失败: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def save_scenario_images(self, images: List, scenario_key: str, scenario_name: str) -> List[str]:
        """保存场景图像"""
        try:
            from PIL import Image
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = []
            
            for i, img_data in enumerate(images):
                # 转换图像数据
                img_array = np.array(img_data)
                
                # 确保数据在正确范围内
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
                
                # 创建PIL图像
                img = Image.fromarray(img_array)
                
                # 保存图像
                filename = f"{scenario_key}_{timestamp}_{i+1}.png"
                filepath = self.output_dir / filename
                img.save(filepath)
                
                saved_files.append(str(filepath))
                print(f"图像已保存: {filepath}")
            
            return saved_files
            
        except Exception as e:
            print(f"保存图像失败: {e}")
            return []
    
    def generate_all_scenarios(self, images_per_scenario: int = 2) -> Dict[str, Any]:
        """生成所有预定义场景的图像"""
        print("\n=== 生成所有生态场景图像 ===")
        
        results = {}
        total_images = 0
        successful_scenarios = 0
        
        for scenario_key in self.ecology_scenarios.keys():
            result = self.generate_scenario_images(scenario_key, images_per_scenario)
            results[scenario_key] = result
            
            if result['success']:
                successful_scenarios += 1
                total_images += result['num_images']
        
        # 保存生成报告
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_scenarios": len(self.ecology_scenarios),
            "successful_scenarios": successful_scenarios,
            "total_images_generated": total_images,
            "device_used": self.device,
            "generation_mode": self.generator.generation_mode,
            "results": results
        }
        
        report_file = self.output_dir / f"generation_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== 生成完成 ===")
        print(f"成功场景: {successful_scenarios}/{len(self.ecology_scenarios)}")
        print(f"总图像数: {total_images}")
        print(f"报告已保存: {report_file}")
        
        return report
    
    def generate_custom_scenario(self, name: str, conditions: List[float], description: str = "", num_images: int = 2) -> Dict[str, Any]:
        """生成自定义场景图像"""
        print(f"\n=== 生成自定义场景: {name} ===")
        print(f"描述: {description}")
        print(f"条件: {conditions}")
        
        try:
            # 准备输入数据
            input_data = {
                "conditions": conditions
            }
            
            # 生成图像
            result = self.generator.predict(input_data, num_images=num_images)
            
            if "error" in result:
                print(f"生成失败: {result['error']}")
                return {"success": False, "error": result['error']}
            
            # 保存图像
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            scenario_key = f"custom_{name.replace(' ', '_').lower()}"
            
            saved_files = self.save_scenario_images(
                result['generated_images'], 
                scenario_key, 
                name
            )
            
            print(f"成功生成 {len(result['generated_images'])} 张自定义图像")
            
            return {
                "success": True,
                "scenario": name,
                "num_images": len(result['generated_images']),
                "saved_files": saved_files,
                "generation_mode": result.get('generation_mode', 'unknown')
            }
            
        except Exception as e:
            print(f"生成自定义场景图像失败: {e}")
            return {"success": False, "error": str(e)}

def main():
    """主函数"""
    print("生态警示图像生成系统")
    print("=" * 50)
    
    # 初始化系统
    system = EcologyImageGenerationSystem(device="cpu")
    system.initialize_system()
    
    # 生成所有预定义场景
    report = system.generate_all_scenarios(images_per_scenario=2)
    
    # 生成一些自定义场景
    print("\n=== 生成自定义场景 ===")
    
    # 自定义场景1：城市绿化
    system.generate_custom_scenario(
        name="城市绿化",
        conditions=[0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.3, 0.8, 0.7],
        description="展示城市绿化对环境改善的作用",
        num_images=2
    )
    
    # 自定义场景2：海洋保护
    system.generate_custom_scenario(
        name="海洋保护",
        conditions=[0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.9, 0.8],
        description="保护海洋生态系统的重要性",
        num_images=2
    )
    
    print("\n=== 所有图像生成完成 ===")
    print(f"图像保存目录: {system.output_dir}")
    
    # 显示生成统计
    if report['successful_scenarios'] > 0:
        print(f"\n✅ 成功生成了 {report['total_images_generated']} 张生态警示图像")
        print("这些图像可用于:")
        print("- 环保教育宣传")
        print("- 生态警示展示")
        print("- 气候变化科普")
        print("- 可持续发展教育")
    else:
        print("\n❌ 图像生成失败，请检查系统配置")

if __name__ == "__main__":
    main()