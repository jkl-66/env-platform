#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境保护警示图像生成器

基于 Stable Diffusion 3.5 Large Turbo 模型
支持用户自然语言输入，生成环境保护警示意义的图像
"""

import os
import sys
import torch
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import numpy as np

try:
    from diffusers import StableDiffusion3Pipeline
    import transformers
except ImportError as e:
    print(f"❌ 缺少必要的依赖包: {e}")
    print("请运行: pip install diffusers transformers torch accelerate")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentalImageGenerator:
    """环境保护警示图像生成器"""
    
    def __init__(self, 
                 model_id: str = "stabilityai/stable-diffusion-3.5-large-turbo",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        初始化环境图像生成器
        
        Args:
            model_id: Hugging Face 模型ID
            device: 计算设备 (cuda/cpu)，None时自动检测
            cache_dir: 模型缓存目录
        """
        self.model_id = model_id
        self.device = self._setup_device(device)
        self.cache_dir = cache_dir or str(Path.cwd() / "cache" / "huggingface")
        
        # 设置Hugging Face环境
        self._setup_huggingface_env()
        
        # 初始化管道
        self.pipeline = None
        self.is_loaded = False
        
        # 环境保护提示词模板
        self.environmental_prompts = self._load_environmental_prompts()
        
        logger.info(f"环境图像生成器初始化完成 - 模型: {model_id}, 设备: {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> str:
        """设置计算设备"""
        if device:
            return device
        
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"检测到CUDA设备: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.warning("未检测到CUDA设备，使用CPU（生成速度较慢）")
        
        return device
    
    def _setup_huggingface_env(self):
        """设置Hugging Face环境变量"""
        # 设置缓存目录
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        os.environ['HF_HOME'] = str(cache_path)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_path)
        os.environ['HF_HUB_CACHE'] = str(cache_path)
        
        # 设置镜像（可选）
        # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        logger.info(f"Hugging Face缓存目录: {cache_path}")
    
    def _load_environmental_prompts(self) -> Dict[str, Dict[str, Any]]:
        """加载环境保护提示词模板"""
        return {
            "air_pollution": {
                "base_prompt": "industrial air pollution, smoggy city skyline, thick smoke from factories, poor air quality, environmental warning",
                "keywords": ["pollution", "smog", "factory", "smoke", "air quality"],
                "style_suffix": "dramatic lighting, high contrast, warning atmosphere"
            },
            "water_pollution": {
                "base_prompt": "polluted river with industrial waste, contaminated water, dead fish, environmental disaster",
                "keywords": ["water pollution", "contaminated", "industrial waste", "toxic"],
                "style_suffix": "dark tones, environmental crisis, documentary style"
            },
            "deforestation": {
                "base_prompt": "massive deforestation, cut down trees, environmental destruction, loss of biodiversity",
                "keywords": ["deforestation", "logging", "forest destruction", "habitat loss"],
                "style_suffix": "before and after contrast, environmental impact"
            },
            "climate_change": {
                "base_prompt": "effects of climate change, melting glaciers, rising sea levels, extreme weather",
                "keywords": ["climate change", "global warming", "melting ice", "sea level rise"],
                "style_suffix": "dramatic environmental change, scientific documentation"
            },
            "plastic_pollution": {
                "base_prompt": "ocean plastic pollution, marine life affected by plastic waste, environmental crisis",
                "keywords": ["plastic pollution", "ocean waste", "marine life", "microplastics"],
                "style_suffix": "underwater scene, environmental awareness"
            },
            "renewable_energy": {
                "base_prompt": "renewable energy solutions, solar panels, wind turbines, clean environment, sustainable future",
                "keywords": ["renewable energy", "solar power", "wind energy", "sustainability"],
                "style_suffix": "bright, hopeful, clean technology, positive environmental message"
            }
        }
    
    def load_model(self) -> bool:
        """加载Stable Diffusion模型"""
        try:
            logger.info(f"正在加载模型: {self.model_id}")
            
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=self.cache_dir,
                use_safetensors=True
            )
            
            # 移动到指定设备
            self.pipeline = self.pipeline.to(self.device)
            
            # 启用内存优化
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_attention_slicing()
            
            self.is_loaded = True
            logger.info("✅ 模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def _enhance_environmental_prompt(self, user_input: str, category: Optional[str] = None) -> str:
        """增强用户输入的提示词
        
        Args:
            user_input: 用户输入的自然语言描述
            category: 环境类别 (可选)
            
        Returns:
            增强后的提示词
        """
        # 检测环境类别
        if not category:
            category = self._detect_environmental_category(user_input)
        
        # 获取对应的提示词模板
        template = self.environmental_prompts.get(category, {
            "base_prompt": "",
            "style_suffix": "environmental awareness, high quality, detailed"
        })
        
        # 构建增强提示词
        enhanced_prompt = f"{user_input}, {template['base_prompt']}, {template['style_suffix']}"
        
        # 添加质量和风格描述
        quality_terms = "high quality, detailed, professional photography, environmental documentary style, 4k resolution"
        
        final_prompt = f"{enhanced_prompt}, {quality_terms}"
        
        logger.info(f"原始输入: {user_input}")
        logger.info(f"增强提示词: {final_prompt}")
        
        return final_prompt
    
    def enhance_prompt(self, user_input: str, category: Optional[str] = None) -> str:
        """增强提示词的公共接口"""
        return self._enhance_environmental_prompt(user_input, category)
    
    def _detect_environmental_category(self, user_input: str) -> str:
        """检测用户输入的环境类别"""
        user_input_lower = user_input.lower()
        
        for category, template in self.environmental_prompts.items():
            keywords = template.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in user_input_lower:
                    return category
        
        # 默认返回通用环境类别
        return "air_pollution"
    
    def generate_image(self, 
                      user_input: Optional[str] = None,
                      prompt: Optional[str] = None, 
                      num_images: int = 1,
                      guidance_scale: float = 7.5,
                      num_inference_steps: int = 28,
                      height: int = 1024,
                      width: int = 1024,
                      seed: Optional[int] = None) -> Dict[str, Any]:
        """生成图像
        
        Args:
            user_input: 用户自然语言输入 (优先使用)
            prompt: 直接提示词 (当user_input为None时使用)
            num_images: 生成图像数量
            guidance_scale: 引导强度
            num_inference_steps: 推理步数
            height: 图像高度
            width: 图像宽度
            seed: 随机种子
            
        Returns:
            包含生成结果的字典
        """
        # 自动加载模型
        if not self.is_loaded:
            if not self.load_model():
                return {
                    "success": False,
                    "error": "模型加载失败",
                    "image_paths": [],
                    "images": []
                }
        
        # 处理输入参数
        if user_input:
            final_prompt = self._enhance_environmental_prompt(user_input)
        elif prompt:
            final_prompt = prompt
        else:
            return {
                "success": False,
                "error": "必须提供 user_input 或 prompt 参数",
                "image_paths": [],
                "images": []
            }
        
        try:
            # 设置随机种子
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            logger.info(f"开始生成图像 - 数量: {num_images}, 尺寸: {width}x{height}")
            logger.info(f"使用提示词: {final_prompt}")
            
            start_time = datetime.now()
            
            # 生成图像
            with torch.autocast(self.device):
                result = self.pipeline(
                    prompt=final_prompt,
                    num_images_per_prompt=num_images,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width
                )
            
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            images = result.images
            logger.info(f"✅ 成功生成 {len(images)} 张图像，耗时 {generation_time:.2f} 秒")
            
            # 保存图像
            output_dir = "outputs/environmental_images"
            saved_paths = self.save_images(images, user_input or final_prompt, output_dir)
            
            return {
                "success": True,
                "images": images,
                "image_paths": saved_paths,
                "output_path": output_dir,
                "prompt": final_prompt,
                "generation_time": generation_time,
                "parameters": {
                    "num_images": num_images,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "height": height,
                    "width": width,
                    "seed": seed
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 图像生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_paths": [],
                "images": []
            }
    
    def save_images(self, 
                   images: List[Image.Image], 
                   prompt: str,
                   output_dir: str = "outputs/environmental_images") -> List[str]:
        """保存生成的图像
        
        Args:
            images: 图像列表
            prompt: 原始提示词
            output_dir: 输出目录
            
        Returns:
            保存的文件路径列表
        """
        if not images:
            return []
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_prompt = safe_prompt.replace(" ", "_")
        
        saved_paths = []
        
        for i, image in enumerate(images, 1):
            filename = f"{safe_prompt}_{timestamp}_{i}.png"
            file_path = output_path / filename
            
            image.save(file_path, "PNG")
            saved_paths.append(str(file_path))
            
            logger.info(f"图像已保存: {file_path}")
        
        return saved_paths
    
    def generate_and_save(self, 
                         user_input: str,
                         category: Optional[str] = None,
                         num_images: int = 1,
                         output_dir: str = "outputs/environmental_images",
                         **generation_kwargs) -> Dict[str, Any]:
        """生成并保存图像的便捷方法
        
        Args:
            user_input: 用户自然语言输入
            category: 环境类别
            num_images: 生成图像数量
            output_dir: 输出目录
            **generation_kwargs: 其他生成参数
            
        Returns:
            生成结果信息
        """
        # 增强提示词
        enhanced_prompt = self.enhance_prompt(user_input, category)
        
        # 生成图像
        images = self.generate_image(
            prompt=enhanced_prompt,
            num_images=num_images,
            **generation_kwargs
        )
        
        if not images:
            return {
                "success": False,
                "error": "图像生成失败",
                "user_input": user_input,
                "enhanced_prompt": enhanced_prompt
            }
        
        # 保存图像
        saved_paths = self.save_images(images, user_input, output_dir)
        
        # 生成报告
        report = {
            "success": True,
            "user_input": user_input,
            "enhanced_prompt": enhanced_prompt,
            "category": category or self._detect_environmental_category(user_input),
            "num_images": len(images),
            "saved_files": saved_paths,
            "timestamp": datetime.now().isoformat(),
            "model_id": self.model_id,
            "device": self.device
        }
        
        # 保存生成报告
        self._save_generation_report(report, output_dir)
        
        return report
    
    def _save_generation_report(self, report: Dict[str, Any], output_dir: str):
        """保存生成报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"generation_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"生成报告已保存: {report_file}")
    
    def list_environmental_categories(self) -> Dict[str, str]:
        """列出所有环境类别"""
        return {
            category: template["base_prompt"]
            for category, template in self.environmental_prompts.items()
        }
    
    def update_environmental_prompts(self, new_prompts: Dict[str, Dict[str, Any]]):
        """更新环境提示词模板
        
        Args:
            new_prompts: 新的提示词模板字典
        """
        self.environmental_prompts.update(new_prompts)
        logger.info("环境提示词模板已更新")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "cache_dir": self.cache_dir,
            "environmental_categories": list(self.environmental_prompts.keys())
        }

def main():
    """主函数 - 演示基本用法"""
    print("🌍 环境保护警示图像生成器")
    print("=" * 50)
    
    # 初始化生成器
    generator = EnvironmentalImageGenerator()
    
    # 加载模型
    if not generator.load_model():
        print("❌ 模型加载失败，程序退出")
        return
    
    # 交互式生成
    while True:
        try:
            print("\n请输入您想要生成的环境场景描述（输入 'quit' 退出）:")
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
            
            if not user_input:
                print("❌ 输入不能为空")
                continue
            
            print(f"\n🎨 正在生成图像: {user_input}")
            
            # 生成并保存图像
            result = generator.generate_and_save(
                user_input=user_input,
                num_images=1
            )
            
            if result["success"]:
                print(f"✅ 生成成功！")
                print(f"📁 保存位置: {result['saved_files'][0]}")
                print(f"🏷️  类别: {result['category']}")
            else:
                print(f"❌ 生成失败: {result.get('error', '未知错误')}")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户取消操作")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
    
    print("\n感谢使用环境保护警示图像生成器！")

if __name__ == "__main__":
    main()