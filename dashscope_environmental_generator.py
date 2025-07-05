#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于阿里云 DashScope 的环境保护警示图像生成器

使用 Qwen 聊天模型生成专业的环境警示 prompt
使用 Flux 图像生成模型生成高质量的环境警示图像
支持用户输入碳排放量、污染指数等环境数据
"""

import os
import sys
import json
import logging
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from PIL import Image
from io import BytesIO
import requests
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath

# 导入 DashScope
try:
    from dashscope import Generation, ImageSynthesis
except ImportError:
    print("❌ 请安装 dashscope: pip install dashscope")
    sys.exit(1)

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 导入配置
try:
    from src.utils.config import get_settings
    settings = get_settings()
except ImportError:
    settings = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashScopeEnvironmentalGenerator:
    """基于阿里云 DashScope 的环境保护警示图像生成器"""
    
    def __init__(self, 
                 dashscope_api_key: Optional[str] = None,
                 chat_model: str = "qwen-turbo",
                 image_model: str = "flux-schnell"):
        """
        初始化 DashScope 环境图像生成器
        
        Args:
            dashscope_api_key: DashScope API Key
            chat_model: 聊天模型名称
            image_model: 图像生成模型名称
        """
        self.api_key = dashscope_api_key or os.getenv('DASHSCOPE_API_KEY')
        self.chat_model = chat_model
        self.image_model = image_model
        
        if not self.api_key:
            raise ValueError("❌ 未设置 DASHSCOPE_API_KEY，请在 .env 文件中配置")
        
        # 设置 API Key
        os.environ['DASHSCOPE_API_KEY'] = self.api_key
        
        # 环境数据类型定义（包含当前世界正常数据作为默认值）
        self.environmental_data_types = {
            "carbon_emission": {
                "name": "碳排放量",
                "unit": "吨CO2当量",
                "default_value": 150,  # 当前世界平均碳排放量
                "thresholds": {
                    "low": 100,
                    "medium": 500,
                    "high": 1000,
                    "critical": 2000
                }
            },
            "air_quality_index": {
                "name": "空气质量指数",
                "unit": "AQI",
                "default_value": 75,  # 当前世界平均AQI
                "thresholds": {
                    "good": 50,
                    "moderate": 100,
                    "unhealthy_sensitive": 150,
                    "unhealthy": 200,
                    "very_unhealthy": 300,
                    "hazardous": 500
                }
            },
            "water_pollution_index": {
                "name": "水污染指数",
                "unit": "WPI",
                "default_value": 35,  # 当前世界平均水污染指数
                "thresholds": {
                    "clean": 25,
                    "slightly_polluted": 50,
                    "moderately_polluted": 75,
                    "heavily_polluted": 100
                }
            },
            "noise_level": {
                "name": "噪音水平",
                "unit": "分贝(dB)",
                "default_value": 55,  # 当前世界平均噪音水平
                "thresholds": {
                    "quiet": 40,
                    "moderate": 55,
                    "loud": 70,
                    "very_loud": 85,
                    "harmful": 100
                }
            },
            "deforestation_rate": {
                "name": "森林砍伐率",
                "unit": "公顷/年",
                "default_value": 3000,  # 当前世界平均森林砍伐率
                "thresholds": {
                    "low": 1000,
                    "medium": 5000,
                    "high": 10000,
                    "critical": 20000
                }
            },
            "plastic_waste": {
                "name": "塑料废物量",
                "unit": "吨/年",
                "default_value": 250,  # 当前世界平均塑料废物量
                "thresholds": {
                    "low": 100,
                    "medium": 500,
                    "high": 1000,
                    "critical": 2000
                }
            }
        }
        
        # 图像风格定义
        self.image_styles = {
            "general": {
                "style": "realistic environmental documentary photography",
                "mood": "serious and informative",
                "color_palette": "natural colors with dramatic contrast"
            },
            "educators": {
                "style": "professional educational illustration",
                "mood": "clear and instructional",
                "color_palette": "balanced colors with good visibility"
            },
            "parents": {
                "style": "approachable realistic photography",
                "mood": "concerning but not frightening",
                "color_palette": "warm tones with clear messaging"
            },
            "students": {
                "style": "cartoon illustration, animated style",
                "mood": "engaging and educational",
                "color_palette": "bright and vibrant colors"
            }
        }
        
        logger.info(f"✅ DashScope 环境图像生成器初始化完成")
        logger.info(f"🤖 聊天模型: {self.chat_model}")
        logger.info(f"🎨 图像模型: {self.image_model}")
    
    def _calculate_deviation_analysis(self, data: Dict[str, Union[float, int]]) -> Dict[str, Any]:
        """
        计算环境数据与默认值的偏差分析
        
        Args:
            data: 环境数据字典
            
        Returns:
            偏差分析结果
        """
        deviation_analysis = {
            "primary_concerns": [],  # 主要关注点（偏差最大的）
            "secondary_concerns": [],  # 次要关注点
            "normal_factors": [],  # 正常范围内的因素
            "deviation_scores": {}  # 偏差分数
        }
        
        for data_type, value in data.items():
            if data_type not in self.environmental_data_types:
                continue
                
            default_value = self.environmental_data_types[data_type]["default_value"]
            
            # 计算偏差比例
            if default_value > 0:
                deviation_ratio = (value - default_value) / default_value
            else:
                deviation_ratio = value / 100  # 避免除零错误
            
            # 计算偏差分数（绝对值，用于排序）
            deviation_score = abs(deviation_ratio)
            
            deviation_info = {
                "type": data_type,
                "name": self.environmental_data_types[data_type]["name"],
                "current_value": value,
                "default_value": default_value,
                "deviation_ratio": deviation_ratio,
                "deviation_score": deviation_score,
                "unit": self.environmental_data_types[data_type]["unit"]
            }
            
            deviation_analysis["deviation_scores"][data_type] = deviation_info
            
            # 分类偏差程度
            if deviation_score >= 1.0:  # 偏差100%以上
                deviation_analysis["primary_concerns"].append(deviation_info)
            elif deviation_score >= 0.3:  # 偏差30%以上
                deviation_analysis["secondary_concerns"].append(deviation_info)
            else:
                deviation_analysis["normal_factors"].append(deviation_info)
        
        # 按偏差分数排序
        deviation_analysis["primary_concerns"].sort(key=lambda x: x["deviation_score"], reverse=True)
        deviation_analysis["secondary_concerns"].sort(key=lambda x: x["deviation_score"], reverse=True)
        
        return deviation_analysis
    
    def _analyze_environmental_data(self, data: Dict[str, Union[float, int]]) -> Dict[str, Any]:
        """
        分析环境数据，确定污染等级和影响
        
        Args:
            data: 环境数据字典，键为数据类型，值为数值
            
        Returns:
            分析结果字典
        """
        analysis = {
            "overall_severity": "low",
            "critical_factors": [],
            "environmental_impacts": [],
            "severity_scores": {},
            "deviation_analysis": self._calculate_deviation_analysis(data)  # 添加偏差分析
        }
        
        total_severity_score = 0
        valid_factors = 0
        
        for data_type, value in data.items():
            if data_type not in self.environmental_data_types:
                logger.warning(f"⚠️ 未知的环境数据类型: {data_type}")
                continue
            
            data_config = self.environmental_data_types[data_type]
            thresholds = data_config["thresholds"]
            
            # 确定严重程度
            if data_type == "air_quality_index":
                if value <= thresholds["good"]:
                    severity = "good"
                    score = 1
                elif value <= thresholds["moderate"]:
                    severity = "moderate"
                    score = 2
                elif value <= thresholds["unhealthy_sensitive"]:
                    severity = "unhealthy_sensitive"
                    score = 3
                elif value <= thresholds["unhealthy"]:
                    severity = "unhealthy"
                    score = 4
                elif value <= thresholds["very_unhealthy"]:
                    severity = "very_unhealthy"
                    score = 5
                else:
                    severity = "hazardous"
                    score = 6
            else:
                # 通用阈值判断
                threshold_keys = list(thresholds.keys())
                if value <= thresholds[threshold_keys[0]]:
                    severity = threshold_keys[0]
                    score = 1
                elif value <= thresholds[threshold_keys[1]]:
                    severity = threshold_keys[1]
                    score = 2
                elif value <= thresholds[threshold_keys[2]]:
                    severity = threshold_keys[2]
                    score = 3
                else:
                    severity = threshold_keys[3] if len(threshold_keys) > 3 else threshold_keys[2]
                    score = 4
            
            analysis["severity_scores"][data_type] = {
                "value": value,
                "severity": severity,
                "score": score,
                "unit": data_config["unit"]
            }
            
            total_severity_score += score
            valid_factors += 1
            
            # 识别关键因素
            if score >= 4:
                analysis["critical_factors"].append({
                    "type": data_type,
                    "name": data_config["name"],
                    "value": value,
                    "unit": data_config["unit"],
                    "severity": severity
                })
        
        # 计算总体严重程度
        if valid_factors > 0:
            avg_score = total_severity_score / valid_factors
            if avg_score >= 5:
                analysis["overall_severity"] = "critical"
            elif avg_score >= 4:
                analysis["overall_severity"] = "high"
            elif avg_score >= 3:
                analysis["overall_severity"] = "medium"
            else:
                analysis["overall_severity"] = "low"
        
        return analysis
    
    def _generate_professional_prompt(self, 
                                    environmental_data: Dict[str, Union[float, int]],
                                    user_description: Optional[str] = None,
                                    target_audience: str = "general") -> str:
        """
        使用 Qwen 模型生成专业的环境警示图像 prompt
        
        Args:
            environmental_data: 环境数据
            user_description: 用户描述
            target_audience: 目标受众 (general, educators, parents, students)
            
        Returns:
            生成的专业 prompt
        """
        # 分析环境数据
        analysis = self._analyze_environmental_data(environmental_data)
        
        # 获取偏差分析和图像风格
        deviation_analysis = analysis.get("deviation_analysis", {})
        style_config = self.image_styles.get(target_audience, self.image_styles["general"])
        
        # 构建主次关注点描述
        primary_concerns = deviation_analysis.get("primary_concerns", [])
        secondary_concerns = deviation_analysis.get("secondary_concerns", [])
        
        primary_desc = ""
        secondary_desc = ""
        
        if primary_concerns:
            primary_items = []
            for concern in primary_concerns[:2]:  # 最多取前2个主要关注点
                deviation_pct = abs(concern["deviation_ratio"]) * 100
                primary_items.append(f"{concern['name']} (偏差{deviation_pct:.0f}%)")
            primary_desc = f"主要关注点：{', '.join(primary_items)}"
        
        if secondary_concerns:
            secondary_items = []
            for concern in secondary_concerns[:3]:  # 最多取前3个次要关注点
                deviation_pct = abs(concern["deviation_ratio"]) * 100
                secondary_items.append(f"{concern['name']} (偏差{deviation_pct:.0f}%)")
            secondary_desc = f"次要关注点：{', '.join(secondary_items)}"
        
        # 构建系统提示
        system_prompt = f"""
你是一位专业的环境保护教育专家和视觉设计师。你的任务是根据提供的环境数据，生成一个专业的、具有教育意义的环境警示图像描述prompt。

要求：
1. 基于具体的环境数据数值，创建真实可信的场景
2. 突出环境问题的严重性和紧迫性
3. 适合教育用途，能够引起观众的环保意识
4. 描述要具体、生动，包含视觉细节
5. 避免过于恐怖或极端的内容
6. 包含希望和解决方案的元素
7. **重要限制：图像中绝对不能包含人物、人脸、人体或任何人类形象**
8. **重要限制：图像中不能包含任何文字、标签、标识或文本元素**
9. 图像风格：{style_config['style']}
10. 情绪氛围：{style_config['mood']}
11. 色彩搭配：{style_config['color_palette']}

prompt应该包含：
- 具体的环境场景描述
- 污染或环境问题的视觉表现
- 对动物或生态系统的影响（不包含人类）
- 专业的摄影风格描述
- 适当的色彩和光线描述

图像主次控制：
- 如果有主要关注点，应该在图像中占据主导地位（60-70%的视觉重点）
- 次要关注点作为背景或辅助元素（20-30%的视觉重点）
- 正常范围内的因素可以作为环境背景（10%的视觉重点）

请用英文生成prompt，长度控制在200-300个单词。
"""
        
        # 构建用户输入
        data_description = []
        for data_type, score_info in analysis["severity_scores"].items():
            data_config = self.environmental_data_types[data_type]
            data_description.append(
                f"{data_config['name']}: {score_info['value']} {score_info['unit']} (严重程度: {score_info['severity']})"
            )
        
        user_input = f"""
环境数据分析结果：
{chr(10).join(data_description)}

总体严重程度：{analysis['overall_severity']}
关键问题因素：{len(analysis['critical_factors'])}个

偏差分析结果：
{primary_desc}
{secondary_desc}

目标受众：{target_audience}
图像风格要求：{style_config['style']} 风格，{style_config['mood']} 氛围，{style_config['color_palette']} 色调
"""
        
        if user_description:
            user_input += f"\n\n用户补充描述：{user_description}"
        
        user_input += "\n\n请生成一个专业的环境警示图像描述prompt（英文），确保图像主次分明，不包含人物和文字。"
        
        try:
            logger.info("🤖 正在使用 Qwen 模型生成专业 prompt...")
            
            response = Generation.call(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                result_format='message'
            )
            
            if response.status_code == HTTPStatus.OK:
                generated_prompt = response.output.choices[0].message.content.strip()
                logger.info(f"✅ 专业 prompt 生成成功")
                logger.info(f"📝 生成的 prompt: {generated_prompt[:100]}...")
                return generated_prompt
            else:
                logger.error(f"❌ Qwen 模型调用失败: {response.message}")
                return self._fallback_prompt_generation(environmental_data, analysis)
                
        except Exception as e:
            logger.error(f"❌ 生成专业 prompt 时发生错误: {e}")
            return self._fallback_prompt_generation(environmental_data, analysis)
    
    def _fallback_prompt_generation(self, 
                                  environmental_data: Dict[str, Union[float, int]], 
                                  analysis: Dict[str, Any]) -> str:
        """
        备用 prompt 生成方法
        
        Args:
            environmental_data: 环境数据
            analysis: 环境数据分析结果
            
        Returns:
            备用生成的 prompt
        """
        severity_map = {
            "low": "mild environmental concern",
            "medium": "moderate environmental pollution",
            "high": "severe environmental degradation",
            "critical": "critical environmental crisis"
        }
        
        base_prompt = f"Environmental warning scene showing {severity_map.get(analysis['overall_severity'], 'environmental issues')}"
        
        # 添加具体的环境问题
        if "carbon_emission" in environmental_data:
            base_prompt += ", industrial emissions and carbon pollution"
        if "air_quality_index" in environmental_data:
            base_prompt += ", smoggy air and poor visibility"
        if "water_pollution_index" in environmental_data:
            base_prompt += ", contaminated water bodies"
        if "deforestation_rate" in environmental_data:
            base_prompt += ", deforested landscapes"
        
        base_prompt += ", professional environmental documentary photography, high contrast, dramatic lighting, educational purpose, realistic style, 4k quality"
        
        return base_prompt
    
    def _generate_image_with_flux(self, prompt: str, size: str = '1024*1024') -> Optional[Image.Image]:
        """
        使用 Flux 模型生成图像
        
        Args:
            prompt: 图像生成 prompt
            size: 图像尺寸
            
        Returns:
            生成的 PIL Image 对象
        """
        try:
            logger.info(f"🎨 正在使用 Flux 模型生成图像...")
            logger.info(f"📝 Prompt: {prompt}")
            
            response = ImageSynthesis.call(
                model=self.image_model,
                prompt=prompt,
                size=size
            )
            
            if response.status_code == HTTPStatus.OK:
                logger.info(f"✅ 图像生成成功")
                logger.info(f"📊 使用情况: {response.usage}")
                
                # 下载并处理图像
                for result in response.output.results:
                    file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
                    image_content = requests.get(result.url).content
                    image = Image.open(BytesIO(image_content))
                    return image
                    
            else:
                logger.error(f"❌ Flux 模型调用失败: status_code={response.status_code}, code={response.code}, message={response.message}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 图像生成时发生错误: {e}")
            return None
    
    def generate_environmental_warning_image(self,
                                           environmental_data: Dict[str, Union[float, int]],
                                           user_description: Optional[str] = None,
                                           target_audience: str = "general",
                                           image_size: str = '1024*1024',
                                           auto_open: bool = True) -> Dict[str, Any]:
        """
        生成环境警示图像
        
        Args:
            environmental_data: 环境数据字典
            user_description: 用户补充描述
            target_audience: 目标受众
            image_size: 图像尺寸
            auto_open: 是否自动打开图片
            
        Returns:
            生成结果字典
        """
        start_time = datetime.now()
        
        try:
            # 1. 分析环境数据
            logger.info("📊 分析环境数据...")
            analysis = self._analyze_environmental_data(environmental_data)
            
            # 2. 生成专业 prompt
            logger.info("🤖 生成专业 prompt...")
            professional_prompt = self._generate_professional_prompt(
                environmental_data, user_description, target_audience
            )
            
            # 3. 生成图像
            logger.info("🎨 生成环境警示图像...")
            image = self._generate_image_with_flux(professional_prompt, image_size)
            
            if not image:
                return {
                    "success": False,
                    "error": "图像生成失败",
                    "analysis": analysis,
                    "prompt": professional_prompt
                }
            
            # 4. 保存图像
            output_dir = "outputs/environmental_images"
            saved_paths = self._save_images([image], environmental_data, output_dir, auto_open)
            
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            # 5. 生成报告
            result = {
                "success": True,
                "environmental_data": environmental_data,
                "analysis": analysis,
                "professional_prompt": professional_prompt,
                "image": image,
                "saved_paths": saved_paths,
                "generation_time": generation_time,
                "target_audience": target_audience,
                "timestamp": datetime.now().isoformat(),
                "models_used": {
                    "chat_model": self.chat_model,
                    "image_model": self.image_model
                }
            }
            
            # 保存生成报告
            self._save_generation_report(result, output_dir)
            
            logger.info(f"✅ 环境警示图像生成完成，耗时 {generation_time:.2f} 秒")
            return result
            
        except Exception as e:
            logger.error(f"❌ 生成环境警示图像时发生错误: {e}")
            return {
                "success": False,
                "error": str(e),
                "environmental_data": environmental_data
            }
    
    def _save_images(self, 
                    images: List[Image.Image], 
                    environmental_data: Dict[str, Union[float, int]],
                    output_dir: str = "outputs/environmental_images",
                    auto_open: bool = True) -> List[str]:
        """
        保存生成的图像
        
        Args:
            images: 图像列表
            environmental_data: 环境数据
            output_dir: 输出目录
            auto_open: 是否自动打开图片
            
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
        data_summary = "_".join([f"{k}_{v}" for k, v in list(environmental_data.items())[:2]])
        safe_summary = "".join(c for c in data_summary if c.isalnum() or c in ('_', '-'))[:30]
        
        saved_paths = []
        
        for i, image in enumerate(images, 1):
            filename = f"env_warning_{safe_summary}_{timestamp}_{i}_dashscope.png"
            file_path = output_path / filename
            
            image.save(file_path, "PNG")
            saved_paths.append(str(file_path))
            
            logger.info(f"图像已保存: {file_path}")
            
            # 自动打开图片
            if auto_open and i == 1:  # 只打开第一张图片
                self._open_image(file_path)
        
        return saved_paths
    
    def _open_image(self, file_path: Path):
        """
        自动打开图片文件
        
        Args:
            file_path: 图片文件路径
        """
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(str(file_path))
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(file_path)])
            elif system == "Linux":
                subprocess.run(["xdg-open", str(file_path)])
            else:
                logger.warning(f"不支持的操作系统: {system}，无法自动打开图片")
                return
            
            logger.info(f"已自动打开图片: {file_path}")
        except Exception as e:
            logger.warning(f"无法自动打开图片 {file_path}: {e}")
    
    def _save_generation_report(self, result: Dict[str, Any], output_dir: str):
        """
        保存生成报告
        
        Args:
            result: 生成结果
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"environmental_report_{timestamp}.json"
        
        # 准备报告数据（移除不能序列化的对象）
        report_data = result.copy()
        if "image" in report_data:
            del report_data["image"]  # PIL Image 对象不能序列化
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"生成报告已保存: {report_file}")
    
    def get_supported_data_types(self) -> Dict[str, Dict[str, Any]]:
        """
        获取支持的环境数据类型
        
        Returns:
            环境数据类型字典
        """
        return self.environmental_data_types
    
    def test_connection(self) -> Dict[str, Any]:
        """
        测试 DashScope 连接
        
        Returns:
            测试结果
        """
        try:
            # 测试聊天模型
            chat_response = Generation.call(
                model=self.chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                result_format='message'
            )
            
            chat_success = chat_response.status_code == HTTPStatus.OK
            
            # 测试图像生成模型
            image_response = ImageSynthesis.call(
                model=self.image_model,
                prompt="test image",
                size='512*512'
            )
            
            image_success = image_response.status_code == HTTPStatus.OK
            
            return {
                "success": chat_success and image_success,
                "chat_model_status": "OK" if chat_success else "Failed",
                "image_model_status": "OK" if image_success else "Failed",
                "chat_model": self.chat_model,
                "image_model": self.image_model
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

def main():
    """
    主函数 - 演示基本用法
    """
    print("🌍 基于 DashScope 的环境保护警示图像生成器")
    print("=" * 60)
    
    try:
        # 初始化生成器
        generator = DashScopeEnvironmentalGenerator()
        
        # 测试连接
        print("🔗 测试 DashScope 连接...")
        test_result = generator.test_connection()
        if not test_result["success"]:
            print(f"❌ 连接测试失败: {test_result.get('error', '未知错误')}")
            return
        
        print("✅ DashScope 连接成功！")
        print(f"🤖 聊天模型状态: {test_result['chat_model_status']}")
        print(f"🎨 图像模型状态: {test_result['image_model_status']}")
        
        # 显示支持的数据类型
        print("\n📊 支持的环境数据类型:")
        for data_type, config in generator.get_supported_data_types().items():
            print(f"  - {config['name']} ({config['unit']})")
        
        # 示例数据
        example_data = {
            "carbon_emission": 1500,  # 吨CO2当量
            "air_quality_index": 180,  # AQI
            "water_pollution_index": 85  # WPI
        }
        
        print(f"\n🧪 使用示例数据生成环境警示图像:")
        for key, value in example_data.items():
            data_config = generator.get_supported_data_types()[key]
            print(f"  - {data_config['name']}: {value} {data_config['unit']}")
        
        # 生成图像
        result = generator.generate_environmental_warning_image(
            environmental_data=example_data,
            user_description="工业区域的严重污染情况，需要引起公众关注",
            target_audience="educators"
        )
        
        if result["success"]:
            print(f"\n✅ 图像生成成功！")
            print(f"📁 保存位置: {result['saved_paths'][0]}")
            print(f"⏱️  生成时间: {result['generation_time']:.2f} 秒")
            print(f"🎯 总体严重程度: {result['analysis']['overall_severity']}")
            print(f"⚠️  关键问题: {len(result['analysis']['critical_factors'])} 个")
        else:
            print(f"❌ 图像生成失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    
    print("\n感谢使用 DashScope 环境保护警示图像生成器！")

if __name__ == "__main__":
    main()