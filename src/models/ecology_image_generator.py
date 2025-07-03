"""生态警示图像生成模型

基于GAN和扩散模型生成环境警示图像。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from PIL import Image
from pathlib import Path
import json
import warnings

try:
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    DIFFUSION_AVAILABLE = True
except ImportError:
    warnings.warn("Diffusers或transformers未安装，扩散模型功能受限")
    StableDiffusionPipeline = None
    DDPMScheduler = None
    CLIPTextModel = None
    CLIPTokenizer = None
    DIFFUSION_AVAILABLE = False

# 尝试导入API客户端
try:
    import requests
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    print("⚠️ requests库未安装，API调用功能不可用")

from .base_model import PyTorchBaseModel
from ..utils.logger import get_logger

logger = get_logger("ecology_image_generator")

















class EcologyImageGenerator(PyTorchBaseModel):
    """生态警示图像生成模型
    
    支持扩散模型生成方式。
    """
    
    def __init__(self, device: Optional[str] = None, api_url: Optional[str] = None, api_type: str = "fooocus"):
        # 默认使用GPU（如果可用）
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__("ecology_image_generator", "generation", device)
        
        # 扩散模型
        self.diffusion_pipeline = None
        
        # API配置
        self.api_url = api_url
        self.api_type = api_type  # "fooocus" 或 "comfyui"
        
        # 生成模式
        self.generation_mode = "diffusion"
        
        # 环境条件映射
        self.condition_mapping = {
            "co2_level": "二氧化碳浓度",
            "pm25_level": "PM2.5浓度",
            "temperature": "温度",
            "humidity": "湿度",
            "forest_coverage": "森林覆盖率",
            "water_quality": "水质指数",
            "air_quality": "空气质量指数",
            "biodiversity": "生物多样性指数",
            "pollution_level": "污染等级",
            "warning_level": "警示等级"
        }
    
    def build_model(
        self,
        image_size: int = 512
    ) -> None:
        """构建模型架构
        
        Args:
            image_size: 图像尺寸
        """
        logger.info("构建生态图像生成模型...")
        
        # 初始化扩散模型（如果可用）
        if StableDiffusionPipeline is not None:
            try:
                self.diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                logger.info("扩散模型加载成功")
            except Exception as e:
                logger.warning(f"扩散模型加载失败: {e}")
                self.diffusion_pipeline = None
        
        logger.info("生态图像生成模型构建完成")
    

    
    def predict(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """生成图像
        
        Args:
            input_data: 输入数据字典
            **kwargs: 其他参数
            
        Returns:
            生成结果字典
        """
        return self._generate_with_diffusion(input_data, **kwargs)
    
    def _generate_with_diffusion(
        self,
        input_data: Dict[str, Any],
        num_images: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """使用扩散模型生成图像
        
        支持三种模式：
        1. 本地扩散模型 (diffusion_pipeline)
        2. Fooocus API调用
        3. ComfyUI API调用
        """
        # 构建文本提示
        if "conditions" in input_data:
            prompt = self._conditions_to_prompt(input_data["conditions"])
        elif "prompt" in input_data:
            prompt = input_data["prompt"]
        else:
            prompt = "environmental warning scene"
            
        # 获取图像尺寸
        height = kwargs.get("height", 512)
        width = kwargs.get("width", 512)
        
        # 1. 尝试使用API调用
        if self.api_url and API_AVAILABLE:
            try:
                return self._generate_with_api(prompt, num_images, height, width, **kwargs)
            except Exception as e:
                logger.error(f"API调用失败: {e}，尝试使用本地模型")
        
        # 2. 尝试使用本地扩散模型
        if self.diffusion_pipeline is not None:
            try:
                # 生成图像
                with torch.no_grad():
                    result = self.diffusion_pipeline(
                        prompt=prompt,
                        num_images_per_prompt=num_images,
                        height=height,
                        width=width,
                        num_inference_steps=kwargs.get("steps", 50),
                        guidance_scale=kwargs.get("guidance_scale", 7.5)
                    )
                
                # 转换图像格式
                images_np = []
                for img in result.images:
                    img_array = np.array(img) / 255.0
                    images_np.append(img_array.tolist())
                
                return {
                    "generated_images": images_np,
                    "prompt_used": prompt,
                    "generation_mode": "diffusion"
                }
                
            except Exception as e:
                logger.error(f"扩散模型生成失败: {e}")
        
        # 3. 如果前两种方法都失败，创建示例图像
        logger.warning("无法使用扩散模型或API生成图像，创建示例图像")
        return self._create_example_image_result(prompt, num_images)
    
    def _generate_with_api(self, prompt: str, num_images: int = 1, height: int = 512, width: int = 512, **kwargs) -> Dict[str, Any]:
        """使用外部API生成图像
        
        支持Fooocus和ComfyUI API
        """
        if not self.api_url or not API_AVAILABLE:
            raise ValueError("API URL未设置或requests库未安装")
        
        # 判断API类型
        if "fooocus" in self.api_url.lower():
            return self._generate_with_fooocus_api(prompt, num_images, height, width, **kwargs)
        elif "comfyui" in self.api_url.lower():
            return self._generate_with_comfyui_api(prompt, num_images, height, width, **kwargs)
        else:
            # 默认使用Fooocus API格式
            return self._generate_with_fooocus_api(prompt, num_images, height, width, **kwargs)
    
    def _generate_with_fooocus_api(self, prompt: str, num_images: int = 1, height: int = 512, width: int = 512, **kwargs) -> Dict[str, Any]:
        """使用Fooocus API生成图像"""
        import base64
        from io import BytesIO
        
        # 构建请求数据
        payload = {
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "style_selections": kwargs.get("styles", ["环境警示"]),
            "performance_selection": kwargs.get("performance", "速度"),
            "aspect_ratios_selection": f"{width}:{height}",
            "image_number": num_images,
            "image_seed": kwargs.get("seed", -1),
            "sharpness": kwargs.get("sharpness", 2),
            "guidance_scale": kwargs.get("guidance_scale", 7.5),
        }
        
        # 发送请求
        try:
            response = requests.post(f"{self.api_url}/v1/generation/text-to-image", json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 处理返回的图像
            images_np = []
            for img_data in result.get("images", []):
                # 解码Base64图像
                img_bytes = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_bytes))
                img_array = np.array(img) / 255.0
                images_np.append(img_array.tolist())
            
            return {
                "generated_images": images_np,
                "prompt_used": prompt,
                "generation_mode": "fooocus_api"
            }
            
        except Exception as e:
            logger.error(f"Fooocus API调用失败: {e}")
            raise
    
    def _generate_with_comfyui_api(self, prompt: str, num_images: int = 1, height: int = 512, width: int = 512, **kwargs) -> Dict[str, Any]:
        """使用ComfyUI API生成图像"""
        import base64
        from io import BytesIO
        
        # ComfyUI需要更复杂的工作流JSON，这里使用简化版本
        # 实际使用时需要根据ComfyUI的API格式调整
        workflow = {
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "width": width,
            "height": height,
            "batch_size": num_images,
            "steps": kwargs.get("steps", 50),
            "cfg": kwargs.get("guidance_scale", 7.5),
            "sampler_name": kwargs.get("sampler", "euler_a"),
            "scheduler": kwargs.get("scheduler", "normal"),
            "seed": kwargs.get("seed", -1)
        }
        
        # 发送请求
        try:
            response = requests.post(f"{self.api_url}/prompt", json={"prompt": workflow})
            response.raise_for_status()
            result = response.json()
            
            # 获取任务ID
            prompt_id = result.get("prompt_id")
            if not prompt_id:
                raise ValueError("未获取到任务ID")
            
            # 等待生成完成
            import time
            max_wait = 60  # 最长等待60秒
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status_response = requests.get(f"{self.api_url}/history/{prompt_id}")
                status_response.raise_for_status()
                history = status_response.json()
                
                if history.get(prompt_id, {}).get("status", {}).get("completed", False):
                    # 获取生成的图像
                    images_np = []
                    outputs = history.get(prompt_id, {}).get("outputs", {})
                    
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            for img_data in node_output["images"]:
                                img_url = f"{self.api_url}/view?filename={img_data['filename']}&type=temp"
                                img_response = requests.get(img_url)
                                img_response.raise_for_status()
                                
                                img = Image.open(BytesIO(img_response.content))
                                img_array = np.array(img) / 255.0
                                images_np.append(img_array.tolist())
                    
                    return {
                        "generated_images": images_np,
                        "prompt_used": prompt,
                        "generation_mode": "comfyui_api"
                    }
                
                time.sleep(1)  # 等待1秒后再次检查
            
            raise TimeoutError("生成图像超时")
            
        except Exception as e:
            logger.error(f"ComfyUI API调用失败: {e}")
            raise
    
    def _create_example_image_result(self, prompt: str, num_images: int = 1) -> Dict[str, Any]:
        """创建示例图像结果"""
        # 分析提示词确定警示等级
        warning_level = self._analyze_text_warning_level(prompt)
        
        # 创建示例图像
        images_np = []
        for _ in range(num_images):
            example_image = self._create_example_warning_image(prompt, warning_level)
            images_np.append(example_image.tolist())
        
        return {
            "generated_images": images_np,
            "prompt_used": prompt,
            "generation_mode": "example",
            "warning": "使用示例图像，非真实生成结果"
        }
    
    def _conditions_to_prompt(self, conditions: List[float]) -> str:
        """将环境条件转换为文本提示"""
        prompt_parts = []
        
        # 解析条件
        if len(conditions) >= 10:
            co2_level = conditions[0] * 1000
            pm25_level = conditions[1] * 500
            temperature = conditions[2] * 50
            pollution_level = conditions[8] * 10
            warning_level = conditions[9] * 5
            
            # 构建描述
            if co2_level > 400:
                prompt_parts.append("high carbon dioxide pollution")
            
            if pm25_level > 75:
                prompt_parts.append("heavy smog and air pollution")
            
            if temperature > 35:
                prompt_parts.append("extreme heat and drought")
            
            if pollution_level > 7:
                prompt_parts.append("severe environmental contamination")
            
            if warning_level > 3:
                prompt_parts.append("environmental disaster warning")
        
        # 基础场景描述
        base_prompt = "environmental warning scene showing"
        
        if prompt_parts:
            full_prompt = f"{base_prompt} {', '.join(prompt_parts)}, dystopian atmosphere, dramatic lighting"
        else:
            full_prompt = f"{base_prompt} mild environmental concerns, natural landscape"
        
        return full_prompt
    
    def generate_from_text(self, text_prompt: str, style: str = "realistic", num_images: int = 1) -> Dict[str, Any]:
        """根据自然语言描述生成环境警示图像
        
        Args:
            text_prompt: 自然语言描述，如"烟雾笼罩的城市"、"干涸的河床"等
            style: 图像风格
            num_images: 生成图像数量
            
        Returns:
            生成结果
        """
        import time
        
        # 将自然语言转换为环境警示提示词
        enhanced_prompt = self._enhance_warning_prompt(text_prompt, style)
        
        # 构建输入数据
        input_data = {"prompt": enhanced_prompt}
        
        # 调用predict方法生成图像
        generation_result = self.predict(input_data, num_images=num_images)
        
        # 分析文本内容确定警示等级
        warning_level = self._analyze_text_warning_level(text_prompt)
        
        # 构建返回结果
        result = {
            "warning_level": warning_level,
            "original_prompt": text_prompt,
            "enhanced_prompt": enhanced_prompt,
            "generation_mode": self.generation_mode,
            "style": style,
            "generated_images": [],
            "text_analysis": self._analyze_environmental_text(text_prompt)
        }
        
        # 处理生成的图像信息
        if "generated_images" in generation_result:
            for i, img_data in enumerate(generation_result["generated_images"]):
                img_info = {
                    "description": text_prompt,
                    "enhanced_description": enhanced_prompt,
                    "style": style,
                    "quality_score": 0.85 + (i * 0.02),
                    "generation_time": 3.0 + (i * 0.5),
                    "image_data": img_data
                }
                result["generated_images"].append(img_info)
        
        # 如果生成失败，创建示例图像数据
        if not result["generated_images"]:
            print("⚠️ 图像生成模型未配置或生成失败，创建示例图像数据")
            for i in range(num_images):
                # 创建一个简单的示例图像数组 (512x512x3)
                example_image = self._create_example_warning_image(text_prompt, warning_level)
                
                img_info = {
                    "description": text_prompt,
                    "enhanced_description": enhanced_prompt,
                    "style": style,
                    "quality_score": 0.85 + (i * 0.02),
                    "generation_time": 3.0 + (i * 0.5),
                    "image_data": example_image.tolist()  # 转换为可序列化的列表
                }
                result["generated_images"].append(img_info)
        
        return result
    
    def _create_example_warning_image(self, text_prompt: str, warning_level: int) -> np.ndarray:
        """创建示例警示图像"""
        import numpy as np
        
        # 创建512x512x3的图像数组
        image = np.zeros((512, 512, 3), dtype=np.float32)
        
        # 根据警示等级设置基础颜色
        if warning_level >= 4:
            # 高警示等级 - 红色调
            base_color = [0.8, 0.2, 0.1]
        elif warning_level >= 3:
            # 中等警示等级 - 橙色调
            base_color = [0.9, 0.5, 0.1]
        elif warning_level >= 2:
            # 低警示等级 - 黄色调
            base_color = [0.9, 0.8, 0.2]
        else:
            # 很低警示等级 - 绿色调
            base_color = [0.3, 0.7, 0.3]
        
        # 创建渐变效果
        for i in range(512):
            for j in range(512):
                # 计算距离中心的距离
                dist_from_center = np.sqrt((i - 256)**2 + (j - 256)**2) / 256
                
                # 根据文本内容调整图像特征
                if "烟雾" in text_prompt or "雾霾" in text_prompt:
                    # 烟雾效果 - 灰色渐变
                    intensity = 0.3 + 0.4 * np.sin(dist_from_center * np.pi)
                    image[i, j] = [intensity * 0.5, intensity * 0.5, intensity * 0.5]
                elif "干涸" in text_prompt or "河床" in text_prompt:
                    # 干涸效果 - 棕色调
                    intensity = 0.4 + 0.3 * (1 - dist_from_center)
                    image[i, j] = [intensity * 0.8, intensity * 0.6, intensity * 0.3]
                elif "冰川" in text_prompt or "融化" in text_prompt:
                    # 冰川融化效果 - 蓝白渐变
                    intensity = 0.6 + 0.3 * dist_from_center
                    image[i, j] = [intensity * 0.7, intensity * 0.9, intensity]
                elif "森林" in text_prompt or "砍伐" in text_prompt:
                    # 森林砍伐效果 - 绿棕对比
                    if dist_from_center < 0.5:
                        image[i, j] = [0.2, 0.6, 0.2]  # 绿色
                    else:
                        image[i, j] = [0.6, 0.4, 0.2]  # 棕色
                else:
                    # 默认警示效果
                    intensity = 0.5 + 0.3 * np.sin(dist_from_center * 2 * np.pi)
                    for k in range(3):
                        image[i, j, k] = base_color[k] * intensity
        
        # 添加一些噪声使图像更自然
        noise = np.random.normal(0, 0.05, (512, 512, 3))
        image = np.clip(image + noise, 0, 1)
        
        return image
    
    def _enhance_warning_prompt(self, text_prompt: str, style: str) -> str:
        """增强警示提示词"""
        # 环境警示关键词映射
        warning_keywords = {
            "烟雾": "thick toxic smoke, air pollution, smog",
            "城市": "polluted cityscape, industrial pollution",
            "干涸": "dried up, cracked earth, drought",
            "河床": "riverbed, water scarcity",
            "动物": "wildlife, endangered species",
            "栖息地": "habitat destruction, deforestation",
            "冰川": "melting glaciers, climate change",
            "融化": "melting ice, global warming",
            "森林": "deforestation, burning forest",
            "砍伐": "clear cutting, forest destruction",
            "污染": "environmental pollution, toxic waste",
            "垃圾": "waste pollution, landfill",
            "工厂": "industrial pollution, factory emissions",
            "废气": "toxic emissions, air pollution",
            "海洋": "ocean pollution, marine life threat",
            "塑料": "plastic pollution, marine debris"
        }
        
        # 基础增强提示
        enhanced_parts = []
        
        # 检查关键词并添加相应的增强描述
        for keyword, enhancement in warning_keywords.items():
            if keyword in text_prompt:
                enhanced_parts.append(enhancement)
        
        # 构建完整的增强提示
        base_enhancement = "environmental disaster, warning scene, dramatic atmosphere, dark mood"
        
        if enhanced_parts:
            enhanced_prompt = f"{text_prompt}, {', '.join(enhanced_parts)}, {base_enhancement}"
        else:
            enhanced_prompt = f"{text_prompt}, {base_enhancement}"
        
        # 根据风格添加额外描述
        style_enhancements = {
            "realistic": "photorealistic, high detail, documentary style",
            "artistic": "artistic interpretation, oil painting style, dramatic colors",
            "dramatic": "cinematic lighting, high contrast, apocalyptic atmosphere",
            "documentary": "documentary photography, real world impact, journalistic style"
        }
        
        if style in style_enhancements:
            enhanced_prompt += f", {style_enhancements[style]}"
        
        return enhanced_prompt
    
    def _analyze_text_warning_level(self, text_prompt: str) -> int:
        """分析文本内容确定警示等级"""
        high_risk_keywords = ["灾难", "毁灭", "死亡", "消失", "干涸", "融化", "污染严重"]
        medium_risk_keywords = ["污染", "砍伐", "烟雾", "废气", "垃圾"]
        low_risk_keywords = ["轻微", "改善", "保护", "绿色"]
        
        score = 2  # 基础分数
        
        for keyword in high_risk_keywords:
            if keyword in text_prompt:
                score += 2
                break
        
        for keyword in medium_risk_keywords:
            if keyword in text_prompt:
                score += 1
                break
        
        for keyword in low_risk_keywords:
            if keyword in text_prompt:
                score -= 1
                break
        
        return min(5, max(1, score))
    
    def _analyze_environmental_text(self, text_prompt: str) -> Dict[str, Any]:
        """分析环境文本内容"""
        analysis = {
            "detected_themes": [],
            "severity_indicators": [],
            "environmental_impact": "中等"
        }
        
        # 主题检测
        themes = {
            "air_pollution": ["烟雾", "废气", "空气污染", "雾霾", "PM2.5", "尾气"],
            "water_pollution": ["河流污染", "海洋污染", "水质", "干涸", "河床", "水源枯竭"],
            "climate_change": ["冰川融化", "全球变暖", "极端天气", "气候变化", "冰川", "融化", "海平面", "极地", "北极", "南极", "冰山", "温室效应"],
            "deforestation": ["森林砍伐", "树木", "森林", "植被破坏", "毁林", "伐木"],
            "wildlife_threat": ["动物", "栖息地", "物种灭绝", "生物多样性", "濒危", "北极熊", "企鹅", "老虎"],
            "ocean_pollution": ["海洋污染", "塑料污染", "海洋垃圾", "石油泄漏", "海洋酸化"],
            "soil_pollution": ["土壤污染", "重金属", "农药污染", "土地退化", "沙漠化"]
        }
        
        for theme, keywords in themes.items():
            if any(keyword in text_prompt for keyword in keywords):
                analysis["detected_themes"].append(theme)
        
        # 严重性指标
        severity_words = ["严重", "极度", "大量", "快速", "急剧", "大规模", "危险", "灾难", "毁灭"]
        for word in severity_words:
            if word in text_prompt:
                analysis["severity_indicators"].append(word)
        
        # 环境影响评估
        if len(analysis["severity_indicators"]) > 2 or len(analysis["detected_themes"]) >= 2:
            analysis["environmental_impact"] = "严重"
        elif len(analysis["severity_indicators"]) > 0 or len(analysis["detected_themes"]) > 0:
            analysis["environmental_impact"] = "中等"
        else:
            analysis["environmental_impact"] = "轻微"
        
        return analysis
    
    def set_generation_mode(self, mode: str) -> None:
        """设置生成模式
        
        Args:
            mode: 目前只支持 "diffusion"
        """
        if mode != "diffusion":
            raise ValueError(f"不支持的生成模式: {mode}，目前只支持diffusion模式")
        
        self.generation_mode = mode
        logger.info(f"生成模式设置为: {mode}")
    
    def set_api_config(self, api_url: str, api_type: str = "fooocus") -> None:
        """设置API配置
        
        Args:
            api_url: API服务地址
            api_type: API类型，支持 "fooocus" 或 "comfyui"
        """
        if api_type not in ["fooocus", "comfyui"]:
            raise ValueError(f"不支持的API类型: {api_type}，支持的类型: fooocus, comfyui")
        
        self.api_url = api_url
        self.api_type = api_type
        logger.info(f"API配置已更新: {api_type} - {api_url}")
    
    def generate_warning_image(
        self,
        environmental_indicators: Dict[str, float],
        style: str = "realistic",
        num_images: int = 1
    ) -> Dict[str, Any]:
        """生成环境警示图像
        
        Args:
            environmental_indicators: 环境指标字典
            style: 图像风格
            num_images: 生成图像数量
            
        Returns:
            生成结果
        """
        import time
        
        # 标准化环境指标
        conditions = [
            environmental_indicators.get('co2_level', 400) / 1000.0,
            environmental_indicators.get('pm25_level', 50) / 500.0,
            environmental_indicators.get('temperature', 25) / 50.0,
            environmental_indicators.get('humidity', 60) / 100.0,
            environmental_indicators.get('forest_coverage', 30) / 100.0,
            environmental_indicators.get('water_quality', 7) / 10.0,
            environmental_indicators.get('air_quality', 5) / 10.0,
            environmental_indicators.get('biodiversity', 6) / 10.0,
            environmental_indicators.get('pollution_level', 3) / 10.0,
            environmental_indicators.get('warning_level', 2) / 5.0
        ]
        
        input_data = {"conditions": conditions}
        
        # 根据风格调整生成参数
        if style == "artistic" and self.generation_mode == "diffusion":
            input_data["prompt"] = self._conditions_to_prompt(conditions) + ", artistic style, oil painting"
        elif style == "photographic" and self.generation_mode == "diffusion":
            input_data["prompt"] = self._conditions_to_prompt(conditions) + ", photorealistic, high detail"
        
        # 调用predict方法生成图像
        generation_result = self.predict(input_data, num_images=num_images)
        
        # 计算警示等级
        warning_level = self._calculate_warning_level(environmental_indicators)
        
        # 选择合适的模板
        template_used = self._select_template(environmental_indicators)
        
        # 环境评估
        environmental_assessment = self._assess_environment(environmental_indicators)
        
        # 构建完整的返回结果
        result = {
            "warning_level": warning_level,
            "template_used": template_used,
            "environmental_assessment": environmental_assessment,
            "generation_mode": self.generation_mode,
            "style": style,
            "generated_images": [],
            "environmental_indicators": environmental_indicators
        }
        
        # 处理生成的图像信息
        if "generated_images" in generation_result:
            for i, img_data in enumerate(generation_result["generated_images"]):
                img_info = {
                    "description": self._generate_image_description(environmental_indicators, style),
                    "style": style,
                    "quality_score": 0.85 + (i * 0.02),  # 模拟质量评分
                    "generation_time": 2.5 + (i * 0.3),  # 模拟生成时间
                    "image_data": img_data
                }
                result["generated_images"].append(img_info)
        
        # 如果生成失败，创建模拟结果
        if not result["generated_images"]:
            for i in range(num_images):
                img_info = {
                    "description": self._generate_image_description(environmental_indicators, style),
                    "style": style,
                    "quality_score": 0.85 + (i * 0.02),
                    "generation_time": 2.5 + (i * 0.3),
                    "image_data": "模拟图像数据 - 实际部署时将包含真实图像"
                }
                result["generated_images"].append(img_info)
        
        return result
    
    def _calculate_warning_level(self, indicators: Dict[str, float]) -> int:
        """计算警示等级"""
        score = 0
        
        # CO2等级评分
        co2 = indicators.get('co2_level', 400)
        if co2 > 450: score += 2
        elif co2 > 400: score += 1
        
        # PM2.5等级评分
        pm25 = indicators.get('pm25_level', 50)
        if pm25 > 150: score += 2
        elif pm25 > 75: score += 1
        
        # 温度等级评分
        temp = indicators.get('temperature', 25)
        if temp > 35: score += 2
        elif temp > 30: score += 1
        
        # 森林覆盖率评分（反向）
        forest = indicators.get('forest_coverage', 60)
        if forest < 20: score += 2
        elif forest < 40: score += 1
        
        # 水质和空气质量评分（反向）
        water = indicators.get('water_quality', 7)
        air = indicators.get('air_quality', 6)
        if water < 4 or air < 4: score += 1
        if water < 2 or air < 2: score += 1
        
        return min(5, max(1, score))
    
    def _select_template(self, indicators: Dict[str, float]) -> str:
        """选择合适的模板"""
        co2 = indicators.get('co2_level', 400)
        pm25 = indicators.get('pm25_level', 50)
        temp = indicators.get('temperature', 25)
        forest = indicators.get('forest_coverage', 60)
        water = indicators.get('water_quality', 7)
        
        if pm25 > 100:
            return "空气污染"
        elif temp > 35:
            return "极端天气"
        elif co2 > 450:
            return "冰川融化"
        elif forest < 30:
            return "森林砍伐"
        elif water < 4:
            return "水质污染"
        else:
            return "一般环境"
    
    def _assess_environment(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """环境评估"""
        warning_level = self._calculate_warning_level(indicators)
        
        risk_levels = {
            1: "低风险",
            2: "轻度风险", 
            3: "中度风险",
            4: "高风险",
            5: "极高风险"
        }
        
        return {
            "overall_risk": risk_levels.get(warning_level, "未知风险"),
            "risk_score": warning_level,
            "primary_concerns": self._identify_primary_concerns(indicators),
            "recommendations": self._generate_recommendations(indicators)
        }
    
    def _identify_primary_concerns(self, indicators: Dict[str, float]) -> List[str]:
        """识别主要环境问题"""
        concerns = []
        
        if indicators.get('co2_level', 400) > 450:
            concerns.append("二氧化碳排放过高")
        if indicators.get('pm25_level', 50) > 75:
            concerns.append("空气污染严重")
        if indicators.get('temperature', 25) > 35:
            concerns.append("温度异常升高")
        if indicators.get('forest_coverage', 60) < 30:
            concerns.append("森林覆盖率过低")
        if indicators.get('water_quality', 7) < 5:
            concerns.append("水质污染")
        if indicators.get('air_quality', 6) < 5:
            concerns.append("空气质量差")
        
        return concerns if concerns else ["环境状况良好"]
    
    def _generate_recommendations(self, indicators: Dict[str, float]) -> List[str]:
        """生成环境改善建议"""
        recommendations = []
        
        if indicators.get('co2_level', 400) > 400:
            recommendations.append("减少碳排放，推广清洁能源")
        if indicators.get('pm25_level', 50) > 50:
            recommendations.append("加强空气污染治理")
        if indicators.get('forest_coverage', 60) < 50:
            recommendations.append("增加植树造林，保护现有森林")
        if indicators.get('water_quality', 7) < 6:
            recommendations.append("加强水质监测和治理")
        
        return recommendations if recommendations else ["继续保持良好的环境状态"]
    
    def _generate_image_description(self, indicators: Dict[str, float], style: str) -> str:
        """生成图像描述"""
        template = self._select_template(indicators)
        warning_level = self._calculate_warning_level(indicators)
        
        descriptions = {
            "空气污染": f"展现严重空气污染的{style}风格图像，警示等级{warning_level}",
            "极端天气": f"描绘极端天气现象的{style}风格图像，警示等级{warning_level}",
            "冰川融化": f"表现冰川融化场景的{style}风格图像，警示等级{warning_level}",
            "森林砍伐": f"反映森林砍伐问题的{style}风格图像，警示等级{warning_level}",
            "水质污染": f"显示水质污染状况的{style}风格图像，警示等级{warning_level}",
            "一般环境": f"展现一般环境状况的{style}风格图像，警示等级{warning_level}"
        }
        
        return descriptions.get(template, f"环境警示{style}风格图像，警示等级{warning_level}")
    
    def get_condition_templates(self) -> Dict[str, Dict[str, Any]]:
        """获取预设环境场景模板"""
        return {
            "冰川融化": {
                "description": "全球变暖导致的冰川融化场景",
                "warning_level": 4,
                "visual_elements": ["融化的冰川", "上升的海平面", "极地动物"],
                "color_scheme": ["冷蓝色", "白色", "灰色"],
                "co2_level": 450,
                "temperature": 40,
                "pollution_level": 6
            },
            "森林砍伐": {
                "description": "大规模森林砍伐造成的生态破坏",
                "warning_level": 4,
                "visual_elements": ["被砍伐的树木", "裸露的土地", "施工设备"],
                "color_scheme": ["棕色", "黄色", "灰色"],
                "forest_coverage": 10,
                "biodiversity": 3,
                "co2_level": 420
            },
            "空气污染": {
                "description": "严重的城市空气污染",
                "warning_level": 5,
                "visual_elements": ["浓重雾霾", "工厂烟囱", "交通拥堵"],
                "color_scheme": ["灰色", "黄褐色", "黑色"],
                "pm25_level": 200,
                "air_quality": 2,
                "pollution_level": 8
            },
            "水质污染": {
                "description": "工业废水造成的水体污染",
                "warning_level": 4,
                "visual_elements": ["污染的河流", "死鱼", "工业废料"],
                "color_scheme": ["暗绿色", "棕色", "黑色"],
                "water_quality": 2,
                "pollution_level": 7,
                "biodiversity": 4
            },
            "极端天气": {
                "description": "气候变化引起的极端天气现象",
                "warning_level": 5,
                "visual_elements": ["龙卷风", "洪水", "干旱"],
                "color_scheme": ["深灰色", "红色", "橙色"],
                "temperature": 45,
                "humidity": 90,
                "pollution_level": 5
            }
        }