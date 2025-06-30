"""生态警示图像生成模型

基于GAN和扩散模型生成环境警示图像。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
import json
import warnings

try:
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError:
    warnings.warn("Diffusers或transformers未安装，扩散模型功能受限")
    StableDiffusionPipeline = None
    DDPMScheduler = None
    CLIPTextModel = None
    CLIPTokenizer = None

from .base_model import PyTorchBaseModel
from ..utils.logger import get_logger

logger = get_logger("ecology_image_generator")


class Generator(nn.Module):
    """StyleGAN3风格的生成器"""
    
    def __init__(
        self,
        latent_dim: int = 512,
        condition_dim: int = 10,
        image_size: int = 256,
        channels: int = 3
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.image_size = image_size
        
        # 条件嵌入层
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # 计算初始特征图大小
        self.init_size = image_size // 16  # 16 = 2^4 (4个上采样层)
        
        # 主干网络
        self.fc = nn.Linear(latent_dim * 2, 512 * self.init_size * self.init_size)
        
        # 上采样块
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(512, 256),  # 16x16 -> 32x32
            self._make_conv_block(256, 128),  # 32x32 -> 64x64
            self._make_conv_block(128, 64),   # 64x64 -> 128x128
            self._make_conv_block(64, 32),    # 128x128 -> 256x256
        ])
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(32, channels, 3, padding=1),
            nn.Tanh()
        )
        
        # 自注意力层
        self.attention = SelfAttention(128)
    
    def _make_conv_block(self, in_channels: int, out_channels: int):
        """创建卷积块"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, noise: torch.Tensor, conditions: torch.Tensor):
        # 条件嵌入
        condition_embed = self.condition_embedding(conditions)
        
        # 连接噪声和条件
        combined = torch.cat([noise, condition_embed], dim=1)
        
        # 全连接层
        x = self.fc(combined)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)
        
        # 上采样
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
            
            # 在中间层添加自注意力
            if i == 1:  # 64x64分辨率时
                x = self.attention(x)
        
        # 输出
        output = self.output_conv(x)
        
        return output


class Discriminator(nn.Module):
    """判别器"""
    
    def __init__(
        self,
        image_size: int = 256,
        channels: int = 3,
        condition_dim: int = 10
    ):
        super().__init__()
        
        self.condition_dim = condition_dim
        
        # 条件嵌入
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, image_size * image_size),
            nn.ReLU()
        )
        
        # 卷积层
        self.conv_blocks = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(channels + 1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(1024, 1, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, images: torch.Tensor, conditions: torch.Tensor):
        batch_size = images.size(0)
        
        # 条件嵌入并reshape为图像格式
        condition_embed = self.condition_embedding(conditions)
        condition_embed = condition_embed.view(batch_size, 1, images.size(2), images.size(3))
        
        # 连接图像和条件
        combined = torch.cat([images, condition_embed], dim=1)
        
        # 卷积处理
        features = self.conv_blocks(combined)
        
        # 输出
        output = self.output(features)
        
        return output.view(batch_size, -1)


class SelfAttention(nn.Module):
    """自注意力模块"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.in_channels = in_channels
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 计算query, key, value
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        # 注意力权重
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


class PerceptualLoss(nn.Module):
    """感知损失（基于VGG16）"""
    
    def __init__(self):
        super().__init__()
        
        # 加载预训练的VGG16
        vgg = models.vgg16(pretrained=True).features
        
        # 提取特定层的特征
        self.feature_layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[:9],   # relu2_2
            vgg[:16],  # relu3_3
            vgg[:23],  # relu4_3
        ])
        
        # 冻结参数
        for layer in self.feature_layers:
            for param in layer.parameters():
                param.requires_grad = False
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss = 0.0
        
        for layer in self.feature_layers:
            pred_features = layer(pred)
            target_features = layer(target)
            loss += self.mse_loss(pred_features, target_features)
        
        return loss


class EcologyImageDataset(Dataset):
    """生态图像数据集"""
    
    def __init__(
        self,
        image_dir: str,
        annotations_file: str,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        # 加载标注
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # 加载图像
        image_path = self.image_dir / item['image_file']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 环境条件（标准化到0-1范围）
        conditions = torch.FloatTensor([
            item.get('co2_level', 0) / 1000.0,      # CO2浓度
            item.get('pm25_level', 0) / 500.0,      # PM2.5浓度
            item.get('temperature', 0) / 50.0,      # 温度
            item.get('humidity', 0) / 100.0,        # 湿度
            item.get('forest_coverage', 0) / 100.0, # 森林覆盖率
            item.get('water_quality', 0) / 10.0,    # 水质指数
            item.get('air_quality', 0) / 10.0,      # 空气质量指数
            item.get('biodiversity', 0) / 10.0,     # 生物多样性指数
            item.get('pollution_level', 0) / 10.0,  # 污染等级
            item.get('warning_level', 0) / 5.0      # 警示等级
        ])
        
        return image, conditions


class EcologyImageGenerator(PyTorchBaseModel):
    """生态警示图像生成模型
    
    支持GAN和扩散模型两种生成方式。
    """
    
    def __init__(self, device: Optional[str] = None):
        super().__init__("ecology_image_generator", "generation", device)
        
        # GAN模型
        self.generator = None
        self.discriminator = None
        
        # 扩散模型
        self.diffusion_pipeline = None
        
        # 损失函数
        self.adversarial_loss = nn.BCELoss()
        self.perceptual_loss = PerceptualLoss()
        
        # 生成模式
        self.generation_mode = "gan"  # "gan" 或 "diffusion"
        
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
        latent_dim: int = 512,
        condition_dim: int = 10,
        image_size: int = 256,
        channels: int = 3
    ) -> None:
        """构建模型架构
        
        Args:
            latent_dim: 潜在空间维度
            condition_dim: 条件维度
            image_size: 图像尺寸
            channels: 图像通道数
        """
        logger.info("构建生态图像生成模型...")
        
        # 构建GAN模型
        self.generator = Generator(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            image_size=image_size,
            channels=channels
        ).to(self.device)
        
        self.discriminator = Discriminator(
            image_size=image_size,
            channels=channels,
            condition_dim=condition_dim
        ).to(self.device)
        
        # 设置主模型为生成器
        self.model = self.generator
        
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
    
    def train(
        self,
        train_data: Dict[str, Any],
        validation_data: Optional[Dict[str, Any]] = None,
        epochs: int = 100,
        batch_size: int = 16,
        learning_rate: float = 0.0002
    ) -> Dict[str, Any]:
        """训练GAN模型
        
        Args:
            train_data: 训练数据字典
            validation_data: 验证数据字典
            epochs: 训练轮次
            batch_size: 批次大小
            learning_rate: 学习率
            
        Returns:
            训练结果
        """
        logger.info("开始训练生态图像生成模型...")
        
        # 准备数据加载器
        if "dataset" in train_data:
            train_loader = DataLoader(
                train_data["dataset"],
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )
        else:
            raise ValueError("训练数据中缺少dataset")
        
        # 设置优化器
        g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999)
        )
        
        d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.999)
        )
        
        # 训练历史
        g_losses = []
        d_losses = []
        
        # 固定噪声用于生成样本
        fixed_noise = torch.randn(16, 512).to(self.device)
        fixed_conditions = torch.rand(16, 10).to(self.device)
        
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            
            for batch_idx, (real_images, conditions) in enumerate(train_loader):
                batch_size = real_images.size(0)
                real_images = real_images.to(self.device)
                conditions = conditions.to(self.device)
                
                # 真实和虚假标签
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # ==================
                # 训练判别器
                # ==================
                d_optimizer.zero_grad()
                
                # 真实图像
                real_output = self.discriminator(real_images, conditions)
                d_real_loss = self.adversarial_loss(real_output, real_labels)
                
                # 生成虚假图像
                noise = torch.randn(batch_size, 512).to(self.device)
                fake_images = self.generator(noise, conditions)
                fake_output = self.discriminator(fake_images.detach(), conditions)
                d_fake_loss = self.adversarial_loss(fake_output, fake_labels)
                
                # 判别器总损失
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                d_optimizer.step()
                
                # ==================
                # 训练生成器
                # ==================
                g_optimizer.zero_grad()
                
                # 生成器损失
                fake_output = self.discriminator(fake_images, conditions)
                g_adversarial_loss = self.adversarial_loss(fake_output, real_labels)
                
                # 感知损失
                g_perceptual_loss = self.perceptual_loss(fake_images, real_images)
                
                # 总生成器损失
                g_loss = g_adversarial_loss + 0.1 * g_perceptual_loss
                g_loss.backward()
                g_optimizer.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
            
            # 记录平均损失
            avg_g_loss = epoch_g_loss / len(train_loader)
            avg_d_loss = epoch_d_loss / len(train_loader)
            
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            
            # 记录训练历史
            self.add_training_record(epoch, {
                "g_loss": avg_g_loss,
                "d_loss": avg_d_loss
            })
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}")
                
                # 生成样本图像
                with torch.no_grad():
                    sample_images = self.generator(fixed_noise, fixed_conditions)
                    # 这里可以保存样本图像
        
        self.is_trained = True
        logger.info("生态图像生成模型训练完成")
        
        return {
            "g_losses": g_losses,
            "d_losses": d_losses,
            "final_g_loss": g_losses[-1],
            "final_d_loss": d_losses[-1]
        }
    
    def predict(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """生成图像
        
        Args:
            input_data: 输入数据字典
            **kwargs: 其他参数
            
        Returns:
            生成结果字典
        """
        if self.generation_mode == "gan":
            return self._generate_with_gan(input_data, **kwargs)
        elif self.generation_mode == "diffusion":
            return self._generate_with_diffusion(input_data, **kwargs)
        else:
            raise ValueError(f"不支持的生成模式: {self.generation_mode}")
    
    def _generate_with_gan(
        self,
        input_data: Dict[str, Any],
        num_images: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """使用GAN生成图像"""
        if self.generator is None:
            return {"error": "GAN生成器未初始化"}
        
        self.generator.eval()
        
        with torch.no_grad():
            # 准备条件
            if "conditions" in input_data:
                conditions = torch.FloatTensor(input_data["conditions"]).to(self.device)
                if conditions.dim() == 1:
                    conditions = conditions.unsqueeze(0)
            else:
                # 使用随机条件
                conditions = torch.rand(num_images, 10).to(self.device)
            
            # 生成噪声
            noise = torch.randn(conditions.size(0), 512).to(self.device)
            
            # 生成图像
            generated_images = self.generator(noise, conditions)
            
            # 转换为numpy数组
            images_np = generated_images.cpu().numpy()
            images_np = (images_np + 1) / 2  # 从[-1,1]转换到[0,1]
            images_np = np.transpose(images_np, (0, 2, 3, 1))  # NCHW -> NHWC
            
            return {
                "generated_images": images_np.tolist(),
                "conditions_used": conditions.cpu().numpy().tolist(),
                "generation_mode": "gan"
            }
    
    def _generate_with_diffusion(
        self,
        input_data: Dict[str, Any],
        num_images: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """使用扩散模型生成图像"""
        if self.diffusion_pipeline is None:
            return {"error": "扩散模型未初始化"}
        
        # 构建文本提示
        if "conditions" in input_data:
            prompt = self._conditions_to_prompt(input_data["conditions"])
        elif "prompt" in input_data:
            prompt = input_data["prompt"]
        else:
            prompt = "environmental warning scene"
        
        try:
            # 生成图像
            with torch.no_grad():
                result = self.diffusion_pipeline(
                    prompt=prompt,
                    num_images_per_prompt=num_images,
                    height=512,
                    width=512,
                    num_inference_steps=50,
                    guidance_scale=7.5
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
            return {"error": str(e)}
    
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
    
    def set_generation_mode(self, mode: str) -> None:
        """设置生成模式
        
        Args:
            mode: "gan" 或 "diffusion"
        """
        if mode not in ["gan", "diffusion"]:
            raise ValueError(f"不支持的生成模式: {mode}")
        
        self.generation_mode = mode
        logger.info(f"生成模式设置为: {mode}")
    
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
        
        return self.predict(input_data, num_images=num_images)
    
    def get_condition_templates(self) -> Dict[str, Dict[str, float]]:
        """获取预设环境场景模板"""
        return {
            "冰川融化": {
                "co2_level": 450,
                "temperature": 40,
                "warning_level": 4,
                "pollution_level": 6
            },
            "森林砍伐": {
                "forest_coverage": 10,
                "biodiversity": 3,
                "warning_level": 4,
                "co2_level": 420
            },
            "空气污染": {
                "pm25_level": 200,
                "air_quality": 2,
                "warning_level": 5,
                "pollution_level": 8
            },
            "水质污染": {
                "water_quality": 2,
                "pollution_level": 7,
                "warning_level": 4,
                "biodiversity": 4
            },
            "极端天气": {
                "temperature": 45,
                "humidity": 90,
                "warning_level": 5,
                "pollution_level": 5
            }
        }