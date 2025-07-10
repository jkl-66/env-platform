#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environmental Protection Warning Image Generator based on Alibaba Cloud DashScope

Uses Qwen chat model to generate professional environmental warning prompts
Uses Flux image generation model to generate high-quality environmental warning images
Supports user input of carbon emissions, pollution indices and other environmental data
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

# Import DashScope
try:
    from dashscope import Generation, ImageSynthesis
except ImportError:
    print("❌ Please install dashscope: pip install dashscope")
    sys.exit(1)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import configuration
try:
    from src.utils.config import get_settings
    settings = get_settings()
except ImportError:
    settings = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashScopeEnvironmentalGenerator:
    """Environmental Protection Warning Image Generator based on Alibaba Cloud DashScope"""
    
    def __init__(self, 
                 dashscope_api_key: Optional[str] = None,
                 chat_model: str = "qwen-turbo",
                 image_model: str = "flux-schnell"):
        """
        Initialize DashScope Environmental Image Generator
        
        Args:
            dashscope_api_key: DashScope API Key
            chat_model: Chat model name
            image_model: Image generation model name
        """
        self.api_key = dashscope_api_key or os.getenv('DASHSCOPE_API_KEY')
        self.chat_model = chat_model
        self.image_model = image_model
        
        if not self.api_key:
            raise ValueError("❌ DASHSCOPE_API_KEY not set, please configure it in .env file")
        
        # Set API Key
        os.environ['DASHSCOPE_API_KEY'] = self.api_key
        
        # Environmental data type definitions (including current world normal data as default values)
        self.environmental_data_types = {
            "carbon_emission": {
                "name": "Carbon Emissions",
                "unit": "billion tons",
                "default_value": 416,  # Global carbon emissions in 2024
                "description": "Global carbon emissions, approximately 41.6 billion tons in 2024",
                "thresholds": {
                    "low": 300,
                    "medium": 400,
                    "high": 500,
                    "critical": 600
                }
            },
            "air_quality_index": {
                "name": "Air Quality Index",
                "unit": "AQI",
                "default_value": 75,  # Current world average AQI
                "description": "Air quality levels: Good(AQI≤50), Moderate(AQI≤100), Unhealthy for Sensitive Groups(AQI≤150), Unhealthy(AQI≤200), Very Unhealthy(AQI≤300), Hazardous(AQI>300)",
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
                "name": "Water Pollution Index",
                "unit": "WPI",
                "default_value": 35,  # Current world average water pollution index
                "description": "Below 60 is suitable for most normal aquatic ecosystems and general water use needs",
                "thresholds": {
                    "clean": 25,
                    "slightly_polluted": 50,
                    "moderately_polluted": 75,
                    "heavily_polluted": 100
                }
            },
            "noise_level": {
                "name": "Noise Level",
                "unit": "decibels(dB)",
                "default_value": 55,  # Current world average noise level
                "description": "30-40 decibels is a relatively quiet normal environment; over 50 decibels affects sleep and rest. Above 70 decibels interferes with conversation, causes irritability, lack of concentration, affects work efficiency, and may even cause accidents; long-term work or living in noise environments above 90 decibels will seriously affect hearing and cause other diseases",
                "thresholds": {
                    "quiet": 40,
                    "moderate": 55,
                    "loud": 70,
                    "very_loud": 85,
                    "harmful": 100
                }
            },
            "deforestation_rate": {
                "name": "Deforestation Area",
                "unit": "10k hectares/year",
                "default_value": 660,  # Current world deforestation area
                "description": "Global annual deforestation area, currently about 6.6 million hectares/year",
                "thresholds": {
                    "low": 300,
                    "medium": 500,
                    "high": 800,
                    "critical": 1000
                }
            },
            "plastic_waste": {
                "name": "Plastic Waste",
                "unit": "10k tons/year",
                "default_value": 1116.8,  # Current world plastic waste amount
                "description": "Global annual plastic waste generation, currently about 11.168 million tons/year",
                "thresholds": {
                    "low": 500,
                    "medium": 800,
                    "high": 1200,
                    "critical": 1500
                }
            }
        }
        
        # Image style definitions
        self.image_styles = {
            "realistic": "Realistic style",
            "artistic": "Artistic style", 
            "scientific": "Scientific visualization",
            "infographic": "Infographic style"
        }
        
        logger.info(f"✅ DashScope Environmental Image Generator initialization completed")
        logger.info(f"🤖 Chat model: {self.chat_model}")
        logger.info(f"🎨 Image model: {self.image_model}")
    

    
    def _analyze_environmental_data(self, data: Dict[str, Union[float, int]]) -> Dict[str, Any]:
        """
        Analyze environmental data to determine pollution levels and impacts
        
        Args:
            data: Environmental data dictionary, keys are data types, values are numerical values
            
        Returns:
            Analysis result dictionary
        """
        analysis = {
            "overall_severity": "low",
            "critical_factors": [],
            "environmental_impacts": [],
            "severity_scores": {}
        }
        
        total_severity_score = 0
        valid_factors = 0
        
        for data_type, value in data.items():
            if data_type not in self.environmental_data_types:
                logger.warning(f"⚠️ Unknown environmental data type: {data_type}")
                continue
            
            data_config = self.environmental_data_types[data_type]
            thresholds = data_config["thresholds"]
            
            # Determine severity level
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
                # Generic threshold assessment
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
            
            # Identify critical factors
            if score >= 4:
                analysis["critical_factors"].append({
                    "type": data_type,
                    "name": data_config["name"],
                    "value": value,
                    "unit": data_config["unit"],
                    "severity": severity
                })
        
        # Calculate overall severity
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
    
    def _calculate_data_deviations(self, environmental_data: Dict[str, Union[float, int]]) -> Dict[str, Any]:
        """
        Calculate deviations from default values for each environmental factor
        
        Args:
            environmental_data: Environmental data dictionary
            
        Returns:
            Deviation analysis results
        """
        deviations = []
        
        for data_type, value in environmental_data.items():
            if data_type in self.environmental_data_types:
                data_config = self.environmental_data_types[data_type]
                default_val = data_config['default_value']
                
                # Calculate absolute and percentage deviation
                abs_deviation = abs(value - default_val)
                pct_deviation = abs((value - default_val) / default_val) * 100
                
                deviations.append({
                    'data_type': data_type,
                    'name': data_config['name'],
                    'value': value,
                    'default_value': default_val,
                    'abs_deviation': abs_deviation,
                    'pct_deviation': pct_deviation,
                    'unit': data_config['unit'],
                    'is_above_default': value > default_val
                })
        
        # Sort by percentage deviation (highest first)
        deviations.sort(key=lambda x: x['pct_deviation'], reverse=True)
        
        # Build top deviations text
        top_deviations_text = []
        for i, dev in enumerate(deviations[:3]):  # Top 3 deviations
            direction = "above" if dev['is_above_default'] else "below"
            top_deviations_text.append(
                f"{i+1}. {dev['name']}: {dev['pct_deviation']:.1f}% {direction} normal "
                f"({dev['value']} vs {dev['default_value']} {dev['unit']})"
            )
        
        return {
            'deviations': deviations,
            'top_deviations': deviations[:3],
            'top_deviations_text': '\n'.join(top_deviations_text)
        }
    
    def _build_emphasis_instructions(self, deviation_analysis: Dict[str, Any]) -> str:
        """
        Build visual emphasis instructions based on data deviations
        
        Args:
            deviation_analysis: Deviation analysis results
            
        Returns:
            Visual emphasis instructions string
        """
        instructions = []
        
        for i, deviation in enumerate(deviation_analysis['top_deviations']):
            data_type = deviation['data_type']
            name = deviation['name']
            pct_dev = deviation['pct_deviation']
            is_above = deviation['is_above_default']
            
            # Generate specific visual instructions based on data type and severity
            if data_type == 'carbon_emission':
                if is_above and pct_dev > 20:
                    instructions.append(f"- HEAVILY emphasize thick, dark smoke and industrial pollution covering the sky")
                elif is_above:
                    instructions.append(f"- Show visible air pollution and smog in the atmosphere")
            
            elif data_type == 'air_quality_index':
                if is_above and pct_dev > 50:
                    instructions.append(f"- Make the air visibly thick with smog, reduced visibility, hazy atmosphere")
                elif is_above:
                    instructions.append(f"- Show polluted air with visible particles and reduced clarity")
            
            elif data_type == 'water_pollution_index':
                if is_above and pct_dev > 30:
                    instructions.append(f"- Show severely contaminated water bodies with visible pollution, dead fish, toxic colors")
                elif is_above:
                    instructions.append(f"- Display polluted water with murky, discolored appearance")
            
            elif data_type == 'noise_level':
                if is_above and pct_dev > 25:
                    instructions.append(f"- Emphasize industrial noise sources, heavy machinery, urban chaos")
                elif is_above:
                    instructions.append(f"- Show busy, noisy urban environment with traffic and construction")
            
            elif data_type == 'deforestation_rate':
                if is_above and pct_dev > 40:
                    instructions.append(f"- PROMINENTLY show massive deforestation, clear-cut areas, destroyed forest landscapes")
                elif is_above:
                    instructions.append(f"- Display areas of forest loss and environmental degradation")
            
            elif data_type == 'plastic_waste':
                if is_above and pct_dev > 35:
                    instructions.append(f"- Show overwhelming plastic waste pollution, garbage-filled landscapes")
                elif is_above:
                    instructions.append(f"- Include visible plastic waste and pollution in the environment")
        
        # Add priority instruction
        if instructions:
            priority_instruction = f"HIGHEST PRIORITY: Focus most prominently on {deviation_analysis['top_deviations'][0]['name']} ({deviation_analysis['top_deviations'][0]['pct_deviation']:.1f}% deviation)"
            instructions.insert(0, priority_instruction)
        
        return '\n'.join(instructions) if instructions else "Show general environmental warning imagery"
    
    def _generate_professional_prompt(self, 
                                    environmental_data: Dict[str, Union[float, int]],
                                    user_description: Optional[str] = None,
                                    target_audience: str = "general") -> str:
        """
        Generate professional environmental warning image prompt using Qwen model
        
        Args:
            environmental_data: Environmental data
            user_description: User description
            target_audience: Target audience (general, educators, parents, students)
            
        Returns:
            Generated professional prompt
        """
        # Analyze environmental data
        analysis = self._analyze_environmental_data(environmental_data)
        
        # Calculate deviation from default values and build emphasis instructions
        deviation_analysis = self._calculate_data_deviations(environmental_data)
        emphasis_instructions = self._build_emphasis_instructions(deviation_analysis)
        
        # Build enhanced system prompt with data-driven visual emphasis
        system_prompt = f"""
You are a professional environmental protection education expert and visual designer. Please generate a professional environmental warning image description prompt based on the provided environmental data.

CRITICAL REQUIREMENTS:
1. Create realistic and credible scenes that DIRECTLY reflect the specific environmental data values provided
2. The image MUST visually emphasize environmental factors that deviate most from normal levels
3. Use the deviation analysis to determine which environmental issues should be most prominent in the image
4. Generate WARNING imagery that clearly shows environmental damage and consequences
5. Make the image educational and impactful for raising environmental awareness
6. Descriptions should be specific, vivid, and include detailed visual elements
7. Images cannot contain people, faces or human bodies
8. Images cannot contain any text, labels or text elements
9. Focus on creating a sense of urgency and environmental crisis

VISUAL EMPHASIS INSTRUCTIONS:
{emphasis_instructions}

Please generate the prompt in English, keeping the length between 200-300 words.
"""
        
        # Build enhanced user input with detailed data analysis
        data_description = []
        for data_type, value in environmental_data.items():
            if data_type in self.environmental_data_types:
                data_config = self.environmental_data_types[data_type]
                default_val = data_config['default_value']
                deviation_pct = ((value - default_val) / default_val) * 100
                severity_info = analysis['severity_scores'].get(data_type, {})
                
                data_description.append(
                    f"{data_config['name']}: {value} {data_config['unit']} "
                    f"(Default: {default_val}, Deviation: {deviation_pct:+.1f}%, "
                    f"Severity: {severity_info.get('severity', 'unknown')})"
                )
        
        user_input = f"""
DETAILED ENVIRONMENTAL DATA ANALYSIS:
{chr(10).join(data_description)}

Overall Environmental Severity: {analysis['overall_severity']}
Critical Factors Requiring Visual Emphasis: {', '.join([f['name'] for f in analysis['critical_factors']])}

Most Significant Deviations from Normal Levels:
{deviation_analysis['top_deviations_text']}
"""
        
        if user_description:
            user_input += f"\n\nUser's Additional Requirements: {user_description}"
        
        user_input += "\n\nPlease generate a professional environmental warning image description prompt that emphasizes the most critical environmental issues based on the data provided."
        
        try:
            logger.info("🤖 Using Qwen model to generate professional prompt...")
            
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
                logger.info(f"✅ Professional prompt generation successful")
                logger.info(f"📝 Generated prompt: {generated_prompt[:100]}...")
                return generated_prompt
            else:
                logger.error(f"❌ Qwen model call failed: {response.message}")
                return self._fallback_prompt_generation(environmental_data, analysis)
                
        except Exception as e:
            logger.error(f"❌ Error occurred while generating professional prompt: {e}")
            return self._fallback_prompt_generation(environmental_data, analysis)
    
    def _fallback_prompt_generation(self, 
                                  environmental_data: Dict[str, Union[float, int]], 
                                  analysis: Dict[str, Any]) -> str:
        """
        Fallback prompt generation method
        
        Args:
            environmental_data: Environmental data
            analysis: Environmental data analysis results
            
        Returns:
            Fallback generated prompt
        """
        base_prompt = "Environmental warning scene showing environmental issues"
        base_prompt += ", professional environmental documentary photography, realistic style, high quality"
        return base_prompt
    
    def _generate_image_with_flux(self, prompt: str, size: str = '1024*1024') -> Optional[Image.Image]:
        """
        Generate image using Flux model
        
        Args:
            prompt: Image generation prompt
            size: Image size
            
        Returns:
            Generated PIL Image object
        """
        try:
            logger.info(f"🎨 Generating image using Flux model...")
            logger.info(f"📝 Prompt: {prompt}")
            
            response = ImageSynthesis.call(
                model=self.image_model,
                prompt=prompt,
                size=size
            )
            
            if response.status_code == HTTPStatus.OK:
                logger.info(f"✅ Image generation successful")
                logger.info(f"📊 Usage: {response.usage}")
                
                # Download and process image
                for result in response.output.results:
                    file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
                    image_content = requests.get(result.url).content
                    image = Image.open(BytesIO(image_content))
                    return image
                    
            else:
                logger.error(f"❌ Flux model call failed: status_code={response.status_code}, code={response.code}, message={response.message}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error occurred during image generation: {e}")
            return None
    
    def generate_environmental_warning_image(self,
                                           environmental_data: Dict[str, Union[float, int]],
                                           user_description: Optional[str] = None,
                                           style: str = "realistic",
                                           image_size: str = '1024*1024') -> Dict[str, Any]:
        """
        Generate environmental warning image
        
        Args:
            environmental_data: Environmental data dictionary
            user_description: User additional description
            style: Image style
            image_size: Image size
            
        Returns:
            Generation result dictionary
        """
        try:
            # Analyze environmental data
            analysis = self._analyze_environmental_data(environmental_data)
            
            # Generate professional prompt
            professional_prompt = self._generate_professional_prompt(
                environmental_data, user_description
            )
            
            # Generate image
            image = self._generate_image_with_flux(professional_prompt, image_size)
            
            if not image:
                # Use fallback prompt generation
                fallback_prompt = self._fallback_prompt_generation(environmental_data, analysis)
                image = self._generate_image_with_flux(fallback_prompt, image_size)
                
                if not image:
                    return {
                        "success": False,
                        "error": "Image generation failed",
                        "analysis": analysis
                    }
            
            # Save image
            output_dir = "outputs/environmental_images"
            saved_paths = self._save_images([image], environmental_data, output_dir)
            
            return {
                "success": True,
                "environmental_data": environmental_data,
                "analysis": analysis,
                "image": image,
                "saved_paths": saved_paths
            }
            
        except Exception as e:
            logger.error(f"Error occurred while generating environmental warning image: {e}")
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
        Save generated images
        
        Args:
            images: Image list
            environmental_data: Environmental data
            output_dir: Output directory
            auto_open: Whether to automatically open images
            
        Returns:
            List of saved file paths
        """
        if not images:
            return []
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_summary = "_".join([f"{k}_{v}" for k, v in list(environmental_data.items())[:2]])
        safe_summary = "".join(c for c in data_summary if c.isalnum() or c in ('_', '-'))[:30]
        
        saved_paths = []
        
        for i, image in enumerate(images, 1):
            filename = f"env_warning_{safe_summary}_{timestamp}_{i}_dashscope.png"
            file_path = output_path / filename
            
            image.save(file_path, "PNG")
            saved_paths.append(str(file_path))
            
            logger.info(f"Image saved: {file_path}")
            
            # Automatically open image if requested
            if auto_open:
                self._open_image(file_path)
        
        return saved_paths
    
    def _open_image(self, file_path: Path):
        """
        Automatically open image file
        
        Args:
            file_path: Image file path
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
                logger.warning(f"Unsupported operating system: {system}, cannot automatically open image")
                return
            
            logger.info(f"Image automatically opened: {file_path}")
        except Exception as e:
            logger.warning(f"Cannot automatically open image {file_path}: {e}")
    
    def _save_generation_report(self, result: Dict[str, Any], output_dir: str):
        """
        Save generation report
        
        Args:
            result: Generation results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"environmental_report_{timestamp}.json"
        
        # Prepare report data (remove non-serializable objects)
        report_data = result.copy()
        if "image" in report_data:
            del report_data["image"]  # PIL Image objects cannot be serialized
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generation report saved: {report_file}")
    
    def get_supported_data_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Get supported environmental data types
        
        Returns:
            Environmental data types dictionary
        """
        return self.environmental_data_types
    
    def get_default_environmental_data(self):
        """Get default environmental data"""
        environmental_data = {}
        for data_type, config in self.environmental_data_types.items():
            environmental_data[data_type] = config['default_value']
        return environmental_data
    
    def interactive_data_input(self):
        """Interactive environmental data input function"""
        print("\n=== Environmental Data Input Mode Selection ===")
        print("1. Interactive Input - Enter environmental data item by item")
        print("2. Quick Mode - Use current world average data")
        print("3. View Data Descriptions")
        
        while True:
            try:
                choice = input("\nPlease select mode (1/2/3): ").strip()
                
                if choice == "1":
                    return self._detailed_interactive_input()
                elif choice == "2":
                    return self._quick_mode_input()
                elif choice == "3":
                    self._show_data_descriptions()
                    continue
                else:
                    print("Please enter a valid choice (1/2/3)")
                    
            except KeyboardInterrupt:
                print("\nUser cancelled input")
                return None
    
    def _show_data_descriptions(self):
        """Show data descriptions"""
        print("\n=== Environmental Data Descriptions ===")
        for data_type, config in self.environmental_data_types.items():
            print(f"\n📊 {config['name']} ({config['unit']})")
            print(f"   Description: {config['description']}")
            print(f"   Current world average: {config['default_value']}")
            print("   Reference thresholds:")
            for level, value in config['thresholds'].items():
                level_name = {
                    'low': 'Low', 'medium': 'Medium', 'high': 'High', 'critical': 'Critical',
                    'good': 'Good', 'moderate': 'Moderate', 'unhealthy_sensitive': 'Unhealthy for Sensitive',
                    'unhealthy': 'Unhealthy', 'very_unhealthy': 'Very Unhealthy', 'hazardous': 'Hazardous',
                    'clean': 'Clean', 'slightly_polluted': 'Slightly Polluted', 'moderately_polluted': 'Moderately Polluted',
                    'heavily_polluted': 'Heavily Polluted', 'quiet': 'Quiet', 'loud': 'Loud', 'very_loud': 'Very Loud',
                    'harmful': 'Harmful'
                }.get(level, level)
                print(f"     {level_name}: {value}")
    
    def _quick_mode_input(self):
        """Quick mode - use default values"""
        print("\n=== Quick Mode - Using Current World Average Data ===")
        environmental_data = self.get_default_environmental_data()
        
        print("Environmental data to be used:")
        for data_type, value in environmental_data.items():
            config = self.environmental_data_types[data_type]
            print(f"  {config['name']}: {value} {config['unit']}")
        
        confirm = input("\nConfirm using this data? (y/n): ").strip().lower()
        if confirm in ['y', 'yes', '']:
            return environmental_data
        else:
            return None
    
    def _detailed_interactive_input(self):
        """Detailed interactive input"""
        print("\n=== Environmental Data Interactive Input ===")
        print("Please enter environmental data, press Enter to use default values\n")
        
        environmental_data = {}
        
        for data_type, config in self.environmental_data_types.items():
            print(f"\n--- {config['name']} ---")
            print(f"Description: {config['description']}")
            print(f"Unit: {config['unit']}")
            print(f"Default value: {config['default_value']}")
            
            # Show threshold references
            print("Reference thresholds:")
            for level, value in config['thresholds'].items():
                level_name = {
                    'low': 'Low', 'medium': 'Medium', 'high': 'High', 'critical': 'Critical',
                    'good': 'Good', 'moderate': 'Moderate', 'unhealthy_sensitive': 'Unhealthy for Sensitive',
                    'unhealthy': 'Unhealthy', 'very_unhealthy': 'Very Unhealthy', 'hazardous': 'Hazardous',
                    'clean': 'Clean', 'slightly_polluted': 'Slightly Polluted', 'moderately_polluted': 'Moderately Polluted',
                    'heavily_polluted': 'Heavily Polluted', 'quiet': 'Quiet', 'loud': 'Loud', 'very_loud': 'Very Loud',
                    'harmful': 'Harmful'
                }.get(level, level)
                print(f"  {level_name}: {value}")
            
            while True:
                try:
                    user_input = input(f"\nPlease enter {config['name']} value (default: {config['default_value']}): ").strip()
                    
                    if user_input == "":
                        # Use default value
                        environmental_data[data_type] = config['default_value']
                        print(f"Using default value: {config['default_value']} {config['unit']}")
                        break
                    else:
                        # Try to convert user input
                        value = float(user_input)
                        environmental_data[data_type] = value
                        print(f"Set: {value} {config['unit']}")
                        break
                        
                except ValueError:
                    print("Please enter a valid numeric value!")
                except KeyboardInterrupt:
                    print("\nUser cancelled input")
                    return None
        
        print("\n=== Input Complete ===")
        print("Final environmental data:")
        for data_type, value in environmental_data.items():
            config = self.environmental_data_types[data_type]
            print(f"{config['name']}: {value} {config['unit']}")
        
        return environmental_data
    
    def get_additional_prompt(self):
        """
        Get additional user prompt for image generation
        
        Returns:
            Additional prompt string or None
        """
        print("\n=== Additional Prompt Input (Optional) ===")
        print("You can add additional descriptions to customize the generated image.")
        print("For example: 'Show a city skyline', 'Include people wearing masks', 'Dark and apocalyptic style', etc.")
        print("Press Enter to skip this step.")
        
        try:
            additional_prompt = input("\nPlease enter additional prompt: ").strip()
            
            if additional_prompt:
                print(f"Additional prompt added: {additional_prompt}")
                return additional_prompt
            else:
                print("No additional prompt added")
                return None
                
        except KeyboardInterrupt:
            print("\nUser cancelled input")
            return None
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test DashScope connection
        
        Returns:
            Test results
        """
        try:
            # Test chat model
            chat_response = Generation.call(
                model=self.chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                result_format='message'
            )
            
            chat_success = chat_response.status_code == HTTPStatus.OK
            
            # Test image generation model
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
    Main function - Interactive environmental data input and image generation
    """
    print("=== DashScope Environmental Warning Image Generator ===")
    print("This tool can generate warning images based on environmental data")
    
    try:
        # Initialize generator
        generator = DashScopeEnvironmentalGenerator()
        
        # Test connection
        print("\nTesting API connection...")
        connection_result = generator.test_connection()
        if not connection_result["success"]:
            print(f"API connection failed: {connection_result.get('error', 'Unknown error')}")
            return
        print("API connection successful!")
        
        # Interactive data input
        environmental_data = generator.interactive_data_input()
        
        if environmental_data is None:
            print("User cancelled operation")
            return
        
        # Get additional prompt from user
        additional_prompt = generator.get_additional_prompt()
        
        print("\nStarting environmental warning image generation...")
        
        # Generate environmental warning image
        result = generator.generate_environmental_warning_image(
            environmental_data, 
            user_description=additional_prompt
        )
        
        if result["success"]:
            print(f"\n✅ Image generation successful!")
            print(f"📁 Saved paths: {result['saved_paths']}")
            if 'analysis' in result:
                analysis = result['analysis']
                print(f"📊 Environmental analysis: Severity level {analysis.get('overall_severity', 'N/A')}, Critical factors: {', '.join([f['name'] for f in analysis.get('critical_factors', [])])}")
        else:
            print(f"\n❌ Image generation failed: {result.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print("\n\nUser interrupted program")
    except Exception as e:
        print(f"\nProgram execution error: {str(e)}")
        logging.error(f"Main function error: {str(e)}")

if __name__ == "__main__":
    main()