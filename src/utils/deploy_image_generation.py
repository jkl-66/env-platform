import os
import sys
import subprocess
import platform
import argparse
import logging
from pathlib import Path
import json
import urllib.request
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("deploy_image_generation")

# 定义常量
FOOOCUS_REPO = "https://github.com/lllyasviel/Fooocus.git"
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"

# 本地模型配置
LOCAL_MODELS = {
    "stable_diffusion_v1_5": {
        "name": "Stable Diffusion 3.5 Large",
        "model_id": "stabilityai/stable-diffusion-3.5-large",
        "size": "约8GB",
        "description": "最新的Stable Diffusion 3.5模型，图像质量和细节表现优秀"
    },
    "stable_diffusion_v2_1": {
        "name": "Stable Diffusion v2.1",
        "model_id": "stabilityai/stable-diffusion-2-1",
        "size": "约5GB",
        "description": "改进版本，图像质量更高"
    },
    "stable_diffusion_xl": {
        "name": "Stable Diffusion XL",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "size": "约7GB",
        "description": "最新版本，支持更高分辨率图像生成"
    }
}


def check_gpu():
    """检查GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"检测到GPU: {gpu_name} ({gpu_memory:.1f}GB显存)")
            return True, gpu_memory
        else:
            logger.warning("未检测到GPU，图像生成速度可能较慢")
            return False, 0
    except ImportError:
        logger.warning("未安装PyTorch，无法检测GPU")
        return False, 0


def check_disk_space(path, required_gb=10):
    """检查磁盘空间"""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        if free_gb >= required_gb:
            logger.info(f"磁盘空间充足: {free_gb:.1f}GB 可用")
            return True
        else:
            logger.warning(f"磁盘空间不足: 需要{required_gb}GB，仅有{free_gb:.1f}GB可用")
            return False
    except Exception as e:
        logger.error(f"检查磁盘空间失败: {e}")
        return False


def install_local_dependencies():
    """安装本地模型运行所需的依赖"""
    dependencies = [
        "torch",
        "torchvision", 
        "diffusers>=0.21.0",
        "transformers>=4.21.0",
        "accelerate",
        "safetensors",
        "pillow",
        "numpy"
    ]
    
    logger.info("安装本地模型运行依赖...")
    try:
        for dep in dependencies:
            logger.info(f"安装 {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--upgrade"])
        logger.info("依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"依赖安装失败: {e}")
        return False


def download_local_model(model_key, cache_dir="./models"):
    """下载本地模型"""
    if model_key not in LOCAL_MODELS:
        logger.error(f"未知模型: {model_key}")
        return False
    
    model_info = LOCAL_MODELS[model_key]
    model_id = model_info["model_id"]
    
    logger.info(f"开始下载模型: {model_info['name']} ({model_info['size']})")
    logger.info(f"模型描述: {model_info['description']}")
    
    try:
        # 使用diffusers库下载模型
        from diffusers import StableDiffusionPipeline
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 下载模型
        logger.info("正在下载模型文件，这可能需要几分钟...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # 保存到本地
        local_path = os.path.join(cache_dir, model_key)
        pipeline.save_pretrained(local_path)
        
        logger.info(f"模型下载完成，保存到: {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"模型下载失败: {e}")
        return False


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        logger.info(f"Python版本兼容: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        logger.error(f"Python版本不兼容: {version.major}.{version.minor}.{version.micro}，需要Python 3.8+")
        return False


def install_requirements(requirements_file):
    """安装依赖"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        logger.info("依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"依赖安装失败: {e}")
        return False


def clone_repository(repo_url, target_dir):
    """克隆仓库"""
    if os.path.exists(target_dir):
        logger.info(f"目录已存在: {target_dir}，跳过克隆")
        return True
    
    try:
        subprocess.check_call(["git", "clone", repo_url, target_dir])
        logger.info(f"仓库克隆成功: {repo_url} -> {target_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"仓库克隆失败: {e}")
        return False


def deploy_fooocus(install_dir, launch=True):
    """部署Fooocus"""
    install_dir = os.path.abspath(install_dir)
    
    # 克隆仓库
    if not clone_repository(FOOOCUS_REPO, install_dir):
        return False
    
    # 安装依赖
    os.chdir(install_dir)
    if platform.system() == "Windows":
        setup_script = os.path.join(install_dir, "install.bat")
        if os.path.exists(setup_script):
            try:
                logger.info("运行Fooocus安装脚本...")
                subprocess.check_call([setup_script])
                logger.info("Fooocus安装成功")
            except subprocess.CalledProcessError as e:
                logger.error(f"Fooocus安装脚本运行失败: {e}")
                return False
    else:  # Linux/Mac
        setup_script = os.path.join(install_dir, "install.sh")
        if os.path.exists(setup_script):
            try:
                logger.info("运行Fooocus安装脚本...")
                subprocess.check_call(["bash", setup_script])
                logger.info("Fooocus安装成功")
            except subprocess.CalledProcessError as e:
                logger.error(f"Fooocus安装脚本运行失败: {e}")
                return False
    
    # 启动Fooocus
    if launch:
        launch_fooocus(install_dir)
    
    return True


def launch_fooocus(install_dir):
    """启动Fooocus"""
    os.chdir(install_dir)
    if platform.system() == "Windows":
        launch_script = os.path.join(install_dir, "run.bat")
    else:  # Linux/Mac
        launch_script = os.path.join(install_dir, "run.sh")
    
    if os.path.exists(launch_script):
        try:
            logger.info("启动Fooocus...")
            if platform.system() == "Windows":
                subprocess.Popen([launch_script])
            else:
                subprocess.Popen(["bash", launch_script])
            logger.info("Fooocus已启动，请在浏览器中访问: http://127.0.0.1:7865")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Fooocus启动失败: {e}")
            return False
    else:
        logger.error(f"启动脚本不存在: {launch_script}")
        return False


def deploy_comfyui(install_dir, launch=True):
    """部署ComfyUI"""
    install_dir = os.path.abspath(install_dir)
    
    # 克隆仓库
    if not clone_repository(COMFYUI_REPO, install_dir):
        return False
    
    # 安装依赖
    os.chdir(install_dir)
    requirements_file = os.path.join(install_dir, "requirements.txt")
    if os.path.exists(requirements_file):
        if not install_requirements(requirements_file):
            return False
    
    # 启动ComfyUI
    if launch:
        launch_comfyui(install_dir)
    
    return True


def launch_comfyui(install_dir):
    """启动ComfyUI"""
    os.chdir(install_dir)
    try:
        logger.info("启动ComfyUI...")
        if platform.system() == "Windows":
            subprocess.Popen([sys.executable, "main.py"])
        else:
            subprocess.Popen(["python", "main.py"])
        logger.info("ComfyUI已启动，请在浏览器中访问: http://127.0.0.1:8188")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ComfyUI启动失败: {e}")
        return False


def deploy_local_model(model_key="stable_diffusion_v1_5", cache_dir="./models"):
    """部署本地模型（推荐方式）"""
    logger.info("=== 部署本地图像生成模型 ===")
    
    # 检查环境
    if not check_python_version():
        return False
    
    has_gpu, gpu_memory = check_gpu()
    
    # 根据GPU情况推荐模型
    if has_gpu and gpu_memory >= 8:
        recommended_model = "stable_diffusion_xl"
        logger.info("检测到高性能GPU，推荐使用Stable Diffusion XL")
    elif has_gpu and gpu_memory >= 6:
        recommended_model = "stable_diffusion_v2_1"
        logger.info("检测到中等性能GPU，推荐使用Stable Diffusion v2.1")
    else:
        recommended_model = "stable_diffusion_v1_5"
        logger.info("推荐使用Stable Diffusion v1.5（兼容性最好）")
    
    # 如果用户没有指定模型，使用推荐模型
    if model_key == "stable_diffusion_v1_5" and recommended_model != "stable_diffusion_v1_5":
        logger.info(f"自动选择推荐模型: {recommended_model}")
        model_key = recommended_model
    
    # 检查磁盘空间
    required_space = 10 if model_key == "stable_diffusion_xl" else 8
    if not check_disk_space(cache_dir, required_space):
        logger.error("磁盘空间不足，无法下载模型")
        return False
    
    # 安装依赖
    if not install_local_dependencies():
        return False
    
    # 下载模型
    model_path = download_local_model(model_key, cache_dir)
    if not model_path:
        return False
    
    # 创建配置文件
    config = {
        "model_type": "local",
        "model_key": model_key,
        "model_path": model_path,
        "device": "cuda" if has_gpu else "cpu",
        "torch_dtype": "float16" if has_gpu else "float32"
    }
    
    config_path = os.path.join(cache_dir, "model_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"本地模型部署成功！配置文件: {config_path}")
    
    # 打印使用指南
    print("\n=== 本地模型使用指南 ===")
    print("\n1. 直接使用本地模型（推荐）:")
    print("```python")
    print("from models.ecology_image_generator import EcologyImageGenerator")
    print("")
    print("# 创建生成器实例（自动使用本地模型）")
    print("generator = EcologyImageGenerator()")
    print("generator.build_model()  # 加载本地模型")
    print("")
    print("# 生成图像")
    print("result = generator.generate_from_text('工业污染导致的河流污染场景', style='realistic')")
    print("```")
    
    print("\n2. 手动指定模型路径:")
    print("```python")
    print("from models.ecology_image_generator import EcologyImageGenerator")
    print("")
    print(f"# 使用指定的本地模型")
    print(f"generator = EcologyImageGenerator(local_model_path='{model_path}')")
    print("result = generator.generate_from_text('森林砍伐警示图像')")
    print("```")
    
    print("\n3. 优势:")
    print("   ✅ 完全本地运行，无需网络连接")
    print("   ✅ 数据隐私安全")
    print("   ✅ 生成速度快（特别是有GPU时）")
    print("   ✅ 无API调用限制")
    print("   ✅ 可离线使用")
    
    return True


def list_available_models():
    """列出可用的本地模型"""
    print("\n=== 可用的本地模型 ===")
    for key, info in LOCAL_MODELS.items():
        print(f"\n{key}:")
        print(f"  名称: {info['name']}")
        print(f"  大小: {info['size']}")
        print(f"  描述: {info['description']}")


def main():
    parser = argparse.ArgumentParser(description="部署图像生成工具")
    parser.add_argument("--mode", type=str, choices=["local", "api"], default="local",
                        help="部署模式: local(本地模型,推荐) 或 api(API服务)")
    parser.add_argument("--tool", type=str, choices=["fooocus", "comfyui"], default="fooocus",
                        help="API模式下要部署的工具: fooocus 或 comfyui")
    parser.add_argument("--model", type=str, choices=list(LOCAL_MODELS.keys()), 
                        default="stable_diffusion_v1_5",
                        help="本地模式下要使用的模型")
    parser.add_argument("--install-dir", type=str, default="./image_generation",
                        help="安装目录")
    parser.add_argument("--cache-dir", type=str, default="./models",
                        help="模型缓存目录")
    parser.add_argument("--no-launch", action="store_true",
                        help="API模式下安装后不启动")
    parser.add_argument("--list-models", action="store_true",
                        help="列出可用的本地模型")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    if args.mode == "local":
        # 本地模型部署（推荐）
        success = deploy_local_model(args.model, args.cache_dir)
        if success:
            logger.info("🎉 本地模型部署成功！")
            print("\n💡 提示: 本地模型是推荐的部署方式，具有更好的隐私性和稳定性")
        else:
            logger.error("❌ 本地模型部署失败")
            sys.exit(1)
    
    else:  # API模式
        logger.info("使用API模式部署（需要额外的服务器资源）")
        
        # 检查环境
        if not check_python_version():
            sys.exit(1)
        
        check_gpu()  # 只是检查，不阻止安装
        
        # 创建安装目录
        install_dir = os.path.abspath(args.install_dir)
        os.makedirs(install_dir, exist_ok=True)
        
        # 部署选择的工具
        if args.tool == "fooocus":
            success = deploy_fooocus(os.path.join(install_dir, "fooocus"), not args.no_launch)
        else:  # comfyui
            success = deploy_comfyui(os.path.join(install_dir, "comfyui"), not args.no_launch)
        
        if success:
            logger.info(f"{args.tool.capitalize()} API服务部署成功!")
            
            # 打印集成指南
            print("\n=== API模式集成指南 ===")
            print("```python")
            if args.tool == "fooocus":
                print("from models.ecology_image_generator import EcologyImageGenerator")
                print("")
                print("# 创建生成器实例并配置API")
                print("generator = EcologyImageGenerator(api_url='http://127.0.0.1:7865', api_type='fooocus')")
                print("")
                print("# 生成图像")
                print("result = generator.generate_from_text('工业污染导致的河流污染场景', style='realistic')")
            else:
                print("from models.ecology_image_generator import EcologyImageGenerator")
                print("")
                print("# 创建生成器实例并配置API")
                print("generator = EcologyImageGenerator(api_url='http://127.0.0.1:8188', api_type='comfyui')")
                print("")
                print("# 生成图像")
                print("result = generator.generate_from_text('工业污染导致的河流污染场景', style='realistic')")
            print("```")
            
            print("\n💡 提示: 如果不需要Web界面，建议使用 --mode local 进行本地部署")
        else:
            logger.error(f"{args.tool.capitalize()} 部署失败!")
            sys.exit(1)


if __name__ == "__main__":
    main()