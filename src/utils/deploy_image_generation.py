import os
import sys
import subprocess
import platform
import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("deploy_image_generation")

# 定义常量
FOOOCUS_REPO = "https://github.com/lllyasviel/Fooocus.git"
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"


def check_gpu():
    """检查GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"检测到GPU: {gpu_name}")
            return True
        else:
            logger.warning("未检测到GPU，图像生成速度可能较慢")
            return False
    except ImportError:
        logger.warning("未安装PyTorch，无法检测GPU")
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


def main():
    parser = argparse.ArgumentParser(description="部署图像生成工具")
    parser.add_argument("--tool", type=str, choices=["fooocus", "comfyui"], default="fooocus",
                        help="要部署的工具: fooocus 或 comfyui")
    parser.add_argument("--install-dir", type=str, default="./image_generation",
                        help="安装目录")
    parser.add_argument("--no-launch", action="store_true",
                        help="安装后不启动")
    
    args = parser.parse_args()
    
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
        logger.info(f"{args.tool.capitalize()} 部署成功!")
        
        # 打印集成指南
        print("\n集成到生态图像生成器的示例代码:")
        print("```python")
        if args.tool == "fooocus":
            print("from models.ecology_image_generator import EcologyImageGenerator")
            print("\n# 创建生成器实例并配置API")
            print("generator = EcologyImageGenerator(api_url='http://127.0.0.1:7865', api_type='fooocus')")
            print("\n# 生成图像")
            print("result = generator.generate_from_text('工业污染导致的河流污染场景', style='realistic')")
        else:
            print("from models.ecology_image_generator import EcologyImageGenerator")
            print("\n# 创建生成器实例并配置API")
            print("generator = EcologyImageGenerator(api_url='http://127.0.0.1:8188', api_type='comfyui')")
            print("\n# 生成图像")
            print("result = generator.generate_from_text('工业污染导致的河流污染场景', style='realistic')")
        print("```")
    else:
        logger.error(f"{args.tool.capitalize()} 部署失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()