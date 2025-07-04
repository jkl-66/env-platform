@echo off
echo 开始下载SD3.5模型文件...

set MODEL_DIR=.\cache\huggingface\models--stabilityai--stable-diffusion-3.5-large-turbo\snapshots\ec07796fc06b096cc56de9762974a28f4c632eda
set BASE_URL=https://hf-mirror.com/stabilityai/stable-diffusion-3.5-large-turbo/resolve/main

REM 创建目录
mkdir "%MODEL_DIR%\scheduler" 2>nul
mkdir "%MODEL_DIR%	ext_encoder" 2>nul
mkdir "%MODEL_DIR%	ext_encoder_2" 2>nul
mkdir "%MODEL_DIR%	ext_encoder_3" 2>nul
mkdir "%MODEL_DIR%	okenizer" 2>nul
mkdir "%MODEL_DIR%	okenizer_2" 2>nul
mkdir "%MODEL_DIR%	okenizer_3" 2>nul
mkdir "%MODEL_DIR%	ransformer" 2>nul
mkdir "%MODEL_DIR%ae" 2>nul

REM 下载文件（需要安装curl或wget）
echo 请手动下载以下文件到对应目录：
echo.
echo 根目录 (%MODEL_DIR%):
echo   - model_index.json
echo.
echo scheduler目录:
echo   - scheduler_config.json
echo.
echo text_encoder目录:
echo   - config.json
echo   - model.safetensors
echo.
echo text_encoder_2目录:
echo   - config.json
echo   - model.safetensors
echo.
echo text_encoder_3目录:
echo   - config.json
echo   - model.safetensors
echo.
echo tokenizer目录:
echo   - tokenizer_config.json
echo   - tokenizer.json
echo.
echo tokenizer_2目录:
echo   - tokenizer_config.json
echo   - tokenizer.json
echo.
echo tokenizer_3目录:
echo   - tokenizer_config.json
echo   - tokenizer.json
echo.
echo transformer目录:
echo   - config.json
echo   - diffusion_pytorch_model-00001-of-00002.safetensors
echo   - diffusion_pytorch_model-00002-of-00002.safetensors
echo   - diffusion_pytorch_model.safetensors.index.json
echo.
echo vae目录:
echo   - config.json
echo   - diffusion_pytorch_model.safetensors
echo.
echo 从以下地址下载: %BASE_URL%/[文件路径]
echo.
pause
