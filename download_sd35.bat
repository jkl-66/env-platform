@echo off
echo ��ʼ����SD3.5ģ���ļ�...

set MODEL_DIR=.\cache\huggingface\models--stabilityai--stable-diffusion-3.5-large-turbo\snapshots\ec07796fc06b096cc56de9762974a28f4c632eda
set BASE_URL=https://hf-mirror.com/stabilityai/stable-diffusion-3.5-large-turbo/resolve/main

REM ����Ŀ¼
mkdir "%MODEL_DIR%\scheduler" 2>nul
mkdir "%MODEL_DIR%	ext_encoder" 2>nul
mkdir "%MODEL_DIR%	ext_encoder_2" 2>nul
mkdir "%MODEL_DIR%	ext_encoder_3" 2>nul
mkdir "%MODEL_DIR%	okenizer" 2>nul
mkdir "%MODEL_DIR%	okenizer_2" 2>nul
mkdir "%MODEL_DIR%	okenizer_3" 2>nul
mkdir "%MODEL_DIR%	ransformer" 2>nul
mkdir "%MODEL_DIR%ae" 2>nul

REM �����ļ�����Ҫ��װcurl��wget��
echo ���ֶ����������ļ�����ӦĿ¼��
echo.
echo ��Ŀ¼ (%MODEL_DIR%):
echo   - model_index.json
echo.
echo schedulerĿ¼:
echo   - scheduler_config.json
echo.
echo text_encoderĿ¼:
echo   - config.json
echo   - model.safetensors
echo.
echo text_encoder_2Ŀ¼:
echo   - config.json
echo   - model.safetensors
echo.
echo text_encoder_3Ŀ¼:
echo   - config.json
echo   - model.safetensors
echo.
echo tokenizerĿ¼:
echo   - tokenizer_config.json
echo   - tokenizer.json
echo.
echo tokenizer_2Ŀ¼:
echo   - tokenizer_config.json
echo   - tokenizer.json
echo.
echo tokenizer_3Ŀ¼:
echo   - tokenizer_config.json
echo   - tokenizer.json
echo.
echo transformerĿ¼:
echo   - config.json
echo   - diffusion_pytorch_model-00001-of-00002.safetensors
echo   - diffusion_pytorch_model-00002-of-00002.safetensors
echo   - diffusion_pytorch_model.safetensors.index.json
echo.
echo vaeĿ¼:
echo   - config.json
echo   - diffusion_pytorch_model.safetensors
echo.
echo �����µ�ַ����: %BASE_URL%/[�ļ�·��]
echo.
pause
