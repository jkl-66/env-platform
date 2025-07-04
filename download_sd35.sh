#!/bin/bash
# SD3.5模型下载脚本

echo "开始下载SD3.5模型文件..."

# 设置变量
MODEL_DIR="./cache/huggingface/models--stabilityai--stable-diffusion-3.5-large-turbo/snapshots/ec07796fc06b096cc56de9762974a28f4c632eda"
BASE_URL="https://hf-mirror.com/stabilityai/stable-diffusion-3.5-large-turbo/resolve/main"

# 创建目录
mkdir -p "$MODEL_DIR/scheduler"
mkdir -p "$MODEL_DIR/text_encoder"
mkdir -p "$MODEL_DIR/text_encoder_2"
mkdir -p "$MODEL_DIR/text_encoder_3"
mkdir -p "$MODEL_DIR/tokenizer"
mkdir -p "$MODEL_DIR/tokenizer_2"
mkdir -p "$MODEL_DIR/tokenizer_3"
mkdir -p "$MODEL_DIR/transformer"
mkdir -p "$MODEL_DIR/vae"

# 下载函数
download_file() {
    local file_path="$1"
    local url="$BASE_URL/$file_path"
    local output="$MODEL_DIR/$file_path"
    
    echo "下载: $file_path"
    curl -L -o "$output" "$url" || wget -O "$output" "$url"
    
    if [ $? -eq 0 ]; then
        echo "✅ $file_path 下载完成"
    else
        echo "❌ $file_path 下载失败"
    fi
}

# 下载所有文件
download_file "model_index.json"
download_file "scheduler/scheduler_config.json"
download_file "text_encoder/config.json"
download_file "text_encoder/model.safetensors"
download_file "text_encoder_2/config.json"
download_file "text_encoder_2/model.safetensors"
download_file "text_encoder_3/config.json"
download_file "text_encoder_3/model.safetensors"
download_file "tokenizer/tokenizer_config.json"
download_file "tokenizer/tokenizer.json"
download_file "tokenizer_2/tokenizer_config.json"
download_file "tokenizer_2/tokenizer.json"
download_file "tokenizer_3/tokenizer_config.json"
download_file "tokenizer_3/tokenizer.json"
download_file "transformer/config.json"
download_file "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
download_file "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
download_file "transformer/diffusion_pytorch_model.safetensors.index.json"
download_file "vae/config.json"
download_file "vae/diffusion_pytorch_model.safetensors"

echo "下载完成！"
