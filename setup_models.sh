#!/bin/bash

# 下载Janus-7B-Pro模型到本地（假设通过huggingface-cli下载）
echo "开始下载Janus-7B-Pro..."
huggingface-cli download --repo-id https://huggingface.co/deepseek-ai/Janus-Pro-7B --local-dir ./Janus-7B-Pro --local-dir-use-symlinks False

echo "Janus-7B-Pro下载完成，保存至 ./Janus-7B-Pro 文件夹。"

# 检查是否安装了Ollama，没有则进行安装
if ! command -v ollama &> /dev/null
then
    echo "未检测到 Ollama，正在安装 Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama已安装，跳过安装步骤。"
fi

# 使用Ollama下载qwq32b模型
echo "开始使用Ollama下载 qwq32b..."
ollama pull qwq:32b

echo "qwq32b下载完成。"

echo "所有任务已完成。"
