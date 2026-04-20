#!/bin/bash

echo "🚀 Starting RepoMind System Setup..."

# 1. Install System Dependencies
echo "📦 Installing system dependencies (zstd, redis)..."
sudo apt-get update
sudo apt-get install -y zstd redis-server git curl

# 2. Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "🧠 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Ollama already installed."
fi

# 3. Pull AI Models
echo "📥 Pulling Qwen2.5-Coder model..."
ollama pull qwen2.5-coder:1.5b

# 4. Python Environment Setup
echo "🐍 Setting up Python virtual environment..."
python3 -m venv repo_mind_env
source repo_mind_env/bin/activate
pip install --upgrade pip
pip install llama-index-llms-ollama llama-index-embeddings-huggingface llama-index-vector-stores-redis redis fastapi uvicorn

# 5. Start Redis
echo "🔄 Ensuring Redis is running..."
sudo service redis-server start

echo "✨ Setup Complete! Run 'source repo_mind_env/bin/activate' to start."