import os
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader


def setup_engine():
    # 1. Setup the LLM (The Brain)
    # Using the model I just pulled
    llm = Ollama(model="qwen2.5-coder:1.5b", request_timeout=120.0)

    # 2. Setup the Embedding Model (The Translator)
    # This turns my code into vectors for the Semantic Cache
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 3. Configure LlamaIndex to use these tools globally
    Settings.llm = llm
    Settings.embed_model = embed_model

    print("✅ AI Engine Initialized (CPU Mode)")


def load_repository(path="/home/akshaybajetha/Desktop/core_projects/bot/api_new"):
    print(f"📂 Scanning directory: {path}...")

    # This reads my local files while ignoring the virtual env
    documents = SimpleDirectoryReader(
        path,
        recursive=True,
        required_exts=[".py", ".md", ".txt"],
        exclude=["repo_mind_env/*", ".git/*", "node_modules/*", "venv/*", "__pycache__/*", "*.pyc", "dist/*", "build/*"]
    ).load_data()
    print(f"Found {len(documents)} document chunks. Starting Indexing...")
    # Create a searchable index from my files
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index.as_query_engine()


if __name__ == "__main__":
    setup_engine()
    # Simple test to see if it can read its own code
    query_engine = load_repository()
    response = query_engine.query("Find where the environment variables or secrets are loaded and check if they are handled safely.")
    print(f"\n🤖 AI Response:\n{response}")