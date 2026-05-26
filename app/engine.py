import os
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import PromptTemplate

from cache import init_persistent_index

def setup_engine():
    # 1. Setup the LLM (The Brain)
    # Using the model I just pulled
    llm = Ollama(model="qwen2.5-coder:1.5b", request_timeout=600.0)

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
    repo_name = os.path.basename(path)
    print(repo_name)
    index = init_persistent_index(documents, index_name=f"repo_{repo_name}")

    print(f"Found {len(documents)} document chunks. Starting Indexing...")
    template = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
        "If the answer is not present in the code context, "
        "state that you do not know. Do not use external knowledge."
    )
    custom_qa_template = PromptTemplate(template)

    # --- INITIALIZE QUERY ENGINE WITH STREAMING ---
    # similarity_top_k=5 is the 'sweet spot' for CPU speed vs accuracy
    return index.as_query_engine(
        text_qa_template=custom_qa_template,
        similarity_top_k=5,
        streaming=True
    )
    # Create a searchable index from my files
    # index = VectorStoreIndex.from_documents(documents, show_progress=True)
    # return index.as_query_engine()

def test_pure_model_speed():
    print("Testing model speed WITHOUT repository context...")

    # Just initialize the LLM directly, skipping the index
    # (Assuming you're using LlamaIndex/LangChain)
    llm = Ollama(model="qwen2.5-coder:1.5b", streaming=True)

    response = llm.stream_complete("Hello, can you hear me?")
    for token in response:
        print(token.delta, end="", flush=True)


def test_pure_model_speed_2():
    print("\n⚡ Testing LlamaIndex Direct Query (No Context)...")

    # Access the LLM from the global Settings
    llm = Settings.llm

    # 'complete' is for a single prompt, 'chat' is for a conversation history
    # We use stream_complete to see the word-by-word speed
    response = llm.stream_complete("Explain the difference between a Python list and a tuple.")

    print("🤖 AI Response: ", end="", flush=True)
    for token in response:
        # 'delta' contains the new piece of text generated in this step
        print(token.delta, end="", flush=True)
    print("\n")
import time

if __name__ == "__main__":
    setup_engine()
    # Simple test to see if it can read its own code
    # test_pure_model_speed()

    start = time.time()
    query_engine = load_repository()
    print(f"Repository loaded in {time.time() - start:.2f} seconds")

    start = time.time()
    response = query_engine.query("Review the error handling in my api_new logic and suggest improvements.")
    print(f"Retrieval finished in {time.time() - start:.2f} seconds")
    response.print_response_stream()
    print("\n")



