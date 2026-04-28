import os
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema


class RedisCacheManager:
    def __init__(self, host="127.0.0.1", port=6379, index_name="repo_mind_core"):
        self.redis_url = f"redis://{host}:{port}"
        self.index_name = index_name

    def get_vector_store(self):
        """
        Sets up the Redis Vector Store using the official IndexSchema.
        """
        # Define the schema structure required by RedisVL and LlamaIndex
        custom_schema = IndexSchema.from_dict({
            "index": {
                "name": self.index_name,
                "prefix": "repomind"
            },
            "fields": [
                # Required fields for LlamaIndex
                {"name": "id", "type": "tag"},
                {"name": "doc_id", "type": "tag"},
                {"name": "text", "type": "text"},
                # Vector field for BAAI/bge-small-en-v1.5 (384 dimensions)
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "dims": 384,
                        "algorithm": "hnsw",  # HNSW is better for speed/scaling
                        "distance_metric": "cosine"
                    }
                }
            ]
        })

        return RedisVectorStore(
            schema=custom_schema,
            redis_url=self.redis_url,
            overwrite=False
        )

    def get_storage_context(self):
        """Creates the storage context linked to Redis."""
        vector_store = self.get_vector_store()
        return StorageContext.from_defaults(vector_store=vector_store)


def init_persistent_index(documents, index_name="repo_mind_core"):
    cache_manager = RedisCacheManager(index_name=index_name)
    vector_store = cache_manager.get_vector_store()

    # CHECK: Does the index actually have data in Redis?
    # We use the internal redis_client from redisvl
    if vector_store._index.exists():
        print(f"🚀 [CACHE HIT] Found existing vectors for {index_name}. Loading instantly...")
        # We initialize the index directly from the vector_store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )
    else:
        print(f"⚠️ [CACHE MISS] No vectors found for {index_name}. Embedding now...")
        storage_context = cache_manager.get_storage_context()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

    return index