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

    # BETTER CHECK: Does the index exist AND have documents?
    index_exists = vector_store._index.exists()

    # Check if there are actually keys in this index
    doc_count = 0
    if index_exists:
        # Get count of documents in the index
        stats = vector_store._index.info()
        doc_count = int(stats.get('num_docs', 0))

    if index_exists and doc_count > 0:
        print(f"🚀 [CACHE HIT] Found {doc_count} vectors. Loading...")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    else:
        print(f"⚠️ [CACHE MISS] Index empty or missing. Embedding {len(documents)} docs...")
        # Force a fresh index if it was empty
        storage_context = cache_manager.get_storage_context()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

    return index