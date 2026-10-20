"""
Builds a LlamaIndex VectorStoreIndex over the policy documents using the
T5EncoderEmbedding wrapper, and exposes a retrieve() function returning
top-k policy chunks with similarity scores.
"""
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from src.embeddings import T5EncoderEmbedding

POLICY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "policies")

_index = None


def get_index():
    global _index
    if _index is None:
        Settings.embed_model = T5EncoderEmbedding()
        docs = SimpleDirectoryReader(POLICY_DIR).load_data()
        _index = VectorStoreIndex.from_documents(docs)
    return _index


def retrieve(query: str, top_k: int = 2):
    index = get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    results = []
    for n in nodes:
        results.append(
            {
                "text": n.node.get_content(),
                "score": float(n.score) if n.score is not None else None,
                "file_name": n.node.metadata.get("file_name", "unknown"),
            }
        )
    return results
