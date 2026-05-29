"""Backend implementations: FAISS vector store, ChromaDB vector store, TF-IDF embedder, SQLite knowledge graph."""

from alt_memory.backends.embedder import (
    NumpyEmbedder, SentenceTransformerEmbedder, SpacyGloveEmbedder, get_embedder,
)
from alt_memory.backends.faiss_store import FaissStore
from alt_memory.backends.chroma_store import ChromaStore
from alt_memory.backends.knowledge_graph import KnowledgeGraph

__all__ = [
    "NumpyEmbedder",
    "SentenceTransformerEmbedder",
    "SpacyGloveEmbedder",
    "get_embedder",
    "FaissStore",
    "ChromaStore",
    "KnowledgeGraph",
]
