"""Shared fixtures for alt-memory tests.

Sets the default embedder to ``numpy`` so tests run without downloading
SentenceTransformer models.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Force numpy embedder for all tests — avoids model downloads
os.environ.setdefault("ALT_DEFAULT_EMBEDDER", "numpy")


@pytest.fixture(scope="function")
def dim_path() -> str:
    """Return a temporary directory path for a dimension."""
    tmp = tempfile.mkdtemp(prefix="_test_dim_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="function")
def fresh_dim(dim_path: str):
    """Return an initialized Dimension with numpy embedder."""
    from alt_memory.dimension import Dimension

    d = Dimension(path=dim_path, backend="faiss")
    d.init()
    yield d
    try:
        d.close()
    except Exception:
        pass


@pytest.fixture(scope="function")
def dim_with_data(fresh_dim):
    """Add a few entities to the dimension for search/query tests."""
    ids = []
    for i, text in enumerate([
        "hello world this is a test document",
        "machine learning and artificial intelligence",
        "python programming for data science",
        "the quick brown fox jumps over the lazy dog",
        "deep neural networks and backpropagation",
        "natural language processing with transformers",
        "computer vision and image recognition",
        "reinforcement learning for game playing",
        "statistical analysis and probability theory",
        "database systems and query optimization",
    ]):
        rid = fresh_dim.add_entity(
            "test", "demo", text,
            metadata={"idx": i, "topic": "tech" if i < 7 else "math"},
        )
        ids.append(rid)
    return fresh_dim, ids
