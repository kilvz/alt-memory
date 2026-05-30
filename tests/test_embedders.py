"""Tests for all embedder variants with both backends."""

import os
import shutil
import tempfile

import pytest


BACKENDS = ["faiss", "chroma"]
EMBEDDERS = ["numpy", "minilm", "bert"]


def _make_dim(tmpdir: str, backend: str, embedder: str):
    """Create a dimension with the given backend and embedder."""
    from alt_memory.dimension import Dimension

    d = Dimension(path=tmpdir, backend=backend)
    d.init()
    d.set_embedder(embedder, reindex=False)
    return d


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("embedder", EMBEDDERS)
def test_embedder_add_and_search(backend: str, embedder: str):
    tmp = tempfile.mkdtemp(prefix=f"_test_{embedder}_{backend}_")
    try:
        d = _make_dim(tmp, backend, embedder)
        eid = d.add_entity("test", "demo", f"hello from {embedder} on {backend}")
        assert eid is not None
        results = d.search("hello", n_results=5)
        assert len(results) >= 1
        # Second add to verify consistency
        eid2 = d.add_entity("test", "demo", "second entity for overlap")
        assert eid2 is not None
        results2 = d.search("second", n_results=5)
        assert len(results2) >= 1
        d.close()
    except Exception as e:
        pytest.fail(f"{embedder} + {backend} failed: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_switch_embedder_mid_life(backend: str):
    """Switch embedders on an existing dimension with data."""
    tmp = tempfile.mkdtemp(prefix=f"_test_switch_{backend}_")
    try:
        from alt_memory.dimension import Dimension

        d = Dimension(path=tmp, backend=backend)
        d.init()
        # Add with numpy first
        d.set_embedder("numpy", reindex=False)
        d.add_entity("test", "demo", "numpy embedded entity")
        # Switch to minilm and reindex
        d.set_embedder("minilm", reindex=True)
        results = d.search("numpy", n_results=5)
        assert len(results) >= 1
        d.close()
    except Exception as e:
        pytest.fail(f"switch_embedder {backend} failed: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
