"""Tests for FaissStore with IndexIDMap (v4.3.1+)."""

from pathlib import Path

import faiss
import numpy as np
import pytest

from alt_memory.backends.faiss_store import FaissStore
from alt_memory.backends.types import DEFAULT_DIM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> FaissStore:
    s = FaissStore(str(tmp_path), dimension=DEFAULT_DIM)
    return s


@pytest.fixture
def populated_store(store: FaissStore) -> FaissStore:
    texts = [f"doc {i} content here" for i in range(10)]
    ids = [store.next_id() for _ in range(10)]
    metas = [{"idx": i} for i in range(10)]
    emb = _fake_embeddings(10)
    store.add(ids, texts, metas, emb)
    return store


def _fake_embeddings(n: int, dim: int = DEFAULT_DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    faiss.normalize_L2(emb)
    return emb


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_indexidmap(self, store: FaissStore):
        assert isinstance(store.index, faiss.IndexIDMap)

    def test_persists_and_reloads(self, tmp_path: Path):
        s1 = FaissStore(str(tmp_path))
        s1.close()
        s2 = FaissStore(str(tmp_path))
        assert isinstance(s2.index, faiss.IndexIDMap)
        s2.close()

    def test_ntotal_zero(self, store: FaissStore):
        assert store.index.ntotal == 0
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Add / Search
# ---------------------------------------------------------------------------


class TestAddAndSearch:
    def test_add_single(self, store: FaissStore):
        eid = store.next_id()
        emb = _fake_embeddings(1)
        store.add([eid], ["hello world"], [{"k": "v"}], emb)
        assert store.count() == 1

    def test_search_returns_results(self, populated_store: FaissStore):
        q = _fake_embeddings(1)
        ids, texts, dists, metas = populated_store.search(q, n_results=5)
        assert len(ids) == 5
        assert len(texts) == 5
        assert len(dists) == 5
        assert len(metas) == 5

    def test_search_empty(self, store: FaissStore):
        q = _fake_embeddings(1)
        ids, texts, dists, metas = store.search(q)
        assert ids == []
        assert texts == []
        assert dists == []
        assert metas == []

    def test_search_with_where(self, populated_store: FaissStore):
        q = _fake_embeddings(1)
        ids, texts, dists, metas = populated_store.search(q, n_results=10, where={"idx": 0})
        assert len(ids) >= 1
        for m in metas:
            assert m.get("idx") == 0

    def test_search_preserves_order(self, populated_store: FaissStore):
        q = _fake_embeddings(1)
        ids, texts, dists, metas = populated_store.search(q, n_results=10)
        if len(dists) >= 2:
            # For inner product, higher distance = more similar
            for i in range(len(dists) - 1):
                assert dists[i] >= dists[i + 1] - 1e-6, (
                    f"dist[{i}]={dists[i]} < dist[{i+1}]={dists[i+1]}"
                )


# ---------------------------------------------------------------------------
# Delete (O(1) with IndexIDMap — no full rebuild)
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_by_id(self, populated_store: FaissStore):
        count_before = populated_store.count()
        ids, _, _, _ = populated_store.search(_fake_embeddings(1), n_results=10)
        target = ids[0]
        populated_store.delete(ids=[target])
        assert populated_store.count() == count_before - 1
        # Verify not in search results
        ids2, _, _, _ = populated_store.search(_fake_embeddings(1), n_results=10)
        assert target not in ids2

    def test_delete_all(self, populated_store: FaissStore):
        ids, _, _, _ = populated_store.search(_fake_embeddings(1), n_results=10)
        populated_store.delete(ids=ids)
        assert populated_store.count() == 0
        assert populated_store.index.ntotal == 0  # IndexIDMap should be empty

    def test_delete_nonexistent(self, populated_store: FaissStore):
        count_before = populated_store.count()
        populated_store.delete(ids=["nonexistent"])
        assert populated_store.count() == count_before

    def test_delete_by_where(self, populated_store: FaissStore):
        populated_store.delete(where={"idx": 0})
        ids, texts, dists, metas = populated_store.search(
            _fake_embeddings(1), n_results=10,
        )
        # None of the remaining docs should have idx=0
        for m in metas:
            assert m.get("idx") != 0, f"idx=0 still present: {m}"


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


class TestUpsert:
    def test_upsert_new(self, store: FaissStore):
        eid = store.next_id()
        emb = _fake_embeddings(1)
        store.upsert([eid], ["new doc"], [{"k": "v"}], emb)
        assert store.count() == 1

    def test_upsert_existing(self, populated_store: FaissStore):
        count_before = populated_store.count()
        ids, _, _, _ = populated_store.search(_fake_embeddings(1), n_results=10)
        target = ids[0]
        # Upsert with same ID but different content
        new_emb = _fake_embeddings(1)
        populated_store.upsert([target], ["updated content"], [{"k": "v2"}], new_emb)
        # Count should stay the same (replace, not add)
        assert populated_store.count() == count_before

    def test_upsert_mixed(self, populated_store: FaissStore):
        count_before = populated_store.count()
        ids, _, _, _ = populated_store.search(_fake_embeddings(1), n_results=10)
        # Upsert 2 existing + 2 new
        mixed_ids = ids[:2] + [populated_store.next_id(), populated_store.next_id()]
        new_emb = _fake_embeddings(4)
        populated_store.upsert(mixed_ids, ["a", "b", "c", "d"], [{}, {}, {}, {}], new_emb)
        # Total = existing(10) - replaced(2) + new(4) = 12
        assert populated_store.count() == count_before + 2


# ---------------------------------------------------------------------------
# Persistence / Rebuild
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_rebuild_from_sqlite(self, tmp_path: Path):
        s1 = FaissStore(str(tmp_path))
        ids = [s1.next_id() for _ in range(5)]
        texts = [f"persist test {i}" for i in range(5)]
        metas = [{"i": i} for i in range(5)]
        emb = _fake_embeddings(5)
        s1.add(ids, texts, metas, emb)
        s1.close()

        # Reopen and check recovery path
        s2 = FaissStore(str(tmp_path))
        assert s2.count() == 5
        # Manually corrupt index to test rebuild
        path = Path(tmp_path) / "index.faiss"
        path.unlink()
        s2._rebuild_index()
        assert s2.index.ntotal == 5
        q = _fake_embeddings(1)
        found_ids, _, _, _ = s2.search(q, n_results=10)
        assert len(found_ids) == 5
        s2.close()


# ---------------------------------------------------------------------------
# Schema migration (pos_map → faiss_id)
# ---------------------------------------------------------------------------


class TestMigration:
    def test_legacy_pos_map_is_dropped(self, tmp_path: Path):
        """Simulate an old DB without faiss_id column, verify migration."""
        # Create legacy-style DB
        import sqlite3
        db_path = Path(tmp_path) / "metadata.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE docs (id TEXT PRIMARY KEY, text TEXT, metadata TEXT, vector BLOB, rowid INTEGER)")
        conn.execute("CREATE TABLE pos_map (faiss_pos INTEGER PRIMARY KEY, doc_id TEXT NOT NULL)")
        conn.execute("INSERT INTO docs (id, text, metadata, rowid) VALUES ('legacy_1', 'test', '{}', 1)")
        conn.commit()
        conn.close()

        # Open with new code — should migrate
        store = FaissStore(str(tmp_path))
        # Verify pos_map is gone
        tables = set(
            r[0] for r in store._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            ).fetchall()
        )
        assert "pos_map" not in tables, "legacy pos_map table should be dropped"
        # Verify faiss_id column exists
        cols = {r[1] for r in store._db.execute("PRAGMA table_info(docs)").fetchall()}
        assert "faiss_id" in cols
        store.close()
