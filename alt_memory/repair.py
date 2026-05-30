"""FAISS-specific repair — scan, prune, rebuild, health status.

FAISS FlatIP + SQLite repair operations.
No onnxruntime dependency — works with NumpyEmbedder only.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from alt_memory.repair_utils import (
    rebuild_fts5,
    run_vacuum,
    sqlite_entity_count,
    sqlite_integrity_errors,
)

logger = logging.getLogger(__name__)


def _get_dim_path(dim_path: Optional[str] = None) -> str:
    if dim_path:
        return dim_path
    default = os.path.join(os.path.expanduser("~"), ".alt-memory")
    return os.environ.get("ALT_MEMORY_PATH", default)


def _open_dimension_db(dim_path: str) -> sqlite3.Connection:
    base = Path(dim_path).expanduser().resolve()
    conn = sqlite3.connect(str(base / "dimension.db"), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _get_data_dir(dim_path: str) -> str:
    return str(Path(dim_path).expanduser().resolve() / "data")


def _open_meta_db(data_dir: str) -> sqlite3.Connection:
    db_path = Path(data_dir) / "metadata.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"metadata.db not found at {db_path} — dimension not initialized or data directory missing"
        )
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class TruncationDetected(Exception):
    """Raised by check_extraction_safety when extraction looks truncated."""

    def __init__(self, message: str, sqlite_count: Optional[int], extracted: int):
        super().__init__(message)
        self.message = message
        self.sqlite_count = sqlite_count
        self.extracted = extracted


# ── Status ────────────────────────────────────────────────────────────────


def status(dim_path: Optional[str] = None) -> dict:
    """Health check: compare SQLite vs vector store counts, run integrity checks.

    Supports both faiss and chroma backends. Returns a dict with ``ok``,
    ``entities_sqlite``, ``vectors`` (vector store count), ``backend``,
    ``integrity_errors``, and a human-readable ``message``.
    """
    dim_path = _get_dim_path(dim_path)
    base = Path(dim_path).expanduser().resolve()

    sqlite_db = base / "dimension.db"
    data_dir = base / "data"

    result: dict = {
        "path": str(base),
        "ok": False,
        "entities_sqlite": 0,
        "vectors": 0,
        "backend": "unknown",
        "integrity_errors": [],
        "message": "",
    }

    integrity = sqlite_integrity_errors(str(sqlite_db))
    if integrity:
        result["integrity_errors"] = integrity
        result["message"] = f"SQLite integrity errors: {integrity[0][:80]}"
        return result
    result["entities_sqlite"] = sqlite_entity_count(str(sqlite_db), table="entities") or 0

    config_path = base / "dimension.json"
    backend = "faiss"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            backend = cfg.get("backend", "faiss")
        except Exception:
            pass
    result["backend"] = backend

    store = None
    try:
        if backend == "chroma":
            chroma_sqlite = data_dir / "chroma.sqlite3"
            if not chroma_sqlite.exists():
                result["message"] = "ChromaDB data not found — dimension not initialized"
                return result
            from alt_memory.backends.chroma_store import ChromaStore
            store = ChromaStore(str(data_dir))
        else:
            index_path = data_dir / "index.faiss"
            if not index_path.exists():
                result["message"] = f"{backend.upper()} index not found — dimension not initialized"
                return result
            from alt_memory.backends.faiss_store import FaissStore
            store = FaissStore(str(data_dir))

        result["vectors"] = store.count()
        store.close()
        store = None
    except Exception as e:
        if store:
            try:
                store.close()
            except Exception:
                pass
        result["message"] = f"{backend.upper()} error: {e}"
        return result

    if result["entities_sqlite"] != result["vectors"]:
        result["message"] = (
            f"Mismatch: SQLite {result['entities_sqlite']} entities "
            f"vs {backend.upper()} {result['vectors']} vectors"
        )
    else:
        result["ok"] = True
        result["message"] = "Healthy"
    return result


# ── Scan ──────────────────────────────────────────────────────────────────


def scan_dimension(dim_path: Optional[str] = None) -> tuple[set[str], set[str]]:
    """Cross-reference SQLite entities vs vector store.

    Probes the vector store in batches with per-ID fallback on failure —
    a single corrupt entry can't cascade and hide other IDs in the same batch.

    Returns ``(good_ids, bad_ids)`` where ``bad_ids`` are IDs present in
    one store but not the other. Writes ``corrupt_ids.txt`` to the dimension
    directory when bad IDs are found.
    """
    dim_path = _get_dim_path(dim_path)
    base = Path(dim_path).expanduser().resolve()
    data_dir = _get_data_dir(dim_path)
    print(f"\n  Dimension: {base}")

    sqlite_ids: set[str] = set()
    db = _open_dimension_db(str(base))
    try:
        for row in db.execute("SELECT id FROM entities"):
            sqlite_ids.add(row[0])
    finally:
        db.close()
    print(f"  SQLite entities: {len(sqlite_ids):,}")

    if not sqlite_ids:
        print("  Nothing to scan.")
        return set(), set()

    config_path = base / "dimension.json"
    backend = "faiss"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            backend = cfg.get("backend", "faiss")
        except Exception:
            pass

    vector_ids: set[str] = set()
    print(f"\n  Probing {backend.upper()} store (batches of 100)...")
    t0 = time.time()

    if backend == "chroma":
        from alt_memory.backends.chroma_store import ChromaStore
        store = ChromaStore(str(data_dir))
        try:
            all_sorted = sorted(sqlite_ids)
            batch = 100
            for i in range(0, len(all_sorted), batch):
                chunk = all_sorted[i : i + batch]
                try:
                    ids, _, _ = store.get(ids=chunk)
                    for got in ids:
                        vector_ids.add(got)
                except Exception:
                    for sid in chunk:
                        try:
                            ids, _, _ = store.get(ids=[sid])
                            if ids:
                                vector_ids.add(sid)
                        except Exception:
                            pass
        finally:
            store.close()
    else:
        meta = _open_meta_db(data_dir)
        try:
            all_sorted = sorted(sqlite_ids)
            batch = 100
            for i in range(0, len(all_sorted), batch):
                chunk = all_sorted[i : i + batch]
                ph = ",".join("?" * len(chunk))
                try:
                    for row in meta.execute(
                        f"SELECT id FROM docs WHERE id IN ({ph})", chunk
                    ):
                        vector_ids.add(row[0])
                except Exception:
                    for sid in chunk:
                        try:
                            row = meta.execute(
                                "SELECT id FROM docs WHERE id = ?", (sid,)
                            ).fetchone()
                            if row:
                                vector_ids.add(sid)
                        except Exception:
                            pass
        finally:
            meta.close()

    elapsed = time.time() - t0
    print(f"  Probed in {elapsed:.1f}s")
    print(f"  {backend.upper()} vectors found: {len(vector_ids):,}")

    good = sqlite_ids & vector_ids
    bad = (sqlite_ids - vector_ids) | (vector_ids - sqlite_ids)

    print("\n  Scan complete.")
    print(f"  GOOD: {len(good):,}")
    print(f"  BAD:  {len(bad):,}")

    if bad:
        bad_file = base / "corrupt_ids.txt"
        with open(str(bad_file), "w") as f:
            for bid in sorted(bad):
                f.write(bid + "\n")
        print(f"  Bad IDs written to: {bad_file}")

    return good, bad


# ── Prune ─────────────────────────────────────────────────────────────────


def prune_corrupt(dim_path: Optional[str] = None, confirm: bool = False) -> None:
    """Delete corrupt IDs listed in ``corrupt_ids.txt``.

    Removes from both SQLite (``entities`` + ``entities_fts``) and FAISS.
    """
    dim_path = _get_dim_path(dim_path)
    base = Path(dim_path).expanduser().resolve()
    data_dir = _get_data_dir(dim_path)
    bad_file = base / "corrupt_ids.txt"

    if not bad_file.exists():
        print("  No corrupt_ids.txt found — run scan first.")
        return

    with open(str(bad_file)) as f:
        bad_ids = [line.strip() for line in f if line.strip()]
    print(f"  {len(bad_ids):,} corrupt IDs queued for deletion")

    if not confirm:
        print("\n  DRY RUN — no deletions performed.")
        print("  Re-run with --confirm to actually delete.")
        return

    db = _open_dimension_db(str(base))

    config_path = base / "dimension.json"
    backend = "faiss"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            backend = cfg.get("backend", "faiss")
        except Exception:
            pass

    if backend == "chroma":
        from alt_memory.backends.chroma_store import ChromaStore
        store = ChromaStore(data_dir)
    else:
        from alt_memory.backends.faiss_store import FaissStore
        store = FaissStore(data_dir)
    try:
        before = store.count()
        print(f"  Collection size before: {before:,}")

        batch = 100
        for i in range(0, len(bad_ids), batch):
            chunk = bad_ids[i : i + batch]
            ph = ",".join("?" * len(chunk))
            store.delete(ids=chunk)
            db.execute(f"DELETE FROM entities WHERE id IN ({ph})", chunk)
            db.execute(f"DELETE FROM entities_fts WHERE id IN ({ph})", chunk)
            db.commit()

        after = store.count()
        print(f"\n  Deleted: {len(bad_ids):,}")
        print(f"  Collection size: {before:,} \u2192 {after:,}")
        bad_file.unlink()
        print("  corrupt_ids.txt removed.")
    finally:
        store.close()
        db.close()


# ── Extraction safety ─────────────────────────────────────────────────────


def check_extraction_safety(dim_path: str, extracted: int) -> None:
    """Verify extraction count matches SQLite ground truth.

    Raises :class:`TruncationDetected` when extracted < SQLite count.
    """
    sqlite_count = sqlite_entity_count(
        str(Path(dim_path).expanduser().resolve() / "dimension.db")
    )
    if sqlite_count is None:
        print("  WARNING: cannot read SQLite count — skipping safety check")
        return
    if extracted < sqlite_count:
        msg = (
            f"Extraction returned {extracted} entities but SQLite has "
            f"{sqlite_count}. This may indicate truncation."
        )
        raise TruncationDetected(msg, sqlite_count, extracted)
    print(f"  Extraction verified: {extracted} == {sqlite_count} (SQLite)")


# ── Rebuild index (from stored vectors) ────────────────────────────────────


def rebuild_index(
    dim_path: Optional[str] = None,
    rebuild_fts: bool = True,
    vacuum: bool = True,
) -> int:
    """Rebuild FAISS index from stored vector blobs.

    Uses ``FaissStore._rebuild_index()`` which reads vectors from
    ``metadata.db`` and rebuilds ``index.faiss``. Also optionally
    rebuilds FTS5 and vacuums ``dimension.db``.

    Returns the number of vectors rebuilt.
    """
    dim_path = _get_dim_path(dim_path)
    base = Path(dim_path).expanduser().resolve()
    data_dir = _get_data_dir(dim_path)
    dim_db = base / "dimension.db"

    from alt_memory.backends.faiss_store import FaissStore

    print(f"\n  Rebuilding FAISS index at: {data_dir}")

    store = FaissStore(data_dir)
    try:
        before = store.count()
        print(f"  Vectors before rebuild: {before:,}")

        store._rebuild_index()
        store._save()
        after = store.count()

        print(f"  Vectors after rebuild: {after:,}")

        if rebuild_fts:
            print("  Rebuilding FTS5 index...")
            rebuild_fts5(str(dim_db), fts_table="entities_fts")

        if vacuum:
            print("  Running VACUUM...")
            run_vacuum(str(dim_db))

        return after
    finally:
        store.close()


# ── Full rebuild from SQLite (re-embed) ────────────────────────────────────


def rebuild_from_sqlite(
    dim_path: Optional[str] = None,
    rebuild_fts: bool = True,
) -> int:
    """Full rebuild: re-embed all entities from SQLite ground truth.

    Reads every entity from ``dimension.db``, re-embeds with
    :class:`NumpyEmbedder`, and rebuilds the FAISS index from scratch.
    Slower than :func:`rebuild_index` but recovers from corrupted vector
    blobs or dimensional mismatches.

    Returns the number of vectors rebuilt.
    """
    dim_path = _get_dim_path(dim_path)
    base = Path(dim_path).expanduser().resolve()
    data_dir = _get_data_dir(dim_path)
    dim_db = base / "dimension.db"

    from alt_memory.backends.embedder import get_embedder
    from alt_memory.backends.faiss_store import FaissStore

    embedder = get_embedder()

    print(f"\n  Full rebuild from SQLite: {base}")

    db = _open_dimension_db(str(base))
    try:
        rows = db.execute(
            "SELECT id, content, metadata FROM entities ORDER BY rowid"
        ).fetchall()
    finally:
        db.close()

    if not rows:
        print("  No entities found — nothing to rebuild.")
        return 0

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    metadatas = [json.loads(r[2] or "{}") for r in rows]

    print(f"  Loaded {len(ids):,} entities from SQLite")
    print(f"  Embedding {len(texts):,} texts...")
    t0 = time.time()
    all_embeddings = embedder.embed(texts)
    elapsed = time.time() - t0
    rate = len(texts) / max(elapsed, 0.01)
    print(f"  Embedded in {elapsed:.1f}s ({rate:.0f}/s)")

    store = FaissStore(data_dir)
    try:
        for tbl in ("pos_map", "docs"):
            store._db.execute(f"DELETE FROM {tbl}")
        store._db.commit()

        import faiss

        store.index = faiss.IndexFlatIP(store.dimension)
        store.add(ids=ids, texts=texts, metadatas=metadatas, embeddings=all_embeddings)

        after = store.count()
        print(f"  Rebuilt: {after:,} vectors in FAISS")

        if rebuild_fts:
            print("  Rebuilding FTS5 index...")
            rebuild_fts5(str(dim_db), fts_table="entities_fts")

        return after
    finally:
        store.close()
