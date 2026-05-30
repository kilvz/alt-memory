"""FAISS vector store with SQLite metadata + vector persistence.

Uses ``IndexIDMap`` so delete/upsert are O(1) ``remove_ids()`` instead of
full index rebuilds. The ``pos_map`` table was dropped in v4.3.1 — each doc
stores its stable ``faiss_id`` directly.
"""

import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def _sql_val(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return v
    return str(v)


from alt_memory.backends.base import DEFAULT_DIM


class FaissStore:
    """Persistent vector store backed by ``IndexIDMap(IndexFlatIP)`` + SQLite."""

    def __init__(self, path: str, dimension: int = DEFAULT_DIM):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self._lock = threading.Lock()

        self._db_path = self.path / "metadata.db"
        self._init_db()
        self._migrate_schema()

        index_path = self.path / "index.faiss"
        if index_path.exists():
            loaded = faiss.read_index(str(index_path))
            if isinstance(loaded, faiss.IndexIDMap):
                self.index = loaded
            else:
                logger.info(
                    "Wrapping legacy IndexFlatIP in IndexIDMap (%d vectors)",
                    loaded.ntotal,
                )
                self.index = faiss.IndexIDMap(loaded)
            logger.info("Loaded FAISS index with %d vectors", self.index.ntotal)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
            logger.info("Created new IndexIDMap(IndexFlatIP) (dim=%d)", dimension)

        self._seq_path = self.path / "seq.txt"
        self._seq = self._load_seq()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict],
            embeddings: np.ndarray) -> None:
        if len(ids) == 0:
            return
        embeddings = np.asarray(embeddings, dtype=np.float32).copy()
        faiss.normalize_L2(embeddings)
        with self._lock:
            faiss_ids = self._alloc_faiss_ids(len(ids))
            self.index.add_with_ids(embeddings, faiss_ids)
            self._write_docs(ids, texts, metadatas, embeddings, faiss_ids)
            self._save()

    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict],
               embeddings: np.ndarray) -> None:
        if len(ids) == 0:
            return
        embeddings = np.asarray(embeddings, dtype=np.float32).copy()
        faiss.normalize_L2(embeddings)
        with self._lock:
            existing = set(self._get_existing_ids(ids))
            if existing:
                self._remove_by_doc_ids(list(existing))
            faiss_ids = self._alloc_faiss_ids(len(ids))
            self.index.add_with_ids(embeddings, faiss_ids)
            self._write_docs(ids, texts, metadatas, embeddings, faiss_ids)
            self._save()

    def search(self, query_emb: np.ndarray, n_results: int = 10,
               where: Optional[dict] = None,
               where_document: Optional[dict] = None
               ) -> tuple[list[str], list[str], list[float], list[dict]]:
        query_emb = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_emb)
        if where_document:
            logger.warning("FaissStore.search ignores where_document")
        with self._lock:
            k = min(n_results, self.index.ntotal)
            if k == 0:
                return [], [], [], []
            distances, indices = self.index.search(query_emb, k)
            faiss_ids = indices[0].tolist()
            dists = distances[0].tolist()
            rows = self._get_rows_by_faiss_id(faiss_ids)
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        metadatas = [json.loads(r[2]) if r[2] else {} for r in rows]
        if where:
            filtered = [(ids[i], texts[i], dists[i], metadatas[i])
                        for i in range(len(ids))
                        if self._matches_where(metadatas[i], where)]
            if filtered:
                ids = [f[0] for f in filtered]
                texts = [f[1] for f in filtered]
                dists = [f[2] for f in filtered]
                metadatas = [f[3] for f in filtered]
            else:
                return [], [], [], []
        return list(ids), list(texts), list(dists), list(metadatas)

    def get(self, ids: Optional[list[str]] = None, where: Optional[dict] = None,
            where_document: Optional[dict] = None, limit: Optional[int] = None,
            offset: Optional[int] = 0) -> tuple[list[str], list[str], list[dict]]:
        with self._lock:
            if ids:
                placeholders = ",".join("?" * len(ids))
                cur = self._db.execute(
                    f"SELECT id, text, metadata FROM docs WHERE id IN ({placeholders})",
                    ids,
                )
            else:
                conditions = []
                params = []
                if where:
                    cond, p = self._where_to_sql(where)
                    if cond != "1=1":
                        conditions.append(cond)
                        params.extend(p)
                if where_document:
                    cond, p = self._where_doc_to_sql(where_document)
                    if cond != "1=1":
                        conditions.append(cond)
                        params.extend(p)
                sql = "SELECT id, text, metadata FROM docs"
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                sql += " ORDER BY rowid"
                if limit is not None:
                    sql += " LIMIT ?"
                    params.append(limit)
                if offset:
                    sql += " OFFSET ?"
                    params.append(offset)
                cur = self._db.execute(sql, params)
            rows = cur.fetchall()
        return [r[0] for r in rows], [r[1] for r in rows], \
               [json.loads(r[2]) if r[2] else {} for r in rows]

    def delete(self, ids: Optional[list[str]] = None,
               where: Optional[dict] = None) -> None:
        with self._lock:
            if ids:
                self._remove_by_doc_ids(ids)
            elif where:
                cond, params = self._where_to_sql(where)
                del_ids = [
                    r[0] for r in self._db.execute(
                        f"SELECT id FROM docs WHERE {cond}", params,
                    ).fetchall()
                ]
                if del_ids:
                    self._remove_by_doc_ids(del_ids)
            else:
                return
            self._save()

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            if hasattr(self, '_db'):
                self._db.close()

    def next_id(self) -> str:
        with self._lock:
            self._seq += 1
            self._save_seq()
            return f"doc_{self._seq}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""CREATE TABLE IF NOT EXISTS docs (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                vector BLOB,
                faiss_id INTEGER
        )""")
        self._db.commit()

    def _migrate_schema(self) -> None:
        # v4.3.1: add faiss_id column if missing (upgrade from <=4.3.0)
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(docs)").fetchall()}
        if "faiss_id" not in cols:
            logger.info("Migrating schema: adding faiss_id column to docs")
            self._db.execute("ALTER TABLE docs ADD COLUMN faiss_id INTEGER")
            # Populate faiss_id from existing rowid values
            rows = self._db.execute(
                "SELECT id, rowid FROM docs WHERE faiss_id IS NULL",
            ).fetchall()
            for doc_id, rid in rows:
                self._db.execute(
                    "UPDATE docs SET faiss_id = ? WHERE id = ?", (int(rid), doc_id),
                )
            self._db.commit()

        # Drop legacy pos_map if present (replaced by faiss_id column)
        legacy_tables = {
            r[0] for r in self._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            ).fetchall()
        }
        if "pos_map" in legacy_tables:
            logger.info("Migrating schema: dropping legacy pos_map table")
            self._db.execute("DROP TABLE IF EXISTS pos_map")
            self._db.commit()

    def _alloc_faiss_ids(self, count: int) -> np.ndarray:
        """Allocate ``count`` unique int64 FAISS IDs from the sequence counter."""
        start = self._seq + 1
        self._seq += count
        self._save_seq()
        return np.arange(start, start + count, dtype=np.int64)

    def _write_docs(self, ids: list[str], texts: list[str],
                    metadatas: list[dict], embeddings: np.ndarray,
                    faiss_ids: np.ndarray) -> None:
        for doc_id, text, meta, emb, fid in zip(ids, texts, metadatas,
                                                 embeddings, faiss_ids):
            vec_blob = emb.tobytes()
            self._db.execute(
                "INSERT OR REPLACE INTO docs "
                "(id, text, metadata, vector, faiss_id) VALUES (?, ?, ?, ?, ?)",
                (doc_id, text, json.dumps(meta), vec_blob, int(fid)),
            )

    def _get_rows_by_faiss_id(self, faiss_ids: list[int]) -> list[tuple]:
        """Look up docs by their FAISS ID, preserving search-result order."""
        valid = [fid for fid in faiss_ids if fid != -1]
        if not valid:
            return []
        placeholders = ",".join("?" * len(valid))
        order_clause = " ".join(
            f"WHEN {int(fid)} THEN {int(i)}" for i, fid in enumerate(valid)
        )
        cur = self._db.execute(
            f"SELECT id, text, metadata FROM docs "
            f"WHERE faiss_id IN ({placeholders}) "
            f"ORDER BY CASE faiss_id {order_clause} END",
            valid,
        )
        return cur.fetchall()

    def _get_existing_ids(self, ids: list[str]) -> list[str]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        cur = self._db.execute(
            f"SELECT id FROM docs WHERE id IN ({placeholders})", ids,
        )
        return [r[0] for r in cur.fetchall()]

    def _get_faiss_ids_for_docs(self, doc_ids: list[str]) -> list[int]:
        """Return stable faiss_id values for the given doc IDs."""
        if not doc_ids:
            return []
        placeholders = ",".join("?" * len(doc_ids))
        cur = self._db.execute(
            f"SELECT faiss_id FROM docs WHERE id IN ({placeholders}) "
            f"AND faiss_id IS NOT NULL",
            doc_ids,
        )
        return [r[0] for r in cur.fetchall()]

    def _remove_by_doc_ids(self, doc_ids: list[str]) -> None:
        """Remove docs from FAISS + SQLite. No full-rebuild needed."""
        faiss_ids = self._get_faiss_ids_for_docs(doc_ids)
        if faiss_ids:
            selector = faiss.IDSelectorArray(np.array(faiss_ids, dtype=np.int64))
            self.index.remove_ids(selector)
        placeholders = ",".join("?" * len(doc_ids))
        self._db.execute(f"DELETE FROM docs WHERE id IN ({placeholders})", doc_ids)

    def _rebuild_index(self) -> None:
        """Full rebuild from SQLite vectors (crash recovery)."""
        cur = self._db.execute(
            "SELECT id, vector, faiss_id FROM docs "
            "WHERE vector IS NOT NULL AND faiss_id IS NOT NULL "
            "ORDER BY faiss_id",
        )
        rows = cur.fetchall()
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
        if not rows:
            logger.info("FAISS index rebuilt (empty)")
            return
        vectors = []
        faiss_ids = []
        for doc_id, vec_blob, fid in rows:
            vec = np.frombuffer(vec_blob, dtype=np.float32)
            if len(vec) != self.dimension:
                logger.warning("Skipping doc %s: wrong vector dim %d", doc_id, len(vec))
                continue
            vectors.append(vec)
            faiss_ids.append(int(fid))
        if not vectors:
            logger.info("FAISS index rebuilt (all vectors skipped)")
            return
        all_vecs = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(all_vecs)
        self.index.add_with_ids(
            all_vecs, np.array(faiss_ids, dtype=np.int64),
        )
        logger.info("FAISS index rebuilt with %d vectors", len(vectors))

    def _save(self) -> None:
        try:
            self._db.commit()
            faiss.write_index(self.index, str(self.path / "index.faiss"))
        except Exception:
            logger.exception("FAISS index write failed")
            raise

    # ------------------------------------------------------------------
    # seq counter
    # ------------------------------------------------------------------

    def _load_seq(self) -> int:
        if self._seq_path.exists():
            try:
                return int(self._seq_path.read_text().strip())
            except (ValueError, OSError):
                pass
        try:
            cur = self._db.execute(
                "SELECT COALESCE(MAX(faiss_id), 0) FROM docs",
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return int(row[0])
        except sqlite3.OperationalError:
            pass
        try:
            cur = self._db.execute(
                "SELECT id FROM docs WHERE id GLOB 'doc_*' "
                "ORDER BY CAST(SUBSTR(id, 5) AS INTEGER) DESC LIMIT 1",
            )
            row = cur.fetchone()
            if row:
                parts = row[0].split('_')
                if len(parts) >= 2 and parts[-1].isdigit():
                    return int(parts[-1])
        except sqlite3.OperationalError:
            pass
        return 0

    def _save_seq(self) -> None:
        tmp = self.path / "seq.tmp"
        tmp.write_text(str(self._seq))
        tmp.replace(self._seq_path)

    # ------------------------------------------------------------------
    # Filter helpers (unchanged from v4.3.0)
    # ------------------------------------------------------------------

    def _matches_where(self, metadata: dict, where: dict) -> bool:
        for key, condition in where.items():
            if key == "$and":
                if not all(self._matches_where(metadata, c) for c in condition):
                    return False
            elif key == "$or":
                if not any(self._matches_where(metadata, c) for c in condition):
                    return False
            else:
                val = metadata.get(key)
                if isinstance(condition, dict):
                    for op, op_val in condition.items():
                        if op == "$eq" and val != op_val:
                            return False
                        elif op == "$ne" and val == op_val:
                            return False
                        elif op == "$in" and val not in op_val:
                            return False
                        elif op == "$nin" and val in op_val:
                            return False
                        elif op in ("$gt", "$gte", "$lt", "$lte"):
                            if val is None:
                                return False
                            try:
                                if op == "$gt" and not val > op_val:
                                    return False
                                if op == "$gte" and not val >= op_val:
                                    return False
                                if op == "$lt" and not val < op_val:
                                    return False
                                if op == "$lte" and not val <= op_val:
                                    return False
                            except TypeError:
                                return False
                else:
                    if val != condition:
                        return False
        return True

    def _where_to_sql(self, where: dict, prefix: str = "") -> tuple[str, list]:
        conditions = []
        params = []
        meta_col = f"{prefix}metadata" if prefix else "metadata"
        for key, condition in where.items():
            if key == "$and":
                sub_conds = []
                for c in condition:
                    sub, p = self._where_to_sql(c, prefix)
                    sub_conds.append(f"({sub})")
                    params.extend(p)
                if not sub_conds:
                    sub_conds.append("1=1")
                conditions.append("(" + " AND ".join(sub_conds) + ")")
            elif key == "$or":
                sub_conds = []
                for c in condition:
                    sub, p = self._where_to_sql(c, prefix)
                    sub_conds.append(f"({sub})")
                    params.extend(p)
                if not sub_conds:
                    sub_conds.append("1=1")
                conditions.append("(" + " OR ".join(sub_conds) + ")")
            else:
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', key):
                    raise ValueError(f"Invalid metadata key: {key!r}")
                if isinstance(condition, dict):
                    for op, op_val in condition.items():
                        if op in ("$eq", "$ne", "$in", "$nin",
                                  "$gt", "$gte", "$lt", "$lte"):
                            json_path = f"$.{key}"
                            if op == "$eq":
                                conditions.append(
                                    f"json_extract({meta_col}, ?) = ?")
                                params.extend([json_path, _sql_val(op_val)])
                            elif op == "$ne":
                                conditions.append(
                                    f"json_extract({meta_col}, ?) != ?")
                                params.extend([json_path, _sql_val(op_val)])
                            elif op == "$in":
                                ph = ",".join("?" * len(op_val))
                                conditions.append(
                                    f"json_extract({meta_col}, ?) IN ({ph})")
                                params.extend([json_path] + list(op_val))
                            elif op == "$nin":
                                ph = ",".join("?" * len(op_val))
                                conditions.append(
                                    f"json_extract({meta_col}, ?) NOT IN ({ph})")
                                params.extend([json_path] + list(op_val))
                            elif op == "$gt":
                                conditions.append(
                                    f"json_extract({meta_col}, ?) > ?")
                                params.extend([json_path, op_val])
                            elif op == "$gte":
                                conditions.append(
                                    f"json_extract({meta_col}, ?) >= ?")
                                params.extend([json_path, op_val])
                            elif op == "$lt":
                                conditions.append(
                                    f"json_extract({meta_col}, ?) < ?")
                                params.extend([json_path, op_val])
                            elif op == "$lte":
                                conditions.append(
                                    f"json_extract({meta_col}, ?) <= ?")
                                params.extend([json_path, op_val])
                else:
                    conditions.append(
                        f"json_extract({meta_col}, ?) = ?")
                    params.extend([f"$.{key}", _sql_val(condition)])
        return " AND ".join(conditions) if conditions else "1=1", params

    def _where_doc_to_sql(self, where_doc: dict) -> tuple[str, list]:
        conditions = []
        params = []
        for key, condition in where_doc.items():
            if key == "$contains":
                conditions.append("text LIKE ?")
                params.append(f"%{condition}%")
        return " AND ".join(conditions) if conditions else "1=1", params
