"""ChromaDB vector store — drop-in replacement for FaissStore.

Same public interface (add, upsert, search, get, delete, count, close, next_id)
but backed by ChromaDB PersistentClient. Follows upstream patterns:
metadata sanitization, HNSW bloat guards, write serialization.
"""

from __future__ import annotations

import logging
import pathlib
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DIM = 384


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms

_HNSW_BLOAT_GUARD = {
    "hnsw:batch_size": 50_000,
    "hnsw:sync_threshold": 50_000,
}


def _sanitize_metadatas(metadatas: Optional[list[Optional[dict]]]) -> Optional[list[Optional[dict]]]:
    if metadatas is None:
        return None
    return [
        (m if (isinstance(m, dict) and len(m) > 0) else {"_repaired_empty_meta": True})
        for m in metadatas
    ]


class ChromaStore:
    """Persistent vector store backed by ChromaDB — FaissStore-compatible interface."""

    def __init__(self, path: str, dimension: int = _DEFAULT_DIM):
        self._store_path = path
        self._path = pathlib.Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self._lock = threading.Lock()

        import chromadb
        self._client = chromadb.PersistentClient(path=str(self._path))
        self._collection = self._client.get_or_create_collection(
            name="vectors",
            metadata={
                "hnsw:space": "ip",
                "hnsw:num_threads": 1,
                **_HNSW_BLOAT_GUARD,
            },
        )

        self._seq_path = self._path / "seq.txt"
        self._seq = self._load_seq()

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict],
            embeddings: np.ndarray) -> None:
        if len(ids) == 0:
            return
        embeddings = np.asarray(embeddings, dtype=np.float32).copy()
        embeddings = _l2_normalize(embeddings)
        sanitized = _sanitize_metadatas(metadatas)
        kwargs = {
            "ids": ids,
            "documents": texts,
            "embeddings": embeddings.tolist(),
        }
        if sanitized is not None:
            kwargs["metadatas"] = sanitized
        with self._lock:
            self._collection.add(**kwargs)

    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict],
               embeddings: np.ndarray) -> None:
        if len(ids) == 0:
            return
        embeddings = np.asarray(embeddings, dtype=np.float32).copy()
        embeddings = _l2_normalize(embeddings)
        sanitized = _sanitize_metadatas(metadatas)
        kwargs = {
            "ids": ids,
            "documents": texts,
            "embeddings": embeddings.tolist(),
        }
        if sanitized is not None:
            kwargs["metadatas"] = sanitized
        with self._lock:
            self._collection.upsert(**kwargs)

    def search(self, query_emb: np.ndarray, n_results: int = 10,
               where: Optional[dict] = None,
               where_document: Optional[dict] = None) -> tuple[list[str], list[str], list[float], list[dict]]:
        if n_results <= 0:
            return [], [], [], []
        query_emb = np.asarray(query_emb, dtype=np.float32).copy().reshape(1, -1)
        query_emb = _l2_normalize(query_emb)
        k = n_results
        with self._lock:
            result = self._collection.query(
                query_embeddings=query_emb.tolist(),
                n_results=k,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"],
            )
        if not result["ids"] or not result["ids"][0]:
            return [], [], [], []
        ids = result["ids"][0]
        texts = result["documents"][0] if result.get("documents") else []
        dists = result["distances"][0] if result.get("distances") else []
        metas_raw = result["metadatas"][0] if result.get("metadatas") else []
        metadatas = [dict(m) if m else {} for m in metas_raw]
        return list(ids), list(texts), list(dists), metadatas

    def get(self, ids: Optional[list[str]] = None, where: Optional[dict] = None,
            where_document: Optional[dict] = None, limit: Optional[int] = None,
            offset: Optional[int] = 0) -> tuple[list[str], list[str], list[dict]]:
        with self._lock:
            result = self._collection.get(
                ids=ids,
                where=where,
                where_document=where_document,
                limit=limit,
                offset=offset or 0,
                include=["documents", "metadatas"],
            )
        ids_out = result.get("ids", []) or []
        docs = result.get("documents", []) or []
        metas_raw = result.get("metadatas", []) or []
        metadatas = [dict(m) if m else {} for m in metas_raw]
        return list(ids_out), list(docs), metadatas

    def delete(self, ids: Optional[list[str]] = None, where: Optional[dict] = None) -> None:
        with self._lock:
            self._collection.delete(ids=ids, where=where)

    def count(self) -> int:
        with self._lock:
            return self._collection.count()

    def close(self) -> None:
        try:
            self._client.close()
        except Exception as exc:
            logger.warning("ChromaStore.close error: %s", exc)

    def next_id(self) -> str:
        with self._lock:
            self._seq += 1
            self._save_seq()
            return f"doc_{self._seq}"

    def _load_seq(self) -> int:
        if self._seq_path.exists():
            try:
                return int(self._seq_path.read_text().strip())
            except (ValueError, OSError):
                pass
        try:
            max_num = 0
            offset = 0
            batch_size = 1000
            while True:
                existing = self._collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=[],
                )
                if not existing or not existing.get("ids"):
                    break
                for eid in existing["ids"]:
                    parts = eid.split("_")
                    if len(parts) >= 2 and parts[-1].isdigit():
                        max_num = max(max_num, int(parts[-1]))
                if len(existing["ids"]) < batch_size:
                    break
                offset += batch_size
            return max_num
        except Exception:
            pass
        return 0

    def _save_seq(self) -> None:
        tmp = self._path / "seq.tmp"
        tmp.write_text(str(self._seq))
        tmp.replace(self._seq_path)
