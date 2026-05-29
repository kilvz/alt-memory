"""ChromaBackend + ChromaCollection — RFC 001 backend contract for ChromaDB.

Wraps :class:`ChromaStore` in the :class:`BaseCollection` / :class:`BaseBackend`
ABCs so the registry and repair code can consume the backend through a uniform
interface.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np

from alt_memory.backends.base import (
    BaseBackend,
    BaseCollection,
    DimensionRef,
    GetResult,
    HealthStatus,
    QueryResult,
    _IncludeSpec,
)
from alt_memory.backends.chroma_hnsw import _pin_hnsw_threads
from alt_memory.backends.chroma_store import ChromaStore
from alt_memory.backends.types import (
    DimensionNotFoundError,
    UnsupportedFilterError,
)

logger = logging.getLogger(__name__)


_REQUIRED_OPERATORS = frozenset({"$eq", "$ne", "$in", "$nin", "$and", "$or", "$contains"})
_OPTIONAL_OPERATORS = frozenset({"$gt", "$gte", "$lt", "$lte"})
_SUPPORTED_OPERATORS = _REQUIRED_OPERATORS | _OPTIONAL_OPERATORS


def _validate_where(where: Optional[dict]) -> None:
    """Scan a where-clause for unknown operators and raise ``UnsupportedFilterError``."""
    if not where:
        return
    stack = [where]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        for k, v in node.items():
            if k.startswith("$") and k not in _SUPPORTED_OPERATORS:
                raise UnsupportedFilterError(f"operator {k!r} not supported by chroma backend")
            if isinstance(v, dict):
                stack.append(v)
            elif isinstance(v, list):
                stack.extend(x for x in v if isinstance(x, dict))


class ChromaCollection(BaseCollection):
    """Adapter wrapping :class:`ChromaStore` in the :class:`BaseCollection` ABC.

    When ``dim_path`` is set, all write methods (``add``, ``upsert``,
    ``delete``) acquire ``mine_dimension_lock(dim_path)`` for the
    duration of the underlying chromadb call. This serializes MCP and
    other direct-backend writers against each other, closing the race
    that triggers ChromaDB's multi-threaded HNSW corruption.

    ``dim_path=None`` disables the wrapping, preserving the legacy
    no-lock behaviour for callers that construct a ``ChromaCollection``
    directly without going through ``ChromaBackend``.
    """

    def __init__(self, store: ChromaStore, dim_path: Optional[str] = None):
        self._store = store
        self._dim_path = dim_path

    @contextlib.contextmanager
    def _write_lock(self):
        """Acquire ``mine_dimension_lock`` for the configured dimension, if any.

        No-op (yields immediately) when ``self._dim_path`` is None.
        """
        if self._dim_path is None:
            yield
            return
        from alt_memory.dimension import mine_dimension_lock

        with mine_dimension_lock(self._dim_path):
            yield

    def add(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        if embeddings is None:
            raise ValueError("ChromaCollection.add requires embeddings")
        emb_array = np.asarray(embeddings, dtype=np.float32)
        with self._write_lock():
            self._store.add(
                ids=ids,
                texts=documents,
                metadatas=metadatas or [{} for _ in range(len(ids))],
                embeddings=emb_array,
            )

    def upsert(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        if embeddings is None:
            raise ValueError("ChromaCollection.upsert requires embeddings")
        emb_array = np.asarray(embeddings, dtype=np.float32)
        with self._write_lock():
            self._store.upsert(
                ids=ids,
                texts=documents,
                metadatas=metadatas or [{} for _ in range(len(ids))],
                embeddings=emb_array,
            )

    def query(
        self,
        *,
        query_texts: Optional[list[str]] = None,
        query_embeddings: Optional[list[list[float]]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[list[str]] = None,
    ) -> QueryResult:
        _validate_where(where)
        _validate_where(where_document)

        spec = _IncludeSpec.resolve(include, default_distances=True)

        if query_embeddings is not None:
            qe = np.asarray(query_embeddings, dtype=np.float32)
        elif query_texts is not None:
            raise ValueError(
                "ChromaCollection.query requires query_embeddings; "
                "text-to-embed conversion is the caller's responsibility"
            )
        else:
            raise ValueError("query requires query_texts or query_embeddings")

        all_ids: list[list[str]] = []
        all_docs: list[list[str]] = []
        all_dists: list[list[float]] = []
        all_metas: list[list[dict]] = []

        for q in range(qe.shape[0]):
            qv = qe[q : q + 1]
            ids, texts, dists, metadatas = self._store.search(
                qv, n_results=n_results, where=where, where_document=where_document
            )
            all_ids.append(ids)
            all_docs.append(texts if spec.documents else [])
            all_dists.append(dists if spec.distances else [])
            all_metas.append(metadatas if spec.metadatas else [])

        return QueryResult(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas,
            distances=all_dists,
            embeddings=None,
        )

    def get(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list[str]] = None,
    ) -> GetResult:
        _validate_where(where)
        _validate_where(where_document)

        spec = _IncludeSpec.resolve(include, default_distances=False)
        g_ids, g_docs, g_metas = self._store.get(
            ids=ids, where=where, where_document=where_document,
            limit=limit, offset=offset or 0,
        )
        return GetResult(
            ids=g_ids,
            documents=g_docs if spec.documents else [],
            metadatas=g_metas if spec.metadatas else [],
            embeddings=None,
        )

    def delete(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
    ) -> None:
        _validate_where(where)
        with self._write_lock():
            self._store.delete(ids=ids, where=where)

    def count(self) -> int:
        return self._store.count()

    def close(self) -> None:
        self._store.close()

    def health(self) -> HealthStatus:
        try:
            n = self._store.count()
            return HealthStatus.healthy(detail=f"ChromaDB healthy, {n} vectors")
        except Exception as e:
            return HealthStatus.unhealthy(detail=str(e))


class ChromaBackend(BaseBackend):
    """RFC 001 backend wrapping :class:`ChromaStore`.

    One ``ChromaStore`` per dimension; cached by ``data_dir``.

    Maintains ``self._freshness`` — ``data_dir -> (inode, mtime)`` of
    ``chroma.sqlite3`` at cache time. If the file changes on disk
    (rebuild, restore, etc.) the cached store is invalidated on the
    next :meth:`get_collection` call.
    """

    name: ClassVar[str] = "chroma"
    spec_version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset()

    def __init__(self):
        self._stores: dict[str, ChromaStore] = {}
        self._freshness: dict[str, tuple[int, float]] = {}
        self._stores_lock = threading.Lock()
        self._closed = False

    @staticmethod
    def _resolve_embedding_function():
        """Return an embedding function wrapping alt-memory's embedder.

        Both ``get_collection`` and ``get_or_create_collection`` must receive
        the EF explicitly — ChromaDB 1.x does not persist it with the
        collection, so a reader that omits the argument silently gets the
        library default and its queries won't match the writer's vectors.
        """
        try:
            from alt_memory.backends.embedder import get_embedder

            embedder = get_embedder()

            def _ef(texts):
                result = embedder.embed(texts)
                return result.tolist()

            return _ef
        except Exception:
            logger.exception("Failed to build embedding function; using chromadb default")
            return None

    @staticmethod
    def _explain_ef_mismatch(error: Exception, data_dir: str) -> Optional[str]:
        """If ``error`` looks like a ChromaDB EF-name mismatch, return a
        user-friendly explanation. Otherwise return None.
        """
        msg = str(error)
        if "Embedding function conflict" not in msg and "embedding function" not in msg.lower():
            return None
        return (
            f"Embedding model mismatch reading dimension at {data_dir!r}.\n"
            f"  Underlying ChromaDB error: {msg}\n"
            f"  The dimension was built with a different embedding model. Either:\n"
            f"    (a) revert to the previous model, or\n"
            f"    (b) rebuild the index with the current model."
        )

    @staticmethod
    def _db_stat(data_dir: str) -> tuple[int, float]:
        """Return ``(inode, mtime)`` of ``chroma.sqlite3`` or ``(0, 0.0)`` if absent."""
        db_path = os.path.join(data_dir, "chroma.sqlite3")
        try:
            st = os.stat(db_path)
            return (st.st_ino, st.st_mtime)
        except OSError:
            return (0, 0.0)

    def _data_dir(self, dimension: DimensionRef) -> str:
        """Resolve a dimension ref to its data directory path."""
        data_dir = dimension.local_path
        if data_dir is None:
            data_dir = str(Path(dimension.id).expanduser().resolve() / "data")
        return data_dir

    def get_collection(
        self,
        *,
        dimension: DimensionRef,
        collection_name: str,
        create: bool = False,
        options: Optional[dict] = None,
    ) -> ChromaCollection:
        if self._closed:
            raise RuntimeError("ChromaBackend is closed")

        data_dir = self._data_dir(dimension)

        with self._stores_lock:
            store = self._stores.get(data_dir)
            dim_path = str(Path(dimension.id).expanduser().resolve())

            # Freshness check: has chroma.sqlite3 changed on disk?
            if store is not None:
                current_stat = self._db_stat(data_dir)
                cached_stat = self._freshness.get(data_dir, (0, 0.0))
                db_path = os.path.join(data_dir, "chroma.sqlite3")

                db_missing = cached_stat != (0, 0.0) and not os.path.isfile(db_path)
                inode_changed = (
                    current_stat[0] != 0
                    and cached_stat[0] != 0
                    and current_stat[0] != cached_stat[0]
                )
                mtime_appeared = cached_stat[1] == 0.0 and current_stat[1] != 0.0
                mtime_changed = (
                    current_stat[1] != 0.0
                    and cached_stat[1] != 0.0
                    and abs(current_stat[1] - cached_stat[1]) > 0.01
                )

                if db_missing or inode_changed or mtime_changed or mtime_appeared:
                    logger.debug(
                        "chroma.sqlite3 changed for %s (inode=%s mtime=%.3f); "
                        "rebuilding store cache",
                        data_dir,
                        current_stat[0],
                        current_stat[1],
                    )
                    store.close()
                    del self._stores[data_dir]
                    self._freshness.pop(data_dir, None)
                    store = None

            if store is None:
                if not os.path.isdir(data_dir):
                    if not create:
                        raise DimensionNotFoundError(f"Dimension not found: {data_dir}")
                    os.makedirs(data_dir, exist_ok=True)
                store = ChromaStore(data_dir)
                self._stores[data_dir] = store
                self._freshness[data_dir] = self._db_stat(data_dir)

        col = ChromaCollection(store, dim_path=dim_path)
        _pin_hnsw_threads(col._store._collection)
        return col

    def close_dimension(self, dimension: DimensionRef) -> None:
        data_dir = self._data_dir(dimension)
        with self._stores_lock:
            store = self._stores.pop(data_dir, None)
            self._freshness.pop(data_dir, None)
        if store:
            store.close()

    def close(self) -> None:
        with self._stores_lock:
            stores = list(self._stores.values())
            self._stores.clear()
            self._freshness.clear()
        for store in stores:
            try:
                store.close()
            except Exception:
                logger.exception("error closing ChromaStore")
        self._closed = True

    def health(self, dimension: Optional[DimensionRef] = None) -> HealthStatus:
        if not dimension:
            return HealthStatus.healthy(detail="ChromaDB backend ready")
        data_dir = self._data_dir(dimension)
        with self._stores_lock:
            store = self._stores.get(data_dir)
            created = store is None
            if created:
                store = ChromaStore(data_dir)
        try:
            n = store.count()
            return HealthStatus.healthy(detail=f"ChromaDB {n} vectors")
        except Exception as e:
            return HealthStatus.unhealthy(detail=str(e))
        finally:
            if created:
                store.close()

    @classmethod
    def detect(cls, path: str) -> bool:
        """Detect a ChromaDB dimension by the presence of ``data/chroma.sqlite3``."""
        return (Path(path) / "data" / "chroma.sqlite3").exists()
