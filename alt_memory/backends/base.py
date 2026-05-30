"""Storage backend contract — ABCs every backend must implement.

Typed results, error
classes, and value objects live in ``types.py``; this module defines the
two core ABCs (:class:`BaseCollection`, :class:`BaseBackend`) and the
``_IncludeSpec`` resolver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional

from alt_memory.backends.types import (
    BackendError,
    DimensionMismatchError,
    GetResult,
    HealthStatus,
    DimensionNotFoundError,
    QueryResult,
    UnsupportedFilterError,
    _DictCompatMixin,
)

DEFAULT_DIM = 384

__all__ = [
    "BaseBackend",
    "BaseCollection",
    "BackendError",
    "DimensionMismatchError",
    "GetResult",
    "HealthStatus",
    "DimensionNotFoundError",
    "QueryResult",
    "UnsupportedFilterError",
    "_DictCompatMixin",
    "_IncludeSpec",
]


# ── DimensionRef ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DimensionRef:
    """A handle to a dimension, consumed by backends."""

    id: str
    local_path: Optional[str] = None
    namespace: Optional[str] = None


# ── Include spec resolver ─────────────────────────────────────────────────


_VALID_INCLUDE_KEYS = frozenset({"documents", "metadatas", "distances", "embeddings"})


@dataclass
class _IncludeSpec:
    """Resolve an ``include=`` parameter with spec-mandated defaults."""

    documents: bool = True
    metadatas: bool = True
    distances: bool = True
    embeddings: bool = False

    @classmethod
    def resolve(
        cls, include: Optional[list[str]], *, default_distances: bool = True
    ) -> _IncludeSpec:
        if include is None:
            return cls(
                documents=True,
                metadatas=True,
                distances=default_distances,
                embeddings=False,
            )
        keys = {k for k in include if k in _VALID_INCLUDE_KEYS}
        return cls(
            documents="documents" in keys,
            metadatas="metadatas" in keys,
            distances="distances" in keys,
            embeddings="embeddings" in keys,
        )


# ── Collection contract ──────────────────────────────────────────────────


class BaseCollection(ABC):
    """Per-collection read/write surface every backend must implement."""

    @abstractmethod
    def add(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None: ...

    @abstractmethod
    def upsert(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None: ...

    @abstractmethod
    def query(
        self,
        *,
        query_texts: Optional[list[str]] = None,
        query_embeddings: Optional[list[list[float]]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[list[str]] = None,
    ) -> QueryResult: ...

    @abstractmethod
    def get(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list[str]] = None,
    ) -> GetResult: ...

    @abstractmethod
    def delete(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
    ) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    def estimated_count(self) -> int:
        return self.count()

    def close(self) -> None:
        return None

    def health(self) -> HealthStatus:
        return HealthStatus.healthy()

    def update(
        self,
        *,
        ids: list[str],
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        if embeddings is None:
            raise ValueError(
                "update requires embeddings (backends cannot re-embed internally)"
            )
        if documents is None and metadatas is None:
            raise ValueError("update requires at least one of documents, metadatas")
        n = len(ids)
        for label, value in (
            ("documents", documents),
            ("metadatas", metadatas),
            ("embeddings", embeddings),
        ):
            if value is not None and len(value) != n:
                raise ValueError(f"{label} length {len(value)} does not match ids length {n}")
        existing = self.get(ids=ids, include=["documents", "metadatas"])
        by_id = {
            rid: (existing.documents[i], existing.metadatas[i])
            for i, rid in enumerate(existing.ids)
        }
        merged_docs: list[str] = []
        merged_metas: list[dict] = []
        for i, rid in enumerate(ids):
            prev_doc, prev_meta = by_id.get(rid, ("", {}))
            merged_docs.append(documents[i] if documents is not None else prev_doc)
            new_meta = dict(prev_meta or {})
            if metadatas is not None:
                new_meta.update(metadatas[i] or {})
            merged_metas.append(new_meta)
        self.upsert(
            documents=merged_docs,
            ids=list(ids),
            metadatas=merged_metas,
            embeddings=embeddings,
        )


# ── Backend contract ─────────────────────────────────────────────────────


class BaseBackend(ABC):
    """Long-lived factory serving many palaces."""

    name: ClassVar[str]
    spec_version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def get_collection(
        self,
        *,
        dimension: DimensionRef,
        collection_name: str,
        create: bool = False,
        options: Optional[dict] = None,
    ) -> BaseCollection: ...

    def close_dimension(self, dimension: DimensionRef) -> None:
        return None

    def close(self) -> None:
        return None

    def health(self, dimension: Optional[DimensionRef] = None) -> HealthStatus:
        return HealthStatus.healthy()

    @classmethod
    def detect(cls, path: str) -> bool:
        return False
