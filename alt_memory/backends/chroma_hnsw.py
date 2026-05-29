"""HNSW safety pipeline for ChromaDB-backed dimensions.

Adapted from upstream MemPalace's ``chroma.py`` — detects corrupt HNSW
segments BEFORE ChromaDB opens them, preventing SIGSEGV in native code.
All ``palace_path`` references are mapped to ``data_dir`` and
``collection_name`` defaults to ``"vectors"`` (alt-memory's collection).

Standalone functions — no alt-memory internal imports needed beyond the
stdlib.
"""

import datetime as _dt
import logging
import os
import pickle
import sqlite3
from numbers import Integral
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A healthy HNSW payload should keep link_lists.bin proportional to
# data_level0.bin. When link_lists.bin grows orders of magnitude larger than
# data_level0.bin, Chroma/HNSW can segfault while opening the segment even if
# index_metadata.pickle is structurally valid.
_HNSW_LINK_TO_DATA_MAX_RATIO = 10.0

# HNSW tuning to prevent link_lists.bin bloat on large mines.
_HNSW_BLOAT_GUARD = {
    "hnsw:batch_size": 50_000,
    "hnsw:sync_threshold": 50_000,
}

# Missing index_metadata.pickle is normal only while a segment is still fresh
# or effectively empty. Once data_level0.bin has non-trivial payload, a
# missing metadata pickle means the segment was interrupted after writing HNSW
# data but before writing its metadata.
_HNSW_MISSING_METADATA_DATA_FLOOR = 1024

# Divergence threshold: chromadb's HNSW flushes asynchronously, so HNSW
# typically lags sqlite by up to ``sync_threshold`` records under active
# write load. Two synchronization windows worth (2 x sync_threshold) is a
# safe steady-state ceiling.
_HNSW_DIVERGENCE_FALLBACK_FLOOR = 2000
_HNSW_DIVERGENCE_FRACTION = 0.10

_BLOB_FIX_MARKER = ".blob_seq_ids_migrated"


# ── Size-ratio helpers ───────────────────────────────────────────────────


def _hnsw_link_to_data_ratio(seg_dir: str) -> Optional[float]:
    """Return link_lists.bin / data_level0.bin size ratio for a segment.

    ``None`` means the ratio is not meaningful, usually because one file is
    missing or data_level0.bin is empty. ``float("inf")`` means the files were
    present but could not be statted safely.
    """
    link_path = os.path.join(seg_dir, "link_lists.bin")
    data_path = os.path.join(seg_dir, "data_level0.bin")

    if not (os.path.isfile(link_path) and os.path.isfile(data_path)):
        return None

    try:
        data_size = os.path.getsize(data_path)
        link_size = os.path.getsize(link_path)
    except OSError:
        return float("inf")

    if data_size <= 0:
        return None

    return link_size / data_size


def _hnsw_link_lists_is_usable_for_payload(seg_dir: str) -> bool:
    """Return False when a non-trivial HNSW payload lacks usable link lists."""
    data_path = os.path.join(seg_dir, "data_level0.bin")
    link_path = os.path.join(seg_dir, "link_lists.bin")

    try:
        if not os.path.isfile(data_path):
            return True

        data_size = os.path.getsize(data_path)
        if data_size <= _HNSW_MISSING_METADATA_DATA_FLOOR:
            return True

        return os.path.isfile(link_path) and os.path.getsize(link_path) > 0
    except OSError:
        return False


def _hnsw_payload_appears_sane(seg_dir: str) -> bool:
    """Return False when HNSW payload files are structurally implausible."""
    if not _hnsw_link_lists_is_usable_for_payload(seg_dir):
        return False

    ratio = _hnsw_link_to_data_ratio(seg_dir)
    return ratio is None or ratio <= _HNSW_LINK_TO_DATA_MAX_RATIO


# ── Segment health checks ────────────────────────────────────────────────


def _segment_appears_healthy(seg_dir: str) -> bool:
    """Return True if a chromadb HNSW segment dir looks intact.

    Sniff-tests the chromadb-written segment metadata file
    (``index_metadata.pickle``) for its expected format bytes without
    parsing it. ChromaDB writes that file after a successful HNSW flush;
    a complete write starts with byte ``0x80`` and ends with byte ``0x2e``.

    Missing metadata is healthy only while the segment still looks fresh or
    empty. If ``data_level0.bin`` already has non-trivial payload but
    ``index_metadata.pickle`` is missing, the segment is partially flushed.
    """
    if not _hnsw_payload_appears_sane(seg_dir):
        return False

    meta_path = os.path.join(seg_dir, "index_metadata.pickle")
    if not os.path.isfile(meta_path):
        data_path = os.path.join(seg_dir, "data_level0.bin")
        try:
            if (
                os.path.isfile(data_path)
                and os.path.getsize(data_path) > _HNSW_MISSING_METADATA_DATA_FLOOR
            ):
                return False
        except OSError:
            return False

        return True

    try:
        size = os.path.getsize(meta_path)
        if size < 16:
            return False
        with open(meta_path, "rb") as f:
            head = f.read(2)
            f.seek(-1, 2)
            tail = f.read(1)
    except OSError:
        return False
    return len(head) == 2 and head[0] == 0x80 and tail == b"\x2e"


# ── Quarantine (rename unsafe segments out of the way) ──────────────────


def quarantine_stale_hnsw(data_dir: str, stale_seconds: float = 300.0) -> list[str]:
    """Rename HNSW segment dirs that look unsafe to open.

    This catches two classes of HNSW corruption before ChromaDB opens the
    native segment reader:

    1. stale-by-mtime segments whose ``index_metadata.pickle`` fails the
       existing format sniff-test;
    2. structurally impossible HNSW payloads where ``link_lists.bin`` is much
       larger than ``data_level0.bin``.

    The original directory is renamed, not deleted, so recovery remains
    possible if the heuristic ever misfires.
    """
    db_path = os.path.join(data_dir, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return []

    try:
        sqlite_mtime = os.path.getmtime(db_path)
    except OSError:
        return []

    moved: list[str] = []

    try:
        entries = os.listdir(data_dir)
    except OSError:
        return []

    for name in entries:
        if "-" not in name or name.startswith(".") or ".drift-" in name:
            continue

        seg_dir = os.path.join(data_dir, name)
        if not os.path.isdir(seg_dir):
            continue

        hnsw_bin = os.path.join(seg_dir, "data_level0.bin")
        if not os.path.isfile(hnsw_bin):
            continue

        try:
            hnsw_mtime = os.path.getmtime(hnsw_bin)
        except OSError:
            continue

        payload_ratio = _hnsw_link_to_data_ratio(seg_dir)
        payload_corrupt = payload_ratio is not None and payload_ratio > _HNSW_LINK_TO_DATA_MAX_RATIO

        if not payload_corrupt and sqlite_mtime - hnsw_mtime < stale_seconds:
            continue

        if not payload_corrupt and _segment_appears_healthy(seg_dir):
            logger.info(
                "HNSW mtime gap %.0fs on %s exceeds threshold but segment "
                "metadata and payload size are intact - flush-lag, not "
                "corruption. Leaving in place.",
                sqlite_mtime - hnsw_mtime,
                seg_dir,
            )
            continue

        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        target = f"{seg_dir}.drift-{stamp}"

        if payload_corrupt:
            reason = (
                f"link_lists.bin/data_level0.bin ratio {payload_ratio:.1f}x "
                f"exceeds {_HNSW_LINK_TO_DATA_MAX_RATIO:.1f}x"
            )
        else:
            reason = (
                f"sqlite {sqlite_mtime - hnsw_mtime:.0f}s newer than HNSW "
                "and integrity check failed"
            )

        try:
            os.rename(seg_dir, target)
            moved.append(target)
            logger.warning(
                "Quarantined corrupt HNSW segment %s (%s); renamed to %s",
                seg_dir,
                reason,
                target,
            )
        except OSError:
            logger.exception("Failed to quarantine corrupt HNSW segment %s", seg_dir)

    return moved


# ── SQLite helpers ────────────────────────────────────────────────────────


def _vector_segment_id(data_dir: str, collection_name: str = "vectors") -> Optional[str]:
    """Return the VECTOR segment UUID for ``collection_name`` or ``None``."""
    db_path = os.path.join(data_dir, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                """
                SELECT s.id
                FROM segments s
                JOIN collections c ON s.collection = c.id
                WHERE c.name = ? AND s.scope = 'VECTOR'
                LIMIT 1
                """,
                (collection_name,),
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def _sqlite_embedding_count(data_dir: str, collection_name: str = "vectors") -> Optional[int]:
    """Count rows in chroma.sqlite3.embeddings for ``collection_name``."""
    db_path = os.path.join(data_dir, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM embeddings e
                JOIN segments s ON e.segment_id = s.id
                JOIN collections c ON s.collection = c.id
                WHERE c.name = ?
                """,
                (collection_name,),
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def _read_sync_threshold(data_dir: str, collection_name: str = "vectors") -> int:
    """Return the ``hnsw:sync_threshold`` for a collection, or 1000 default."""
    db_path = os.path.join(data_dir, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return 1000
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT cm.int_value
                FROM collection_metadata cm
                JOIN collections c ON cm.collection_id = c.id
                WHERE c.name = ? AND cm.key = 'hnsw:sync_threshold'
                """,
                (collection_name,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return int(row[0])
            return 1000
        finally:
            conn.close()
    except Exception:
        logger.debug("_read_sync_threshold failed", exc_info=True)
        return 1000


def _fix_blob_seq_ids(data_dir: str) -> None:
    """Fix ChromaDB 0.6.x -> 1.5.x migration bug: BLOB seq_ids -> INTEGER."""
    db_path = os.path.join(data_dir, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return
    marker = os.path.join(data_dir, _BLOB_FIX_MARKER)
    if os.path.isfile(marker):
        return
    try:
        with sqlite3.connect(db_path) as conn:
            try:
                rows = conn.execute(
                    "SELECT rowid, seq_id FROM embeddings WHERE typeof(seq_id) = 'blob'"
                ).fetchall()
            except sqlite3.OperationalError:
                return
            safe_rows = [(rowid, blob) for rowid, blob in rows if not blob.startswith(b"\x11\x11")]
            skipped = len(rows) - len(safe_rows)
            if skipped:
                logger.warning(
                    "Skipped %d sysdb-10-format BLOB seq_id(s) in embeddings (not converting)",
                    skipped,
                )
            if safe_rows:
                updates = [
                    (int.from_bytes(blob, byteorder="big"), rowid) for rowid, blob in safe_rows
                ]
                conn.executemany("UPDATE embeddings SET seq_id = ? WHERE rowid = ?", updates)
                logger.info("Fixed %d BLOB seq_ids in embeddings", len(updates))
                conn.commit()
    except Exception:
        logger.exception("Could not fix BLOB seq_ids in %s", db_path)
        return
    try:
        Path(marker).touch()
    except OSError:
        logger.exception("Could not write migration marker %s", marker)


# ── Safe unpickling for index_metadata.pickle ────────────────────────────


class _PersistentDataStub:
    """Minimal stand-in for chromadb's ``PersistentData`` during safe unpickling.

    Accepts any constructor args so pickle's REDUCE opcode succeeds,
    captures ``__setstate__`` into ``__dict__``. Only used by
    :func:`_hnsw_element_count` and :func:`quarantine_invalid_hnsw_metadata`
    - never persisted, never re-pickled.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            self.__dict__.update(state[1])


class _SafePersistentDataUnpickler:
    """Whitelist-only unpickler for ``index_metadata.pickle``.

    Allows only ``PersistentData`` from chromadb's HNSW module; everything
    else raises ``UnpicklingError``.
    """

    _ALLOWED = frozenset(
        {
            (
                "chromadb.segment.impl.vector.local_persistent_hnsw",
                "PersistentData",
            ),
        }
    )

    @classmethod
    def load(cls, path: str):
        import pickle

        class _Restricted(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if (module, name) in cls._ALLOWED:
                    return _PersistentDataStub
                raise pickle.UnpicklingError(f"disallowed class: {module}.{name}")

        with open(path, "rb") as f:
            return _Restricted(f).load()


# ── HNSW element count ────────────────────────────────────────────────────


def _hnsw_element_count(data_dir: str, segment_id: str) -> Optional[int]:
    """Return the element count chromadb thinks the HNSW segment holds.

    Reads ``index_metadata.pickle`` via a tight-allowlist unpickler and
    counts ``id_to_label`` entries.

    Returns ``None`` when the file is absent (fresh / never-flushed
    segment) or the unpickle fails.
    """
    pickle_path = os.path.join(data_dir, segment_id, "index_metadata.pickle")
    if not os.path.isfile(pickle_path):
        return None
    try:
        pd = _SafePersistentDataUnpickler.load(pickle_path)
        if isinstance(pd, dict):
            id_to_label = pd.get("id_to_label")
        else:
            id_to_label = getattr(pd, "id_to_label", None)
        if isinstance(id_to_label, dict):
            return len(id_to_label)
        return None
    except Exception:
        logger.debug("_hnsw_element_count failed for %s", pickle_path, exc_info=True)
        return None


# ── Capacity divergence probe ─────────────────────────────────────────────


def hnsw_capacity_status(data_dir: str, collection_name: str = "vectors") -> dict:
    """Compare sqlite embedding count against HNSW element count.

    The #1222 failure mode: ``max_elements`` froze at 16 384 while sqlite
    accumulated far more embeddings. Every subsequent tool call segfaulted
    when chromadb tried to load the undersized HNSW. This probe runs
    *before* anything touches the segment so we can warn instead of
    crashing.

    Returns a dict with:

    * ``segment_id``       - VECTOR segment UUID, or ``None`` if no dimension
    * ``sqlite_count``     - embeddings present in chroma.sqlite3
    * ``hnsw_count``       - elements chromadb's pickle knows about
    * ``divergence``       - ``sqlite_count - hnsw_count`` when both known
    * ``diverged``         - True when divergence exceeds the threshold
    * ``status``           - ``"ok"`` | ``"diverged"`` | ``"unknown"``
    * ``message``          - human-readable summary

    Never raises.
    """
    out: dict[str, Any] = {
        "segment_id": None,
        "sqlite_count": None,
        "hnsw_count": None,
        "divergence": None,
        "diverged": False,
        "status": "unknown",
        "message": "",
    }

    try:
        seg_id = _vector_segment_id(data_dir, collection_name)
        out["segment_id"] = seg_id

        sqlite_count = _sqlite_embedding_count(data_dir, collection_name)
        out["sqlite_count"] = sqlite_count

        if seg_id is None or sqlite_count is None:
            out["message"] = "dimension state unreadable; skipping HNSW capacity check"
            return out

        hnsw_count = _hnsw_element_count(data_dir, seg_id)
        out["hnsw_count"] = hnsw_count

        sync_threshold = _read_sync_threshold(data_dir, collection_name)
        divergence_floor = max(_HNSW_DIVERGENCE_FALLBACK_FLOOR, 2 * sync_threshold)

        if hnsw_count is None:
            out["message"] = (
                "HNSW capacity unavailable: metadata has not been flushed; "
                "leaving vector search enabled"
            )
            return out

        divergence = sqlite_count - hnsw_count
        out["divergence"] = divergence
        threshold = max(divergence_floor, int(sqlite_count * _HNSW_DIVERGENCE_FRACTION))
        if divergence > threshold:
            out["status"] = "diverged"
            out["diverged"] = True
            pct = 100.0 * divergence / max(sqlite_count, 1)
            out["message"] = (
                f"HNSW index holds {hnsw_count:,} elements but sqlite has "
                f"{sqlite_count:,} embeddings - {divergence:,} ({pct:.0f}%) "
                "are invisible to vector search. Run `alt-memory repair` to rebuild."
            )
        else:
            out["status"] = "ok"
            out["message"] = (
                f"HNSW {hnsw_count:,} / sqlite {sqlite_count:,} (within flush-lag tolerance)"
            )
    except Exception:
        logger.debug("hnsw_capacity_status failed", exc_info=True)
        out["message"] = "HNSW capacity probe raised; skipping"
    return out


# ── Thread pinning ────────────────────────────────────────────────────────


def _pin_hnsw_threads(collection) -> None:
    """Best-effort retrofit: pin ``hnsw:num_threads=1`` on an existing collection.

    Must run on every ``get_collection`` call because the metadata does not
    persist across ``PersistentClient`` reopens in chromadb 1.5.x.
    """
    try:
        from chromadb.api.collection_configuration import (
            UpdateCollectionConfiguration,
            UpdateHNSWConfiguration,
        )
    except ImportError:
        logger.debug("_pin_hnsw_threads skipped: chromadb too old", exc_info=True)
        return
    try:
        collection.modify(
            configuration=UpdateCollectionConfiguration(hnsw=UpdateHNSWConfiguration(num_threads=1))
        )
    except Exception:
        logger.debug("_pin_hnsw_threads modify failed", exc_info=True)


# ── Metadata validation helpers ────────────────────────────────────────────


def _valid_dimensionality(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and int(value) > 0


def _persisted_metadata_value(obj: object, name: str) -> object:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _persisted_metadata_fields(obj: object) -> tuple[object, object]:
    return _persisted_metadata_value(obj, "dimensionality"), _persisted_metadata_value(
        obj, "id_to_label"
    )


def _missing_dimensionality_appears_recoverable(
    persisted: object, id_to_label: dict, seg_dir: str
) -> bool:
    total = _persisted_metadata_value(persisted, "total_elements_added")
    label_to_id = _persisted_metadata_value(persisted, "label_to_id")
    data_path = os.path.join(seg_dir, "data_level0.bin")
    link_path = os.path.join(seg_dir, "link_lists.bin")

    if not isinstance(total, Integral) or isinstance(total, bool):
        return False
    if not isinstance(label_to_id, dict):
        return False
    try:
        if not (
            os.path.isfile(data_path)
            and os.path.isfile(link_path)
            and os.path.getsize(data_path) > _HNSW_MISSING_METADATA_DATA_FLOOR
        ):
            return False
    except OSError:
        return False
    if not _hnsw_payload_appears_sane(seg_dir):
        return False

    label_count = len(id_to_label)
    if int(total) != label_count or len(label_to_id) != label_count:
        return False
    try:
        return all(label_to_id.get(label) == item_id for item_id, label in id_to_label.items())
    except TypeError:
        return False


# ── Invalid metadata quarantine ────────────────────────────────────────────


def quarantine_invalid_hnsw_metadata(data_dir: str) -> list[str]:
    """Quarantine segment dirs whose ``index_metadata.pickle`` is unreadable or invalid.

    Chroma's persisted HNSW metadata is untrusted disk state. If a segment has
    labels but invalid or partial metadata, current Chroma versions can accept
    the pickle and crash later in the Rust loader. We rename the entire segment
    out of the way before ``PersistentClient`` opens so Chroma can rebuild
    cleanly.
    """
    try:
        entries = os.listdir(data_dir)
    except OSError:
        return []

    moved: list[str] = []
    for name in entries:
        if "-" not in name or name.startswith(".") or ".drift-" in name or ".corrupt-" in name:
            continue
        seg_dir = os.path.join(data_dir, name)
        if not os.path.isdir(seg_dir):
            continue

        meta_path = os.path.join(seg_dir, "index_metadata.pickle")
        if not os.path.isfile(meta_path):
            continue

        reason = None
        try:
            persisted = _SafePersistentDataUnpickler.load(meta_path)
        except (EOFError, OSError):
            logger.debug(
                "Skipping invalid-HNSW quarantine for transient metadata read in %s",
                meta_path,
                exc_info=True,
            )
            continue
        except pickle.UnpicklingError as exc:
            if "truncated" in str(exc).lower() or "ran out of input" in str(exc).lower():
                logger.debug(
                    "Skipping invalid-HNSW quarantine for transient metadata read in %s",
                    meta_path,
                    exc_info=True,
                )
                continue
            reason = f"invalid index_metadata.pickle: {exc}"
        except Exception as exc:
            reason = f"invalid index_metadata.pickle: {exc}"
        else:
            if not isinstance(persisted, dict) and not (
                hasattr(persisted, "dimensionality") or hasattr(persisted, "id_to_label")
            ):
                reason = f"unrecognized index_metadata.pickle payload: {type(persisted).__name__}"
            else:
                dimensionality, id_to_label = _persisted_metadata_fields(persisted)
                if id_to_label is not None and not isinstance(id_to_label, dict):
                    reason = f"invalid id_to_label type {type(id_to_label).__name__}"
                else:
                    has_labels = bool(id_to_label)
                    if (
                        has_labels
                        and dimensionality is None
                        and not _missing_dimensionality_appears_recoverable(
                            persisted, id_to_label, seg_dir
                        )
                    ):
                        reason = (
                            "labels present but dimensionality is missing or invalid "
                            f"({dimensionality!r})"
                        )
                    elif (
                        has_labels
                        and dimensionality is not None
                        and not _valid_dimensionality(dimensionality)
                    ):
                        reason = (
                            "labels present but dimensionality is missing or invalid "
                            f"({dimensionality!r})"
                        )
                    elif dimensionality is not None and not _valid_dimensionality(dimensionality):
                        reason = f"invalid dimensionality {dimensionality!r}"

        if reason is None:
            continue

        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        target = f"{seg_dir}.corrupt-{stamp}"
        try:
            os.rename(seg_dir, target)
            moved.append(target)
            logger.warning("Quarantined invalid HNSW metadata in %s: %s", seg_dir, reason)
        except OSError:
            logger.exception("Failed to quarantine invalid HNSW metadata in %s", seg_dir)

    return moved
