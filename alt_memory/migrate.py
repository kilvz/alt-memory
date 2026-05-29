"""Dimension schema migration and FAISS rebuild.

Detects the current dimension schema version, applies pending migrations,
and provides disaster recovery — rebuild the FAISS index from SQLite
when vectors are lost or corrupted.

Schema versions
---------------
v0 — unversioned dimension (pre-migrate): realms, domains, entities (with FTS5), closets
v1 — adds _meta version tracking table and content_date column to entities
v2 — renames wings→realms, rooms→domains, drawers→entities in DB tables
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from alt_memory.backends.embedder import get_embedder

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 2
META_TABLE = "_meta"


# ── Version tracking ───────────────────────────────────────────────────────


def _open_db(dim_path: str) -> sqlite3.Connection:
    base = Path(dim_path).expanduser().resolve()
    dim_db = base / "dimension.db"
    legacy_db = base / "palace.db"
    db_path = str(dim_db) if dim_db.exists() else str(legacy_db)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_meta_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {META_TABLE} (key TEXT PRIMARY KEY, value TEXT)"
    )


def get_dimension_version(dim_path: str) -> int:
    """Return the schema version of an existing dimension, or 0 if unversioned."""
    conn = _open_db(dim_path)
    try:
        _ensure_meta_table(conn)
        row = conn.execute(
            f"SELECT value FROM {META_TABLE} WHERE key='dimension_version'"
        ).fetchone()
        if row:
            return int(row["value"])
        # Fallback to legacy key
        row = conn.execute(
            f"SELECT value FROM {META_TABLE} WHERE key='palace_version'"
        ).fetchone()
        return int(row["value"]) if row else 0
    except (sqlite3.Error, ValueError):
        return 0
    finally:
        conn.close()


def set_dimension_version(dim_path: str, version: int) -> None:
    conn = _open_db(dim_path)
    try:
        _ensure_meta_table(conn)
        conn.execute(
            f"INSERT OR REPLACE INTO {META_TABLE} (key, value) VALUES ('dimension_version', ?)",
            (str(version),),
        )
        conn.commit()
    finally:
        conn.close()


# ── Schema migrations ──────────────────────────────────────────────────────


def _migrate_v0_to_v1(dim_path: str, conn: sqlite3.Connection) -> None:
    """v0 -> v1: add _meta version table and content_date column."""
    _ensure_meta_table(conn)
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(drawers)").fetchall()
    }
    if "content_date" not in existing:
        conn.execute("ALTER TABLE drawers ADD COLUMN content_date TEXT DEFAULT ''")
        logger.info("migrate: added content_date column to drawers")


def _migrate_v1_to_v2(dim_path: str, conn: sqlite3.Connection) -> None:
    """v1 -> v2: rename wings→realms, rooms→domains, drawers→entities."""
    existing_tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    table_renames = {"wings": "realms", "rooms": "domains", "drawers": "entities"}
    for old, new in table_renames.items():
        if old in existing_tables:
            conn.execute(f"ALTER TABLE {old} RENAME TO {new}")
            logger.info("migrate: renamed table %s → %s", old, new)

    # Rename columns inside the new entities table
    entities_cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(entities)").fetchall()
    }
    if "wing" in entities_cols:
        conn.execute("ALTER TABLE entities RENAME COLUMN wing TO realm")
    if "room" in entities_cols:
        conn.execute("ALTER TABLE entities RENAME COLUMN room TO domain")

    # Migrate _meta key from palace_version to dimension_version
    _ensure_meta_table(conn)
    row = conn.execute(
        f"SELECT value FROM {META_TABLE} WHERE key='palace_version'"
    ).fetchone()
    if row:
        conn.execute(
            f"INSERT OR REPLACE INTO {META_TABLE} (key, value) VALUES ('dimension_version', ?)",
            (row["value"],),
        )
        conn.execute(f"DELETE FROM {META_TABLE} WHERE key='palace_version'")


_MIGRATIONS = {
    1: _migrate_v0_to_v1,
    2: _migrate_v1_to_v2,
}


def migrate_schema(dim_path: str, dry_run: bool = False) -> dict:
    """Run all pending schema migrations and return a report."""
    current = get_dimension_version(dim_path)
    report = {
        "path": str(Path(dim_path).expanduser().resolve()),
        "version_before": current,
        "version_after": current,
        "migrations_applied": [],
        "dry_run": dry_run,
    }

    if current >= CURRENT_SCHEMA_VERSION:
        report["version_after"] = current
        return report

    conn = _open_db(dim_path)
    try:
        for target_version in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
            fn = _MIGRATIONS.get(target_version)
            if fn is None:
                logger.warning("No migration defined for v%d -> v%d", target_version - 1, target_version)
                continue
            label = f"v{target_version - 1}_to_v{target_version}"
            logger.info("Applying migration %s", label)
            if not dry_run:
                fn(dim_path, conn)
                set_dimension_version(dim_path, target_version)
            report["migrations_applied"].append(label)
            report["version_after"] = target_version

        if not dry_run:
            conn.commit()
    finally:
        conn.close()

    return report


# ── FAISS rebuild ──────────────────────────────────────────────────────────


def rebuild_faiss(dim_path: str) -> dict:
    """Rebuild the FAISS index from dimension.db SQLite content.

    Reads all entity texts, re-embeds them, and writes a fresh
    ``data/index.faiss`` + ``data/metadata.db``. Existing data is
    replaced. Useful when the FAISS index is corrupted or missing.
    """
    from alt_memory.backends.faiss_store import FaissStore

    base = Path(dim_path).expanduser().resolve()
    data_dir = base / "data"

    # Read all entities
    conn = _open_db(dim_path)
    try:
        rows = conn.execute(
            "SELECT id, content, metadata FROM entities ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {"ok": True, "vectors_rebuilt": 0, "message": "No entities to rebuild"}

    ids = []
    texts = []
    metadatas = []
    for r in rows:
        ids.append(r["id"])
        texts.append(r["content"])
        try:
            metadatas.append(json.loads(r["metadata"]) if r["metadata"] else {})
        except (json.JSONDecodeError, TypeError):
            metadatas.append({})

    embedder = get_embedder()
    vectors = embedder.embed(texts)

    data_dir.mkdir(parents=True, exist_ok=True)
    store = FaissStore(str(data_dir), dimension=vectors.shape[1])
    store.clear()
    store.add(ids, texts, metadatas, vectors)

    n = store.count()
    store.close()

    logger.info("Rebuilt FAISS index with %d vectors", n)
    return {"ok": True, "vectors_rebuilt": n}


# ── Orchestrator ────────────────────────────────────────────────────────────


def migrate(dim_path: str, dry_run: bool = False, confirm: bool = False) -> dict:
    """Run all pending schema migrations.

    Returns a dict with ``migrations_applied``, ``version_before``,
    ``version_after``, and ``rebuild_triggered``.
    """
    result = migrate_schema(dim_path, dry_run=dry_run)
    return result


# ── CLI integration ────────────────────────────────────────────────────────


def status(dim_path: str) -> dict:
    """Detailed schema-version status of the dimension."""
    base = Path(dim_path).expanduser().resolve()
    dim_db = base / "dimension.db"
    legacy_db = base / "palace.db"
    dim_db_path = dim_db if dim_db.exists() else legacy_db
    data_dir = base / "data"
    index_file = data_dir / "index.faiss"
    meta_file = data_dir / "metadata.db"

    info: dict = {
        "path": str(base),
        "dim_db_exists": dim_db_path.exists(),
        "index_exists": index_file.exists(),
        "metadata_db_exists": meta_file.exists(),
    }

    if not dim_db_path.exists():
        info["version"] = 0
        info["message"] = "Dimension not initialized"
        return info

    version = get_dimension_version(str(dim_path))
    info["version"] = version
    info["latest_version"] = CURRENT_SCHEMA_VERSION
    info["up_to_date"] = version >= CURRENT_SCHEMA_VERSION

    # Count items
    conn = _open_db(dim_path)
    try:
        info["entities"] = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        info["realms"] = conn.execute("SELECT COUNT(*) FROM realms").fetchone()[0]
        info["domains"] = conn.execute("SELECT COUNT(*) FROM domains").fetchone()[0]
    except sqlite3.Error:
        pass
    finally:
        conn.close()

    # FAISS count
    if index_file.exists() and meta_file.exists():
        try:
            store_conn = sqlite3.connect(str(meta_file))
            row = store_conn.execute("SELECT COUNT(*) FROM pos_map").fetchone()
            info["vectors"] = row[0] if row else 0
            store_conn.close()
        except sqlite3.Error:
            info["vectors"] = None
    else:
        info["vectors"] = 0

    return info
