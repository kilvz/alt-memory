"""Dimension schema migration and FAISS rebuild.

Detects the current dimension schema version, applies pending migrations,
and provides disaster recovery — rebuild the FAISS index from SQLite
when vectors are lost or corrupted.

Schema versions
---------------
v0 — unversioned dimension (pre-migrate)
v1 — current: adds _meta version tracking table and content_date column
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 1
META_TABLE = "_meta"
_VALID_META_TABLES = frozenset({"_meta"})


def _validate_meta_table(table: str) -> None:
    if table not in _VALID_META_TABLES:
        raise ValueError(f"Invalid meta table name: {table!r}")


# ── Version tracking ───────────────────────────────────────────────────────


def _open_db(dim_path: str) -> sqlite3.Connection:
    base = Path(dim_path).expanduser().resolve()
    db_path = str(base / "dimension.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_meta_table(conn: sqlite3.Connection) -> None:
    _validate_meta_table(META_TABLE)
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
        for row in conn.execute("PRAGMA table_info(entities)").fetchall()
    }
    if "content_date" not in existing:
        conn.execute("ALTER TABLE entities ADD COLUMN content_date TEXT DEFAULT ''")
        logger.info("migrate: added content_date column to entities")


_MIGRATIONS = {
    1: _migrate_v0_to_v1,
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
            report["migrations_applied"].append(label)
            report["version_after"] = target_version

        if not dry_run:
            _ensure_meta_table(conn)
            conn.execute(
                f"INSERT OR REPLACE INTO {META_TABLE} (key, value) VALUES ('dimension_version', ?)",
                (str(CURRENT_SCHEMA_VERSION),),
            )
            conn.commit()
    finally:
        conn.close()

    return report


# ── FAISS rebuild ──────────────────────────────────────────────────────────
