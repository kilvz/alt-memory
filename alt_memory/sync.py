"""sync.py — Gitignore-aware entity prune using SQLite backend."""

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, TypedDict

from alt_memory.dimension import Dimension

logger = logging.getLogger(__name__)
_BATCH = 1000


class SyncReport(TypedDict):
    scanned: int
    kept: int
    gitignored: int
    missing: int
    no_source: int
    out_of_scope: int
    removed_entities: int
    removed_nodes: int
    dry_run: bool
    by_source: dict[str, int]


def _resolve_project_root(source_file: Path, project_roots: list) -> Optional[Path]:
    for root in project_roots:
        try:
            source_file.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def _has_git_marker(path: Path) -> bool:
    return (path / ".git").exists() or (path / ".gitignore").is_file()


def _ancestor_matchers(source_file: Path, root: Path, matcher_cache: dict) -> list:
    matchers: list = []
    try:
        parts = source_file.relative_to(root).parts
    except ValueError:
        return matchers
    cursor = root
    matcher = _load_gi_matcher(cursor, matcher_cache)
    if matcher is not None:
        matchers.append((root, matcher))
    for part in parts[:-1]:
        cursor = cursor / part
        matcher = _load_gi_matcher(cursor, matcher_cache)
        if matcher is not None:
            matchers.append((cursor, matcher))
    return matchers


def _load_gi_matcher(directory: Path, cache: dict):
    cache_key = str(directory.resolve())
    if cache_key in cache:
        return cache[cache_key]
    gi_file = directory / ".gitignore"
    if not gi_file.is_file():
        cache[cache_key] = None
        return None
    try:
        import pathspec
        with open(gi_file, "r") as f:
            spec = pathspec.PathSpec.from_lines("gitwildmatch", f.readlines())
        cache[cache_key] = spec
        return spec
    except ImportError:
        cache[cache_key] = None
        return None
    except Exception:
        logger.debug("sync gitignore spec load failed", exc_info=True)
        cache[cache_key] = None
        return None


def is_gitignored(path: Path, matchers: list, is_dir: bool = False) -> bool:
    for root, matcher in matchers:
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            continue
        if matcher.match_file(rel):
            return True
    return False


def _is_registry_row(meta: dict, entity_id: str) -> bool:
    if (meta or {}).get("domain") == "_registry":
        return True
    if (meta or {}).get("ingest_mode") == "registry":
        return True
    if entity_id and entity_id.startswith("_reg_"):
        return True
    return False


def _classify_entity(
    meta: dict, matcher_cache: dict, project_roots: list, entity_id: str = ""
) -> str:
    if _is_registry_row(meta, entity_id):
        return "kept"
    source_file = (meta or {}).get("source_file")
    if not source_file:
        return "no_source"
    src = Path(source_file)
    if not src.is_absolute():
        return "no_source"
    src = src.resolve(strict=False)
    root = _resolve_project_root(src, project_roots)
    if root is None:
        return "out_of_scope"
    if not src.exists():
        return "missing"
    matchers = _ancestor_matchers(src, root, matcher_cache)
    if matchers and is_gitignored(src, matchers, is_dir=False):
        return "gitignored"
    return "kept"


def _iter_entity_metadata(dimension, realm: Optional[str], conn=None):
    if conn is None:
        conn = sqlite3.connect(str(dimension._base / "dimension.db"))
        conn.row_factory = sqlite3.Row
        should_close = True
    else:
        should_close = False
    try:
        base_sql = "SELECT id, realm, domain, content, metadata, source_file, created_at FROM entities"
        params = []
        if realm:
            base_sql += " WHERE realm = ?"
            params.append(realm)
        base_sql += " ORDER BY created_at DESC"
        offset = 0
        while True:
            sql = base_sql + " LIMIT ? OFFSET ?"
            page_params = params + [_BATCH, offset]
            rows = conn.execute(sql, page_params).fetchall()
            if not rows:
                return
            for r in rows:
                meta = json.loads(r["metadata"] or "{}")
                yield r["id"], meta
            if len(rows) < _BATCH:
                return
            offset += len(rows)
    finally:
        if should_close:
            conn.close()


def _auto_detect_project_roots(dimension, realm: Optional[str], conn=None) -> list:
    if conn is None:
        conn = sqlite3.connect(str(dimension._base / "dimension.db"))
        conn.row_factory = sqlite3.Row
        should_close = True
    else:
        should_close = False
    try:
        sql = "SELECT source_file FROM entities"
        params = []
        if realm:
            sql += " WHERE realm = ?"
            params.append(realm)
        rows = conn.execute(sql, params).fetchall()
    finally:
        if should_close:
            conn.close()
    roots: set = set()
    seen_sources: set = set()
    for r in rows:
        source_file = r["source_file"]
        if not source_file or source_file in seen_sources:
            continue
        seen_sources.add(source_file)
        src = Path(source_file)
        if not src.is_absolute():
            continue
        for parent in src.parents:
            if _has_git_marker(parent):
                roots.add(parent.resolve(strict=False))
                break
    return sorted(roots, key=lambda p: (-len(str(p)), str(p)))


def _normalize_project_dirs(project_dirs) -> list:
    resolved = [Path(p).resolve(strict=False) for p in project_dirs]
    return sorted(resolved, key=lambda p: (-len(str(p)), str(p)))


def _delete_in_batches(dimension, ids: list, batch_size: int = 999, wal_log: Optional[Callable] = None):
    import datetime as dt
    wal_dir = Path.home() / ".alt-memory" / "wal"
    wal_dir.mkdir(parents=True, exist_ok=True)
    wal_path = wal_dir / "sync_log.jsonl"
    deleted = 0
    for i in range(0, len(ids), batch_size):
        chunk = ids[i:i + batch_size]
        ph = ",".join("?" * len(chunk))
        dimension._store.delete(ids=chunk)
        dimension._db.execute(f"DELETE FROM entities WHERE id IN ({ph})", chunk)
        dimension._db.execute(f"DELETE FROM entities_fts WHERE id IN ({ph})", chunk)
        dimension._db.commit()
        deleted += len(chunk)
        if wal_log is not None:
            entry = json.dumps({
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "dimension": str(dimension._base),
                "ids_deleted": chunk,
                "count": len(chunk),
            })
            try:
                with open(wal_path, "a") as f:
                    f.write(entry + "\n")
            except OSError:
                pass
    return deleted


def sync_dimension(
    dimension_path: str,
    project_dirs: Optional[list] = None,
    realm: Optional[str] = None,
    dry_run: bool = True,
    batch_size: int = _BATCH,
    wal_log: Optional[Callable] = None,
) -> SyncReport:
    if not dry_run and not realm and not project_dirs:
        raise ValueError(
            "sync apply requires explicit realm= or project_dirs= so it cannot "
            "auto-prune every realm in a multi-project dimension; pass --realm or "
            "a project directory"
        )
    if project_dirs is not None and not project_dirs:
        raise ValueError(
            "project_dirs was provided but is empty; pass at least one project "
            "root or pass project_dirs=None to auto-detect from entity metadata"
        )

    counts = {
        "scanned": 0, "kept": 0, "gitignored": 0,
        "missing": 0, "no_source": 0, "out_of_scope": 0,
    }
    by_source: dict = defaultdict(int)
    removable_ids: list = []
    removable_sources: set = set()

    dimension = Dimension(dimension_path)
    dimension.init()
    try:
        conn = dimension._db

        if project_dirs is not None:
            roots = _normalize_project_dirs(project_dirs)
        else:
            roots = _auto_detect_project_roots(dimension, realm, conn=conn)

        matcher_cache: dict = {}
        classification_cache: dict = {}

        for entity_id, meta in _iter_entity_metadata(dimension, realm, conn=conn):
            counts["scanned"] += 1
            meta = meta or {}
            source_file = meta.get("source_file")

            if _is_registry_row(meta, entity_id):
                bucket = "kept"
            elif source_file and source_file in classification_cache:
                bucket = classification_cache[source_file]
            else:
                bucket = _classify_entity(meta, matcher_cache, roots, entity_id)
                if source_file:
                    classification_cache[source_file] = bucket

            counts[bucket] += 1
            if bucket in ("gitignored", "missing"):
                removable_ids.append(entity_id)
                if source_file:
                    removable_sources.add(source_file)
                    by_source[source_file] += 1

        report: SyncReport = {
            **counts,
            "removed_entities": 0,
            "removed_nodes": 0,
            "dry_run": dry_run,
            "by_source": dict(by_source),
        }

        if dry_run or not removable_ids:
            return report

        report["removed_entities"] = _delete_in_batches(dimension, removable_ids, wal_log=wal_log)

        node_rows = 0
        if removable_sources:
            placeholders = ",".join("?" * len(removable_sources))
            dimension._db.execute(f"DELETE FROM nodes WHERE source_file IN ({placeholders})", list(removable_sources))
            node_rows = dimension._db.execute("SELECT changes()").fetchone()[0]
            dimension._db.commit()
        report["removed_nodes"] = node_rows

        return report
    finally:
        dimension.close()


__all__ = [
    "SyncReport",
    "sync_dimension",
]
