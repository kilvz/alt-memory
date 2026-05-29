"""dedup.py — Detect and remove near-duplicate entities using FAISS cosine distance."""

import argparse
import logging
import os
import sqlite3
import time
from collections import defaultdict

from alt_memory.dimension import Dimension

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.15
MIN_ENTITIES_TO_CHECK = 5


def _get_dimension_path():
    try:
        from alt_memory.config import AltMemoryConfig
        return AltMemoryConfig().dimension_path
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".alt-memory")


def _sqlite_conn(dimension_path):
    db_path = os.path.join(dimension_path, "dimension.db")
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_source_groups(dimension, min_count=MIN_ENTITIES_TO_CHECK, source_pattern=None, realm=None):
    conn = _sqlite_conn(dimension._base)
    if conn is None:
        return {}
    try:
        sql = "SELECT id, source_file FROM entities"
        params = []
        conditions = []
        if realm:
            conditions.append("realm = ?")
            params.append(realm)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    groups = defaultdict(list)
    for r in rows:
        src = r["source_file"] or "unknown"
        if source_pattern and source_pattern.lower() not in src.lower():
            continue
        groups[src].append(r["id"])
    return {src: ids for src, ids in groups.items() if len(ids) >= min_count}


def dedup_source_group(dimension, entity_ids, threshold=DEFAULT_THRESHOLD, dry_run=True):
    kept = []
    to_delete = []

    for eid in entity_ids:
        doc = dimension.get_entity(eid)
        if not doc or not doc["content"] or len(doc["content"]) < 20:
            to_delete.append(eid)
            continue

        if not kept:
            kept.append((eid, doc["content"]))
            continue

        try:
            results = dimension.search(doc["content"], n_results=min(len(kept), 5), mode="vector")
            kept_ids_set = {k[0] for k in kept}

            is_dup = False
            for r in results:
                if r.id in kept_ids_set and r.distance < threshold:
                    is_dup = True
                    break

            if is_dup:
                to_delete.append(eid)
            else:
                kept.append((eid, doc["content"]))
        except Exception:
            kept.append((eid, doc["content"]))

    if to_delete and not dry_run:
        for eid in to_delete:
            dimension.delete_entity(eid)

    return [k[0] for k in kept], to_delete


def show_stats(dimension_path=None):
    dimension_path = dimension_path or _get_dimension_path()
    dimension = Dimension(dimension_path)
    dimension.init()
    groups = get_source_groups(dimension)

    total_entities = sum(len(ids) for ids in groups.values())
    print(f"\n  Sources with {MIN_ENTITIES_TO_CHECK}+ entities: {len(groups)}")
    print(f"  Total entities in those sources: {total_entities:,}")

    print("\n  Top 15 by entity count:")
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    for src, ids in sorted_groups[:15]:
        print(f"    {len(ids):4d}  {src[:65]}")

    estimated_dups = sum(int(len(ids) * 0.4) for ids in groups.values() if len(ids) > 20)
    print(f"\n  Estimated duplicates (groups > 20): ~{estimated_dups:,}")


def dedup_dimension(
    dimension_path=None,
    threshold=DEFAULT_THRESHOLD,
    dry_run=True,
    source_pattern=None,
    min_count=MIN_ENTITIES_TO_CHECK,
    realm=None,
):
    dimension_path = dimension_path or _get_dimension_path()
    dimension = Dimension(dimension_path)
    dimension.init()

    print(f"\n{'=' * 55}")
    print("  alt-memory Deduplicator")
    print(f"{'=' * 55}")

    status = dimension.status()
    print(f"  Dimension: {dimension_path}")
    print(f"  Entities: {status['entities']:,}")
    print(f"  Threshold: {threshold}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'─' * 55}")

    if realm:
        print(f"  Realm: {realm}")
    groups = get_source_groups(dimension, min_count, source_pattern, realm=realm)
    print(f"\n  Sources to check: {len(groups)}")

    t0 = time.time()
    total_kept = 0
    total_deleted = 0

    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

    for i, (src, entity_ids) in enumerate(sorted_groups):
        kept, deleted = dedup_source_group(dimension, entity_ids, threshold, dry_run)
        total_kept += len(kept)
        total_deleted += len(deleted)

        if deleted:
            print(
                f"  [{i + 1:3d}/{len(groups)}] "
                f"{src[:50]:50s} {len(entity_ids):4d} \u2192 {len(kept):4d}  "
                f"(-{len(deleted)})"
            )

    elapsed = time.time() - t0

    print(f"\n{'─' * 55}")
    print(f"  Done in {elapsed:.1f}s")
    print(
        f"  Entities: {total_kept + total_deleted:,} \u2192 {total_kept:,}  (-{total_deleted:,} removed)"
    )

    if dry_run:
        print("\n  [DRY RUN] No changes written. Re-run without --dry-run to apply.")

    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate near-identical entities")
    parser.add_argument("--dimension", default=None, help="Dimension directory path")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Cosine distance threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    parser.add_argument("--realm", default=None, help="Scope dedup to a single realm")
    parser.add_argument("--source", default=None, help="Filter by source file pattern")
    args = parser.parse_args()

    path = os.path.expanduser(args.dimension) if args.dimension else None

    if args.stats:
        show_stats(dimension_path=path)
    else:
        dedup_dimension(
            dimension_path=path,
            threshold=args.threshold,
            dry_run=args.dry_run,
            source_pattern=args.source,
            realm=args.realm,
        )
