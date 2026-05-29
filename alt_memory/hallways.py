"""Hallways — within-realm entity-to-entity connectors.

A **hallway** is a connection between two entities (people, projects,
concepts, interests) inside one realm, materialized from their
co-occurrence across that realm's entities. Conceptually:

    REALM -> has ENTITIES (each tagged with entities)
             entities -> connected to other entities by HALLWAYS
                        (within-realm, built from entity co-occurrence)
                        hallways -> are the primitive
                                    tunnels -> use hallways to spawn
                                              cross-realm connections

If Aya and Lumi are both mentioned in 47 entities across the diary,
letters, and ideas domains, there's a hallway between them. If Aya
and "consciousness" co-occur in 19 entities, there's a hallway between
them too. The hallway *is* the structural fact of "these two entities
travel together inside this realm."

FAISS/SQLite (alt-memory).

Persistence mirrors the original: a JSON file under ``~/.alt-memory/``
so the records survive across mines and are inspectable / editable by
hand if needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from typing import Optional
import threading

from alt_memory.config import AltMemoryConfig
from alt_memory.dynamics import initialize_dynamics_fields
from alt_memory.dimension import Dimension

logger = logging.getLogger("alt_memory_hallways")

# Derived from AltMemoryConfig so it respects the configured dim_path.
def _hallway_file() -> str:
    return AltMemoryConfig().hallway_file

_SCHEMA_VERSION = 1

_hallway_lock = threading.Lock()


__all__ = [
    "compute_hallways_for_realm",
    "list_hallways",
    "delete_hallway",
]


# --------------------------------------------------------------------------
# Persistence — JSON file at _hallway_file(), restricted perms (0600) on POSIX
# --------------------------------------------------------------------------


def _load_hallways() -> list[dict]:
    """Read all hallway records. Returns ``[]`` if the file is missing or corrupt."""
    if not os.path.exists(_hallway_file()):
        return []
    try:
        with open(_hallway_file(), encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.debug("hallways: load failed, treating as empty", exc_info=True)
        return []
    if isinstance(raw, dict) and "hallways" in raw:
        return raw.get("hallways") or []
    if isinstance(raw, list):
        return raw
    return []


def _save_hallways(hallways: list[dict]) -> None:
    """Atomically persist hallway records to _hallway_file().

    Uses an os.replace temp-file dance so a crash mid-write doesn't
    corrupt the file. POSIX permission is restricted to 0600 because
    hallways reveal within-realm entity connections that the user may
    not want world-readable.
    """
    directory = os.path.dirname(_hallway_file())
    os.makedirs(directory, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "hallways": list(hallways),
    }
    fd, tmp_path = tempfile.mkstemp(prefix=".hallways-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            # Non-POSIX systems may not support chmod; not fatal.
            pass
        os.replace(tmp_path, _hallway_file())
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------
# Core algorithm — compute entity-pair hallways for one realm
# --------------------------------------------------------------------------


def _parse_entities(value) -> list[str]:
    """Drawer ``entities`` metadata is a semicolon-separated string. Parse it.

    Returns a deterministic *list* (not a set) because order matters for
    the deduplication semantics below: an entity that mentions ``Aya;Aya``
    should only contribute one Aya to the entity set for that entity.
    """
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip() for v in value if str(v).strip()]
    elif isinstance(value, str):
        items = [v.strip() for v in value.split(";") if v.strip()]
    else:
        return []
    # Dedupe while preserving first-seen order so id derivation is stable.
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _hallway_id(realm: str, entity_a: str, entity_b: str) -> str:
    """Deterministic id derived from realm + sorted entity pair.

    Sorting before hashing makes the id symmetric — (Aya, Lumi) and
    (Lumi, Aya) produce the same record. So an idempotent re-mine
    upserts the same hallway instead of creating two parallel records.
    """
    a, b = sorted([entity_a, entity_b])
    key = f"{realm}::{a}::{b}".encode("utf-8")
    suffix = hashlib.sha256(key).hexdigest()[:8]
    return f"hallway_{realm}_{a}_{b}_{suffix}"


def compute_hallways_for_realm(
    realm: str,
    dim_path: str = "~/.alt-memory",
    min_count: int = 2,
) -> list[dict]:
    """Compute entity-pair hallways for one realm.

    Algorithm:
      1. Open the dimension at ``dim_path`` and query entities for ``realm``
          from the SQLite backend.
      2. For each entity with entities, every pair of distinct entities in
         that entity is one co-occurrence. Increment a counter for each
         pair; also record the domain the entity lives in.
      3. For each (entity_a, entity_b) pair whose co-occurrence count is
         ``>= min_count``, materialize a hallway record. The record
          carries the pair, the count, and the set of domains where they
         co-occurred (useful context for navigation).
      4. Persist the full hallway list (records for other realms preserved,
         this realm's records replaced) and return the just-computed list.

    Args:
        realm: realm name to scan.
        dim_path: path to the alt-memory data directory. Defaults to
            ``~/.alt-memory``.
        min_count: minimum co-occurrence count required to materialize a
            hallway between two entities. Default 2 — single co-occurrences
            are noise (entities mentioned together once in one entity);
            two or more is a real signal. Clamped to ``>=1``.

    Returns:
        List of hallway dicts created for this realm. Records for other
        realms already on disk are preserved.
    """
    min_count = max(1, int(min_count))

    # 1. Open dimension and query entities for this realm.
    try:
        dim = Dimension(dim_path)
        dim.init()
        conn = None
        try:
            conn = sqlite3.connect(str(dim._base / "dimension.db"))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
            "SELECT id, content, metadata FROM entities WHERE realm = ?", (realm,)
        ).fetchall()
        finally:
            if conn:
                conn.close()
    except Exception:
        logger.warning(
            "compute_hallways_for_realm: query failed for %s", realm, exc_info=True
        )
        return []

    if not rows:
        return []

    # 2. Walk entities, counting entity-pair co-occurrence + tracking domains.
    # pair_counts: {(entity_a, entity_b): count} — keys always sorted to
    # canonicalize the (a, b) vs (b, a) symmetry.
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    pair_rooms: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in rows:
        try:
            meta = json.loads(row["metadata"])
        except (json.JSONDecodeError, TypeError, ValueError):
            meta = {}
        # Sentinel entities carry no real content — skip them.
        if meta.get("is_sentinel"):
            continue
        entities = _parse_entities(meta.get("entities"))
        if len(entities) < 2:
            # Need at least 2 entities for a pair to exist.
            continue
        domain_name = meta.get("domain")
        domain_str = domain_name if isinstance(domain_name, str) and domain_name.strip() else None

        # Each unordered pair of distinct entities in this entity is one
        # co-occurrence. itertools.combinations already gives unordered
        # pairs without repetition.
        for a, b in combinations(entities, 2):
            # Canonicalize order so (Aya, Lumi) and (Lumi, Aya) are the
            # same key. Skip self-pairs defensively.
            if a == b:
                continue
            key = tuple(sorted([a, b]))
            pair_counts[key] += 1
            if domain_str:
                pair_rooms[key].add(domain_str)

    if not pair_counts:
        return []

    # 3. Materialize hallway records for pairs above the threshold.
    #    Before building, load existing records so we can PRESERVE
    #    dynamics fields (strength, stability, last_activated, access_count)
    #    across recomputes. Without this preservation, every mine wipes
    #    the connection weights accumulated through use — defeating the
    #    living-connection layer entirely.
    with _hallway_lock:
        existing = _load_hallways()
        existing_dynamics_lookup: dict = {}
        for h in existing:
            if h.get("realm") != realm:
                continue
            # Canonicalize the lookup key by sorting the entity pair — must
            # match the symmetric ID generation in _hallway_id (which also
            # sorts). Without this, a persisted record with reversed entity
            # order would silently miss the lookup and lose its accumulated
            # dynamics on every recompute.
            key = tuple(sorted([h.get("entity_a"), h.get("entity_b")], key=lambda x: (x is None, x)))
            # Only copy the fields the dynamics layer cares about; everything
            # else is recomputed deterministically from the entity set.
            existing_dynamics_lookup[key] = {
                k: h[k] for k in ("strength", "stability", "last_activated", "access_count") if k in h
            }

        created: list[dict] = []
        created_at = datetime.now(timezone.utc).isoformat()
        for key in sorted(pair_counts.keys()):
            count = pair_counts[key]
            if count < min_count:
                continue
            entity_a, entity_b = key
            rooms = sorted(pair_rooms.get(key, set()))
            room_summary = ", ".join(rooms[:3]) if rooms else "(no domain tags)"
            if len(rooms) > 3:
                room_summary += f", +{len(rooms) - 3} more"
            record = {
                "id": _hallway_id(realm, entity_a, entity_b),
                "realm": realm,
                "entity_a": entity_a,
                "entity_b": entity_b,
                "co_occurrence_count": count,
                "domains": rooms,
                "label": f"{entity_a} \u2194 {entity_b} (co-occur in {count} entities across {len(rooms) or 'no'} domain{'s' if len(rooms) != 1 else ''}: {room_summary})",
                "created_at": created_at,
                "created_by": "auto",
            }
            # Apply preserved dynamics if this entity pair existed in the
            # prior realm snapshot, then initialize any missing fields.
            preserved = existing_dynamics_lookup.get(key, {})
            record.update(preserved)
            initialize_dynamics_fields(record)
            created.append(record)

        # 4. Persist — preserve other-realm records, replace this realm's records.
        preserved_other_realms = [h for h in existing if h.get("realm") != realm]
        _save_hallways(preserved_other_realms + created)

    return created


# --------------------------------------------------------------------------
# Query API — list_hallways, delete_hallway
# --------------------------------------------------------------------------


def list_hallways(realm: Optional[str] = None) -> list[dict]:
    """List hallway records. Filter by ``realm`` if specified."""
    with _hallway_lock:
        all_hallways = _load_hallways()
        if realm is None:
            return list(all_hallways)
        return [h for h in all_hallways if h.get("realm") == realm]


def delete_hallway(hallway_id: str) -> bool:
    """Remove one hallway record by id. Returns True if a record was removed."""
    with _hallway_lock:
        hallways = _load_hallways()
        filtered = [h for h in hallways if h.get("id") != hallway_id]
        if len(filtered) == len(hallways):
            return False
        _save_hallways(filtered)
        return True
