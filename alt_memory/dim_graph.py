"""
dim_graph.py — Graph traversal layer for Alt Memory
====================================================

Builds a navigable graph from the dimension structure:
  - Nodes = domains (named ideas)
  - Edges = shared domains across realms (tunnels)
   - Edge types = gates (the corridors)

Enables queries like:
  "Start at chromadb-setup in realm_code, walk to realm_myproject"
  "Find all domains connected to riley-college-apps"
  "What topics bridge realm_hardware and realm_myproject?"

No external graph DB needed — built from SQLite metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone

from alt_memory.config import AltMemoryConfig, normalize_realm_name
from alt_memory.dimension import Dimension, mine_lock

logger = logging.getLogger("alt_memory_dim_graph")


def _normalize_realm(realm: str | None) -> str | None:
    if not isinstance(realm, str):
        return None
    realm = realm.strip()
    if not realm:
        return None
    return normalize_realm_name(realm)


_graph_cache_lock = threading.Lock()
_graph_cache_nodes = None
_graph_cache_edges = None
_graph_cache_time = 0.0
_GRAPH_CACHE_TTL = 60.0


def invalidate_graph_cache():
    global _graph_cache_nodes, _graph_cache_edges, _graph_cache_time
    with _graph_cache_lock:
        _graph_cache_nodes = None
        _graph_cache_edges = None
        _graph_cache_time = 0.0


def _get_dimension(config=None):
    config = config or AltMemoryConfig()
    try:
        dimension = Dimension(config.dim_path)
        dimension.init()
        return dimension
    except Exception:
        logger.debug("_get_dimension failed", exc_info=True)
        return None


def build_graph(dimension=None, config=None):
    """
    Build the dimension graph from SQLite metadata.

    Returns cached result if fresh (within TTL). Cache is invalidated
    on writes via invalidate_graph_cache(). Thread-safe via _graph_cache_lock.

    Note: warm cache ignores ``dimension`` and ``config`` arguments — this is
    intentional for the MCP server's single-dimension use case. Callers
    switching collections should call ``invalidate_graph_cache()`` first.

    Returns:
        nodes: dict of {domain: {realms: set, gates: set, count: int}}
        edges: list of {domain, realm_a, realm_b, gate} — one per tunnel crossing
    """
    global _graph_cache_nodes, _graph_cache_edges, _graph_cache_time
    now = time.time()
    with _graph_cache_lock:
        if _graph_cache_nodes is not None and (now - _graph_cache_time) < _GRAPH_CACHE_TTL:
            return _graph_cache_nodes, _graph_cache_edges

        if dimension is None:
            dimension = _get_dimension(config)
        if not dimension:
            return {}, []

        rows = dimension._db.execute(
            "SELECT id, realm, domain, metadata FROM entities"
        ).fetchall()

        domain_data = defaultdict(lambda: {"realms": set(), "gates": set(), "count": 0, "dates": set()})

        for row in rows:
            row_id, row_realm, row_domain, row_meta = row
            meta = json.loads(row_meta or "{}")
            domain = row_domain or meta.get("domain", "")
            realm = row_realm or meta.get("realm", "")
            gate = meta.get("gate", "")
            date = meta.get("date", "")
            if domain and domain != "general" and realm:
                domain_data[domain]["realms"].add(realm)
                if gate:
                    domain_data[domain]["gates"].add(gate)
                if date:
                    domain_data[domain]["dates"].add(date)
                domain_data[domain]["count"] += 1

        edges = []
        for domain, data in domain_data.items():
            realms = sorted(data["realms"])
            if len(realms) >= 2:
                for i, ra in enumerate(realms):
                    for rb in realms[i + 1:]:
                        for gate in data["gates"]:
                            edges.append({
                                "domain": domain,
                                "realm_a": ra,
                                "realm_b": rb,
                                "gate": gate,
                                "count": data["count"],
                            })

        nodes = {}
        for domain, data in domain_data.items():
            nodes[domain] = {
                "realms": sorted(data["realms"]),
                "gates": sorted(data["gates"]),
                "count": data["count"],
                "dates": sorted(data["dates"])[-5:] if data["dates"] else [],
            }

        if nodes or edges:
            _graph_cache_nodes = nodes
            _graph_cache_edges = edges
            _graph_cache_time = time.time()

    return nodes, edges


def traverse(start_domain: str, dimension=None, config=None, max_hops: int = 2):
    """
    Walk the graph from a starting domain. Find connected domains
    through shared realms.

    Returns list of paths: [{domain, realm, gate, hop_distance}]
    """
    nodes, edges = build_graph(dimension, config)

    if start_domain not in nodes:
        return {
            "error": f"Domain '{start_domain}' not found",
            "suggestions": _fuzzy_match(start_domain, nodes),
        }

    start = nodes[start_domain]
    visited = {start_domain}
    results = [{
        "domain": start_domain,
        "realms": start["realms"],
        "gates": start["gates"],
        "count": start["count"],
        "hop": 0,
    }]

    frontier = deque([(start_domain, 0)])
    while frontier:
        current_domain, depth = frontier.popleft()
        if depth >= max_hops:
            continue

        current = nodes.get(current_domain, {})
        current_realms = set(current.get("realms", []))

        for domain, data in nodes.items():
            if domain in visited:
                continue
            shared_realms = current_realms & set(data["realms"])
            if shared_realms:
                visited.add(domain)
                results.append({
                    "domain": domain,
                    "realms": data["realms"],
                    "gates": data["gates"],
                    "count": data["count"],
                    "hop": depth + 1,
                    "connected_via": sorted(shared_realms),
                })
                if depth + 1 < max_hops:
                    frontier.append((domain, depth + 1))

    results.sort(key=lambda x: (x["hop"], -x["count"]))
    return results[:50]


def find_tunnels(realm_a: str = None, realm_b: str = None, dimension=None, config=None):
    """
    Find domains that connect two realms (or all tunnel domains if no realms specified).
    These are the "gateways" — same named idea appearing in multiple domains.
    """
    nodes, edges = build_graph(dimension, config)

    norm_a = _normalize_realm(realm_a)
    norm_b = _normalize_realm(realm_b)

    tunnels = []
    for domain, data in nodes.items():
        realms = data["realms"]
        if len(realms) < 2:
            continue

        if norm_a and norm_a not in realms:
            continue
        if norm_b and norm_b not in realms:
            continue

        tunnels.append({
            "domain": domain,
            "realms": realms,
            "gates": data["gates"],
            "count": data["count"],
            "recent": data["dates"][-1] if data["dates"] else "",
        })

    if not tunnels and (realm_a or realm_b):
        logger.warning(
            "No tunnels found for realm filter(s): realm_a=%r (normalized=%r), realm_b=%r (normalized=%r)",
            realm_a,
            norm_a,
            realm_b,
            norm_b,
        )

    tunnels.sort(key=lambda x: -x["count"])
    return tunnels[:50]


def graph_stats(dimension=None, config=None):
    """Summary statistics about the dimension graph."""
    nodes, edges = build_graph(dimension, config)

    tunnel_domains = sum(1 for n in nodes.values() if len(n["realms"]) >= 2)
    realm_counts = Counter()
    for data in nodes.values():
        for r in data["realms"]:
            realm_counts[r] += 1

    return {
        "total_domains": len(nodes),
        "tunnel_domains": tunnel_domains,
        "total_edges": len(edges),
        "domains_per_realm": dict(realm_counts.most_common()),
        "top_tunnels": [
            {"domain": d, "realms": r["realms"], "count": r["count"]}
            for d, r in sorted(nodes.items(), key=lambda x: -len(x[1]["realms"]))[:10]
            if len(r["realms"]) >= 2
        ],
    }


def _fuzzy_match(query: str, nodes: dict, n: int = 5):
    """Find domains that approximately match a query string."""
    query_lower = query.lower()
    scored = []
    for domain in nodes:
        if query_lower in domain:
            scored.append((domain, 1.0))
        elif any(word in domain for word in query_lower.split("-")):
            scored.append((domain, 0.5))
    scored.sort(key=lambda x: -x[1])
    return [r for r, _ in scored[:n]]


# In-memory cache for tunnels — avoids redundant disk reads on every mutation.
# Keyed by tunnel file path. Invalidated when mtime changes.
_tunnel_cache: dict[str, list[dict]] = {}
_tunnel_cache_mtime: dict[str, int] = {}
_tunnel_cache_lock = threading.Lock()


def _tunnel_file(config=None, dimension=None) -> str:
    """Return the path to the tunnels.json file, derived from AltMemoryConfig or dimension.

    When ``dimension`` is provided the path is derived from the dimension's
    base directory. Otherwise falls back to ``AltMemoryConfig.tunnel_file``.
    """
    if dimension is not None:
        return os.path.join(str(dimension._base.parent), "tunnels.json")
    config = config or AltMemoryConfig()
    return config.tunnel_file


def _load_tunnels(config=None, dimension=None):
    """Load explicit tunnels from disk.

    Returns an empty list if the file is missing or corrupt (e.g. truncated
    by a crash mid-write on a system that lacks atomic-rename semantics).

    ``config`` may be passed in by the caller to avoid re-instantiating
    ``AltMemoryConfig`` (which re-reads config from disk) on every helper
    call within a single create_tunnel cycle.

    ``dimension`` may be passed instead of ``config`` to resolve the tunnel
    file path from the dimension's base directory. Takes precedence if both
    are supplied.
    """
    tunnel_file = _tunnel_file(config=config, dimension=dimension)
    stat_result = None

    # Check in-memory cache (under lock)
    with _tunnel_cache_lock:
        try:
            stat_result = os.stat(tunnel_file)
            cached_mtime = _tunnel_cache_mtime.get(tunnel_file)
            if cached_mtime is not None and stat_result.st_mtime_ns == cached_mtime:
                cached = _tunnel_cache.get(tunnel_file)
                if cached is not None:
                    return list(cached)
        except OSError:
            pass

    try:
        with open(tunnel_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        result = []
    except Exception:
        logger.warning(
            "Tunnels file '%s' is corrupt or unreadable; starting empty.",
            tunnel_file,
        )
        result = []
    else:
        result = data if isinstance(data, list) else []

    # Update cache (under lock)
    with _tunnel_cache_lock:
        _tunnel_cache[tunnel_file] = list(result)
        if stat_result is not None:
            _tunnel_cache_mtime[tunnel_file] = stat_result.st_mtime_ns
        else:
            try:
                _tunnel_cache_mtime[tunnel_file] = os.stat(tunnel_file).st_mtime_ns
            except OSError:
                _tunnel_cache_mtime.pop(tunnel_file, None)
    return result


def _save_tunnels(tunnels, config=None, dimension=None):
    """Persist explicit tunnels atomically.

    Writes to ``tunnels.json.tmp`` then ``os.replace``s it into place, so
    a crash mid-write can never leave a partial/empty tunnels.json that
    silently wipes every tunnel on next read.

    Also restricts the parent directory to 0o700 and the file to 0o600.

    ``config`` may be passed in by the caller to avoid re-instantiating
    ``AltMemoryConfig`` on every save.

    ``dimension`` may be passed instead of ``config`` to resolve the tunnel
    file path from the dimension's base directory. Takes precedence if both
    are supplied.
    """
    tunnel_file = _tunnel_file(config=config, dimension=dimension)
    parent = os.path.dirname(tunnel_file)
    os.makedirs(parent, exist_ok=True)
    try:
        os.chmod(parent, 0o700)
    except (OSError, NotImplementedError):
        pass
    tmp_path = tunnel_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(tunnels, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp_path, tunnel_file)
    try:
        os.chmod(tunnel_file, 0o600)
    except (OSError, NotImplementedError):
        pass

    # Update in-memory cache (under lock)
    with _tunnel_cache_lock:
        _tunnel_cache[tunnel_file] = list(tunnels)
        try:
            _tunnel_cache_mtime[tunnel_file] = os.stat(tunnel_file).st_mtime_ns
        except OSError:
            _tunnel_cache_mtime.pop(tunnel_file, None)


def _endpoint_key(realm: str, domain: str) -> str:
    return f"{realm}/{domain}"


def _canonical_tunnel_id(
    source_realm: str, source_domain: str, target_realm: str, target_domain: str
) -> str:
    src = _endpoint_key(source_realm, source_domain)
    tgt = _endpoint_key(target_realm, target_domain)
    a, b = sorted((src, tgt))
    return hashlib.sha256(f"{a}↔{b}".encode()).hexdigest()[:16]


def _require_name(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _check_domain_exists(realm: str, domain: str, dimension) -> bool:
    """Check if at least one entity exists for the given realm/domain in SQLite."""
    if dimension is None:
        return True
    try:
        row = dimension._db.execute(
            "SELECT COUNT(*) FROM entities WHERE realm=? AND domain=?", (realm, domain)
        ).fetchone()
        return row[0] > 0
    except Exception:
        logger.warning(
            "Error checking domain existence for %s/%s; allowing tunnel creation.",
            realm,
            domain,
            exc_info=True,
        )
        return True


def create_tunnel(
    source_realm: str,
    source_domain: str,
    target_realm: str,
    target_domain: str,
    label: str = "",
    dimension=None,
    source_entity_id: str = None,
    target_entity_id: str = None,
    kind: str = "explicit",
):
    """Create an explicit (symmetric) tunnel between two locations in the dimension.

    Tunnels are undirected: ``create_tunnel(A, B)`` and ``create_tunnel(B, A)``
    resolve to the same canonical ID. A second call with the same endpoints
    updates the stored label (and entity IDs, if provided) rather than
    creating a duplicate. Endpoints are compared **verbatim** — ``"my-realm"``
    and ``"my_realm"`` are distinct (see Note below and #1504).

    The ``source`` / ``target`` fields on the returned dict preserve the
    argument order the caller used, so callers can display it directionally
    if they like. The ID and dedup are symmetric.

    Args:
        source_realm: Realm of the source (e.g., "project_api").
        source_domain: Domain in the source realm.
        target_realm: Realm of the target (e.g., "project_database").
        target_domain: Domain in the target realm.
        label: Description of the connection.
        source_entity_id: Optional specific entity ID.
        target_entity_id: Optional specific entity ID.
        kind: Tunnel category — ``"explicit"`` (default, user-created link
            between real domains) or ``"topic"`` (auto-generated cross-realm
            topical link where domains are synthetic ``topic:<name>``
            identifiers).

    Returns:
        The stored tunnel dict.

    Raises:
        ValueError: if any realm or domain is empty or non-string, or if an explicit
                    tunnel points to a nonexistent domain.

    Note:
        Realm slugs are stored verbatim — passing ``"my-realm"`` and ``"my_realm"``
        produces two distinct tunnels (canonical IDs differ). Read-path helpers
        (``list_tunnels`` / ``follow_tunnels``) normalize both sides at compare
        time so legacy underscore data and explicit-flag hyphen data both
        match queries in either form. See #1504.
    """
    # NOTE: must NOT be called while already holding mine_lock —
    # this function acquires it internally (line 484), and
    # mine_lock is a file lock, not re-entrant.
    source_realm = _require_name(source_realm, "source_realm")
    source_domain = _require_name(source_domain, "source_domain")
    target_realm = _require_name(target_realm, "target_realm")
    target_domain = _require_name(target_domain, "target_domain")

    if kind == "explicit":
        if dimension is None:
            dimension = _get_dimension()
        if not _check_domain_exists(source_realm, source_domain, dimension):
            raise ValueError(f"Source domain '{source_domain}' does not exist in realm '{source_realm}'")
        if not _check_domain_exists(target_realm, target_domain, dimension):
            raise ValueError(f"Target domain '{target_domain}' does not exist in realm '{target_realm}'")

    tunnel_id = _canonical_tunnel_id(source_realm, source_domain, target_realm, target_domain)

    tunnel = {
        "id": tunnel_id,
        "source": {"realm": source_realm, "domain": source_domain},
        "target": {"realm": target_realm, "domain": target_domain},
        "label": label,
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if source_entity_id:
        tunnel["source"]["entity_id"] = source_entity_id
    if target_entity_id:
        tunnel["target"]["entity_id"] = target_entity_id

    with mine_lock(_tunnel_file(dimension=dimension)):
        tunnels = _load_tunnels(dimension=dimension)
        for existing in tunnels:
            if existing.get("id") == tunnel_id:
                tunnel["created_at"] = existing.get("created_at", tunnel["created_at"])
                tunnel["updated_at"] = datetime.now(timezone.utc).isoformat()
                existing.clear()
                existing.update(tunnel)
                _save_tunnels(tunnels, dimension=dimension)
                return existing
        tunnels.append(tunnel)
        _save_tunnels(tunnels, dimension=dimension)
    return tunnel


def list_tunnels(realm: str = None, dimension=None):
    """List all explicit tunnels, optionally filtered by realm.

    Returns tunnels where ``realm`` appears as either source or target
    (tunnels are symmetric, so either endpoint is a valid filter match).

    Pass ``dimension`` to load tunnels from a non-default dimension's
    ``tunnels.json``.
    """
    norm_realm = _normalize_realm(realm)
    tunnels = _load_tunnels(dimension=dimension)
    if norm_realm:
        tunnels = [
            t
            for t in tunnels
            if _normalize_realm((t.get("source") or {}).get("realm")) == norm_realm
            or _normalize_realm((t.get("target") or {}).get("realm")) == norm_realm
        ]
    return tunnels


def delete_tunnel(tunnel_id: str, dimension=None):
    """Delete an explicit tunnel by ID. Returns ``{"deleted": <id>}``.

    Pass ``dimension`` to delete from a non-default dimension's
    ``tunnels.json``.
    """
    with mine_lock(_tunnel_file(dimension=dimension)):
        tunnels = _load_tunnels(dimension=dimension)
        tunnels = [t for t in tunnels if t.get("id") != tunnel_id]
        _save_tunnels(tunnels, dimension=dimension)
    return {"deleted": tunnel_id}


def follow_tunnels(realm: str, domain: str, dimension=None, config=None):
    """Follow explicit tunnels from a domain — returns connected entities.

    Given a location (realm/domain), finds all tunnels leading from or to it,
    and optionally fetches the connected entity content.
    """
    norm_realm = _normalize_realm(realm) or realm
    tunnels = _load_tunnels(config=config, dimension=dimension)
    connections = []

    for t in tunnels:
        src = t.get("source") or {}
        tgt = t.get("target") or {}

        if _normalize_realm(src.get("realm")) == norm_realm and src.get("domain") == domain:
            connections.append({
                "direction": "outgoing",
                "connected_realm": tgt["realm"],
                "connected_domain": tgt["domain"],
                "label": t.get("label", ""),
                "entity_id": tgt.get("entity_id"),
                "tunnel_id": t["id"],
            })
        elif _normalize_realm(tgt.get("realm")) == norm_realm and tgt.get("domain") == domain:
            connections.append({
                "direction": "incoming",
                "connected_realm": src["realm"],
                "connected_domain": src["domain"],
                "label": t.get("label", ""),
                "entity_id": src.get("entity_id"),
                "tunnel_id": t["id"],
            })

    if not connections:
        logger.warning("No explicit tunnels found for %s/%s", realm, domain)

    if dimension and connections:
        try:
            entity_ids = [c["entity_id"] for c in connections if c.get("entity_id")]
            if entity_ids:
                placeholders = ",".join("?" * len(entity_ids))
                rows = dimension._db.execute(
                    f"SELECT id, content FROM entities WHERE id IN ({placeholders})",
                    entity_ids,
                ).fetchall()
                entity_map = {r[0]: r[1] for r in rows}
                for c in connections:
                    eid = c.get("entity_id")
                    if eid and eid in entity_map:
                        c["entity_preview"] = entity_map[eid][:300]
        except Exception:
            logger.debug("Entity preview hydration failed", exc_info=True)

    return connections


TOPIC_DOMAIN_PREFIX = "topic:"


def _normalize_topic(name: str) -> str:
    return str(name).strip().lower()


def topic_domain(name: str) -> str:
    return f"{TOPIC_DOMAIN_PREFIX}{name}"


def compute_topic_tunnels(
    topics_by_realm: dict,
    min_count: int = 1,
    label_prefix: str = "shared topic",
    dimension=None,
) -> list[dict]:
    """Create tunnels for every pair of realms that share >= ``min_count`` topics.

    Args:
        topics_by_realm: ``{realm_name: [topic_name, ...]}`` mapping. Topic
            names are compared case-insensitively; the first observed
            casing is used for the tunnel domain name.
        min_count: minimum number of overlapping topics required to drop
            any tunnel between a realm pair.
        label_prefix: human-readable string prefixed to the tunnel label.

    Returns:
        List of tunnel dicts as returned by ``create_tunnel`` — one per
        (realm_a, realm_b, topic) triple that crossed the threshold.

    No-op semantics:
      - empty/None ``topics_by_realm`` returns ``[]``.
      - realms whose topic list is empty are skipped.
      - ``min_count <= 0`` is clamped to 1.
    """
    # NOTE: must NOT be called while already holding mine_lock —
    # this function acquires it internally (line 643), and
    # mine_lock is a file lock, not re-entrant.
    if not topics_by_realm:
        return []

    min_count = max(1, int(min_count))

    realm_topics: dict[str, dict[str, str]] = {}
    for realm, names in topics_by_realm.items():
        if not isinstance(realm, str) or not realm.strip():
            continue
        if not isinstance(names, (list, tuple)):
            continue
        bucket: dict[str, str] = {}
        for n in names:
            if not isinstance(n, str):
                continue
            key = _normalize_topic(n)
            if not key:
                continue
            bucket.setdefault(key, n.strip())
        if bucket:
            realm_topics[normalize_realm_name(realm.strip())] = bucket

    realms = sorted(realm_topics.keys())
    created: list[dict] = []
    with mine_lock(_tunnel_file(dimension=dimension)):
        tunnels = _load_tunnels(dimension=dimension)
        for i, ra in enumerate(realms):
            topics_a = realm_topics[ra]
            for rb in realms[i + 1:]:
                topics_b = realm_topics[rb]
                shared_keys = set(topics_a.keys()) & set(topics_b.keys())
                if len(shared_keys) < min_count:
                    continue
                for key in sorted(shared_keys):
                    topic_name = topics_a[key] if topics_a[key] else topics_b[key]
                    domain = topic_domain(topic_name)
                    tunnel_id = _canonical_tunnel_id(ra, domain, rb, domain)
                    tunnel = {
                        "id": tunnel_id,
                        "source": {"realm": ra, "domain": domain},
                        "target": {"realm": rb, "domain": domain},
                        "label": f"{label_prefix}: {topic_name}",
                        "kind": "topic",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    found = False
                    for existing in tunnels:
                        if existing.get("id") == tunnel_id:
                            tunnel["created_at"] = existing.get("created_at", tunnel["created_at"])
                            tunnel["updated_at"] = datetime.now(timezone.utc).isoformat()
                            existing.clear()
                            existing.update(tunnel)
                            created.append(existing)
                            found = True
                            break
                    if not found:
                        tunnels.append(tunnel)
                        created.append(tunnel)
        _save_tunnels(tunnels, dimension=dimension)
    return created


def topic_tunnels_for_realm(
    realm: str,
    topics_by_realm: dict,
    min_count: int = 1,
    label_prefix: str = "shared topic",
    dimension=None,
) -> list[dict]:
    """Compute topic tunnels involving a single realm.

    Used by the miner to incrementally update tunnels for the realm that
    just finished mining without recomputing pairs that don't involve it.
    Returns the list of tunnels created or refreshed.
    """
    if not topics_by_realm or not isinstance(realm, str) or not realm.strip():
        return []

    realm = normalize_realm_name(realm.strip())
    own = topics_by_realm.get(realm)
    if own is None:
        for k, v in topics_by_realm.items():
            if isinstance(k, str) and normalize_realm_name(k.strip()) == realm:
                own = v
                break
    if not isinstance(own, (list, tuple)) or not own:
        return []

    created: list[dict] = []
    for other, other_topics in topics_by_realm.items():
        if not isinstance(other, str) or not other.strip():
            continue
        if normalize_realm_name(other.strip()) == realm:
            continue
        if not isinstance(other_topics, (list, tuple)) or not other_topics:
            continue
        slice_map = {realm: list(own), other: list(other_topics)}
        created.extend(
            compute_topic_tunnels(
                slice_map,
                min_count=min_count,
                label_prefix=label_prefix,
                dimension=dimension,
            )
        )
    return created


def entity_tunnels_for_realm(
    realm: str,
    gateways: list,
    label_prefix: str = "shared entity",
    dimension=None,
) -> list:
    """Compute entity tunnels involving a single realm.

    An entity tunnel bridges two realms when the same entity (person,
    project, concept, interest) appears in within-realm gateways of both.
    This is the architectural counterpart to ``topic_tunnels_for_realm`` —
    same storage path (``create_tunnel`` → tunnels.json),
    same dedup, same listing API.

    Endpoints use the synthetic domain id ``entity:<name>`` (mirrors
    ``topic:<slug>``) so they can't collide with literal folder-derived
    domains of the same name. Casing of the entity is preserved.

    Topic tunnels are NOT replaced — both systems coexist for one release
    cycle while entity tunnels prove out. Deprecation is a separate PR.
    """
    if not gateways or not isinstance(realm, str) or not realm.strip():
        return []

    realm_norm = normalize_realm_name(realm.strip())

    entity_realms: dict = {}
    for h in gateways:
        if not isinstance(h, dict):
            continue
        h_realm = h.get("realm")
        if not isinstance(h_realm, str) or not h_realm.strip():
            continue
        h_realm_norm = normalize_realm_name(h_realm.strip())
        for ent_key in ("entity_a", "entity_b"):
            ent = h.get(ent_key)
            if not isinstance(ent, str) or not ent.strip():
                continue
            entity_realms.setdefault(ent, {}).setdefault(h_realm_norm, h_realm)

    if not entity_realms:
        return []

    created: list = []
    for entity in sorted(entity_realms.keys()):
        realms_for_entity = entity_realms[entity]
        if realm_norm not in realms_for_entity:
            continue
        own_realm_display = realms_for_entity[realm_norm]
        other_realms_norm = sorted(r for r in realms_for_entity if r != realm_norm)
        for other_norm in other_realms_norm:
            other_display = realms_for_entity[other_norm]
            domain = f"entity:{entity}"
            tunnel = create_tunnel(
                source_realm=own_realm_display,
                source_domain=domain,
                target_realm=other_display,
                target_domain=domain,
                label=f"{label_prefix}: {entity}",
                kind="entity",
                dimension=dimension,
            )
            created.append(tunnel)
    return created
