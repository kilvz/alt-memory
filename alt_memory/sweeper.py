#!/usr/bin/env python3
"""
sweeper.py — Message-granular miner that catches what the file-level
primary miners dropped.

Algorithm, per session:

    cursor = max(timestamp of sweeper-written entities for this session_id)
    For each user/assistant message in the jsonl:
        if cursor is not None and message.timestamp < cursor: skip
        else: upsert an entity keyed by (session_id, message_uuid)

Properties:

  - Idempotent on its own writes: rerunning is a no-op because entity
    IDs are deterministic and existence is pre-checked before counting.
  - Resume-safe: a crash mid-sweep is recovered on the next run — the
    cursor advances to the last ingested timestamp and re-attempts at
    that boundary are de-duped by the deterministic ID.
  - Tie-break safe: uses ``< cursor`` (not ``<=``), so if multiple
    messages share the max timestamp and only some were ingested, the
    rest are still picked up on re-run.
  - No size caps: each entity holds one exchange, ~1-5 KB.

Coordination with the primary file-level miners (``miner.py`` /
``convo_miner.py``) is limited: those miners chunk at a fixed char size
and do not currently stamp ``session_id``/``timestamp`` metadata that
the sweeper can key off. In practice the sweeper coordinates with its
own prior runs, and may ingest content that also got chunked into
primary-miner entities (under different IDs). Follow-up: add uniform
``ingest_mode`` + message metadata to the primary miners so dedup spans
both paths.

Usage:
    from alt_memory.sweeper import sweep
    result = sweep("/path/to/session.jsonl", "/path/to/dimension")
"""

from __future__ import annotations

import fnmatch
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from alt_memory.dimension import Dimension

logger = logging.getLogger(__name__)


# ── JSONL parsing ────────────────────────────────────────────────────


def _flatten_content(content) -> str:
    """Normalize Claude Code's message content to a plain string.

    User messages are strings already; assistant messages are a list of
    content blocks like [{"type": "text", "text": "..."}, {"type":
    "tool_use", ...}]. All blocks are preserved verbatim — the design
    principle is "verbatim always", so tool inputs and results are
    serialized in full, never truncated.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                parts.append(
                    f"[tool_use: {block.get('name', '?')} "
                    f"input={json.dumps(block.get('input', {}), default=str)}]"
                )
            elif btype == "tool_result":
                parts.append(f"[tool_result: {json.dumps(block.get('content', ''), default=str)}]")
            else:
                parts.append(f"[{btype}: {json.dumps(block, default=str)}]")
        return "\n".join(p for p in parts if p)
    return str(content)


def parse_claude_jsonl(path: str) -> Iterator[dict]:
    """Yield user/assistant records from a Claude Code .jsonl file.

    Each yield is:
        {
          "session_id": str,
          "uuid":       str,   # per-message UUID
          "timestamp":  str,   # ISO 8601
          "role":       "user" | "assistant",
          "content":    str,   # flattened text
        }

    Non-message records (progress, file-history-snapshot, system,
    queue-operation, last-prompt) are filtered out. Malformed lines are
    skipped silently — data quality is the transcript writer's problem,
    not ours.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rtype = record.get("type")
            if rtype not in ("user", "assistant"):
                continue
            msg = record.get("message") or {}
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            timestamp = record.get("timestamp")
            if not timestamp:
                continue
            uuid = record.get("uuid")
            if not uuid:
                continue
            session_id = record.get("sessionId") or record.get("session_id")
            if not session_id:
                continue
            content = _flatten_content(msg.get("content", ""))
            if not content.strip():
                continue
            yield {
                "session_id": session_id,
                "uuid": uuid,
                "timestamp": timestamp,
                "role": role,
                "content": content,
            }


# ── Cursor resolution ────────────────────────────────────────────────


def get_dimension_cursor(dimension, session_id: str) -> Optional[str]:
    """Return the max timestamp of entities for this session_id, or None.

    ISO-8601 strings compare lexically in the right order, so we don't
    need to parse them. Queries the dimension SQLite database directly for
    metadata with the matching session_id.

    Backend errors are logged at WARNING and surface as a `None` cursor —
    which makes the caller treat the session as empty and ingest every
    message. That's intentional: a no-cursor sweep is recovered from on
    the next run by deterministic entity IDs, so a degraded cursor never
    causes silent data loss.
    """
    try:
        row = dimension._db.execute(
            "SELECT json_extract(metadata, '$.timestamp') FROM entities "
            "WHERE json_extract(metadata, '$.session_id') = ? "
            "ORDER BY json_extract(metadata, '$.timestamp') DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    except Exception as exc:
        logger.warning(
            "sweeper: cursor lookup failed for session_id=%s (%s); "
            "treating as empty — entities will be re-upserted idempotently.",
            session_id,
            exc,
        )
        return None
    if row and row[0]:
        return row[0]
    return None


# ── Sweep ────────────────────────────────────────────────────────────


def _entity_id_for_message(session_id: str, message_uuid: str) -> str:
    """Deterministic entity ID so upserts at the same message are no-ops.

    Uses the full session_id (not a prefix) to avoid any cross-session
    collision risk if a transcript source ever uses non-UUID session
    identifiers or shares prefixes across sessions.
    """
    return f"sweep_{session_id}_{message_uuid}"


def sweep(jsonl_path: str, dimension_path: str, source_label: Optional[str] = None,
          skip_before: Optional[str] = None,
          exclude_patterns: Optional[list[str]] = None,
          dry_run: bool = False) -> dict:
    """Ingest every user/assistant message not already represented.

    For each message in the jsonl:
      - If timestamp < cursor for that session, skip (strictly earlier
        than anything already in the dimension — already covered).
      - At timestamp == cursor we do NOT skip, because multiple messages
        can share the same ISO-8601 timestamp; if only some of them were
        ingested before a crash, a ``<= cursor`` skip would lose the rest
        forever. Deterministic entity IDs make re-attempting at the
        cursor boundary safe (existing rows are found via a pre-flight
        ``get(ids=...)`` and counted as "already present", not "added").
      - Else, upsert an entity with deterministic ID so reruns dedupe.

    Parameters
    ----------
    skip_before : str, optional
        ISO-8601 timestamp. Messages strictly before this time are skipped
        (e.g. ``"2026-01-01"`` to ignore work from last year).
    exclude_patterns : list[str], optional
        Glob/wildcard patterns matched against message content. Messages
        whose content matches any pattern are skipped (e.g. ``["*git
        push*", "*npm publish*"]``).
    dry_run : bool, default False
        If True, log messages that *would* be written but do not touch the
        dimension. Accepts (counts as added in the return dict) so callers
        can estimate volume without mutation.

    Returns ``{entities_added, entities_already_present, entities_skipped,
    entities_upserted, cursor_by_session, dry_run}``:

    * ``entities_added`` — rows that did not exist before this sweep.
    * ``entities_already_present`` — rows whose deterministic ID was
      already in the dimension and got rewritten idempotently.
    * ``entities_skipped`` — records skipped by the cursor (strictly
      earlier than what's already stored).
    * ``entities_upserted`` — total writes = added + already_present.
    """
    dimension = Dimension(dimension_path)
    dimension.init()
    cursors: dict = {}

    entities_added = 0
    entities_already_present = 0
    entities_skipped = 0
    entities_excluded = 0

    BATCH_SIZE = 64
    batch: list[tuple] = []
    batch_ids_set: set[str] = set()

    def _flush_batch():
        nonlocal entities_added, entities_already_present
        if not batch:
            return
        # Pre-flight: which IDs in this batch already exist?
        present = set()
        if batch_ids_set:
            placeholders = ",".join("?" * len(batch_ids_set))
            rows = dimension._db_execute(
                f"SELECT id FROM entities WHERE id IN ({placeholders})",
                list(batch_ids_set),
            ).fetchall()
            present = {r[0] for r in rows}
        new_entries = [(r, d, c, m, s, eid)
                       for (r, d, c, m, s, eid) in batch
                       if eid not in present]
        already_count = len(batch) - len(new_entries)
        if new_entries:
            dimension.batch_add_entities(new_entries)
        entities_added += len(new_entries)
        entities_already_present += already_count
        batch.clear()
        batch_ids_set.clear()

    for rec in parse_claude_jsonl(jsonl_path):
        sid = rec["session_id"]
        if sid not in cursors:
            cursors[sid] = get_dimension_cursor(dimension, sid)

        cursor = cursors[sid]
        if cursor is not None and rec["timestamp"] < cursor:
            entities_skipped += 1
            continue

        if skip_before is not None and rec["timestamp"] < skip_before:
            entities_skipped += 1
            continue

        if exclude_patterns:
            content = rec["content"]
            if any(fnmatch.fnmatch(content, pat) for pat in exclude_patterns):
                entities_excluded += 1
                continue

        entity_id = _entity_id_for_message(sid, rec["uuid"])
        document = f"{rec['role'].upper()}: {rec['content']}"
        metadata = {
            "session_id": sid,
            "timestamp": rec["timestamp"],
            "message_uuid": rec["uuid"],
            "role": rec["role"],
            "source_file": source_label or jsonl_path,
            "filed_at": datetime.now().isoformat(),
            "ingest_mode": "sweep",
        }

        if dry_run:
            logger.info("sweeper[dry-run]: would upsert %s (%s chars)", entity_id, len(document))
            entities_added += 1
            continue

        batch.append((
            "sweeper", "messages", document, metadata,
            source_label or jsonl_path, entity_id,
        ))
        batch_ids_set.add(entity_id)

        if len(batch) >= BATCH_SIZE:
            _flush_batch()

    _flush_batch()

    return {
        "entities_added": entities_added,
        "entities_already_present": entities_already_present,
        "entities_upserted": entities_added + entities_already_present,
        "entities_skipped": entities_skipped,
        "entities_excluded": entities_excluded,
        "cursor_by_session": cursors,
        "dry_run": dry_run,
    }


def sweep_directory(dir_path: str, dimension_path: str) -> dict:
    """Sweep every .jsonl file in a directory (recursive).

    Returns aggregated summary across all files. ``files_attempted``
    includes files that raised, so the count reflects discovery rather
    than only successes; ``files_succeeded`` is the subset that
    completed without error.
    """
    dir_p = Path(dir_path).expanduser().resolve()
    files = sorted(dir_p.rglob("*.jsonl"))

    total_added = 0
    total_already_present = 0
    total_skipped = 0
    per_file = []

    failures: list[dict] = []
    for f in files:
        try:
            result = sweep(str(f), dimension_path, source_label=str(f))
        except Exception as exc:
            logger.error("sweeper: sweep failed on %s: %s", f, exc)
            print(f"  WARNING: sweep failed on {f}: {exc}", file=sys.stderr)
            failures.append({"file": str(f), "error": str(exc)})
            continue
        total_added += result["entities_added"]
        total_already_present += result.get("entities_already_present", 0)
        total_skipped += result["entities_skipped"]
        per_file.append(
            {
                "file": str(f),
                "added": result["entities_added"],
                "already_present": result.get("entities_already_present", 0),
                "skipped": result["entities_skipped"],
            }
        )

    return {
        "files_attempted": len(files),
        "files_succeeded": len(per_file),
        "entities_added": total_added,
        "entities_already_present": total_already_present,
        "entities_skipped": total_skipped,
        "per_file": per_file,
        "failures": failures,
    }
