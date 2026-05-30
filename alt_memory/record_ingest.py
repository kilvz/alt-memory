"""Ingest daily summary files into the dimension — per-entry entities with chunking."""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from alt_memory.miner import _extract_entities_for_metadata
from alt_memory.dimension import (
    Dimension,
    build_node_lines,
    get_nodes_collection,
    mine_lock,
    purge_file_nodes,
    upsert_node_lines,
)

logger = logging.getLogger(__name__)

RECORD_ENTRY_RE = re.compile(r"^## .+", re.MULTILINE)
NODE_CHAR_LIMIT = 2000


def _state_file_for(dim_path: str, record_dir: Path) -> Path:
    state_root = Path(os.path.expanduser("~")) / ".alt-memory" / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{dim_path}|{record_dir}".encode()).hexdigest()[:24]
    return state_root / f"record_ingest_{key}.json"


def _split_entries(text: str) -> list[dict]:
    parts = RECORD_ENTRY_RE.split(text)
    headers = RECORD_ENTRY_RE.findall(text)
    entries = []
    for i, header in enumerate(headers):
        body = parts[i + 1] if i + 1 < len(parts) else ""
        entries.append({
            "header": header.strip(),
            "body": body.strip(),
            "entry_index": i,
        })
    return entries


def chunk_entry(body: str, max_chars: int = 2000) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    return [body[i:i + max_chars] for i in range(0, len(body), max_chars)]


def _record_entity_id(source_file: str, entry_index: int, chunk_index: int) -> str:
    source_hash = hashlib.sha256(source_file.encode()).hexdigest()[:16]
    return f"record_{source_hash}_{entry_index}_{chunk_index}"


def _record_node_id_base(realm: str, date_str: str) -> str:
    suffix = hashlib.sha256(f"{realm}|{date_str}".encode()).hexdigest()[:24]
    return f"node_record_{suffix}"


def _purge_entities_with_source(dim: Dimension, source_file: str):
    rows = dim._db_execute(
        "SELECT id FROM entities WHERE source_file = ?", (source_file,)
    ).fetchall()
    ids = [r[0] for r in rows]
    if not ids:
        return
    with dim._lock:
        dim._store.delete(ids=ids)
        for i in range(0, len(ids), 999):
            chunk = ids[i:i + 999]
            ph = ",".join("?" * len(chunk))
            dim._db.execute(f"DELETE FROM entities WHERE id IN ({ph})", chunk)
            dim._db.execute(f"DELETE FROM entities_fts WHERE id IN ({ph})", chunk)
        dim._db.commit()


def ingest_records(
    record_dir,
    dim_path,
    realm="record",
    force=False,
):
    record_dir = Path(record_dir).expanduser().resolve()
    if not record_dir.exists():
        print(f"Record directory not found: {record_dir}")
        return {"days_updated": 0, "nodes_created": 0}

    record_files = sorted(record_dir.glob("*.md"))
    if not record_files:
        print(f"No .md files in {record_dir}")
        return {"days_updated": 0, "nodes_created": 0}

    state_file = _state_file_for(str(dim_path), record_dir)
    if force or not state_file.exists():
        state = {}
    else:
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {}

    dim = Dimension(dim_path)
    dim.init()
    nodes_col = get_nodes_collection(dim_path)

    days_updated = 0
    nodes_created = 0

    for record_path in record_files:
        text = record_path.read_text(encoding="utf-8", errors="replace")
        if len(text.strip()) < 50:
            continue

        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", record_path.stem)
        if not date_match:
            continue
        date_str = date_match.group(1)

        state_key = f"{realm}|{record_path.name}"
        prev_entry = state.get(state_key, {})
        prev_hash = prev_entry.get("content_hash")
        curr_size = len(text)
        curr_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not force:
            if prev_hash is not None and curr_hash == prev_hash:
                continue
            elif curr_size == prev_entry.get("size", 0) and prev_entry.get("size", 0) > 0:
                state[state_key] = {**prev_entry, "content_hash": curr_hash}
                continue

        content_changed = prev_hash is not None and curr_hash != prev_hash
        now_iso = datetime.now(timezone.utc).isoformat()
        entities = _extract_entities_for_metadata(text)
        source_file = str(record_path)

        with mine_lock(source_file):
            entries = _split_entries(text)
            prev_entry_count = state.get(state_key, {}).get("entry_count", 0)
            full_rebuild = force or content_changed

            new_entries = entries if full_rebuild else entries[prev_entry_count:]

            if full_rebuild:
                _purge_entities_with_source(dim, source_file)

            if new_entries:
                batch: list[tuple[str, str, str, dict, str, Optional[str]]] = []
                entry_chunks: list[tuple[str, list[str]]] = []
                for entry in new_entries:
                    entry_text = f"{entry['header']}\n{entry['body']}" if entry['body'] else entry['header']
                    chunks = chunk_entry(entry_text, NODE_CHAR_LIMIT)
                    entry_ids = []
                    for chunk_idx, chunk_text in enumerate(chunks):
                        eid = _record_entity_id(source_file, entry['entry_index'], chunk_idx)
                        entry_ids.append(eid)
                        batch.append((
                            realm, "daily", chunk_text,
                            {
                                "entry_index": entry['entry_index'],
                                "entry_header_preview": entry['header'][:120],
                                "chunk_index": chunk_idx,
                                "source_file": source_file,
                                "date": date_str,
                                "filed_at": now_iso,
                            },
                            source_file, eid,
                        ))
                    entry_chunks.append((entry_text, entry_ids))

                dim.batch_add_entities(batch)

                all_lines = []
                for entry_text, entry_ids in entry_chunks:
                    entry_lines = build_node_lines(
                        text=entry_text,
                        existing={},
                        source_line=source_file,
                        drawer_ids=entry_ids,
                    )
                    all_lines.extend(entry_lines)

                if all_lines:
                    node_id_base = _record_node_id_base(realm, date_str)
                    node_meta = {
                        "date": date_str,
                        "realm": realm,
                        "domain": "daily",
                        "source_file": source_file,
                        "filed_at": now_iso,
                    }
                    if entities:
                        node_meta["entities"] = entities
                    if full_rebuild:
                        purge_file_nodes(nodes_col, source_file)
                    n = upsert_node_lines(nodes_col, node_id_base, all_lines, node_meta)
                    nodes_created += n

            state[state_key] = {
                "size": curr_size,
                "content_hash": curr_hash,
                "entry_count": len(entries),
                "ingested_at": now_iso,
            }
        days_updated += 1

    state_file.write_text(json.dumps(state, indent=2))
    if days_updated:
        print(f"Record: {days_updated} days updated, {nodes_created} new nodes")

    return {"days_updated": days_updated, "nodes_created": nodes_created}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest daily summaries into the dimension")
    parser.add_argument("--dir", required=True, help="Path to daily_summaries directory")
    parser.add_argument("--dimension", default=os.path.expanduser("~/.alt-memory"))
    parser.add_argument("--realm", default="record")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ingest_records(args.dir, args.dimension, realm=args.realm, force=args.force)
