"""Ingest daily summary files into the dimension — per-entry entities with chunking."""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from alt_memory.config import AltMemoryConfig
from alt_memory.miner import _extract_entities_for_metadata
from alt_memory.dimension import (
    Dimension,
    build_closet_lines,
    get_closets_collection,
    mine_lock,
    purge_file_closets,
    upsert_closet_lines,
)

logger = logging.getLogger(__name__)

DIARY_ENTRY_RE = re.compile(r"^## .+", re.MULTILINE)
CLOSET_CHAR_LIMIT = 2000


def _state_file_for(dim_path: str, diary_dir: Path) -> Path:
    state_root = Path(os.path.expanduser("~")) / ".alt-memory" / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{dim_path}|{diary_dir}".encode()).hexdigest()[:24]
    return state_root / f"diary_ingest_{key}.json"


def _split_entries(text: str) -> list[dict]:
    parts = DIARY_ENTRY_RE.split(text)
    headers = DIARY_ENTRY_RE.findall(text)
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


def _diary_entity_id(source_file: str, entry_index: int, chunk_index: int) -> str:
    source_hash = hashlib.sha256(source_file.encode()).hexdigest()[:16]
    return f"diary_{source_hash}_{entry_index}_{chunk_index}"


def _diary_closet_id_base(realm: str, date_str: str) -> str:
    suffix = hashlib.sha256(f"{realm}|{date_str}".encode()).hexdigest()[:24]
    return f"closet_diary_{suffix}"


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


def ingest_diaries(
    diary_dir,
    dim_path,
    realm="diary",
    force=False,
):
    diary_dir = Path(diary_dir).expanduser().resolve()
    if not diary_dir.exists():
        print(f"Diary directory not found: {diary_dir}")
        return {"days_updated": 0, "closets_created": 0}

    diary_files = sorted(diary_dir.glob("*.md"))
    if not diary_files:
        print(f"No .md files in {diary_dir}")
        return {"days_updated": 0, "closets_created": 0}

    state_file = _state_file_for(str(dim_path), diary_dir)
    if force or not state_file.exists():
        state = {}
    else:
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {}

    dim = Dimension(dim_path)
    dim.init()
    closets_col = get_closets_collection(dim_path)

    days_updated = 0
    closets_created = 0

    for diary_path in diary_files:
        text = diary_path.read_text(encoding="utf-8", errors="replace")
        if len(text.strip()) < 50:
            continue

        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", diary_path.stem)
        if not date_match:
            continue
        date_str = date_match.group(1)

        state_key = f"{realm}|{diary_path.name}"
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
        source_file = str(diary_path)

        with mine_lock(source_file):
            entries = _split_entries(text)
            prev_entry_count = state.get(state_key, {}).get("entry_count", 0)
            full_rebuild = force or content_changed

            new_entries = entries if full_rebuild else entries[prev_entry_count:]

            if full_rebuild:
                _purge_entities_with_source(dim, source_file)

            if new_entries:
                all_lines = []
                for entry in new_entries:
                    entry_text = f"{entry['header']}\n{entry['body']}" if entry['body'] else entry['header']
                    chunks = chunk_entry(entry_text, CLOSET_CHAR_LIMIT)
                    entry_ids = []
                    for chunk_idx, chunk_text in enumerate(chunks):
                        eid = _diary_entity_id(source_file, entry['entry_index'], chunk_idx)
                        entry_ids.append(eid)
                        dim.add_entity(
                            realm=realm,
                            domain="daily",
                            content=chunk_text,
                            metadata={
                                "entry_index": entry['entry_index'],
                                "entry_header_preview": entry['header'][:120],
                                "chunk_index": chunk_idx,
                                "source_file": source_file,
                                "date": date_str,
                                "filed_at": now_iso,
                            },
                            source_file=source_file,
                            entity_id=eid,
                        )
                    entry_lines = build_closet_lines(
                        text=entry_text,
                        existing={},
                        source_line=source_file,
                        drawer_ids=entry_ids,
                    )
                    all_lines.extend(entry_lines)

                if all_lines:
                    closet_id_base = _diary_closet_id_base(realm, date_str)
                    closet_meta = {
                        "date": date_str,
                        "realm": realm,
                        "domain": "daily",
                        "source_file": source_file,
                        "filed_at": now_iso,
                    }
                    if entities:
                        closet_meta["entities"] = entities
                    if full_rebuild:
                        purge_file_closets(closets_col, source_file)
                    n = upsert_closet_lines(closets_col, closet_id_base, all_lines, closet_meta)
                    closets_created += n

            state[state_key] = {
                "size": curr_size,
                "content_hash": curr_hash,
                "entry_count": len(entries),
                "ingested_at": now_iso,
            }
        days_updated += 1

    state_file.write_text(json.dumps(state, indent=2))
    if days_updated:
        print(f"Diary: {days_updated} days updated, {closets_created} new closets")

    return {"days_updated": days_updated, "closets_created": closets_created}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest daily summaries into the dimension")
    parser.add_argument("--dir", required=True, help="Path to daily_summaries directory")
    parser.add_argument("--dimension", default=os.path.expanduser("~/.alt-memory"))
    parser.add_argument("--realm", default="diary")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ingest_diaries(args.dir, args.dimension, realm=args.realm, force=args.force)
