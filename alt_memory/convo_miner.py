#!/usr/bin/env python3
"""
convo_miner.py — Mine conversations into the dimension.

Ingests chat exports (Claude Code, ChatGPT, Slack, plain text transcripts).
Normalizes format, chunks by exchange pair (Q+A = one unit), files to dimension.

Same dimension as project mining. Different ingest strategy.

    Adapted for alt-memory FAISS-backed Dimension.
"""

import hashlib
import json
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from alt_memory.normalize import normalize
from alt_memory.dimension import Dimension, mine_lock, mine_dimension_lock, MineAlreadyRunning

logger = logging.getLogger("alt_memory")

# Schema version for entity normalization. Bump when the normalization
# pipeline changes in a way that existing drawers should be rebuilt to pick up
# (e.g., new noise-stripping rules). `file_already_mined` treats drawers with
# a missing or stale `normalize_version` as "not mined", so the next mine pass
# silently rebuilds them — users don't need to manually erase + re-mine.
#
# v2 (2026-04): introduced strip_noise() for Claude Code JSONL; previous
#               drawers stored system tags / hook chrome verbatim.
NORMALIZE_VERSION = 2

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "coverage",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    ".eggs",
    "htmlcov",
    "target",
}

# Cached gate keywords — avoids re-reading config per entity
_GATE_KEYWORDS_CACHE = None


def _detect_gate_cached(content: str) -> str:
    """Route content to a gate using cached keywords. Same logic as miner.detect_gate."""
    global _GATE_KEYWORDS_CACHE
    if _GATE_KEYWORDS_CACHE is None:
        from alt_memory.config import AltMemoryConfig

        _GATE_KEYWORDS_CACHE = AltMemoryConfig().gate_keywords
    content_lower = content[:3000].lower()
    scores = {}
    for gate, keywords in _GATE_KEYWORDS_CACHE.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            scores[gate] = score
    return max(scores, key=scores.get) if scores else "general"


# File types that might contain conversations
CONVO_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
}

MIN_CHUNK_SIZE = 30
CHUNK_SIZE = 800  # chars per entity — align with miner.py
_LINE_GROUP_SIZE = 25  # lines per fallback group when no paragraph breaks
_LINE_FALLBACK_MIN_NEWLINES = 20  # trigger line-group fallback above this newline count
ENTITY_UPSERT_BATCH_SIZE = 1000
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB — skip files larger than this.
# Matches miner.py at 500 MB. Long Claude Code sessions, multi-year
# ChatGPT exports, and lifetime Slack dumps routinely exceed 10 MB; the
# cap at that level silently dropped them with `continue`. Per-entity
# size is bounded by CHUNK_SIZE, but larger source files still produce
# more drawers and therefore more embedding/storage work — and content
# is normalized and loaded fully into memory before chunking, so memory
# use also scales with source size.


def _register_file(dim, source_file: str, realm: str, agent: str, extract_mode: str):
    """Write a sentinel so file_already_mined() returns True for 0-chunk files.

    Without this, files that normalize to nothing or produce zero chunks are
    re-read and re-processed on every mine run because nothing was written to
    the dimension on the first pass.
    """
    sentinel_key = f"{source_file}:{extract_mode}"
    sentinel_id = f"_reg_{hashlib.sha256(sentinel_key.encode()).hexdigest()[:24]}"
    meta = {
        "realm": realm,
        "domain": "_registry",
        "source_file": source_file,
        "added_by": agent,
        "filed_at": datetime.now().isoformat(),
        "ingest_mode": "registry",
        "extract_mode": extract_mode,
        "normalize_version": NORMALIZE_VERSION,
    }
    dim._db.execute(
        "INSERT OR REPLACE INTO entities (id, realm, domain, content, metadata, source_file) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (sentinel_id, realm, "_registry", f"[registry] {source_file}", json.dumps(meta), source_file),
    )
    dim._db.execute(
        "INSERT OR REPLACE INTO entities_fts (id, content) VALUES (?, ?)",
        (sentinel_id, f"[registry] {source_file}"),
    )
    dim._db.commit()


def _metadata_matches_extract_mode(meta: dict, extract_mode: Optional[str]) -> bool:
    if extract_mode is None:
        return True
    stored_mode = meta.get("extract_mode")
    return stored_mode == extract_mode or (extract_mode == "exchange" and stored_mode is None)


def _source_file_delete_ids(dim, source_file: str, extract_mode: str) -> list[str]:
    """Collect entity IDs for one source file and extraction mode.

    Legacy conversation drawers did not carry extract_mode; treat those as
    exchange-mode rows so schema rebuilds can still clean them up without
    deleting newer general-mode drawers for the same transcript.
    """
    ids: list[str] = []
    try:
        rows = dim._db.execute(
            "SELECT id, metadata FROM entities WHERE source_file = ?",
            (source_file,),
        ).fetchall()
        for drawer_id, meta_json in rows:
            meta = json.loads(meta_json or "{}")
            if _metadata_matches_extract_mode(meta, extract_mode):
                ids.append(drawer_id)
    except Exception:
        logger.debug("Failed to query source_file delete IDs", exc_info=True)
    return ids


def file_already_mined(
    dim,
    source_file: str,
    check_mtime: bool = False,
    extract_mode: Optional[str] = None,
) -> bool:
    """Check if a file has already been filed in the dimension.

    Returns False (so the file gets re-mined) when:
      - no drawers exist for this source_file
      - the stored `normalize_version` is missing or older than the current
        schema (triggers silent rebuild after a normalization upgrade)
      - `check_mtime=True` and the file's mtime differs from the stored one

    When check_mtime=True (used by project miner), also re-mines on content
    change. When check_mtime=False (used by convo miner), transcripts are
    assumed immutable, so only the version gate triggers a rebuild.

    When extract_mode is set (used by convo miner), idempotency is scoped to
    that extraction mode so exchange-mode and general-mode drawers can coexist
    for the same source transcript. Legacy drawers without extract_mode are
    treated as exchange-mode drawers.
    """
    try:
        rows = dim._db.execute(
            "SELECT metadata FROM entities WHERE source_file = ?",
            (source_file,),
        ).fetchall()
        if not rows:
            return False
        stored_meta = None
        if extract_mode is None:
            stored_meta = json.loads(rows[0][0] or "{}")
        else:
            for row in rows:
                meta = json.loads(row[0] or "{}")
                if _metadata_matches_extract_mode(meta, extract_mode):
                    stored_meta = meta
                    break
        if stored_meta is None:
            return False
        # Pre-v2 drawers have no version field — treat them as stale.
        stored_version = stored_meta.get("normalize_version", 1)
        if stored_version < NORMALIZE_VERSION:
            return False
        if check_mtime:
            stored_mtime = stored_meta.get("source_mtime")
            if stored_mtime is None:
                return False
            current_mtime = os.path.getmtime(source_file)
            return abs(float(stored_mtime) - current_mtime) < 0.001
        return True
    except Exception:
        return False


# =============================================================================
# CHUNKING — exchange pairs for conversations
# =============================================================================


def chunk_exchanges(
    content: str,
    chunk_size: int = None,
    min_chunk_size: int = None,
) -> list:
    """
    Chunk by exchange pair: one > turn + AI response = one unit.
    Falls back to paragraph chunking if no > markers.

    Optional params override module-level defaults when provided.

    Raises ``ValueError`` if ``chunk_size`` is not a positive integer or
    ``min_chunk_size`` is negative. A non-positive ``chunk_size`` would
    cause ``_chunk_by_exchange`` below to loop forever — ``content[:0]``
    is empty, ``content[0:]`` is the whole string, and the remainder
    never shrinks.
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if min_chunk_size is None:
        min_chunk_size = MIN_CHUNK_SIZE

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if min_chunk_size < 0:
        raise ValueError(f"min_chunk_size must be >= 0, got {min_chunk_size}")

    lines = content.split("\n")
    quote_lines = sum(1 for line in lines if line.strip().startswith(">"))

    if quote_lines >= 3:
        return _chunk_by_exchange(lines, chunk_size, min_chunk_size)
    else:
        return _chunk_by_paragraph(content, chunk_size, min_chunk_size)


def _chunk_by_exchange(lines: list, chunk_size: int, min_chunk_size: int) -> list:
    """One user turn (>) + the AI response that follows = one or more chunks.

    The full AI response is preserved verbatim.  When the combined
    user-turn + response exceeds chunk_size the response is split across
    consecutive drawers so nothing is silently discarded.
    """
    chunks = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(">"):
            user_turn = line.strip()
            i += 1

            ai_lines = []
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip().startswith(">") or next_line.strip().startswith("---"):
                    break
                if next_line.strip():
                    ai_lines.append(next_line.strip())
                i += 1

            ai_response = " ".join(ai_lines)
            content = f"{user_turn}\n{ai_response}" if ai_response else user_turn

            _emit_bounded(chunks, content, chunk_size, min_chunk_size)
        else:
            i += 1

    return chunks


def _emit_bounded(
    chunks: list,
    content: str,
    chunk_size: int,
    min_chunk_size: int,
) -> None:
    """Append ``content`` as one or more drawers, none exceeding ``chunk_size``.

    The ``min_chunk_size`` floor gates the WHOLE call (drops the input if
    its stripped length is at or below the floor, treated as noise). Once
    the input passes the floor, every slice is emitted verbatim so a
    small trailing remainder is preserved instead of silently dropped.
    The index-based loop avoids the O(N^2) repeated-substring allocation
    of a ``while content: content = content[chunk_size:]`` shape.
    """
    if len(content.strip()) <= min_chunk_size:
        return
    for i in range(0, len(content), chunk_size):
        chunks.append({"content": content[i : i + chunk_size], "chunk_index": len(chunks)})


def _chunk_by_paragraph(content: str, chunk_size: int, min_chunk_size: int) -> list:
    """Fallback: chunk by paragraph breaks."""
    chunks = []
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    # If no paragraph breaks and long content, chunk by line groups
    if len(paragraphs) <= 1 and content.count("\n") > _LINE_FALLBACK_MIN_NEWLINES:
        lines = content.split("\n")
        for i in range(0, len(lines), _LINE_GROUP_SIZE):
            group = "\n".join(lines[i : i + _LINE_GROUP_SIZE]).strip()
            _emit_bounded(chunks, group, chunk_size, min_chunk_size)
        return chunks

    for para in paragraphs:
        _emit_bounded(chunks, para, chunk_size, min_chunk_size)

    return chunks


# =============================================================================
# ROOM DETECTION — topic-based for conversations
# =============================================================================

TOPIC_KEYWORDS = {
    "technical": [
        "code",
        "python",
        "function",
        "bug",
        "error",
        "api",
        "database",
        "server",
        "deploy",
        "git",
        "test",
        "debug",
        "refactor",
    ],
    "architecture": [
        "architecture",
        "design",
        "pattern",
        "structure",
        "schema",
        "interface",
        "module",
        "component",
        "service",
        "layer",
    ],
    "planning": [
        "plan",
        "roadmap",
        "milestone",
        "deadline",
        "priority",
        "sprint",
        "backlog",
        "scope",
        "requirement",
        "spec",
    ],
    "decisions": [
        "decided",
        "chose",
        "picked",
        "switched",
        "migrated",
        "replaced",
        "trade-off",
        "alternative",
        "option",
        "approach",
    ],
    "problems": [
        "problem",
        "issue",
        "broken",
        "failed",
        "crash",
        "stuck",
        "workaround",
        "fix",
        "solved",
        "resolved",
    ],
}


def detect_convo_domain(content: str) -> str:
    """Score conversation content against topic keywords."""
    content_lower = content[:3000].lower()
    scores = {}
    for domain, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in content_lower)
        if score > 0:
            scores[domain] = score
    if scores:
        return max(scores, key=scores.get)
    return "general"


# =============================================================================
# PER-DIMENSION LOCK
# =============================================================================


# =============================================================================
# SCAN FOR CONVERSATION FILES
# =============================================================================


def scan_convos(convo_dir: str) -> list:
    """Find all potential conversation files.

    Skips symlinks and oversized files. Each skipped symlink is logged to
    ``sys.stderr`` with a ``  SKIP: <relative-path> (symlink)`` line so the
    caller can tell why an apparent conversation directory yielded no files.
    """
    convo_path = Path(convo_dir).expanduser().resolve()
    files = []
    for root, dirs, filenames in os.walk(convo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in filenames:
            if filename.endswith(".meta.json"):
                continue
            filepath = Path(root) / filename
            if filepath.suffix.lower() in CONVO_EXTENSIONS:
                # Skip symlinks and oversized files
                if filepath.is_symlink():
                    rel = filepath.relative_to(convo_path).as_posix()
                    try:
                        print(f"  SKIP: {rel} (symlink)", file=sys.stderr)
                    except OSError:
                        pass
                    continue
                try:
                    if filepath.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                files.append(filepath)
    return files


# =============================================================================
# MINE CONVERSATIONS
# =============================================================================


def _file_chunks_locked(dim, source_file, chunks, realm, domain, agent, extract_mode):
    """Lock the source file, purge stale drawers, and upsert fresh chunks.

    Combines the per-file serialization that prevents concurrent agents from
    duplicating work (via mine_lock) with the normalize-version rebuild
    contract (purge-before-insert so pre-v2 drawers don't survive).

    Returns (drawers_added, domain_counts_delta, skipped).
    """
    domain_counts_delta: dict = defaultdict(int)
    entities_added = 0
    with mine_lock(source_file):
        # Re-check after lock — another agent may have just finished this file
        # at the current schema. A stale-version hit here returns False, so we
        # still fall through to the purge+rebuild path below.
        if file_already_mined(dim, source_file, extract_mode=extract_mode):
            return 0, domain_counts_delta, True

        # Purge stale drawers first. When the normalize schema bumps,
        # file_already_mined() returned False for pre-v2 drawers — clean
        # them out so the source doesn't end up with mixed old/new drawers.
        try:
            delete_ids = _source_file_delete_ids(dim, source_file, extract_mode)
            if delete_ids:
                for did in delete_ids:
                    dim.delete_entity(did)
        except Exception:
            logger.debug("Stale-entity purge failed for %s", source_file, exc_info=True)

        # Batch chunks into bounded upserts so large transcripts keep most of
        # the embedding speedup without one huge request. Keep
        # one filed_at per source file so all transcript drawers share an
        # ingest timestamp.
        filed_at = datetime.now().isoformat()
        for batch_start in range(0, len(chunks), ENTITY_UPSERT_BATCH_SIZE):
            batch_docs: list = []
            batch_ids: list = []
            batch_metas: list = []
            for chunk in chunks[batch_start : batch_start + ENTITY_UPSERT_BATCH_SIZE]:
                chunk_domain = chunk.get("memory_type", "general") if extract_mode == "general" else domain
                if extract_mode == "general":
                    domain_counts_delta[chunk_domain] += 1
                entity_key = f"{source_file}:{extract_mode}:{chunk['chunk_index']}"
                entity_id = (
                    f"entity_{realm}_{chunk_domain}_"
                    f"{hashlib.sha256(entity_key.encode()).hexdigest()[:24]}"
                )
                batch_ids.append(entity_id)
                batch_docs.append(chunk["content"])
                batch_metas.append(
                    {
                        "realm": realm,
                        "domain": chunk_domain,
                        "gate": _detect_gate_cached(chunk["content"]),
                        "source_file": source_file,
                        "chunk_index": chunk["chunk_index"],
                        "added_by": agent,
                        "filed_at": filed_at,
                        "ingest_mode": "convos",
                        "extract_mode": extract_mode,
                        "normalize_version": NORMALIZE_VERSION,
                    }
                )
            try:
                for doc_id, doc, meta in zip(batch_ids, batch_docs, batch_metas):
                    dim.add_entity(
                        realm=meta["realm"],
                        domain=meta["domain"],
                        content=doc,
                        metadata=meta,
                        source_file=meta["source_file"],
                        entity_id=doc_id,
                    )
                entities_added += len(batch_docs)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise
    return entities_added, domain_counts_delta, False


def _is_ai_tool_path(path: Path) -> bool:
    """Return True when `path` lives inside a known AI-tool storage dir.

    Detected paths (exact-segment match — substrings like `.gemini-backup`
    or `.codex-archive` do NOT match):
      - any segment ``.codex`` (Codex CLI sessions / archives)
      - any segment ``.gemini`` (Gemini CLI sessions under ~/.gemini/tmp/...)
      - the consecutive segment pair ``.claude/projects`` (Claude Code).
        ``.claude`` alone is NOT matched — that is the settings/config dir,
        not a conversation source.

    Used by ``_resolve_realm`` to default the destination realm to
    ``realm_api`` when the user hasn't passed an explicit ``--realm``.
    """
    try:
        parts = path.resolve().parts
    except (OSError, RuntimeError):
        return False

    if ".codex" in parts:
        return True
    if ".gemini" in parts:
        return True
    for i in range(len(parts) - 1):
        if parts[i] == ".claude" and parts[i + 1] == "projects":
            return True
    return False


def _resolve_realm(convo_path: Path, realm: Optional[str]) -> str:
    """Determine the destination realm for ``mine_convos``.

    Precedence (first match wins):

      1. Explicit ``realm`` argument from the user — always wins, even on
         an AI-tool path. Empty string is treated as "no realm".
      2. AI-tool path detection — defaults to ``realm_api`` so Claude
         Code / Codex / Gemini conversations group under a single realm
         dedicated to API-sourced content.
       3. Basename fallback — sanitized via ``config.normalize_realm_name``
         (lowercase, spaces/hyphens collapsed to underscores). Shared
         single source of truth with ``cmd_init``,
          ``domain_detector_local``, and ``miner.load_config`` so all
         realm-slug producers stay in sync (per #1194 consolidation).
    """
    from alt_memory.config import normalize_realm_name

    if realm:
        return realm
    if _is_ai_tool_path(convo_path):
        return "realm_api"
    return normalize_realm_name(convo_path.name)


def prefetch_mined_set(dim, extract_mode: Optional[str] = None) -> set[str]:
    """Pre-fetch the set of source_files already mined at the current NORMALIZE_VERSION.

    Mirrors file_already_mined()'s version-gate semantics (check_mtime=False
    branch) but in one bulk pass instead of one query per file.
    Returns a set of source_file paths whose stored drawers are at or above
    NORMALIZE_VERSION; callers do `if path in result_set: skip`.

    When extract_mode is set, mirrors file_already_mined(..., extract_mode=...)
    so conversation mines skip per extraction mode rather than per source file.

    The convo miner walks thousands of transcript files; per-file
    queries against SQLite cost ~2s on a 150k-entity dimension, making a
    2000-file sweep take >1h of pure skip-checking. This helper drops
    that to a single scan plus O(1) lookups.
    """
    mined: set[str] = set()
    try:
        rows = dim._db.execute(
            "SELECT source_file, metadata FROM entities WHERE source_file != ''"
        ).fetchall()
        for src, meta_json in rows:
            meta = json.loads(meta_json or "{}")
            if not _metadata_matches_extract_mode(meta, extract_mode):
                continue
            # Same default as file_already_mined: missing version == 1
            version = meta.get("normalize_version", 1)
            if version >= NORMALIZE_VERSION:
                mined.add(src)
    except Exception:
        logger.warning("prefetch_mined_set: partial fetch, %d files loaded", len(mined))
    return mined


def _validate_dimension_fts5_after_mine(dim_path: str) -> None:
    """Run PRAGMA quick_check on the dimension SQLite DB after a mine."""
    db_path = os.path.join(str(dim_path), "dimension.db")
    if os.path.isfile(db_path):
        try:
            conn = sqlite3.connect(db_path)
            try:
                result = conn.execute("PRAGMA quick_check").fetchone()
                if result and result[0] != "ok":
                    logger.warning("Post-mine integrity check: %s", result[0])
            finally:
                conn.close()
        except Exception:
            logger.debug("Post-mine integrity check skipped", exc_info=True)


def mine_convos(
    convo_dir: str,
    dim_path: str,
    realm: str = None,
    agent: str = "alt-memory",
    limit: int = 0,
    dry_run: bool = False,
    extract_mode: str = "exchange",
):
    """Mine a directory of conversation files into the dimension.

    extract_mode:
        "exchange" — default exchange-pair chunking (Q+A = one unit)
        "general"  — general extractor: decisions, preferences, milestones, problems, emotions

    The real work is in :func:`_mine_convos_impl`; this wrapper holds the
    per-dimension flock around it so two concurrent ``alt-memory mine --mode
    convos`` invocations against the same dimension can't pile up. This
    mirrors the pattern in :func:`alt_memory.miner.mine`. The lock is
    non-blocking: ``MineAlreadyRunning`` propagates to the CLI (which
    renders a holder-aware message and exits non-zero) or to in-process
    callers that expect to coexist with another writer.

    Dry-run skips the lock — it never writes to the dimension and so cannot
    corrupt anything, and skipping the lock lets dry-run probes coexist
    with a live mine.

    Chunking parameters (chunk_size, min_chunk_size) are read from
    AltMemoryConfig inside :func:`_mine_convos_impl` so `config.json`
    governs both this path and the project-file miner in `miner.py`.
    """
    if dry_run:
        return _mine_convos_impl(
            convo_dir,
            dim_path,
            realm=realm,
            agent=agent,
            limit=limit,
            dry_run=dry_run,
            extract_mode=extract_mode,
        )

    with mine_dimension_lock(dim_path):
        return _mine_convos_impl(
            convo_dir,
            dim_path,
            realm=realm,
            agent=agent,
            limit=limit,
            dry_run=dry_run,
            extract_mode=extract_mode,
        )


def _mine_convos_impl(
    convo_dir: str,
    dim_path: str,
    realm: str = None,
    agent: str = "alt-memory",
    limit: int = 0,
    dry_run: bool = False,
    extract_mode: str = "exchange",
):
    from alt_memory.config import AltMemoryConfig

    palace_config = AltMemoryConfig()
    cfg_chunk_size = palace_config.chunk_size
    # Only override convo_miner's MIN_CHUNK_SIZE when the user has set
    # min_chunk_size explicitly. min_chunk_size_explicit returns the
    # validated value or None — None keeps convo's lower 30-char floor
    # (more permissive than the 50-char project default, so short
    # exchanges aren't dropped). Using the validated accessor (not raw
    # _file_config) means a garbage/negative/bool config value can't
    # TypeError the length gate below or ValueError out of
    # chunk_exchanges and abort convo ingest.
    explicit_min = palace_config.min_chunk_size_explicit
    cfg_min_chunk_size = explicit_min if explicit_min is not None else MIN_CHUNK_SIZE

    convo_path = Path(convo_dir).expanduser().resolve()
    realm = _resolve_realm(convo_path, realm)

    files = scan_convos(convo_dir)
    if limit > 0:
        files = files[:limit]

    print(f"\n{'=' * 55}")
    print("  Alt Memory Mine — Conversations")
    print(f"{'=' * 55}")
    print(f"  Realm:    {realm}")
    print(f"  Source:  {convo_path}")
    print(f"  Files:   {len(files)}")
    print(f"  Dimension:  {dim_path}")
    if dry_run:
        print("  DRY RUN — nothing will be filed")
    print(f"{'-' * 55}\n")

    dim = None if dry_run else Dimension(dim_path)
    if dim:
        dim.init()

    # Bulk pre-fetch already-mined set in one paginated pass instead of
    # `len(files)` separate WHERE-source_file queries. On a 150k-entity
    # each per-file query costs ~2s, so a 2000-file sweep used to
    # spend >1h just deciding to skip. prefetch_mined_set() does the same
    # decisions in a single scan; loop body becomes an O(1) set check.
    mined_set: set[str] = (
        prefetch_mined_set(dim, extract_mode=extract_mode) if not dry_run else set()
    )

    total_entities = 0
    files_skipped = 0
    domain_counts = defaultdict(int)

    for i, filepath in enumerate(files, 1):
        source_file = str(filepath)

        # Skip if already filed at current NORMALIZE_VERSION
        if not dry_run and source_file in mined_set:
            files_skipped += 1
            continue

        # Normalize format
        try:
            content = normalize(str(filepath))
        except (OSError, ValueError):
            if not dry_run:
                _register_file(dim, source_file, realm, agent, extract_mode)
            continue

        if not content or len(content.strip()) < cfg_min_chunk_size:
            if not dry_run:
                _register_file(dim, source_file, realm, agent, extract_mode)
            continue

        # Chunk — either exchange pairs or general extraction
        if extract_mode == "general":
            from .general_extractor import extract_memories

            chunks = extract_memories(content, chunk_size=cfg_chunk_size)
            # Each chunk already has memory_type; use it as the domain name
        else:
            chunks = chunk_exchanges(
                content,
                chunk_size=cfg_chunk_size,
                min_chunk_size=cfg_min_chunk_size,
            )

        if not chunks:
            if not dry_run:
                _register_file(dim, source_file, realm, agent, extract_mode)
            continue

        # Detect domain from content (general mode uses memory_type instead)
        if extract_mode != "general":
            domain = detect_convo_domain(content)
        else:
            domain = None  # set per-chunk below

        if dry_run:
            if extract_mode == "general":
                from collections import Counter

                type_counts = Counter(c.get("memory_type", "general") for c in chunks)
                types_str = ", ".join(f"{t}:{n}" for t, n in type_counts.most_common())
                print(f"    [DRY RUN] {filepath.name} \u2192 {len(chunks)} memories ({types_str})")
            else:
                print(f"    [DRY RUN] {filepath.name} \u2192 domain:{domain} ({len(chunks)} entities)")
            total_entities += len(chunks)
            # Track domain counts
            if extract_mode == "general":
                for c in chunks:
                    domain_counts[c.get("memory_type", "general")] += 1
            else:
                domain_counts[domain] += 1
            continue

        # Lock + purge stale + file fresh chunks. Lock serializes concurrent
        # agents; purge removes pre-v2 drawers so the schema bump applies.
        entities_added, domain_delta, skipped = _file_chunks_locked(
            dim, source_file, chunks, realm, domain, agent, extract_mode
        )
        if skipped:
            files_skipped += 1
            continue

        if extract_mode != "general":
            domain_counts[domain] += 1
        for r, n in domain_delta.items():
            domain_counts[r] += n

        total_entities += entities_added
        print(f"  + [{i:4}/{len(files)}] {filepath.name[:50]:50} +{entities_added}")

    if not dry_run:
        _validate_dimension_fts5_after_mine(dim_path)

    print(f"\n{'=' * 55}")
    print("  Done.")
    print(f"  Files processed: {len(files) - files_skipped}")
    print(f"  Files skipped (already filed): {files_skipped}")
    print(f"  Entities filed: {total_entities}")
    if domain_counts:
        print("\n  By domain:")
        label = "entities" if extract_mode == "general" else "files"
        for d, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {d:20} {count} {label}")
    print('\n  Next: alt-memory search "what you\'re looking for"')
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convo_miner.py <convo_dir> [--dim PATH] [--limit N] [--dry-run]")
        sys.exit(1)
    from alt_memory.config import AltMemoryConfig

    mine_convos(sys.argv[1], dim_path=AltMemoryConfig().dim_path)
