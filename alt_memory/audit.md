# Alt Memory — Complete Audit

Generated: 2026-05-30
Scope: F:\alt-memory (entire codebase)

---

## 1. Dead Code Audit

### 1.1 Wholly Dead Modules (4)

| # | File | Lines | What | Why |
|---|------|-------|------|-----|
| 1 | `alt_memory/version.py` | 1–4 | `__version__ = "4.2.0"` | Never imported. Package version is hardcoded in `__init__.py:78` as `4.3.0`. `version.py` still says `4.2.0`. |
| 2 | `alt_memory/_stdio.py` | 1–40 | Windows UTF-8 helper | Never imported anywhere in the codebase. |
| 3 | `alt_memory/node_llm.py` | 1–250+ | `LLMConfig`, `generate_nodes()`, helpers | Never imported anywhere. Not even exported from `__init__.py`. |
| 4 | `alt_memory/sources/transforms.py` | 1–95 | 14 transformation functions + `RESERVED_TRANSFORMATIONS` dict | Never imported. `sources/__init__.py` doesn't re-export it. |

### 1.2 Dead Public API (exported from `__init__.py`, zero internal callers) (6)

| # | File | Line | Symbol | Notes |
|---|------|------|--------|-------|
| 5 | `query_sanitizer.py` | 41 | `sanitize_query()` | Listed in `__all__:108`. Zero internal callers. |
| 6 | `exporter.py` | 68 | `export_dimension()` | Listed in `__all__:111`. Zero internal callers. |
| 7 | `fact_checker.py` | 53 | `check_text()` | Listed in `__all__:130`. Zero internal callers. |
| 8 | `dedup.py` | 98, 120 | `show_stats()`, `dedup_dimension()` | Listed in `__all__:114–115`. Only called from own `__main__` block (lines 199–201). |
| 9 | `convo_scanner.py` | 18, 84 | `is_claude_projects_root()`, `scan_claude_projects()` | Listed in `__all__:140–141`. Only called within own module. |
| 10 | `searcher.py` | 188, 243, 255 | `expand_with_neighbors()`, `render_with_line_numbers()`, `extract_line_range()` | Listed in `__init__.py` exports. None called from outside `searcher.py`. |

### 1.3 Dead Private Functions (7)

| # | File | Line | Symbol | Why |
|---|------|------|--------|-----|
| 11 | `searcher.py` | 93 | `_extract_entity_ids_from_node()` | Never called anywhere, including within `searcher.py`. |
| 12 | `dimension.py` | 247 | `bulk_check_mined()` | Defined but never called. |
| 13 | `dimension.py` | 265 | `prefetch_mined_set()` | Duplicate of `convo_miner.py:605`. Dimension version never called. |
| 14 | `entity_detector.py` | 821 | `confirm_entities()` | Standalone module-level function (not EntityDetector method). Never called. |
| 15 | `gateways.py` | 320, 329 | `list_gateways()`, `delete_gateway()` | Listed in `__all__` but never called from outside `gateways.py`. |
| 16 | `repair.py` | 58, 336 | `TruncationDetected` (exception), `check_extraction_safety()` | Never called. `TruncationDetected` only referenced within dead `check_extraction_safety`. |
| 17 | `record_ingest.py` | 28, 35, 49, 55, 60, 65 | `_state_file_for()`, `_split_entries()`, `chunk_entry()`, `_record_entity_id()`, `_record_node_id_base()`, `_purge_entities_with_source()` | Transitively dead — only used by `ingest_records()` which is dead. |

### 1.4 Dead Code in `backends/chroma_hnsw.py` (6 top-level + ~12 transitive)

Only `_pin_hnsw_threads` (line 532) is live (imported by `chroma_backend.py:28`). Everything else is orphaned:

| # | Line | Symbol | Notes |
|---|------|--------|-------|
| 18 | 154 | `quarantine_stale_hnsw()` | Entire internal call tree (lines 54, 79, 97, 109, 201, 207) transitively dead. |
| 19 | 328 | `_fix_blob_seq_ids()` | Not called from anywhere. |
| 20 | 370 | `_PersistentDataStub` class | Only used by `_SafePersistentDataUnpickler` (line 412), which is itself dead. |
| 21 | 389 | `_SafePersistentDataUnpickler` class | Only used at lines 435 and 637 — both in dead functions. |
| 22 | 451 | `hnsw_capacity_status()` | Entire internal call tree (lines 249, 274, 299, 422) transitively dead. |
| 23 | 609 | `quarantine_invalid_hnsw_metadata()` | Entire internal call tree (lines 557, 561, 567, 573, 594, 662, 670, 681, 687) transitively dead. |

### 1.5 Bug: Wrong Dict Key in `dim_graph.py`

**`dim_graph.py:164`** stores in key `"gates"`:
```python
domain_data[domain]["gates"].add(gate)
```

**`dim_graph.py:173`** reads from key `"halls"` (should be `"gates"`):
```python
"halls": sorted(data["halls"])
```

Because `domain_data` is a `defaultdict(set)`, `data["halls"]` never raises `KeyError` — it returns an empty set every time. The `"halls"` field in output is always `[]`.

### 1.6 Summary

| Category | Count |
|----------|-------|
| Fully dead modules | 4 |
| Dead public API functions | 6 |
| Dead private functions | 7 |
| Dead functions in chroma_hnsw.py | 6 (+ ~12 transitive) |
| Bug (wrong dict key) | 1 |
| **Total dead symbols** | **~23+ top-level** |

---

## 2. Duplicate Code Audit

### 2.1 HIGH Severity (3)

#### Finding 1: `SKIP_DIRS` — defined 7 times, 5 different values

| # | File | Line | Notes |
|---|------|------|-------|
| 1 | `dimension.py` | 31 | `.git`, `__pycache__`, `node_modules`, `.venv`, `venv`, `.opencode`, `.claude` |
| 2 | `entity_detector.py` | 85–97 | Same + `env`, `dist`, `build`, `.next`, `coverage`, `.alt-memory` |
| 3 | `convo_miner.py` | 39–62 | Same as #2 + many more cache/IDE dirs |
| 4 | `format_miner.py` | 41 | Same as #1 (frozenset, identical) |
| 5 | `domain_detector_local.py` | 44 | Same as #2 |
| 6 | `project_scanner.py` | 18–23 | Same as #2 + `.terraform`, `vendor`, `target` |
| 7 | `domain_detector_local.py` | 120 | **Inline duplicate within same file!** Smaller subset |

**Fix:** Merge into `alt_memory/constants.py` or `dimension.py`.

#### Finding 2: `_ENTITY_STOPLIST` — defined 2x, inconsistent

| File | Lines | Contents |
|------|-------|----------|
| `dimension.py` | 35–47 | 14 entries (days, months) + `User, Assistant, System, Tool` |
| `entity_detector.py` | 388–429 | Same 14 + pronouns + many more generic words |

`miner.py` imports from `dimension.py`. `entity_detector.py` has its own independent (larger) copy. Nothing imports entity_detector's version.

**Fix:** entity_detector.py should import from dimension.py's authoritative list.

#### Finding 3: Entity extraction loop — identical in `miner.py` and `dimension.py`

**`miner.py:157–197`** (`_extract_entities_for_metadata`) and **`dimension.py:1448–1476`** (`_candidate_entity_words`) have 90%+ identical logic:

- Same imports (`_get_coca_filter`, `_apply_known_systems_prepass`, `get_entity_patterns`)
- Same coca loading + known_systems prepass
- Same stoplist/stopwords/coca filtering
- Same regex `\b[A-Z][a-zA-Z'\-]...\b`
- Same iteration structure

**Differences:** miner.py counts occurrences (dict), dimension.py deduplicates (seen set). Regex `{2,50}` vs `{1,50}`.

**Fix:** Factor into shared `_extract_candidates(text, min_len=2)` in `entity_detector.py`.

### 2.2 MEDIUM Severity (6)

#### Finding 4: `search()` vs `search_memories()` — 90%+ duplicate logic

**`dimension.py:802–818`** (`search()`) and **`dimension.py:819–973+`** (`search_memories()`).

Duplicated pattern — store-empty + keyword-fallback logic:
```python
if not self._store or self._store.count() == 0:
    if mode == "hybrid":
        return self._keyword_search(query, n_results, realm, domain)
```

`search_memories()` also has a `vector_disabled` branch with the same fallback but returns a slightly different dict format.

**Fix:** `search_memories()` should delegate to `search()` for common dispatch.

#### Finding 5: ISO date regex patterns — #1 and #2 near-identical

| File | Lines | Pattern |
|------|-------|---------|
| `dialect.py` | 178–180 | `\b(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)?)\b` |
| `entity_detector.py` | 468–470 | `\b(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?)\b` (has `Z?` vs `Z`) |

`miner.py:659` and `config.py:52–56` have different patterns for different purposes (parsing vs validation).

**Fix:** Merge #1 and #2 into shared constant.

#### Finding 6: `_DEFAULT_DIM = 384` duplicated

| File | Line |
|------|------|
| `backends/chroma_store.py` | 19 |
| `backends/faiss_store.py` | 27 |

**Fix:** Merge into `backends/__init__.py` or `backends/base.py`.

#### Finding 7: `ENTITY_UPSERT_BATCH_SIZE = 1000` duplicated

| File | Line |
|------|------|
| `convo_miner.py` | 96 |
| `format_miner.py` | 37 |

**Fix:** Merge into shared constant.

#### Finding 8: `MAX_FILE_SIZE = 500 * 1024 * 1024` duplicated

| File | Line | Name |
|------|------|------|
| `convo_miner.py` | 97 | `MAX_FILE_SIZE` |
| `format_miner.py` | 36 | `DEFAULT_MAX_FILE_SIZE` |
| `normalize.py` | 122 | Inline `500 * 1024 * 1024` |

**Fix:** Merge into shared constant.

#### Finding 9: `_KNOWN_LOCATIONS` duplicated

| File | Lines | Scope |
|------|-------|-------|
| `dialect.py` | 84–90 | ~20 major cities (lowercase frozenset) |
| `entity_detector.py` | 260–358 | ~100+ cities, countries, regions, states (mixed-case set) |

**Fix:** entity_detector's list is a superset — dialect.py should reference it.

### 2.3 LOW Severity (5)

#### Finding 10: `CHUNK_SIZE = 800` hardcoded instead of reading config

`convo_miner.py:93` hardcodes `CHUNK_SIZE = 800` with comment "align with miner.py". `config.py:98` already has `DEFAULT_CHUNK_SIZE = 800`.

**Fix:** convo_miner.py should use `AltMemoryConfig().chunk_size`.

#### Finding 11: `_PERSON_TITLES` / `_TITLES` overlapping

`dialect.py:33–36` vs `entity_detector.py:370–373`. 9 common values, 11 unique to dialect, 6 unique to entity_detector.

**Fix:** Merge into single authoritative list.

#### Finding 12: `_TOKEN_RE` — 3 independent implementations

| File | Line | Pattern |
|------|------|---------|
| `searcher.py` | 15 | `\w{2,}` (unicode) |
| `backends/embedder.py` | 22 | `(?u)\b\w[\w'-]*\w\b|\b\w\b` |
| `spellcheck.py` | 158 | `(\S+)` |

**Fix:** Unify #1 and #2 if one tokenizer serves both purposes.

#### Finding 13: Stale comment in `convo_miner.py:68–81`

`_detect_gate_cached` says "Same logic as miner.detect_gate" — but `detect_gate` does not exist in `miner.py`.

**Fix:** Update comment.

#### Finding 14: `_walk_files` + `SKIP_DIRS` pattern repeated 6+ times

Each file independently implements `dirs[:] = [d for d in dirs if d not in SKIP_DIRS]`.

**Fix:** Extract into shared `walk_files()` utility (root cause is SKIP_DIRS duplication).

### 2.4 Summary

| Severity | Count | Key Items |
|----------|-------|-----------|
| HIGH | 3 | SKIP_DIRS (7×), _ENTITY_STOPLIST (2×), entity extraction loop (2×) |
| MEDIUM | 6 | search/search_memories, ISO date regex, _DEFAULT_DIM, BATCH_SIZE, MAX_FILE_SIZE, _KNOWN_LOCATIONS |
| LOW | 5 | CHUNK_SIZE, _PERSON_TITLES, _TOKEN_RE, stale comment, _walk_files pattern |
| INFO | 2 | `from __future__ import annotations` (27 files), `from pathlib import Path` (35 files) |

---

## 3. Performance Audit

### 3.1 N+1 Query Patterns (6 issues)

| # | File:Line | Issue | Severity | Fix |
|---|-----------|-------|----------|-----|
| 1 | `sweeper.py:261–263` | Individual `get_entity(bid)` per 64 IDs | **HIGH** | Replace loop with `SELECT id FROM entities WHERE id IN (...)` batch query |
| 2 | `convo_miner.py:535–543` | `add_entity()` per chunk (up to 1000) instead of `batch_add_entities()` | **CRITICAL** | Accumulate batch and call `batch_add_entities()` once |
| 3 | `convo_miner.py:495–496` | `delete_entity()` per stale ID | **HIGH** | Batch DELETE with `WHERE id IN (...)` |
| 4 | `dimension.py:712–713` | FTS5 DELETE per entity in `batch_add_entities` | **MEDIUM** | Use `executemany` for DELETE, or skip (FTS5 INSERT OR REPLACE) |
| 5 | `dimension.py:1174–1177` | FTS5 INSERT per entity in `rebuild_fts` | **MEDIUM** | Use `executemany` |
| 6 | `sweeper.py:166–171` | `fetchall()` + Python max of JSON metadata | **MEDIUM** | Use `SELECT MAX(json_extract(metadata, '$.timestamp'))` SQL aggregate |

### 3.2 Missing Database Indexes (3 issues)

| # | Table.Column | Severity | Hot Paths | Fix |
|---|-------------|----------|-----------|-----|
| 7 | `entities(source_file)` | **HIGH** | convo_miner, dimension, dim_graph, sync (9+ call sites) | `CREATE INDEX idx_entities_source ON entities(source_file)` |
| 8 | `entities(created_at)` | **MEDIUM** | `list_entities()` (ORDER BY), `memories_filed_away()` | `CREATE INDEX idx_entities_created_at ON entities(created_at)` |
| 9 | FTS5 content-sync table | **LOW** | Manual FTS5 maintenance on every write | Use `content=` option for auto-sync |

### 3.3 Unbounded Memory Usage (4 issues)

| # | File:Line | Issue | Severity | Fix |
|---|-----------|-------|----------|-----|
| 10 | `dim_graph.py:101` | `fetchall()` loads ALL entities (~500MB for 1M) | **HIGH** | Stream via cursor or `LIMIT/OFFSET` |
| 11 | `gateways.py:203–205` | Loads ALL realm entities with full `content` | **HIGH** | Query only `id, metadata` (not `content`); stream via cursor |
| 12 | `convo_miner.py:623–625` | `fetchall()` of ALL source_files + metadata | **HIGH** | Paginated scan or `mined_files` tracking table |
| 13 | `faiss_store.py:226–228` | Loads ALL vector blobs (~150MB for 100K vectors) | **MEDIUM** | Process in pages |

### 3.4 Thread Safety Issues (3 issues)

| # | File:Line | Issue | Severity | Fix |
|---|-----------|-------|----------|-----|
| 14 | `dimension.py:329–333` | `_db_execute` releases lock before caller's `.fetchall()` | **HIGH** | Execute + fetchall inside the lock |
| 15 | `dimension.py:318` + `faiss_store.py:37` | Deadlock risk: Dimension RLock + FaissStore Lock | **MEDIUM** | Document lock ordering (Dimension → FaissStore) |
| 16 | `dimension.py:652, 720, 797` | `_save_embedder()` outside `self._lock` | **MEDIUM** | Move `_save_embedder` inside the lock block |

### 3.5 Inefficient I/O Patterns (4 issues)

| # | File:Line | Issue | Severity | Fix |
|---|-----------|-------|----------|-----|
| 17 | `dimension.py:455–462` | `_save_embedder()` writes entire embedder state (~12M floats as JSON) on EVERY entity write | **HIGH** | Save only after `batch_add_entities` or on `close()` / debounced timer |
| 18 | `dim_graph.py:329–364` | Rewrites entire `tunnels.json` per mutation | **MEDIUM** | Use SQLite for tunnels, or batch writes |
| 19 | `gateways.py:85–114` | Rewrites entire `gateways.json` per computation | **MEDIUM** | Same as above |
| 20 | `faiss_store.py:225–253` | Rebuilds entire FAISS index (O(N)) on every delete/upsert | **HIGH** | Migrate to `IndexIDMap` + `IndexFlatIP` (supports `remove_ids`) |

### 3.6 Expensive Operations on Hot Paths (3 issues)

| # | File:Line | Issue | Severity | Fix |
|---|-----------|-------|----------|-----|
| 21 | `dimension.py:239–244`, `convo_miner.py:639–652` | `PRAGMA quick_check` after every mine | **MEDIUM** | Run only on explicit `--validate` flag or periodic maintenance |
| 22 | `sweeper.py:291` | `fnmatch.fnmatch` recompiles glob patterns per call | **LOW** | Pre-compile with `re.compile(fnmatch.translate(pat))` |
| 23 | `faiss_store.py:398–402` | File write per ID generation (`next_id` → `_save_seq`) | **MEDIUM** | Batch-increment N IDs with one I/O, or use SQLite AUTOINCREMENT |

### 3.7 Inefficient Embedding (2 issues)

| # | File:Line | Issue | Severity | Fix |
|---|-----------|-------|----------|-----|
| 24 | `dimension.py:636` | `add_entity()` embeds 1 text at a time | **HIGH** | Route through `batch_add_entities` internally or buffer |
| 25 | `convo_miner.py:535–543` | Does not use `batch_add_entities` | **HIGH** | See N+1 #2 above |

### 3.8 Inefficient Search (3 issues)

| # | File:Line | Issue | Severity | Fix |
|---|-----------|-------|----------|-----|
| 26 | `dimension.py:1016–1031` | `_vector_search` over-fetches 2× then Python-filters by realm/domain | **MEDIUM** | Add realm/domain filter to FaissStore.search() SQL WHERE |
| 27 | `dimension.py:885–886` | `search_memories` over-fetches 6× for hybrid re-rank (2× × 3×) | **MEDIUM** | Reduce multiplier to 2× or eliminate |
| 28 | `dimension.py:1090–1099` | Allocates new SearchResult objects per result just for normalization | **LOW** | Normalize distance in-place |

### 3.9 Lock Contention (3 issues)

| # | File:Line | Issue | Severity | Fix |
|---|-----------|-------|----------|-----|
| 29 | `dimension.py:318` | Single `RLock` serializes ALL reads and writes | **HIGH** | Split into read-write lock (rwlock) — reads proceed without lock under SQLite WAL |
| 30 | `dim_graph.py:46`, `gateways.py:53`, `embedder.py:195–197` | Global module-level locks serialize across all dimensions | **MEDIUM** | Scope locks to instance or per-realm granularity |
| 31 | `dim_graph.py:487` | `mine_lock` file lock collisions with convo_miner | **LOW** | Document lock hierarchy |

### 3.10 Summary

| Severity | Count | Key Issues |
|----------|-------|------------|
| **CRITICAL** | 1 | convo_miner `add_entity` loop instead of batch (1.2/7.2) |
| **HIGH** | 10 | N+1 queries, missing source_file index, unbounded loads, thread safety, embedder save-per-write, FAISS rebuild-per-delete, single lock |
| **MEDIUM** | 10 | Per-entity FTS5 DELETE, missing created_at index, global locks, full file rewrites, PRAGMA quick_check, next_id I/O, 6× over-fetch |
| **LOW** | 3 | fnmatch recompilation, SearchResult allocation, lock ordering docs |
