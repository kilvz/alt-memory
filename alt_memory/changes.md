# Changes — Alt Memory Audit Cleanup

Generated: 2026-05-30
Scope: F:\alt-memory

---

## 1. dim_graph.py — Bug Fix + Rename

### Bug: `data["halls"]` → `data["gates"]`

**`dim_graph.py:141`**: Data was stored in dict key `"gates"` (line 117: `domain_data[domain]["gates"].add(gate)`) but read from `"halls"` (line 141: `"halls": sorted(data["halls"])`). Because `domain_data` is a `defaultdict(lambda: ...)`, `data["halls"]` never raised KeyError — it silently created a new empty set every time. **Result: the `"halls"` field in every graph output was always `[]` regardless of actual gate data.**

**Fix**: `data["halls"]` → `data["gates"]`. Now returns actual gate data.

### Rename: all `"halls"` keys → `"gates"`

4 occurrences renamed in output dicts across `build_graph()`, `traverse()`, `find_tunnels()`:

- `dim_graph.py:141` — nodes dict construction
- `dim_graph.py:175` — traverse results
- `dim_graph.py:198` — BFS expansion results  
- `dim_graph.py:234` — find_tunnels results

### Comment fix: `dynamics.py:1`

`"halls + tunnels"` → `"gates + tunnels"`

---

## 2. SQL Indexes — dimension.py

Added 2 new indexes after existing `idx_entities_realm_domain` on line 414:

```sql
CREATE INDEX IF NOT EXISTS idx_entities_source ON entities(source_file)
CREATE INDEX IF NOT EXISTS idx_entities_created_at ON entities(created_at)
```

**Impact**: Eliminates full table scans on `WHERE source_file = ?` (9+ call sites across convo_miner, dimension, dim_graph, sync) and `ORDER BY created_at DESC` (list_entities, memories_filed_away).

---

## 3. N+1 Query Fixes

### 3a. sweeper._flush_batch — batch ID pre-flight

**Before (sweeper.py:261-263)**:
```python
for bid in batch_ids_set:
    if dimension.get_entity(bid):   # 64 SQL round-trips
        present.add(bid)
```

**After**:
```python
if batch_ids_set:
    placeholders = ",".join("?" * len(batch_ids_set))
    rows = dimension._db_execute(
        f"SELECT id FROM entities WHERE id IN ({placeholders})",
        list(batch_ids_set),
    ).fetchall()
    present = {r[0] for r in rows}    # 1 SQL round-trip
```

**Behavior**: Identical — both check entity existence by ID. Single query vs 64.

### 3b. sweeper.get_dimension_cursor — SQL aggregate

**Before (sweeper.py:166-185)**:
```python
rows = conn.execute(...).fetchall()  # loads ALL metadata
metas = [json.loads(r[0]) for r in rows]  # parses ALL JSON
return max(m.get("timestamp") for m in metas)  # Python max
```

**After**:
```python
row = conn.execute(
    "SELECT json_extract(metadata, '$.timestamp') FROM entities "
    "WHERE json_extract(metadata, '$.session_id') = ? "
    "ORDER BY json_extract(metadata, '$.timestamp') DESC LIMIT 1"
).fetchone()
return row[0] if row else None
```

**Behavior**: Identical — both return the maximum timestamp. One row instead of all rows.

### 3c. convo_miner._file_chunks_locked — batch_add_entities

**Before (convo_miner.py:535-543)**:
```python
for doc_id, doc, meta in zip(batch_ids, batch_docs, batch_metas):
    dim.add_entity(...)   # 1 embed + 1 SQL INSERT + 1 _save_embedder per chunk
```

**After**:
```python
batch_tuples = [ ... ]
dim.batch_add_entities(batch_tuples)  # batch embed + batch INSERT + 1 _save_embedder
```

**Behavior**: Same entities created with same content/metadata/IDs. Single transaction vs N transactions.

### 3d. convo_miner._file_chunks_locked — delete_entities

**Before (convo_miner.py:495-496)**:
```python
for did in delete_ids:
    dim.delete_entity(did)  # per-ID: SELECT + store.delete + 3× SQL DELETE
```

**After**:
```python
dim.delete_entities(delete_ids)  # batch: store.delete + DELETE WHERE id IN (...)
```

**Behavior**: Same deletes, single lock acquisition.

### 3e. dimension.batch_add_entities — FTS5 DELETE

**Before (dimension.py:712-713)**:
```python
for eid, content in fts_rows:
    self._db.execute("DELETE FROM entities_fts WHERE id = ?", (eid,))
```

**After**:
```python
self._db.executemany("DELETE FROM entities_fts WHERE id = ?",
                     [(eid,) for eid, _ in fts_rows])
```

**Behavior**: Same deletes, single SQLite prepared-statement replay.

### 3f. dimension.rebuild_fts — INSERT

**Before (dimension.py:1175-1177)**:
```python
for rid, content in rows:
    self._db.execute("INSERT INTO entities_fts ...", (rid, content))
```

**After**:
```python
self._db.executemany("INSERT INTO entities_fts ...", rows)
```

**Behavior**: Same inserts, single prepared-statement replay.

---

## 4. New Methods on Dimension

### `batch_add_entities()`

```python
def batch_add_entities(
    self,
    entities: list[tuple[str, str, str, dict, str, Optional[str]]],
) -> list[str]:
```

Each tuple: `(realm, domain, content, metadata, source_file, entity_id)`.

Bulk operation:
- Single embedder call for all texts (batch embedding)
- Single `store.add()` call
- Single `executemany` for SQL INSERT
- Single `executemany` for FTS5 DELETE + INSERT
- Single `_save_embedder()`
- Single transaction

**Callers**: sweeper._flush_batch, convo_miner._file_chunks_locked

### `delete_entities()`

```python
def delete_entities(self, entity_ids: list[str]) -> int:
```

Bulk delete:
- Single `store.delete(ids=entity_ids)`
- Single `DELETE FROM entities WHERE id IN (...)`
- Single `DELETE FROM entities_fts WHERE id IN (...)`
- Single transaction

**Callers**: convo_miner._file_chunks_locked

---

## 5. Duplicate Code Consolidation

### 5a. SKIP_DIRS — 7 copies → 1 authoritative source

**Authoritative source**: `dimension.py:31` (expanded from 7→27 entries)

```python
SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".next", "coverage", ".ruff_cache", ".mypy_cache",
    ".pytest_cache", ".cache", ".tox", ".nox", ".idea", ".vscode",
    ".ipynb_checkpoints", ".eggs", "htmlcov", "target", ".terraform",
    "vendor", ".opencode", ".claude", ".alt-memory",
})
```

**All 6 consumers now import the same object** (verified: `is` check passes):
- `convo_miner.py` — `from alt_memory.dimension import SKIP_DIRS`
- `entity_detector.py` — `from alt_memory.dimension import SKIP_DIRS`
- `domain_detector_local.py` — `from alt_memory.dimension import SKIP_DIRS`
- `format_miner.py` — `from alt_memory.dimension import SKIP_DIRS`
- `project_scanner.py` — `from alt_memory.dimension import SKIP_DIRS`
- `miner.py` — already imported from dimension.py

**Inline duplicate removed**: `domain_detector_local.py:120` had `skip_dirs = {".git", ..., "build"}` — replaced with imported `SKIP_DIRS`.

**Behavioral change**: Some consumers now skip more directories:
- `domain_detector_local.detect_domains_from_files` — previously skipped 7 dirs, now skips 27
- `entity_detector.scan_for_detection` — previously skipped 11 dirs, now skips 27
- `domain_detector_local._walk_files` — previously skipped 10 dirs, now skips 27

All additions are well-known tool/cache dirs that should never be project content.

### 5b. _ENTITY_STOPLIST — 2 copies → 1 authoritative source

**Authoritative source**: `dimension.py:41` (expanded from 11→198 entries)

Now includes pronouns, articles, common verbs, titles, months/days, technical terms (True/False/Null), gerunds, etc. — union of both previous copies.

**entity_detector.py** now does `from alt_memory.dimension import _ENTITY_STOPLIST` instead of defining its own 55-line set.

**Verified**: `entity_detector._ENTITY_STOPLIST is dimension._ENTITY_STOPLIST` → `True`

**Behavioral change**: Both `miner._extract_entities_for_metadata` and `dimension._candidate_entity_words` now filter 198 stop words instead of ~20. Fewer false-positive entity detections (pronouns, article words, gerunds, titles now filtered).

### 5c. Entity extraction loop — 2 copies → 1 shared function

**New function in `entity_detector.py`**:
```python
def _extract_candidate_words(
    text: str, min_len: int = 2, deduplicate: bool = False
) -> dict[str, int] | list[str]:
```

- When `deduplicate=False`: returns `dict[str, int]` counting occurrences (miner.py style)
- When `deduplicate=True`: returns `list[str]` deduplicated (dimension.py style)
- Supports both `min_len=1` (dimension.py) and `min_len=2` (miner.py)
- Shared logic: COCA filter loading, i18n stopwords, known-systems prepass, regex matching

**Callers**:
- `miner._extract_entities_for_metadata()` — calls with `min_len=2`
- `dimension._candidate_entity_words()` — calls with `min_len=1, deduplicate=True`

**Regex equivalence**:
- miner.py old: `\b[A-Z][a-zA-Z'\-]{2,50}\b` → new: `\b[A-Z][a-zA-Z'\-]{2,50}\b` (identical)
- dimension.py old: `\b[A-Z][a-zA-Z'\-]{1,50}\b` → new: `\b[A-Z][a-zA-Z'\-]{1,50}\b` (identical)

### 5d. _DEFAULT_DIM = 384

**Moved** from `chroma_store.py:19` and `faiss_store.py:27` to `backends/base.py:25` as `DEFAULT_DIM`.

Both stores import: `from alt_memory.backends.base import DEFAULT_DIM`

### 5e. ENTITY_UPSERT_BATCH_SIZE, DEFAULT_MAX_FILE_SIZE

**Moved** to `dimension.py` (after `_STORE_BACKUP_FILES`):
- `ENTITY_UPSERT_BATCH_SIZE = 1000`
- `DEFAULT_MAX_FILE_SIZE = 500 * 1024 * 1024`

`format_miner.py` now imports both instead of defining locally.

---

## 6. search/search_memories Consolidation

**`dimension.py:894-914`** — `search_memories()` now delegates early-return cases to `search()`:

```python
mode = "keyword" if vector_disabled else "hybrid"
sr = self.search(query, n_results, realm, domain, mode=mode)
if vector_disabled or not self._store or self._store.count() == 0:
    return {
        "query": query,
        "filters": {"realm": realm, "domain": domain},
        "results": [
            {
                "id": r.id, "text": r.text,
                "distance": None if vector_disabled else r.distance,
                "metadata": r.metadata, "realm": r.realm, "domain": r.domain,
            }
            for r in sr
        ],
        **({"fallback": "keyword_only"} if vector_disabled else {}),
    }
```

**Behavior preserved**:
- `vector_disabled=True` → `distance: None`, `fallback: "keyword_only"` included
- `store empty` → `distance: r.distance` (BM25 score), no fallback key
- Both now route through `search()` dispatch instead of duplicating `_keyword_search()` calls

**Normal hybrid path** (lines 931+) remains unchanged — still calls `_vector_search` + `_keyword_search` directly.

---

## 7. Stale Comment Fix

**`convo_miner.py:69`** — `_detect_gate_cached` docstring:
- Before: `"""... Same logic as miner.detect_gate."""`
- After: `"""..."""` (detect_gate doesn't exist in miner.py)

---

## 8. Deferred (Low Impact)

The following duplicates were identified but left unchanged — merging would change behavior or requires restructuring incompatible output formats:

- `_KNOWN_LOCATIONS`: dialect.py (20 entries) vs entity_detector.py (100+ entries). Different frozenset vs set, different casing. Deferred.
- `_PERSON_TITLES` / `_TITLES`: Overlapping but dialect.py uses lowercase, entity_detector.py uses TitleCase. Deferred.
- ISO date regex patterns: 4 independent regexes, each optimised for different context. Deferred.
- `CHUNK_SIZE = 800` in convo_miner.py — could read from config.py but would change behavior if config differs. Deferred.
- `_TOKEN_RE`: 3 independent tokenizers for different purposes (search, embedding, spellcheck). Deferred.
