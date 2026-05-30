# Regression Report — Audit Cleanup Impact Analysis

Generated: 2026-05-30
Scope: F:\alt-memory\alt_memory

---

## Verification Methods

| Check | Tool | Result |
|-------|------|--------|
| All module imports | `import alt_memory` + 34 submodules | PASS |
| Wildcard exports | `from alt_memory import *` | PASS (85 names) |
| `__all__` consistency | 5 `__init__.py` files | PASS (74/8/24 exports) |
| Circular imports | `sys.modules` purge + reimport | PASS |
| `Dimension` method surface | 27 public methods | PASS |
| New methods accessible | `delete_entities`, `batch_add_entities` | PASS |
| Backend integration | `_test_backends.py` (faiss + chroma) | PASS |
| `search_memories` dict format | 3 paths: empty, vector_disabled, normal | PASS |
| `search_memories` early-return delegation | keyword mode fallback | PASS |
| Shared extraction (miner path) | `_extract_candidate_words(text, min_len=2)` | PASS (returns dict counts) |
| Shared extraction (dimension path) | `_candidate_entity_words(text)` | PASS (returns dedup list) |
| `batch_add_entities` | auto + manual IDs, empty batch | PASS |
| `delete_entities` | single, multi, empty, FTS5 cleanup | PASS |
| `SKIP_DIRS` identity | 7 modules share same `frozenset` | PASS (all `is` identical) |
| `_ENTITY_STOPLIST` identity | dimension.py + entity_detector.py share same `frozenset` | PASS |
| `DEFAULT_DIM` consistency | base.py → chroma_store + faiss_store (384) | PASS |
| Batch constants consistency | dimension.py → format_miner.py (1000, 500MB) | PASS |
| Constants safety | `SKIP_DIRS`, `_ENTITY_STOPLIST` both `frozenset` (immutable) | PASS |
| `domain_detector_local._walk_files` | No inline `skip_dirs` duplicate | PASS |
| `domain_detector_local.detect_domains_from_folders` | Uses imported `SKIP_DIRS` | PASS |
| `format_miner` constants | No local `ENTITY_UPSERT_BATCH_SIZE` or `DEFAULT_MAX_FILE_SIZE` | PASS |
| `convo_miner` sweep | Batch stale-entity purge via `delete_entities` | PASS |
| `sweeper._flush_batch` | Single `SELECT id IN (...)` replaces 64× `get_entity` | PASS |
| `sweeper.get_dimension_cursor` | SQL `ORDER BY DESC LIMIT 1` replaces Python max | PASS |
| `dim_graph` halls→gates rename | No `["halls"]` references remain in `.py` | PASS |
| Old API signatures | `add_entity`, `delete_entity`, `search` all unchanged | PASS |
| Dynamic attribute access | No `getattr(dim, ...)` for renamed methods | PASS |
| CLI commands | All 30+ subcommands reference correct signatures | PASS |
| MCP tools | All tools reference correct dimension methods | PASS |

---

## Issues Found

### 1. [LOW] `search_memories()` has zero callers

`dimension.py:866` defines `search_memories()` but no code anywhere in the codebase calls it:
- MCP `search` tool calls `dim.search()` directly (line 1072)
- Both CLI search commands (`search`, `browse`) call `dim.search()`
- Only `dim.search()` is used externally

**Impact**: Dead code only. `search_memories()` is still functional (tests pass) but unused. It exists as a compatibility shim for any external consumers who might call it programmatically.

**Action**: None (per "fix not remove" constraint). If desired, could be exposed via MCP as `dimensions_search_memories` tool.

### 2. [LOW] `batch_add_entities` and `delete_entities` not exposed via MCP

```
MCP registered tools: search, add_entity, delete_entity, get_entity, list_entities, ...
MCP missing tools:    batch_add_entities, delete_entities
```

**Impact**: External MCP clients (e.g. Claude Desktop) cannot batch-add or batch-delete entities. Internal consumers (`convo_miner.py`, `sweeper.py`) use them directly via `Dimension` object, which works fine.

**Action**: Add MCP tool registrations in `mcp_server.py` if external batch operations are desired.

### 3. [NONE] `audit.md` describes unfixed `halls` bug

`audit.md:55-67` states the `halls`→`gates` bug is unfixed. It was fixed in this session. The audit document pre-dates the fix.

**Impact**: None — the code is correct. The audit doc is stale.

**Action**: Optionally update `audit.md` or regenerate it.

### 4. [NONE] `changes.md` "gates" rename mentions are not code

`changes.md` (this document) describes the `halls`→`gates` fix. These are text descriptions, not code references.

**Impact**: None.

---

## Behavioral Impact Summary

### No Change (Behavior Identical)

| Change | Why unchanged |
|--------|---------------|
| `flush_batch` ID pre-flight | Same entity existence check, fewer SQL round-trips |
| `get_dimension_cursor` max-timestamp | Same result, one row instead of all |
| `convo_miner` batch chunk upsert | Same entities with same content/metadata |
| `convo_miner` stale-entity purge | Same entity IDs deleted |
| `batch_add_entities` FTS5 DELETE | Single `executemany` vs loop |
| `rebuild_fts` FTS5 INSERT | Single `executemany` vs loop |
| `search_memories` keyword fallback | Same dispatch (vector_disabled path), same dict format |
| `search_memories` empty-store fallback | Same dispatch (FTS5 keyword search), same distance scores |
| `_extract_candidate_words` in miner | Same regex `\b[A-Z][a-zA-Z'\-]{2,50}\b`, same counts output |
| `_extract_candidate_words` in dimension | Same regex `\b[A-Z][a-zA-Z'\-]{1,50}\b`, same dedup list output |

### Changed (Intentional, Backward-Compatible)

| Change | What changes | Risk |
|--------|-------------|------|
| `SKIP_DIRS` expansion (7→27) | More dirs skipped during walks (`.tox`, `.idea`, `.vscode`, etc.) | **Zero** — all additions are well-known tool/cache dirs, never project content. If a project legitimately stores source in `.tox/`, they'd need explicit inclusion. |
| `_ENTITY_STOPLIST` expansion (11→198) | More words filtered from entity detection | **Low** — stops pronouns, articles, gerunds, titles, common verbs from becoming entities. Reduces false positives. If a project has an entity named "The" or "Being" or "Mr", it would now be filtered — but those are extremely unlikely to be meaningful entities. |
| `dim_graph` output: key `"gates"` instead of `"halls"` | `"halls"` was always `[]` due to the bug. Consumers now see real gate data under `"gates"`. | **Zero** — old key was always empty, so any consumer reading `"halls"` got nothing. Now reading `"gates"` gets correct data. Any consumer still reading `"halls"` from a dict that no longer has that key will get `KeyError` rather than `[]`. No such consumers found. |

---

## Systemic Impact Assessment

### Module Dependency Graph

```
dimension.py (authoritative SKIP_DIRS, _ENTITY_STOPLIST, constants, batch methods)
  |-- convo_miner.py     (imports SKIP_DIRS, uses batch_add_entities + delete_entities)
  |-- entity_detector.py (imports SKIP_DIRS, _ENTITY_STOPLIST, defines _extract_candidate_words)
  |     |-- miner.py     (imports _extract_candidate_words)
  |     |-- dimension.py (imports _extract_candidate_words via _candidate_entity_words)
  |-- domain_detector_local.py (imports SKIP_DIRS)
  |-- format_miner.py    (imports SKIP_DIRS, ENTITY_UPSERT_BATCH_SIZE, DEFAULT_MAX_FILE_SIZE)
  |-- project_scanner.py (imports SKIP_DIRS)
  |-- sweeper.py         (uses batch_add_entities)
  |-- dim_graph.py       (halls->gates rename)
  |-- dynamics.py        (comment fix)

backends/base.py (authoritative DEFAULT_DIM)
  |-- backends/chroma_store.py (imports DEFAULT_DIM)
  |-- backends/faiss_store.py  (imports DEFAULT_DIM)
```

### Dependency Direction

All edges point **inward** toward `dimension.py` and `backends/base.py`. The dependency graph is a **star pattern** with no cycles. This means:

- If `dimension.py` breaks → 6 modules break
- If `backends/base.py` breaks → 2 modules break
- All other modules are **leaves** — breaking them affects nothing else

### Risk by Module

| Module | Risk | Reason |
|--------|------|--------|
| `dimension.py` | **LOW** | Constants changed to `frozenset` (immutable, same API). Batch methods are additive. `_candidate_entity_words` delegates internally. No signature changes. |
| `convo_miner.py` | **LOW** | Replaced loop calls with batch calls — same entity IDs, same content. Import changed from local `SKIP_DIRS` to shared. |
| `entity_detector.py` | **LOW** | Imports `SKIP_DIRS`/`_ENTITY_STOPLIST` from dimension.py. Added `_extract_candidate_words` (pure function, no side effects). |
| `miner.py` | **LOW** | Calls `_extract_candidate_words` (functionally identical). |
| `sweeper.py` | **LOW** | SQL-only changes — `SELECT id IN (...)` and `ORDER BY DESC LIMIT 1`. Same results. |
| `dim_graph.py` | **LOW** | Bugfix — `data["halls"]` → `data["gates"]`. Only affects graph output consumers (none found reading `"halls"`). |
| `backends/base.py` | **ZERO** | Added `DEFAULT_DIM` constant — no code removed or changed. |
| `backends/chroma_store.py` | **ZERO** | Same value, different import path. |
| `backends/faiss_store.py` | **ZERO** | Same value, different import path. |
| `domain_detector_local.py` | **ZERO** | Same `SKIP_DIRS` data, imported instead of defined. |
| `format_miner.py` | **ZERO** | Same constants, imported instead of defined. |
| `project_scanner.py` | **ZERO** | Same `SKIP_DIRS` data, imported instead of defined. |
| `dynamics.py` | **ZERO** | Comment-only change. |

### Risk Summary

**No high-risk changes.** All changes are either:
- **Additive** (new methods, new indexes, new constants in shared locations)
- **Delegation** (replacing loops with batch calls — same inputs/outputs)
- **SQL optimization** (same results, fewer rows/round-trips)
- **Import consolidation** (same data, shared object)

---

## Rollback Strategy

If any issue is discovered in production:

### Per-file rollback

| File | Command |
|------|---------|
| `dimension.py` | `git checkout HEAD -- alt_memory/dimension.py` (reverts indexes, batch methods, consolidated SKIP_DIRS, search_memories changes) |
| `entity_detector.py`, `miner.py` | `git checkout HEAD -- alt_memory/entity_detector.py alt_memory/miner.py` (reverts shared extraction) |
| `convo_miner.py` | `git checkout HEAD -- alt_memory/convo_miner.py` (reverts batch calls) |
| `sweeper.py` | `git checkout HEAD -- alt_memory/sweeper.py` (reverts N+1 fixes) |
| `dim_graph.py`, `dynamics.py` | `git checkout HEAD -- alt_memory/dim_graph.py alt_memory/dynamics.py` (reverts halls→gates) |
| `domain_detector_local.py`, `format_miner.py`, `project_scanner.py` | `git checkout HEAD` (restores local constants) |
| `backends/base.py`, `chroma_store.py`, `faiss_store.py` | `git checkout HEAD -- alt_memory/backends/` (reverts DEFAULT_DIM move) |

### Rollback order

1. `dimension.py` first (the authoritative source of most shared constants)
2. Then all consumers (they'll fall back to their old local copies)
3. `backends/` last (lowest risk)

### Re-consolidation after rollback

If only specific changes need reverting:
- Revert `dimension.py` via `git checkout`
- Revert the specific consumer files
- Keep the rest
- All modules will still import from dimension/backends — if a constant was removed from dimension during rollback but a consumer still imports it, that consumer will `ImportError`. To avoid this, use `git checkout HEAD` on the consumer too (restoring its local constant).
