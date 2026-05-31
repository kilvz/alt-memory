# Alt Memory Architecture

## Overview

Alt Memory is a **local-first persistent memory system** for AI agents. It combines:

- **SQLite** (metadata, content, knowledge graph, FTS5 full-text search)
- **FAISS or ChromaDB** (vector similarity search)
- **Swappable embedders** (numpy BERT, sentence-transformers, ONNX MiniLM/Gemma, spaCy, TF-IDF+SVD)
- **MCP server** (40+ JSON-RPC tools over stdio/SSE)

The system is organized as a **Dimension** — a directory on disk containing all data. Within a dimension, data is organized hierarchically: **Realms** → **Domains** → **Entities**.

---

## File Map

```
alt_memory/
├── __init__.py              # Package entry, version, exports Dimension
├── cli.py                   # CLI argparser + main() — 30+ subcommands
├── mcp_server.py            # MCP JSON-RPC server (stdio + SSE)
├── dimension.py             # Core Dimension class — orchestrates everything
├── dim_graph.py             # Graph traversal, tunnels, gateways
├── domain_detector_local.py # Local heuristic domain detection
├── entity_detector.py       # Entity extraction from source files
├── project_scanner.py       # Discover people/projects in a codebase
├── corpus_origin.py         # Detect corpus origin (AI dialogue, code, etc.)
├── split_megafiles.py       # Split multi-session transcripts into per-session files
├── sync.py                  # Prune stale entities from deleted/moved sources
├── hooks.py                 # Claude Code / Codex session hooks
├── instructions/            # Skill instructions for AI agents
│   ├── __init__.py
│   ├── skill_init.py
│   ├── skill_search.py
│   ├── skill_mine.py
│   ├── skill_help.py
│   └── skill_status.py
├── backends/
│   ├── __init__.py           # VectorStore ABC + backend auto-dispatch
│   ├── faiss_store.py        # FAISS vector store implementation
│   ├── chroma_store.py       # ChromaDB vector store implementation
│   ├── embedder.py           # All embedder implementations (7 embedders)
│   └── _stdio.py             # Windows Unicode helper (reconfigure_stdio)
├── _mcp_shared.py            # Shared MCP state (tools dict, version info)
└── _version.py               # Version string
```

---

## Layer-by-Layer Breakdown

### 1. Entry Points

#### `cli.py` — CLI entry
- `main()` parses args, creates `Dimension`, dispatches to subcommand
- 30+ subcommands: `init`, `status`, `add`, `search`, `get`, `list`, `delete`, `realms`, `rooms`, `kg-add`, `kg-query`, `kg-invalidate`, `kg-stats`, `record`, `record-read`, `record-ingest`, `mine`, `sync`, `check-dup`, `rebuild-fts`, `mcp`, `repair`, `migrate`, `split`, `aaak`, `hook`, `instructions`, `wake-up`, `sweep`, `backend`
- Each subcommand maps to a `Dimension` method call

#### `mcp_server.py` — MCP server entry
- `main(arg)` starts a JSON-RPC server over stdio or SSE
- 65 tool handlers registered in `TOOLS` dict
- Each handler calls `Dimension` methods
- SSE mode runs HTTP server on `localhost:8316` (configurable)
- Windows: calls `reconfigure_stdio_utf8_on_windows()` at startup to fix Unicode

#### `__init__.py` — Python API entry
- `from alt_memory import Dimension` gives access to all functionality

---

### 2. Core — `dimension.py`

The `Dimension` class is the central orchestrator. It holds:

- **SQLite connection** (`_con`) — all metadata, content, KG facts
- **Vector store** (`_vs`) — FAISS or ChromaDB instance
- **Embedder** (`_embedder`) — current embedding model
- **Config** — loaded from `dimension.json`

Key methods:

| Method | What it does |
|--------|-------------|
| `init()` | Create tables, load config, init backend, auto-pick embedder |
| `status()` | Return entity count, realm/domain breakdown, embedder info |
| `add_entity()` | Insert into SQLite, embed vector, index in FAISS/Chroma |
| `get_entity()` | Fetch by ID from SQLite |
| `update_entity()` | Update content/metadata/realm/domain, re-embed |
| `delete_entity()` | Remove from SQLite + vector index |
| `list_entities()` | List with offset/limit/realm/domain filters |
| `batch_add_entities()` | Bulk insert in a single transaction |
| `search()` | Hybrid/vector/keyword search with optional realm/domain filter |
| `create_realm()` / `delete_realm()` | Realm CRUD |
| `create_domain()` / `delete_domain()` | Domain CRUD |
| `list_realms()` / `list_domains()` | List with counts |
| `check_duplicate()` | Similarity check against existing entities |
| `get_taxonomy()` | Full realm→domain→entity_count tree |
| `export_collection()` / `import_entities()` | JSON import/export |
| `set_backend()` | Hot-swap FAISS ↔ Chroma |
| `set_embedder()` | Hot-swap embedder with auto-reindex |
| `mine_file()` / `mine_text()` / `batch_mine()` | File mining |
| `sync()` | Prune stale entities from deleted/moved sources |
| `rebuild_fts()` | Rebuild FTS5 index |
| `reconnect()` | Force SQLite reconnect after external changes |

**KG integration**: `d.kg` is a `KnowledgeGraph` instance that manages temporal triples in SQLite.

**Diary integration**: `d.diary_write()` and `d.diary_read()` store agent temporal records in the `agent_<name>` realm.

---

### 3. Vector Backends — `backends/`

#### `__init__.py` — ABC + factory
- `VectorStore` abstract base class with methods: `add()`, `search()`, `delete()`, `rebuild()`, `size()`
- `get_backend()` factory: `"faiss"` → `FaissStore`, `"chroma"` → `ChromaStore`
- `ensure_index()` — create or recover an index

#### `faiss_store.py` — FAISS implementation
- Wraps faiss.IndexFlatIP + faiss.IndexIDMap for 384d vectors
- `add(vectors, ids)` — batch insert into FAISS
- `search(query_vec, k)` — cosine similarity (inner product on normalized vectors)
- `delete(ids)` — remove_by_id
- `size()` — `index.ntotal`
- `rebuild()` — rebuild from scratch
- Serializes to `index.faiss` file
- **Known issue**: IndexIDMap API changed in faiss-cpu 1.14.2 (constructor rejects non-empty input). Code now handles this.

#### `chroma_store.py` — ChromaDB implementation
- Wraps chromadb with persistent client
- Collection stored in `~/.alt-memory/chroma/`
- Same interface as FaissStore

---

### 4. Embedders — `backends/embedder.py`

Single file, 7 embedder implementations sharing a 384d output space:

| Embedder | Class | Dependencies | Notes |
|----------|-------|-------------|-------|
| `numpy` | `NumpyEmbedder` | none | TF-IDF + TruncatedSVD, always available |
| `numpy_bert` | `NumpyBERTEmbedder` | tokenizers | all-MiniLM-L6-v2 via pure numpy inference |
| `sentence` | `SentenceEmbedder` | sentence-transformers | PyTorch-backed, best quality |
| `minilm` | `OnnxEmbedder` | onnxruntime, huggingface_hub | ONNX MiniLM, fast |
| `bert` | `OnnxEmbedder` | onnxruntime or tokenizers | ONNX BERT |
| `embeddinggemma` | `OnnxEmbedder` | onnxruntime, huggingface_hub | Google Gemma embedding model |
| `spacy` | `SpacyEmbedder` | spacy + en_core_web_md | |

Key design:
- `_lazy_load()` pattern — model loaded on first call, not at init
- Auto-resolution priority: sentence → numpy_bert → numpy
- ONNX models downloaded from HuggingFace hub (no PyTorch needed)
- `ONNXEmbedder` removed optimum/PyTorch path — always uses direct ONNX to prevent GPU probing (which caused TDR crashes on Windows with NVIDIA GPUs)
- On embed failure, auto-falls back to numpy embedder

---

### 5. Graph — `dim_graph.py`

Manages the "palace graph" — cross-realm connections:

- `add_tunnel(source_realm, source_domain, target_realm, target_domain, ...)` — create a tunnel
- `delete_tunnel(tunnel_id)`
- `list_tunnels(realm)` — list all tunnels, optionally filtered
- `find_tunnels(realm_a, realm_b)` — find bridges between two realms
- `follow_tunnels(realm, domain)` — follow tunnels from a domain
- `traverse(start_domain, max_hops)` — walk the graph, showing connected ideas
- `graph_stats()` — overview of all connections

Stored in SQLite `tunnels` table. The graph forms a multi-edge directed graph between domains.

---

### 6. Detection & Mining

#### `domain_detector_local.py` — Domain detection
- `detect_domains_local(project_dir)` — scans a project directory and recommends domain names
- Pure heuristic: file extensions, directory names, pattern matching
- No ML/AI dependency

#### `entity_detector.py` — Entity detection
- `detect_entities(content)` — extracts people names, project names, key terms from text
- Used during file mining to auto-tag entities
- Regex + pattern-based

#### `project_scanner.py` — Project discovery
- `discover_entities(project_dir)` — scans a project for people references and project names
- Looks at README, CONTRIBUTORS, git log, file headers

#### `corpus_origin.py` — Origin detection
- `detect_origin_heuristic(samples)` — detects if text is AI dialogue, code, documentation, etc.
- Returns `CorpusOrigin` with `likely_ai_dialogue`, `primary_platform` fields
- Used during `init` to auto-tag the dimension

---

### 7. Sync & Maintenance — `sync.py`

- `sync(project_dir, realm, apply)` — dry-run or apply pruning of entities whose source files have been deleted, moved, or gitignored
- Reads `source_file` metadata from entities and checks file existence
- Uses `.gitignore` rules via `pathspec` to filter ignored files
- `apply=True` actually deletes; default is preview

---

### 8. Hooks — `hooks.py`

Session hooks for Claude Code and Codex AI coding agents. Integrates Alt Memory into the agent's workflow lifecycle.

- `run_hook(hook, harness)` — dispatches to hook handler
- Hooks: `session-start`, `stop`, `precompact`
- Runs MCP commands during session lifecycle (e.g., record start/stop, compact memory)

---

### 9. CLI Instructions — `instructions/`

AI skill instructions embedded in the package. Each file provides a prompt template for a specific operation:

- `skill_init.py` — instructions for initializing a dimension
- `skill_search.py` — instructions for search
- `skill_mine.py` — instructions for mining files
- `skill_help.py` — general help instructions
- `skill_status.py` — instructions for checking status

Accessed via `alt-memory instructions <topic>`.

---

### 10. Utilities

#### `_mcp_shared.py`
Holds the MCP tool registry (`TOOLS` dict), version info, and shared state between MCP server and CLI. All tool handlers are defined here and imported by `mcp_server.py`.

#### `_stdio.py`
Windows-only: `reconfigure_stdio_utf8_on_windows()` fixes Unicode output by reconfiguring stdout/stderr to UTF-8 encoding. Dead code unless called explicitly (called in `mcp_server.py` at startup).

#### `_version.py`
Contains `__version__` string (e.g., `"4.5.1"`).

---

## On-Disk Layout

```
~/.alt-memory/
├── entities.db              # SQLite database
│   ├── entities             # Primary table (id, realm, domain, content, metadata, created_at)
│   ├── entities_fts         # FTS5 virtual table (full-text search)
│   ├── entity_registry      # name → id mappings
│   ├── kg_facts             # knowledge graph triples
│   ├── kg_names             # entity name normalization
│   ├── records              # agent diary entries
│   ├── realms               # realm metadata
│   ├── domains              # domain metadata
│   ├── tunnels              # cross-realm connections
│   └── schema_version       # migration tracking
├── index.faiss              # FAISS vector index (binary, if FAISS backend)
├── chroma/                  # ChromaDB persistent storage (if chroma backend)
├── dimension.json           # Config: backend, embedder, model settings
├── nodes/                   # Packed node storage (cross-reference files)
├── entity_registry.json     # JSON backup of entity name → ID
├── knowledge_graph.json     # JSON backup of KG triples
└── aaak_cache.json          # AAAK compression cache
```

---

## Data Flow

### Write Path
```
add_entity(realm, domain, content, meta)
  → SQLite INSERT (entities table)
  → embed(content) → 384d vector
  → FAISS/Chroma add(vector, id)
  → FTS5 index update (auto, SQLite triggers)
```

### Search Path
```
search(query, mode="hybrid")
  → embed(query) → 384d vector
  → FAISS/Chroma search(vector, k) → candidate IDs + distances
  → FTS5 keyword search → candidate IDs + BM25 scores
  → Hybrid: reciprocal rank fusion + rerank
  → SQLite fetch by ID → return results
```

### Delete Path
```
delete_entity(entity_id)
  → SQLite DELETE
  → FAISS/Chroma delete(id)
  → Remove from FTS5
```

### Sync Path
```
sync(project_dir, apply)
  → Scan all entities with source_file metadata
  → Check if source file exists AND is not gitignored
  → Report stale entities (dry-run)
  → If apply: delete each stale entity
```

---

## Embedder Resolution

On `init()` or `set_embedder()`:

1. If `ALT_DEFAULT_EMBEDDER` env var set → use that
2. If `dimension.json` has `default_embedder` → use that
3. Else auto-resolve: `sentence` (if sentence-transformers installed) → `numpy_bert` (if tokenizers installed) → `numpy` (always)

On failure at any step, falls back to `numpy` gracefully.

---

## Backend Switching

`set_backend(new_backend)`:

1. Read all entities from old backend's SQLite
2. Create new backend instance
3. Re-embed all entities and add to new backend
4. Save config
5. Old backend data (faiss or chroma dir) remains on disk for rollback

---

## Cross-Realm Tunnels (Palace Graph)

Tunnels connect domains across realms. They form a directed multi-edge graph:

```
work/bugs ──tunnel──▶ personal/ideas
personal/ideas ──tunnel──▶ work/features
```

- `traverse(start_domain, max_hops)` follows tunnels to discover connected ideas
- `graph_stats()` shows all connections
- Tunnels are explicit (user-created) or automatic (based on shared entities)

---

## Personas (Character Definitions)

A persona is a **character definition** — system prompt + metadata — that defines how an AI agent behaves. Modeled after [Eternal AI's on-chain agent persona](https://github.com/eternalai-org/eternal-ai) (character file minted as ERC-721 NFT, injected as `system` role in LLM calls).

### Persona Schema

```python
{
    "name": "donald_trump",              # unique identifier
    "system_prompt": "Act as if...",     # character system prompt
    "description": "Donald Trump twin",  # short description
    "model": "DeepSeek-R1-Distill-Llama-70B",  # preferred LLM
    "framework": "eternalai",             # agent framework
    "metadata": {"chain": "base"}         # extensible
}
```

### Storage

Personae are stored in `persona.json` in the dimension directory:

```json
{
  "active": "donald_trump",
  "personas": {
    "donald_trump": {
      "name": "donald_trump",
      "system_prompt": "Act as if you are Donald Trump...",
      "description": "A Donald Trump twin",
      "model": "DeepSeek-R1-Distill-Llama-70B",
      "framework": "eternalai",
      "metadata": {"chain": "base"}
    }
  }
}
```

Legacy format `{"persona": "name"}` is auto-upgraded on read.

### Key Methods (`dimension.py:1598-1741`)

| Method | Description |
|--------|-------------|
| `get_persona()` | Return active persona dict (or `{"name": ""}`) |
| `set_persona(name, system_prompt, ...)` | Set active persona + character def, creates `persona_<name>` realm |
| `create_persona(name, ...)` | Create persona without activating (raises if exists) |
| `list_personas()` | List all registered persona characters with `active` flag |
| `delete_persona(name)` | Remove from registry (does NOT delete realm or entities) |
| `get_persona_character(name)` | Get system prompt string for a persona (active by default) |

### Persona Realm Isolation

Setting a persona as active creates a `persona_<name>` realm. All memory operations targeting that realm are isolated from other personas. Switching persona doesn't hide or delete other realms — it just changes the active identity.

Unlike the old system (where persona was just a bare namespace label), the new persona carries a full character definition that an AI agent can inject as its system prompt, matching how Eternal AI agents use character files.

### MCP Tools

| Tool | Description |
|------|-------------|
| `get_persona` | Get active persona with full character definition |
| `set_persona` | Set active persona with optional character fields |
| `switch_persona` | Alias for set_persona |
| `create_persona` | Create persona without activating |
| `list_personas` | List all registered persona characters |
| `delete_persona` | Remove persona from registry |
| `get_persona_character` | Get system prompt string |

---

## Agent Diaries (Records)

Per-agent temporal entries stored in `realm=agent_<name>`, `domain=record`:

| Layer | Scope | Update frequency | Example |
|-------|-------|-----------------|---------|
| L0 | This session | Per-finding | `DEBUG: found bug in auth retry` |
| L1 | This day | End of session | `SESSION:2026-05-31\|fixed.auth\|★★★` |
| L2 | This week/month | Periodic | `PATTERN: auth.timeouts→rate.limit\|★★★★★` |

Access via `record_write` / `record_read` MCP tools or `d.diary_write()` / `d.diary_read()`.

---

## FAQ / Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Search returns 0 results | Embedder mismatch or empty index | `set_embedder("numpy")`, verify `status()` shows entities |
| MCP returns "Missing required parameters" | Wrong parameter name | Check tool schema via `tools/list` |
| Backend swap fails | Missing package | Install `alt-memory[chroma]` or specific backend |
| Rebuild needed after crash | Corrupt vector index | Use `rebuild_fts` tool or `alt-memory repair --rebuild-fts --vacuum` |
| "Empty vocabulary" warning | numpy embedder has no training data yet | Add a few entities, embedder auto-fits |
| Unicode mojibake in MCP | Windows without UTF-8 reconfigure | Server calls `reconfigure_stdio_utf8_on_windows()` at startup |
| GPU TDR crash (0x10E) | ONNX embedder probing GPU via optimum | Fixed in v4.5.1 — optimum path removed, direct ONNX only |
| CWD pollution | AI coding agent launches MCP from project dir | Use wrapper script that `cd`s to Python home before launching |

---

## Performance

Benchmarked on Windows (FAISS + numpy embedder):
- Average tool latency: **~8.2ms**
- MemPalace predecessor: **~55.0ms**
- Speedup: **6.7x**
- 65 tools, all passing
- Stress test: 500 entities, consistent latency

---

## Version History

| Version | Key changes |
|---------|-------------|
| 4.5.1 | GPU TDR fix (remove optimum path), rollback safety on set_embedder, CWD pollution fix, Unicode fix |
| 4.5.0 | FAISS IndexIDMap API fix, dim_graph tuple fix, sync.py row_factory fix, FTS5 hyphen crash fix |
| 4.4.x | Initial MCP port from MemPalace |
