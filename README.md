# Alt Memory v4.3.0

Local-first AI memory system with hybrid search, entity graph, MCP server, and 40+ tools. Store, search, and manage persistent memory for AI agents — runs entirely on your machine.

## Features

- **Hybrid search** — vector + keyword (FTS5) with configurable ranking
- **Dual backends** — FAISS (default, fast) or ChromaDB (optional)
- **Embedder hotswap** — switch between numpy TF-IDF, ONNX MiniLM, BERT, sentence-transformers at runtime
- **MCP server** — 40+ JSON-RPC tools over stdio/SSE for AI agent control
- **Knowledge graph** — temporal entity-relationship store with invalidation
- **Memory layers** — agent diaries with hierarchical L0/L1/L2 stack
- **File mining** — auto-extract entities from code, text, conversations
- **Sync & dedup** — prune stale entities, detect duplicates
- **i18n** — 14 locale translations for agent communication
- **Palace graph** — cross-realm tunnels and traversal
- **Zero external services** — everything runs locally

## Installation

```bash
pip install alt-memory
```

With optional backends:

```bash
pip install "alt-memory[chroma]"     # ChromaDB backend
pip install "alt-memory[onnx]"        # ONNX MiniLM embedder
pip install "alt-memory[all]"         # Everything
```

Requires Python ≥ 3.10. Default install includes `faiss-cpu`, `numpy`, `scipy`.

## Quick Start — CLI

```bash
# Initialize a dimension
alt-memory init

# Add an entity
alt-memory add --realm myproject --domain bugs --content "Login button freezes on Safari"

# Search (hybrid by default)
alt-memory search "login safari" --limit 5

# Get status
alt-memory status

# List realms and domains
alt-memory realms
alt-memory rooms --realm myproject
```

## Backend Switching

Swap between FAISS and ChromaDB at any time — all data persists in SQLite.

```bash
# CLI
alt-memory backend chroma

# Python API
from alt_memory import Dimension
d = Dimension(path="~/.alt-memory", backend="chroma")
d.init()
d.set_backend("faiss")  # hot-swap

# MCP tool
{"method": "tools/call", "params": {"name": "set_backend", "arguments": {"backend": "chroma"}}}
```

The backend is stored per-dimension in `dimension.json`. All entities, metadata, and FTS indexes are shared — only the vector index differs.

## Embedder Hotswap

Switch embedding models without losing data:

| Model | Description | Install |
|-------|-------------|---------|
| `numpy` | TF-IDF + SVD (default, always avail) | built-in |
| `minilm` | ONNX all-MiniLM-L6-v2 (384 dim) | `alt-memory[onnx]` |
| `bert` | ONNX BERT | `alt-memory[onnx]` |
| `embedding-gemma` | Google Embedding Gemma | `alt-memory[onnx]` |
| `sentence-transformers` | any HF sentence-transformers model | `pip install sentence-transformers` |
| `spacy` | spaCy en_core_web_md | `pip install spacy` |

```python
d.set_embedder("minilm")      # re-embeds all entities automatically
d.set_embedder("numpy")       # back to numpy
d.set_embedder("sentence-transformers", model="all-MiniLM-L6-v2")
```

MCP: `{"name": "set_embedder", "arguments": {"model": "minilm"}}`

## CLI Reference

```
init                          Initialize a new dimension
status                        Show dimension status
add --realm -w --domain -r --content -c [--meta --source]
                              Add an entity
search [query] [--realm -w] [--domain -r] [--limit] [--mode vector|keyword|hybrid]
                              Search (default hybrid)
get <entity_id>               Get an entity by ID
list [--realm] [--domain] [--limit] [--offset]
                              List entities
realms                        List all realms
rooms [--realm]               List domains
delete <entity_id>            Delete an entity
kg-add --subject --predicate --object
                              Add KG fact
kg-query [entity]             Query KG
kg-invalidate --subject --predicate --object
                              Invalidate KG fact
kg-stats                      KG statistics
diary --agent --entry         Write diary entry
diary-read --agent            Read diary
aaak <text>                   Compress text to AAAK
mine --realm --domain <file>  Mine file into dimension
sweep <path>                  Sweep .jsonl files
sync [--dry-run] [--apply]    Prune stale entities
repair [--integrity] [--vacuum] [--rebuild-fts]
                              Repair utilities
mcp [--transport stdio|sse] [--port 8316]
                              Run MCP server
```

## MCP Server (AI Agent Control)

Run the MCP server so AI agents can read/write your memory directly:

```bash
alt-memory mcp --transport stdio
```

Or over HTTP (SSE):

```bash
alt-memory mcp --transport sse --port 8316
```

### MCP Tools (40+)

The server exposes all dimension operations as JSON-RPC tools:

| Tool | Description |
|------|-------------|
| `search` | Hybrid/vector/keyword search |
| `get_status` | Dimension status |
| `list_realms` / `create_realm` / `delete_realm` | Realm management |
| `list_domains` / `create_domain` / `delete_domain` | Domain management |
| `add_entity` / `get_entity` / `update_entity` / `delete_entity` / `list_entities` | Entity CRUD |
| `check_duplicate` | Similarity check |
| `set_backend` / `get_backend` | Backend swap |
| `set_embedder` / `get_default_embedder` / `set_default_embedder` | Embedder control |
| `kg_add` / `kg_query` / `kg_invalidate` / `kg_stats` / `kg_timeline` | Knowledge graph |
| `diary_write` / `diary_read` / `memory_write` / `memory_read` / `memory_summarize` | Agent diaries |
| `mine_file` / `mine_text` / `batch_mine` | File mining |
| `aaak_compress` / `aaak_decompress` / `aaak_parse` | AAAK dialect |
| `rebuild_fts` | FTS maintenance |
| `create_tunnel` / `delete_tunnel` / `find_tunnels` / `follow_tunnels` / `list_tunnels` / `traverse` / `graph_stats` / `get_taxonomy` | Palace graph |
| `sync` / `reconnect` / `hook_settings` | Maintenance |
| `list_agents` / `get_people_map` | User management |
| `get_aaak_spec` | AAAK specification |

## Python API

```python
from alt_memory import Dimension

d = Dimension(path="~/.alt-memory")
d.init()

# CRUD
eid = d.add_entity("realm", "domain", "content",
                   metadata={"key": "val"}, source_file="/path/to/file")
entity = d.get_entity(eid)
d.update_entity(eid, content="updated", realm="newrealm")
d.delete_entity(eid)

# Search (returns list of SearchResult with .id, .text, .distance, .realm, .domain, .metadata)
results = d.search("query", n_results=10, mode="hybrid",
                   realm="realm_filter", domain="domain_filter")

# Backend & embedder control
d.set_backend("chroma")
d.set_embedder("minilm")

# Knowledge graph
d.kg.add("Alice", "works_on", "ProjectX", valid_from="2026-01-01")
facts = d.kg.query("Alice")
d.kg.invalidate("Alice", "works_on", "ProjectX")

# Agent diaries
d.diary_write("agent_name", "Entry text", topic="debugging")
entries = d.diary_read("agent_name", last_n=5)

# Status
info = d.status()  # entities, realms, domains, embedder, fts_enabled

# Sync & maintenance
d.sync(project_dir="/path/to/repo", apply=True)
d.check_duplicate("new content")
d.rebuild_fts()
```

## Architecture

```
┌─────────────────────────────────────────────┐
│                  MCP Server                  │
│  (JSON-RPC over stdio/SSE — 40+ tools)      │
├─────────────────────────────────────────────┤
│                  Dimension                   │
│  (orchestrator — realms, domains, entities)  │
├──────────────────┬──────────────────────────┤
│  Vector Store     │  SQLite + FTS5           │
│  (FAISS/Chroma)   │  (metadata, content, KG) │
├──────────────────┴──────────────────────────┤
│  Embedder Layer                              │
│  (numpy / minilm / BERT / sentence-transform)│
└─────────────────────────────────────────────┘
```

Each dimension is a directory containing:
- `entities.db` — SQLite with FTS5 for content + metadata
- `faiss.index` or `chroma/` — vector index
- `dimension.json` — configuration (backend, embedder)
- `closets/` — packed entity storage
- `entity_registry.json` — entity name mappings
- `knowledge_graph.json` — relationship store

## License

MIT
