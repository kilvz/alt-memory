# Alt Memory

**Local-first persistent memory for AI agents.** Store, search, and manage structured memory with hybrid vector/keyword search, a knowledge graph, agent diaries, and an MCP server — all offline, no external services.

```bash
pip install alt-memory
```

---

## Quick Summary

| Feature | What it does |
|---------|-------------|
| **Dimension** | A directory on disk containing SQLite + FAISS/Chroma + config — your persistent memory store |
| **Realms** | Top-level buckets (e.g., `work`, `personal`, `agent_claude`), like databases |
| **Domains** | Categories within a realm (e.g., `bugs`, `ideas`, `code`), like tables |
| **Entities** | Individual memory items with content + metadata |
| **KG facts** | Relationship triples (`subject → predicate → object`) with temporal validity |
| **Nodes** | Cross-reference entries connecting source files to entity references |
| **Tunnels** | Cross-realm links between domains (the "palace graph") |
| **Records** | Per-agent temporal diary entries |
| **AAAK** | Compressed memory format for efficient storage |

All search is **hybrid** by default — vector similarity + FTS5 keyword matching. 40+ MCP tools, all local, no cloud.

Benchmarked at ~8ms/tool (FAISS backend, numpy embedder) — roughly 7x faster than the predecessor system.

---

## Installation

```bash
pip install alt-memory
```

This gives you FAISS + `tokenizers` (for numpy BERT embedder). On first use, it auto-selects the best embedder available.

| Install | Backend | Embedder |
|---------|---------|----------|
| `pip install alt-memory` | FAISS | numpy BERT (all-MiniLM-L6-v2 via pure numpy) |
| `pip install alt-memory[chroma]` | + ChromaDB | same |
| `pip install alt-memory[onnx]` | same | + ONNX MiniLM / Gemma / BERT |
| `pip install alt-memory[all]` | + ChromaDB | + ONNX + sentence-transformers |

### Extras explained

| Extra | Packages |
|-------|----------|
| `[chroma]` | `chromadb>=1.5.4` |
| `[onnx]` | `onnxruntime`, `huggingface_hub`, `transformers` |
| `[all]` | chroma + onnx + `sentence-transformers` |

Requirements: Python ≥ 3.10, Windows/Linux/macOS.

---

## Quick Start — CLI

```bash
# Initialize a new dimension
alt-memory init

# Check status
alt-memory status

# Store a memory (realm=work, domain=bugs)
alt-memory add --realm work --domain bugs --content "Login freezes on Safari 18.2"

# Search — hybrid by default (vector + keyword)
alt-memory search "login safari" --limit 5

# Browse
alt-memory list --realm work
alt-memory realms
alt-memory rooms

# Knowledge graph
alt-memory kg-add --subject LoginBug --predicate affects --object Safari
alt-memory kg-query LoginBug
alt-memory kg-stats

# Agent diary
alt-memory record --agent claude --entry "Investigated the Safari freeze"

# Mine a file (auto-chunk, extract entities)
alt-memory mine --realm work --domain code src/auth.py

# MCP server
alt-memory mcp --transport stdio
```

### Init with project detection

```bash
# Run init in a project directory — auto-detects corpus origin, entities, domains
alt-memory init /path/to/project --llm-provider ollama --llm-model qwen2.5
```

---

## Auto-Detection: Best Embedder, Zero Config

Alt Memory automatically picks the best embedder available at runtime. No configuration needed.

```
Priority: sentence-transformers → numpy BERT → TF-IDF+SVD
Quality:  ★★★★★              → ★★★★         → ★★
Speed:    ★★★★               → ★★★          → ★★★★★
Deps:     PyTorch (~800MB)    → tokenizers   → none
```

**What happens when you `pip install alt-memory`:**
- `tokenizers` is installed (lightweight, pure Python + Rust wheels for all platforms)
- On first `alt-memory init`, the system detects no PyTorch → uses **numpy BERT** (all-MiniLM-L6-v2 via pure numpy inference)
- Same model as sentence-transformers, no heavy deps

**If you also install sentence-transformers:**
```bash
pip install sentence-transformers   # or pip install alt-memory[all]
```
The system auto-detects it on next start and switches to PyTorch-accelerated inference.

**Force a specific embedder:**
```bash
export ALT_DEFAULT_EMBEDDER=numpy       # TF-IDF+SVD (lightest)
export ALT_DEFAULT_EMBEDDER=minilm      # ONNX MiniLM (requires [onnx])
export ALT_DEFAULT_EMBEDDER=numpy_bert  # Pure numpy BERT
```

---

## Embedder Reference

| Name | Quality | Speed | Dependencies | Platform |
|------|---------|-------|-------------|----------|
| `sentence` | ★★★★★ | ★★★★ | sentence-transformers + PyTorch | all |
| `numpy_bert` | ★★★★ | ★★★ | tokenizers (included) | **all incl. Alpine** |
| `minilm` | ★★★★ | ★★★★★ | onnxruntime `[onnx]` | glibc |
| `embeddinggemma` | ★★★★★ | ★★★★ | onnxruntime `[onnx]` | glibc |
| `bert` | ★★★★ | ★★★★ | onnxruntime or tokenizers `[onnx]` | all |
| `spacy` | ★★★ | ★★★ | spacy + en_core_web_md | all |
| `numpy` | ★★ | ★★★★★ | none (always available) | all |

All embedders produce 384-dimensional vectors. Switch at runtime — entities are automatically re-embedded.

---

## Backend Switching

Swap between FAISS and ChromaDB at runtime. All data persists in SQLite — only the vector index changes.

```bash
# CLI
alt-memory backend chroma

# Python
d = Dimension(path="~/.alt-memory", backend="chroma")
d.set_backend("faiss")  # hot-swap

# MCP
{"name": "set_backend", "arguments": {"backend": "chroma"}}
```

---

## MCP Server (AI Agent Integration)

```bash
alt-memory mcp --transport stdio          # for AI coding agents (stdio)
alt-memory mcp --transport sse --port 8316  # HTTP SSE mode
```

The server exposes 40+ JSON-RPC tools for memory operations. Configure it in your AI agent's MCP settings:

```json
{
  "mcpServers": {
    "alt-memory": {
      "command": "alt-memory",
      "args": ["mcp", "--transport", "stdio"]
    }
  }
}
```

### All MCP Tools

| Category | Tool | Description |
|----------|------|-------------|
| **Search** | `search` | Hybrid/vector/keyword search across the dimension |
| | `check_duplicate` | Check if content already exists (similarity threshold) |
| **CRUD** | `add_entity` | Add a new entity to a realm/domain |
| | `get_entity` | Get entity by ID |
| | `update_entity` | Update content, metadata, realm, or domain |
| | `delete_entity` | Delete a single entity |
| | `delete_entities` | Bulk delete by IDs |
| | `list_entities` | List with pagination and realm/domain filter |
| | `batch_add_entities` | Add multiple entities in one transaction |
| **Realms** | `create_realm` | Create a new top-level bucket |
| | `delete_realm` | Delete realm and all its domains/entities |
| | `list_realms` | List all realms |
| | `get_taxonomy` | Full realm → domain → entity count hierarchy |
| | `get_status` | Entity count, realm/domain breakdown, embedder info |
| **Domains** | `create_domain` | Create a new domain in a realm |
| | `delete_domain` | Delete a domain |
| | `list_domains` | List domains (optionally filtered by realm) |
| **KG** | `kg_add` | Add a fact (subject → predicate → object) |
| | `kg_query` | Query facts about an entity |
| | `kg_invalidate` | Mark a fact as no longer true |
| | `kg_stats` | KG statistics |
| | `kg_timeline` | Chronological timeline of facts |
| **Records** | `record_write` | Write a temporal entry for an agent |
| | `record_read` | Read recent entries for an agent |
| | `list_agents` | List all agents with records |
| **Graph** | `create_tunnel` | Link two domains across realms |
| | `delete_tunnel` | Delete a tunnel |
| | `list_tunnels` | List all tunnels |
| | `find_tunnels` | Find bridges between two realms |
| | `follow_tunnels` | Follow tunnels from a domain |
| | `traverse` | Walk the palace graph from a domain |
| | `graph_stats` | Overview of tunnel connections |
| **Backend** | `set_backend` | Switch between FAISS and ChromaDB |
| | `get_backend` | Get current backend |
| **Embedder** | `set_embedder` | Switch embedding model (auto-reindex) |
| | `get_default_embedder` | Get default embedder config |
| **Mine** | `mine_file` | Mine a single file into the dimension |
| | `mine_text` | Mine text content directly |
| | `batch_mine` | Mine all matching files in a directory |
| | `sync` | Prune entities whose source files were deleted |
| **Import/Export** | `import_entities` | Import from JSON list |
| | `export_collection` | Export all entities as JSON |
| **AAAK** | `aaak_compress` | Compress text to AAAK format |
| | `aaak_decompress` | Decompress AAAK text |
| | `aaak_parse` | Parse a single AAAK entry |
| | `get_aaak_spec` | Get the AAAK dialect spec |
| **Persona** | `get_persona` | Get current active persona |
| | `set_persona` | Switch to a persona (creates persona_<name> realm) |
| | `get_people_map` | Get name variant → canonical mappings |
| | `set_people_map` | Set name variant mappings |
| **Hooks** | `hook_settings` | Configure silent_save and desktop_toast |
| | `memories_filed_away` | Check if recent checkpoint was saved |
| **Maintenance** | `rebuild_fts` | Rebuild FTS5 full-text search index |
| | `reconnect` | Force reconnect to dimension database |

---

## Python API

```python
from alt_memory import Dimension

d = Dimension(path="~/.alt-memory")
d.init()

# Store and search
eid = d.add_entity("work", "bugs", "Login freezes on Safari",
                   metadata={"priority": "high"})
results = d.search("login safari", n_results=10, mode="hybrid")

# Knowledge graph
d.kg.add("LoginBug", "affects", "Safari", valid_from="2026-05-01")
facts = d.kg.query("LoginBug")

# Agent diaries
d.diary_write("claude", "Investigated the Safari freeze", topic="debugging")
entries = d.diary_read("claude", last_n=5)

# Backend & embedder control
d.set_backend("chroma")
d.set_embedder("minilm")

# File mining
d.mine_file("work", "code", "src/auth.py")

# Sync (prune stale entities from deleted files)
d.sync(project_dir="/path/to/repo", apply=True)
```

---

## Architecture

```
┌──────────────────────────────────────────────┐
│              MCP Server (stdio/SSE)           │
│         40+ JSON-RPC tools for AI agents      │
├──────────────────────────────────────────────┤
│                  Dimension                    │
│    orchestrates realms, domains, entities     │
├───────────────────┬──────────────────────────┤
│   Vector Store    │   SQLite + FTS5           │
│ (FAISS / Chroma)  │ (metadata, content, KG)  │
├───────────────────┴──────────────────────────┤
│           Embedder Layer (auto-pick)          │
│  sentence ─▶ numpy BERT ─▶ TF-IDF+SVD        │
└──────────────────────────────────────────────┘
```

### On-disk layout

```
~/.alt-memory/
├── entities.db           # SQLite + FTS5 (all content, metadata, KG)
├── index.faiss           # FAISS vector index (or chroma/ directory)
├── dimension.json        # Config (backend, embedder, model settings)
├── nodes/                # Packed cross-reference node storage
├── entity_registry.json  # Entity name → ID mappings
├── knowledge_graph.json  # Relationship store (backup/sync)
└── aaak_cache.json       # AAAK compression cache
```

All data is portable — copy `~/.alt-memory` to another machine and it works.

---

## Key Concepts

### Organization

| Concept | What it is | MCP tool prefix | SQL analogy |
|---------|-----------|----------------|-------------|
| **Dimension** | Top-level memory store | (implicit) | "database server" |
| **Realm** | Top-level bucket | `*_realm` | database |
| **Domain** | Category within a realm | `*_domain` | table |
| **Entity** | A stored memory item | `*_entity` | row |
| **Node** | Cross-reference: source file → entity refs | `add_node` / `*_node` | index |
| **KG fact** | Relationship triple | `kg_*` | edge |
| **Tunnel** | Cross-realm link between domains | `*_tunnel` | foreign key |
| **Record** | Per-agent temporal entry | `record_*` | log |
| **Persona** | Isolated realm per AI persona | `*_persona` | schema |

### AAAK Memory Compression

AAAK is a compressed memory dialect for efficient storage — readable by humans and LLMs without decoding. Uses 3-letter entity codes, emotion markers, pipe-separated fields, and importance ratings.

```bash
# Compress a memory
alt-memory aaak "Alice loves Jordan, they have two kids: Riley (18, into sports) and Max (11, does chess and swimming)"

# Output
FAM: ALC→♡JOR | 2D(kids): RIL(18,sports) MAX(11,chess+swimming) | ★★★★
```

Use in MCP:
```json
{"name": "aaak_compress", "arguments": {"text": "Your long text here"}}
{"name": "get_aaak_spec", "arguments": {}}
```

### File Mining

```bash
# Mine a single file
alt-memory mine src/auth.py --realm myproject --domain code

# Mine a directory (auto-globs common file types)
alt-memory mine /path/to/project --mode projects

# Preview without filing
alt-memory mine /path/to/project --dry-run

# Batch mine via MCP
{"name": "batch_mine", "arguments": {"directory": "/path/to/project", "realm": "myproject"}}
```

Mining auto-chunks long files, extracts entities, and stores them with source file references.

### Sync & Maintenance

```bash
# Dry-run: see what would be pruned
alt-memory sync --project-dir /path/to/repo

# Actually prune stale entities
alt-memory sync --project-dir /path/to/repo --apply

# Repair tools
alt-memory repair --integrity          # Check SQLite integrity
alt-memory repair --vacuum             # VACUUM
alt-memory repair --rebuild-fts        # Rebuild FTS5 index

# Full repair (reset corrupted state)
alt-memory migrate --rebuild-faiss     # Rebuild FAISS from SQLite
```

### Agent Diaries (L0/L1/L2 Memory Layers)

Records are the "temporal" memory layer — per-agent, ordered, searchable:

```python
# Layer 0 — immediate record (this session)
d.diary_write("claude", "DEBUG: found race condition in auth retry", topic="debug")

# Layer 1 — daily summary (end of session)
d.diary_write("claude", "SESSION:2026-05-31|fixed.auth.race.cond|★★★", topic="summary")

# Layer 2 — project-level patterns (weekly)
d.diary_write("claude", "PATTERN: auth.timeouts.peak→rate.limit.insufficient|★★★★★", topic="pattern")
```

Read with:
```bash
alt-memory wake-up --agent claude --last-n 5
```

---

## Docker

```bash
# FAISS + numpy BERT (Alpine, ~117MB content)
docker run -v ~/.alt-memory:/root/.alt-memory kilv/alt-memory:alpine mcp

# Full install (Debian, includes chroma + onnx)
docker run -v ~/.alt-memory:/root/.alt-memory kilv/alt-memory:latest mcp
```

Tags:
- `kilv/alt-memory:latest` — Debian-based, full install
- `kilv/alt-memory:4.5.1` — versioned Debian
- `kilv/alt-memory:alpine` — Alpine-based, FAISS + numpy BERT (~117MB)
- `kilv/alt-memory:4.5.1-alpine` — versioned Alpine

---

## CLI Reference (Full)

```
alt-memory init [dir]              Initialize dimension (optionally scan project)
alt-memory status                  Show dimension status (JSON)
alt-memory add -w <realm> -r <domain> -c <content>  Add entity
alt-memory search <query>          Search (hybrid default, use --mode flag)
alt-memory get <id>                Get entity by ID
alt-memory list [--realm] [--domain]  List entities
alt-memory realms [--verbose]      List realms
alt-memory rooms [--realm]         List domains
alt-memory delete <id>             Delete entity

alt-memory kg-add --subject <s> --predicate <p> --object <o>  Add KG fact
alt-memory kg-query [entity]       Query KG
alt-memory kg-invalidate --subject <s> --predicate <p> --object <o>  Invalidate
alt-memory kg-stats                KG statistics

alt-memory record --agent <a> --entry <e>  Write diary entry
alt-memory record-read --agent <a>   Read diary entries
alt-memory record-ingest --dir <d>   Ingest daily summary files

alt-memory mine <file|dir>         Mine files into dimension
alt-memory sync [--apply]          Prune stale entities
alt-memory check-dup <content>     Check for duplicates
alt-memory rebuild-fts             Rebuild FTS index

alt-memory mcp --transport stdio   Run MCP server
alt-memory repair [--vacuum] [--rebuild-fts]  Repair utilities
alt-memory migrate [--rebuild-faiss]   Schema migration
alt-memory aaak <text>             AAAK compression
alt-memory split [--source]        Split mega-files into sessions

alt-memory hook run --hook <h> --harness <h>  Run hook (claude-code/codex)
alt-memory instructions <topic>    Output skill instructions
alt-memory wake-up [--agent]       L0+L1 wake-up context
```

---

## License

MIT
