# Alt Memory

**Local-first persistent memory for AI agents.** Store, search, and manage structured memory with hybrid vector/keyword search, a knowledge graph, agent diaries, and an MCP server — all offline, no external services.

```
pip install alt-memory
```

---

## Features

- **Hybrid search** — vector similarity + FTS5 keyword matching with configurable ranking
- **Dual vector backends** — FAISS (default, fast) or ChromaDB (HNSW, optional)
- **Auto-picking embedder** — uses best available: sentence-transformers → pure numpy BERT → TF-IDF+SVD
- **Swappable embedders** — switch at runtime between numpy, ONNX MiniLM, BERT, Gemma, spaCy
- **Knowledge graph** — temporal entity-relationship triples with invalidation timelines
- **Agent diaries** — per-agent temporal records with L0/L1/L2 memory layers
- **File mining** — auto-extract entities from code, conversations, text
- **MCP server** — 40+ JSON-RPC tools over stdio/SSE for AI agent integration
- **Palace graph** — cross-realm tunnels with graph traversal
- **Personas** — isolated memory realms per AI persona
- **i18n** — 14 languages for agent-facing communication
- **Docker** — official images on Docker Hub (`kilv/alt-memory`)

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
# Initialize
alt-memory init

# Store a memory
alt-memory add --realm work --domain bugs --content "Login freezes on Safari 18.2"

# Search (hybrid by default)
alt-memory search "login safari" --limit 5

# Add a knowledge graph fact
alt-memory kg-add --subject LoginBug --predicate affects --object Safari
alt-memory kg-query LoginBug

# Write a diary entry
alt-memory record --agent claude --entry "Investigated the Safari freeze"

# Run the MCP server
alt-memory mcp --transport stdio
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
alt-memory mcp --transport stdio          # for AI coding agents
alt-memory mcp --transport sse --port 8316  # HTTP
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

### Key MCP tools

| Tool | Description |
|------|-------------|
| `search` | Hybrid/vector/keyword search |
| `add_entity` / `get_entity` / `update_entity` / `delete_entity` | CRUD |
| `set_backend` / `get_backend` | Backend control |
| `set_embedder` / `get_default_embedder` | Embedder control |
| `kg_add` / `kg_query` / `kg_invalidate` / `kg_stats` | Knowledge graph |
| `record_write` / `record_read` | Agent diaries |
| `mine_file` / `mine_text` | File mining |
| `list_realms` / `list_domains` | Browsing |
| `traverse` / `graph_stats` / `get_taxonomy` | Palace graph |
| `sync` / `check_duplicate` | Maintenance |

Full list: `alt-memory mcp --help` or inspect via `tools/list` after connecting.

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

## Docker

```bash
# FAISS + numpy BERT (Alpine, ~117MB content)
docker run -v ~/.alt-memory:/root/.alt-memory kilv/alt-memory:alpine mcp

# Full install (Debian, includes chroma + onnx)
docker run -v ~/.alt-memory:/root/.alt-memory kilv/alt-memory:latest mcp
```

Tags:
- `kilv/alt-memory:latest` — Debian-based, full install
- `kilv/alt-memory:4.5.0` — versioned Debian
- `kilv/alt-memory:alpine` — Alpine-based, FAISS + numpy BERT (~117MB)
- `kilv/alt-memory:4.5.0-alpine` — versioned Alpine

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

Each dimension is a directory on disk:
```
~/.alt-memory/
├── entities.db           # SQLite + FTS5
├── index.faiss           # FAISS vector index (or chroma/)
├── dimension.json        # config (backend, embedder)
├── nodes/                # packed entity storage
├── entity_registry.json  # entity name → ID mappings
└── knowledge_graph.json  # relationship store
```

---

## CLI Reference

```
alt-memory init                  Initialize a new dimension
alt-memory status                Show dimension status
alt-memory add -w <realm> -r <domain> -c <content>
alt-memory search <query>        Search (hybrid default)
alt-memory get <id>              Get entity by ID
alt-memory list                  List entities
alt-memory realms                List realms
alt-memory rooms                 List domains
alt-memory delete <id>           Delete entity
alt-memory kg-add --subject --predicate --object
alt-memory kg-query <entity>     Query knowledge graph
alt-memory kg-invalidate ...     Invalidate KG fact
alt-memory kg-stats              KG statistics
alt-memory record --agent --entry
alt-memory record-read --agent   Read diary
alt-memory mine -w <realm> -r <domain> <file>
alt-memory sync [--apply]        Prune stale entities
alt-memory mcp                   Run MCP server
alt-memory repair                Repair utilities
alt-memory aaak <text>           AAAK compression
```

---

## License

MIT
