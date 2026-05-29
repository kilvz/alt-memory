# AGENTS.md — For AI agents using Alt Memory

This file tells you (the AI) how to use Alt Memory effectively. Read this before operating the dimension.

## How to get started

The dimension at `~/.alt-memory` is already initialized. Connect via the MCP server:

```
alt-memory mcp --transport stdio
```

This exposes 40+ JSON-RPC tools. The tool names and schemas are advertised via `tools/list`. Use `tools/call` to invoke them.

## Core concepts

| Concept | MCP tool prefix | Notes |
|---------|----------------|-------|
| **Realm** | `*_realm` | Top-level bucket (e.g. `work`, `personal`, `agent_claude`) |
| **Domain** | `*_domain` | Category within a realm (e.g. `bugs`, `ideas`, `diary`) |
| **Entity** | `*_entity` | A stored memory item with content + metadata |
| **KG fact** | `kg_*` | Relationship triple with temporal validity |
| **Tunnel** | `*_tunnel` | Cross-realm link between domains |
| **Diary** | `diary_*` | Agent-specific temporal entries |

## Key operations

### Search — always use hybrid mode

```json
{
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {
      "query": "what was that bug about login",
      "mode": "hybrid",
      "n_results": 10
    }
  }
}
```

Hybrid mode combines vector similarity with FTS5 keyword matching. This gives better results than either alone.

### Storing information

```json
{
  "method": "tools/call",
  "params": {
    "name": "add_entity",
    "arguments": {
      "realm": "work",
      "domain": "bugs",
      "content": "Login button freezes on Safari 18.2 — only affects macOS Sequoia users",
      "metadata": {"priority": "high", "status": "open"}
    }
  }
}
```

Always include meaningful realm/domain organization. Use `list_realms` and `list_domains` to see existing structure before creating new ones.

### Knowledge graph for relationships

Use KG for structured facts, not free text:

```json
{"name": "kg_add", "arguments": {"subject": "LoginBug", "predicate": "affects", "object": "Safari"}}
{"name": "kg_add", "arguments": {"subject": "LoginBug", "predicate": "priority", "object": "high", "valid_from": "2026-05-01"}}
{"name": "kg_query", "arguments": {"entity": "LoginBug"}}
```

Temporal facts (`valid_from`/`valid_to`) let the KG answer "what was true at a given time."

### Agent diaries

Write temporal entries scoped to your agent name:

```json
{"name": "diary_write", "arguments": {"agent": "claude", "entry": "DEBUG: investigated Safari login freeze — suspected CSS :has() compatibility", "topic": "debugging"}}
{"name": "diary_read", "arguments": {"agent": "claude", "last_n": 5}}
```

Entries go to realm `agent_<name>/diary` automatically.

## Backend & embedder control

You can swap the vector backend and embedding model at runtime:

```json
{"name": "set_backend", "arguments": {"backend": "chroma"}}
{"name": "get_backend", "arguments": {}}
{"name": "set_embedder", "arguments": {"model": "minilm"}}
{"name": "get_default_embedder", "arguments": {}}
```

Available backends: `faiss` (default), `chroma` (requires `pip install alt-memory[chroma]`).

Available embedders: `numpy` (default, always), `minilm`, `bert`, `embedding-gemma` (require `pip install alt-memory[onnx]`), `sentence-transformers` (requires `pip install sentence-transformers`), `spacy`.

When you change embedder, all entities are automatically re-embedded. This can take time on large dimensions.

## Sync & maintenance

Before accessing user project files:

```json
{"name": "sync", "arguments": {"project_dir": ["/path/to/project"], "apply": true}}
```

This prunes entities whose source files were deleted or gitignored.

## Checking for duplicates

```json
{"name": "check_duplicate", "arguments": {"content": "new text to check", "threshold": 0.9}}
```

Returns similar existing entities. Default threshold 0.9 (strict).

## Common patterns

### Morning context

```json
[
  {"name": "get_status", "arguments": {}},
  {"name": "diary_read", "arguments": {"agent": "claude", "last_n": 5}},
  {"name": "search", "arguments": {"query": "active tasks", "mode": "hybrid"}},
  {"name": "kg_query", "arguments": {"entity": "claude"}}
]
```

### Investigating a topic

```json
[
  {"name": "search", "arguments": {"query": "topic keywords", "n_results": 20}},
  {"name": "kg_query", "arguments": {"entity": "TopicName"}},
  {"name": "traverse", "arguments": {"start_domain": "topic_domain"}}
]
```

### Storing a discovered bug

```json
[
  {
    "name": "add_entity",
    "arguments": {
      "realm": "project_name",
      "domain": "bugs",
      "content": "Detailed bug description...",
      "metadata": {"severity": "high", "found_in": "v4.3.0"}
    }
  },
  {
    "name": "kg_add",
    "arguments": {"subject": "BugDescription", "predicate": "affects", "object": "ComponentX"}
  },
  {
    "name": "diary_write",
    "arguments": {"agent": "claude", "entry": "Found bug in ComponentX...", "topic": "findings"}
  }
]
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Search returns 0 results | Embedder mismatch or empty index | `set_embedder("numpy")`, verify `get_status()` shows entities |
| MCP returns "Missing required parameters" | Wrong parameter name | Check tool schema via `tools/list` |
| Backend swap fails | Missing package | Install `alt-memory[chroma]` or specific backend |
| Rebuild needed after crash | Corrupt vector index | Use `rebuild_fts` tool or CLI `alt-memory repair --rebuild-fts --vacuum` |
| "Empty vocabulary" warning | numpy embedder has no training data yet | Add a few entities, embedder auto-fits |
