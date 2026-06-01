# Personas — Character Definitions with Isolated Memory

A **persona** in Alt Memory is a character definition — a system prompt with a name — just like [Eternal AI's `.txt` character files](https://github.com/eternalai-org/eternal-ai). The model and framework are deployment choices made by the user at agent creation time, NOT part of the character definition. Alt Memory follows the same separation.

Each persona gets its own `persona_<name>` realm for isolated memory, and its definition is persisted in `persona.json` inside the dimension directory.

---

## Storage: `persona.json`

Persona state lives in a single JSON file at the dimension root:

```json
{
  "active": "donald_trump",
  "personas": {
    "donald_trump": {
      "name": "donald_trump",
      "system_prompt": "Act as if you are Donald Trump, the President of the United States.",
      "description": "A Donald Trump twin agent",
      "metadata": {}
    },
    "coder": {
      "name": "coder",
      "system_prompt": "You are an expert Python developer.",
      "description": "Coding expert",
      "metadata": { "lang": "python" }
    }
  }
}
```

**Legacy upgrade**: The old format `{"persona": "name"}` is auto-upgraded to the new format on first read.

---

## Python API

```python
from alt_memory import Dimension

d = Dimension(path="/path/to/dim")
d.init()
```

### `get_persona()`

Returns the current active persona definition, or `{"name": ""}` if unset.

```python
d.get_persona()
# → {"name": "donald_trump", "system_prompt": "...", ...}
```

### `set_persona(name, system_prompt=None, description=None, metadata=None)`

Set the active persona. Creates a `persona_<name>` realm if needed. If the persona already exists, existing fields are preserved unless overridden.

```python
d.set_persona(
    "donald_trump",
    system_prompt="Act as if you are Donald Trump, the President of the United States. "
                  "Respond with confidence, use superlatives, and never back down. "
                  "Make America great again.",
    description="A Donald Trump twin agent",
)
# → {"name": "donald_trump", "system_prompt": "...", "description": "...", "metadata": {}}
```

### `create_persona(name, system_prompt="", description="", metadata=None)`

Create a persona definition **without activating it**. Raises `ValueError` if the name already exists.

```python
d.create_persona(
    name="coder",
    system_prompt="You are an expert Python developer.",
    description="Coding expert",
    metadata={"lang": "python"},
)
```

### `switch_persona(name, **kwargs)`

Alias for `set_persona`. Switches the active persona.

```python
d.switch_persona("coder", system_prompt="You are an expert Python developer.")
```

### `get_persona_character(name=None)`

**The key method for LLM agents.** Returns the raw system prompt string for a persona. If `name` is omitted, uses the active persona. Returns `""` if not found.

```python
system_prompt = d.get_persona_character("donald_trump")
# → "Act as if you are Donald Trump..."
```

This is what you inject as `{"role": "system", "content": system_prompt}` in your LLM call.

### `list_personas()`

Returns all registered personas with an `active` boolean flag.

```python
d.list_personas()
# → [{"name": "donald_trump", "system_prompt": "...", "active": True},
#     {"name": "coder", "system_prompt": "...", "active": False}]
```

### `delete_persona(name)`

Removes a persona from the registry. Returns `True` if it existed. Does **not** delete the `persona_<name>` realm or its entities.

```python
d.delete_persona("coder")
# → True
```

---

## MCP Tools

| Tool | Required params | Returns |
|------|----------------|---------|
| `get_persona` | — | Active persona dict |
| `set_persona` | `name` | Full persona dict |
| `switch_persona` | `name` | Full persona dict |
| `create_persona` | `name` | Full persona dict |
| `list_personas` | — | `{"personas": [...]}` |
| `delete_persona` | `name` | `{"deleted": true/false}` |
| `get_persona_character` | `name` (optional) | `{"system_prompt": "..."}` |

```json
{"name": "create_persona", "arguments": {"name": "coder", "system_prompt": "You are an expert Python developer."}}
{"name": "set_persona", "arguments": {"name": "coder"}}
{"name": "get_persona", "arguments": {}}
{"name": "get_persona_character", "arguments": {"name": "coder"}}
{"name": "list_personas", "arguments": {}}
{"name": "delete_persona", "arguments": {"name": "coder"}}
```

---

## Persona Realm Isolation

Every persona gets its own realm: `persona_<name>`.

```python
d.set_persona("donald_trump")
d.list_realms()
# → [..., {"name": "persona_donald_trump", "description": "Persona: donald_trump"}]
```

The realm is created but **not auto-populated** — it's a reserved namespace. The AI agent is expected to file persona-scoped memories, KG facts, and diary entries there by specifying `realm="persona_<name>"`:

```python
# File a memory to the persona's realm
d.add_entity("persona_donald_trump", "ideas", "Build a wall and make the AI pay for it")

# Search within the persona's realm only
d.search("wall", realm="persona_donald_trump")

# Add KG facts scoped to this persona
d.kg_add("DonaldTrumpPersona", "opinion", "AI", source="persona_donald_trump")

# Diary entries for this persona
d.diary_write(agent="donald_trump", entry="User asked about AI policy.", realm="persona_donald_trump")
```

Switching persona does not affect other realms — memories in `persona_coder` stay isolated from `persona_donald_trump`.

---

## Memory Retrieval Architecture

This section covers how Alt Memory searches and retrieves memories. Since persona realms are just regular realms, all of these operations work identically on `persona_<name>` — just pass `realm="persona_<name>"` to scope the search.

### Three Search Modes

| Mode | Method | How it works | Best for |
|------|--------|-------------|----------|
| **Vector** | `mode="vector"` | Embed query → FAISS `IndexIDMap.search()` → cosine distance ranking | Semantic similarity ("find related concepts") |
| **Keyword** | `mode="keyword"` | FTS5 token match → BM25 ranking | Exact term lookup ("find where I wrote 'login bug'") |
| **Hybrid** | `mode="hybrid"` (default) | Vector + keyword results merged, re-ranked by weighted BM25 | General-purpose — catches both semantic and exact matches |

### Entry Point: `search()`

```python
def search(query, n_results=10, realm=None, domain=None, mode="hybrid"):
    if no vector store:      fallback to keyword
    if mode == "keyword":    _keyword_search()
    if mode == "vector":     _vector_search()
    if mode == "hybrid":     _hybrid_search()
```

The `realm` and `domain` filters are applied downstream in each search path. The `mode` parameter selects the retrieval strategy.

### Vector Path (`_vector_search`)

```
query text
  → embedder.embed(query) → np.ndarray
  → FaissStore.search(embedding, k=n_results×2)
  → FAISS IndexIDMap returns (ids, texts, distances, metadatas)
  → Python-side filter by realm/domain
  → Return top-k SearchResult items sorted by cosine distance
```

```python
# dimension.py:1062
def _vector_search(self, query, n_results, realm, domain):
    query_emb = self._embedder.embed([query])[0]
    ids, texts, distances, metadatas = self._store.search(query_emb, n_results=n_results * 2)
    # filter realm/domain, return top-n_results
```

The embedder is pluggable — numpy TF-IDF/SVD by default, but can be ONNX MiniLM, BERT, sentence-transformers, or spaCy. Regardless of embedder, the FAISS index stores 384–768 dim float vectors and searches via exact (brute-force) L2/cosine.

### Keyword Path (`_keyword_search`)

```
query: "login safari"
  → _build_fts_query() → "login* AND safari*"
  → FTS5 MATCH on entities_fts table
  → JOIN entities e ON e.id = f.id
  → Optional WHERE e.realm = ? / e.domain = ? filter
  → ORDER BY rank (BM25 built-in)
  → LIMIT n_results
  → Fallback to _vector_search if FTS5 query fails
```

```python
# dimension.py:1082
def _keyword_search(self, query, n_results, realm, domain):
    if not self._fts_enabled:
        return self._vector_search(query, n_results, realm, domain)  # fallback
    fts_query = self._build_fts_query(query)
    sql = """SELECT e.id, e.realm, e.domain, e.content, e.metadata, rank
             FROM entities_fts f JOIN entities e ON e.id = f.id
             WHERE entities_fts MATCH ?"""
    # + optional realm/domain WHERE clauses
    # + ORDER BY rank LIMIT ?
    rows = self._db_execute(sql, params).fetchall()
    # wrap in SearchResult with distance = 1.0 - rank
```

**FTS5 query construction** (`_build_fts_query`, `dimension.py:1112`):

| Input | Output | Rule |
|-------|--------|------|
| `"login safari"` | `"login* AND safari*"` | Bare terms → `term*` prefix wildcards joined by AND |
| `"login AND safari"` | `"login AND safari"` | Operators (AND, OR, NOT, NEAR) passed through verbatim |
| `"\"login safari\""` | `"\"login safari\""` | Phrase queries passed through verbatim |
| `"multi-part"` | `"multi* AND part*"` | Hyphenated terms split on `-` |

### Hybrid Path (`_hybrid_search`)

```
query
  ├── _vector_search(query, n_results×2)   → vec_results
  └── _keyword_search(query, n_results×2)  → kw_results

1. Normalize keyword distances to [0, 1] (match cosine scale)
2. Dedup by entity ID — vec_results first, kw_results fill gaps
3. Sort by distance descending
4. Return top-n_results
```

```python
# dimension.py:1138
def _hybrid_search(self, query, n_results, realm, domain):
    vec_results = self._vector_search(query, n_results * 2, realm, domain)
    kw_results = self._keyword_search(query, n_results * 2, realm, domain)
    # normalize keyword distances, dedup by id, sort, return
```

### Full Pipeline: `search_memories()`

The MCP server's `search` tool uses `search_memories()` (`dimension.py:881`) — a richer pipeline than `search()`:

```
search_memories(query, realm, domain, candidate_strategy, max_distance, vector_disabled)

1. vector_disabled=True? → route to keyword-only, return
2. _vector_search(query, n_results×3)   → vec_candidates
3. _keyword_search(query, n_results×3)  → kw_candidates
4. Apply max_distance threshold on vec_results (filter out distant matches)
5. Merge candidates (strategy: "vector" or "union")
   - "vector":   only vector candidates (default)
   - "union":    append BM25-only results not already in vec candidates
6. Node boost — entities whose source files have cross-reference nodes get a boost
7. Hydration — for boosted results, find best keyword-matching neighbor chunks (±1)
8. BM25 re-rank — _hybrid_rank(hits, query) with weights 0.6 vector + 0.4 BM25
9. Attach node_preview text to results
10. Return top-n_results dict
```

**Node boost** (`dimension.py:976`): When a source file has entity cross-references in the `nodes` table (e.g., "→entity_id1,entity_id2"), all results from that file get a `node_boost` multiplier. This prioritizes memories that are actively cross-referenced over orphaned ones.

**Hydration** (`dimension.py:1005`): For boosted results, the system re-queries the source file's chunks, picks the one with the best BM25 match to the query, and expands it with ±1 neighbor chunks for context.

### Direct Lookups

These bypass the search pipeline entirely:

```python
# By primary key — raw SQL
d.get_entity("entity_abc123")
# → {"id": "entity_abc123", "realm": "...", "domain": "...",
#     "content": "...", "metadata": {...}, "source_file": "...", "created_at": "..."}

# By realm/domain with pagination — SQL with filters, sorted by created_at DESC
d.list_entities(realm="persona_coder", limit=20, offset=0)
d.list_entities(realm="persona_coder", domain="preferences")
```

### Full Pipeline Diagram

```
                    ┌──────────────┐
                    │  search()    │
                    │  or          │
                    │ search_mem() │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Mode check  │
                    └──┬───┬───┬───┘
                       │   │   │
              ┌────────┘   │   └────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ keyword  │ │ vector   │ │ hybrid   │
        │ FTS5     │ │ embed    │ │ vec ×2   │
        │ BM25     │ │ FAISS    │ │ kw  ×2   │
        │ rank     │ │ cosine   │ │ merge    │
        └──────────┘ │ distance │ │ re-rank  │
                     └──────────┘ └──────────┘
                                      │
                              ┌───────▼────────┐
                              │ search_mem()   │
                              │ extra steps:   │
                              │ • max_distance │
                              │ • node boost   │
                              │ • hydration    │
                              │ • BM25 rerank  │
                              └────────────────┘
```

---

## Retrieving Persona Memories

The `persona_<name>` realm is a regular realm — all search, list, and retrieval operations work on it identically, just filtered by `realm=`.

### Search

```python
# Semantic + keyword hybrid search limited to the persona's realm
results = d.search("async patterns", realm="persona_coder")

# Vector-only or keyword-only
d.search("async", realm="persona_coder", mode="vector")
d.search("async", realm="persona_coder", mode="keyword")
```

### List all entities

```python
# All memories in the persona's realm
d.list_entities(realm="persona_coder")

# Filter by domain within the persona's realm
d.list_entities(realm="persona_coder", domain="preferences")
d.list_entities(realm="persona_coder", domain="patterns")
```

### Get entity by ID

```python
# If you have an entity ID from a search result
d.get_entity("entity_abc123")
```

### Knowledge graph queries

```python
# Query KG facts filed under the persona's realm
d.kg_query("DonaldTrumpPersona")

# All KG stats for the dimension (includes persona facts)
d.kg_stats()
```

### Diary entries

```python
# Read recent diary entries filed to the persona's realm
d.diary_read(agent="donald_trump", last_n=5, realm="persona_donald_trump")
```

### Via MCP

```json
{"name": "search", "arguments": {"query": "async", "realm": "persona_coder"}}
{"name": "list_entities", "arguments": {"realm": "persona_coder"}}
{"name": "list_entities", "arguments": {"realm": "persona_coder", "domain": "preferences"}}
{"name": "kg_query", "arguments": {"entity": "DonaldTrumpPersona"}}
{"name": "record_read", "arguments": {"agent": "donald_trump", "last_n": 5, "realm": "persona_donald_trump"}}
```

### Search across all personas

Omit the realm filter to search all realms at once, including all persona realms:

```python
results = d.search("async")
# Returns results from persona_coder, persona_assistant, and any other realm
```

### Search all personas only

No built-in "all personas" wildcard — but you can query multiple persona realms in sequence:

```python
personas = d.list_personas()
all_results = []
for p in personas:
    realm = f"persona_{p['name']}"
    all_results.extend(d.search("async", realm=realm))
```

---

## When to Inject the Persona (and When Not To)

The persona character definition is **only injected on calls where the agent speaks as that character** — typically user-facing chat responses. Internal processing calls should NOT receive the persona.

### ✅ Inject persona on

- **Chat responses** — the agent replies to the user in-character
- **Conversation continuation** — maintaining character across a dialogue
- **Any output visible to the user** where the agent represents that persona

### ❌ Do NOT inject on

- **Memory extraction** — `AutoMemoryLearner`, `HeartbeatMemoryExtractor` need neutral extraction, not roleplay
- **Tool execution** — tool calls (search, fetch, calculate) should be factual, not character-affected
- **Internal reasoning** — planning, chain-of-thought, reflection steps
- **Embedding generation** — semantic vectors should represent content, not character voice
- **KG fact extraction** — extracting `subject→predicate→object` triples is mechanical, not creative

### Why

If you inject a Donald Trump persona into every call, your memory extraction will say things like "tremendous memory about AI, the best memory." Your tool calls will waste tokens on character noise. Your embeddings will cluster around character voice rather than semantic content.

The persona is the **output face** of the agent — it colors what the user sees, not what the system processes internally.

### The standard injection pattern

```python
import urllib.request, json

# Only get and inject the persona for chat responses
system_prompt = d.get_persona_character("donald_trump")

payload = json.dumps({
    "model": "fast-free",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What do you think about AI?"}
    ],
}).encode()

req = urllib.request.Request(
    "https://api.kilv.my.id/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req, timeout=30)
result = json.loads(resp.read())
print(result["choices"][0]["message"]["content"])
# → "Let me tell you, AI is tremendous..."
```

The persona is just the character file — you choose the model, endpoint, temperature, and all other parameters at chat time:

```python
prompt = d.get_persona_character("coder")
# Your agent framework decides:
#   model = "gpt-4o"  or  "claude-sonnet"  or  "mixtral"
#   temperature = 0.7
# None of these are part of the persona.
```

---

## Workflow Example

```python
from alt_memory import Dimension

# 1. Open a dimension
d = Dimension(path="./my-dim")
d.init()

# 2. Create personas
d.create_persona("assistant", system_prompt="You are a helpful, concise assistant.")
d.create_persona("coder", system_prompt="You are an expert Python developer. Write clean, tested code.")

# 3. Activate one
d.set_persona("coder")

# 4. Get the character definition for your LLM
prompt = d.get_persona_character()
print(prompt)  # → "You are an expert Python developer..."

# 5. File persona-scoped memories
d.add_entity("persona_coder", "patterns", "User prefers async Python with asyncio")
d.add_entity("persona_coder", "preferences", "User likes type hints everywhere")

# 6. Switch persona
d.set_persona("assistant")

# 7. Search within a specific persona's context
results = d.search("async", realm="persona_coder")
```

---

## Design Principles

1. **Persona = character definition only** — name + system prompt. No model, no framework.
2. **Model/framework are deployment choices** — chosen by the user when creating/launching the agent, matching Eternal AI's `eai agent create -m <model> -f <framework>`.
3. **Persona isolation via realms** — `persona_<name>` realm keeps each character's memories separate.
4. **The persona registry is just JSON** — `persona.json` in the dimension directory. Easy to version, backup, or edit manually.
5. **Realm survives persona deletion** — deleting a persona from the registry leaves its realm intact. Data is never accidentally lost.
