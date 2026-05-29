# Alt Memory

AI memory system. Store everything, find anything. Local, free, no API key.

---

## Slash Commands

| Command              | Description                    |
|----------------------|--------------------------------|
| /alt-memory:init      | Install and set up Alt Memory   |
| /alt-memory:search    | Search your memories            |
| /alt-memory:mine      | Mine projects and conversations |
| /alt-memory:status    | Dimension overview and stats    |
| /alt-memory:help      | This help message               |

---

## MCP Tools (19)

### Dimension (read)
- alt_memory_status -- Dimension status and stats
- alt_memory_list_wings -- List all realms
- alt_memory_list_rooms -- List domains in a realm
- alt_memory_get_taxonomy -- Get the full taxonomy tree
- alt_memory_search -- Search memories by query
- alt_memory_check_duplicate -- Check if a memory already exists
- alt_memory_get_aaak_spec -- Get the AAAK specification

### Dimension (write)
- alt_memory_add_drawer -- Add a new memory (entity)
- alt_memory_delete_drawer -- Delete a memory (entity)

### Knowledge Graph
- alt_memory_kg_query -- Query the knowledge graph
- alt_memory_kg_add -- Add a knowledge graph entry
- alt_memory_kg_invalidate -- Invalidate a knowledge graph entry
- alt_memory_kg_timeline -- View knowledge graph timeline
- alt_memory_kg_stats -- Knowledge graph statistics

### Navigation
- alt_memory_traverse -- Traverse the dimension structure
- alt_memory_find_tunnels -- Find cross-realm connections
- alt_memory_graph_stats -- Graph connectivity statistics

### Agent Diary
- alt_memory_diary_write -- Write a diary entry
- alt_memory_diary_read -- Read diary entries

---

## CLI Commands

    alt-memory init <dir>                  Initialize a new dimension
    alt-memory mine <dir>                  Mine a project (default mode)
    alt-memory mine <dir> --mode convos    Mine conversation exports
    alt-memory search "query"              Search your memories
    alt-memory split <dir>                 Split large transcript files
    alt-memory wake-up                     Load dimension into context
    alt-memory compress                    Compress dimension storage
    alt-memory status                      Show dimension status
    alt-memory repair                      Rebuild vector index
    alt-memory mcp                         Show MCP setup command
    alt-memory hook run                    Run hook logic (for harness integration)
    alt-memory instructions <name>         Output skill instructions

---

## Auto-Save Hooks

- Stop hook -- Automatically saves memories every 15 messages. Counts human
  messages in the session transcript (skipping command-messages). When the
  threshold is reached, blocks the AI with a save instruction. Uses
  ~/.alt-memory/hook_state/ to track save points per session. If
  stop_hook_active is true, passes through to prevent infinite loops.

- PreCompact hook -- Emergency save before context compaction. Always blocks
  with a comprehensive save instruction because compaction means the AI is
  about to lose detailed context.

Hooks read JSON from stdin and output JSON to stdout. They can be invoked via:

    echo '{"session_id":"abc","stop_hook_active":false,"transcript_path":"..."}' | alt-memory hook run --hook stop --harness claude-code

---

## Architecture

    Realms (projects/people)
      +-- Domains (topics)
            +-- Closets (summaries)
                  +-- Entities (verbatim memories)

    Halls connect domains within a realm.
    Tunnels connect domains across realms.

The dimension is stored locally using ChromaDB for vector search and SQLite for
metadata. No cloud services or API keys required.

---

## Getting Started

1. /alt-memory:init -- Set up your dimension
2. /alt-memory:mine -- Mine a project or conversation
3. /alt-memory:search -- Find what you stored
