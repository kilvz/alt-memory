# Alt Memory Status

Display the current state of the user's memory dimension.

## Step 1: Gather Dimension Status

Check if MCP tools are available (look for alt_memory_status in available tools).

- If MCP is available: Call the alt_memory_status tool to retrieve dimension state.
- If MCP is not available: Run the CLI command: alt-memory status

## Step 2: Display Realm/Domain/Entity Counts

Present the dimension structure counts clearly:
- Number of realms
- Number of domains
- Number of entities
- Total memories stored

Keep the output concise -- use a brief summary format, not verbose tables.

## Step 3: Knowledge Graph Stats (MCP only)

If MCP tools are available, also call:
- alt_memory_kg_stats -- for a knowledge graph overview (triple count, entity
  count, relationship types)
- alt_memory_graph_stats -- for connectivity information (connected components,
  average connections per entity)

Present these alongside the dimension counts in a unified summary.

## Step 4: Suggest Next Actions

Based on the current state, suggest one relevant action:

- Empty dimension (zero memories): Suggest "Try /alt-memory:mine to add data from
  files, URLs, or text."
- Has data but no knowledge graph (memories exist but KG stats show zero
  triples): Suggest "Consider adding knowledge graph triples for richer
  queries."
- Healthy dimension (has memories and KG data): Suggest "Use /alt-memory:search to
  query your memories."

## Output Style

- Be concise and informative -- aim for a quick glance, not a report.
- Use short labels and numbers, not prose paragraphs.
- If any step fails or a tool is unavailable, note it briefly and continue
  with what is available.
