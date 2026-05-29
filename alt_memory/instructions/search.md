# Alt Memory Search

When the user wants to search their Alt Memory memories, follow these steps:

## 1. Parse the Search Query

Extract the core search intent from the user's message. Identify any explicit
or implicit filters:
- Realm -- a top-level category (e.g., "work", "personal", "research")
- Domain -- a sub-category within a realm
- Keywords / semantic query -- the actual search terms

## 2. Determine Wing/Room Filters

If the user mentions a specific domain, topic area, or context, map it to the
appropriate realm and/or domain. If unsure, omit filters to search globally. You
can discover the taxonomy first if needed.

## 3. Use MCP Tools (Preferred)

If MCP tools are available, use them in this priority order:

- alt_memory_search(query, realm, domain) -- Primary search tool. Pass the semantic
  query and any realm/domain filters.
- alt_memory_list_wings -- Discover all available realms. Use when the user asks
  what categories exist or you need to resolve a realm name.
- alt_memory_list_rooms(realm) -- List domains within a specific realm. Use to help
  the user navigate or to resolve a domain name.
- alt_memory_get_taxonomy -- Retrieve the full realm/domain/entity tree. Use when
  the user wants an overview of their entire memory structure.
- alt_memory_traverse(domain) -- Walk the knowledge graph starting from a domain.
  Use when the user wants to explore connections and related memories.
- alt_memory_find_tunnels(realm1, realm2) -- Find cross-realm connections (tunnels)
  between two realms. Use when the user asks about relationships between
  different knowledge domains.

## 4. CLI Fallback

If MCP tools are not available, fall back to the CLI:

    alt-memory search "query" [--realm X] [--domain Y]

## 5. Present Results

When presenting search results:
- Always include source attribution: realm, domain, and entity for each result
- Show relevance or similarity scores if available
- Group results by realm/domain when returning multiple hits
- Quote or summarize the memory content clearly

## 6. Offer Next Steps

After presenting results, offer the user options to go deeper:
- Drill deeper -- search within a specific domain or narrow the query
- Traverse -- explore the knowledge graph from a related domain
- Check tunnels -- look for cross-realm connections if the topic spans domains
- Browse taxonomy -- show the full structure for manual exploration
