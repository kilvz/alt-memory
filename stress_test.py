"""Memory stress test: alt-memory vs MemPalace.
Stores 500 entities, benchmarks throughput, then tests recall accuracy.
"""
import json
import sys
import time
from pathlib import Path
from hashlib import md5

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[reportAttributeAccessIssue]

# ─── Dataset generation ────────────────────────────────────────────────
TOPICS = [
    "machine learning", "neuroscience", "quantum computing", "cryptography",
    "climate science", "materials engineering", "synthetic biology",
    "robotics", "natural language processing", "computer vision",
]
ADJS = ["novel", "scalable", "efficient", "robust", "lightweight",
        "high-performance", "interpretable", "distributed", "real-time", "autonomous"]
NOUNS = ["architecture", "framework", "algorithm", "protocol", "system",
         "model", "approach", "method", "pipeline", "platform"]
YEARS = ["2023", "2024", "2025", "2026"]
CITATION_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def gen_entity(i: int) -> dict:
    t = TOPICS[i % len(TOPICS)]
    adj = ADJS[(i // 7) % len(ADJS)]
    noun = NOUNS[(i // 3) % len(NOUNS)]
    year = YEARS[(i // 13) % len(YEARS)]
    seed = md5(str(i).encode()).hexdigest()
    rank = (i % 100) + 1
    return {
        "realm": "stress_test",
        "domain": "research",
        "content": (
            f"Paper {i}: A {adj} {noun} for {t} (2026). "
            f"Introduced a breakthrough {t} method achieving {rank}% accuracy "
            f"on benchmark datasets. Published in Nature Machine Intelligence, "
            f"led by author_{seed[:8]} from University_{seed[8:12]}. "
            f"Key innovation: {adj}_{noun}_v{year[-2:]} enables real-time "
            f"processing with {rank}ms latency — a {rank}x improvement over "
            f"prior work. Code: github.com/lab_{seed[:4]}/{adj}_{noun}. "
            f"温度: {rank}°C | 精度: {rank}% ✓ | cost: ${rank}M — funded"
        ),
        "metadata": {
            "topic": t,
            "year": year,
            "rank": rank,
            "adj": adj,
            "noun": noun,
            "author": f"author_{seed[:8]}",
        }
    }

ENTITIES = [gen_entity(i) for i in range(500)]
SEARCH_QUERIES = [
    # (query, expected_topic) pairs for recall testing
    ("breakthrough machine learning architecture", "machine learning"),
    ("novel neuroscience framework", "neuroscience"),
    ("efficient quantum computing algorithm", "quantum computing"),
    ("scalable cryptography protocol", "cryptography"),
    ("robust climate science system", "climate science"),
    ("high-performance materials engineering model", "materials engineering"),
    ("lightweight synthetic biology approach", "synthetic biology"),
    ("interpretable robotics method", "robotics"),
    ("distributed natural language processing pipeline", "natural language processing"),
    ("autonomous computer vision platform", "computer vision"),
]

# ─── Helpers ───────────────────────────────────────────────────────────
LOG = []
LOG_FILE = Path.home() / ".alt-memory" / "stress_test_results.txt"

def say(msg):
    LOG.append(msg)
    print(msg)

def log_results():
    LOG_FILE.write_text("\n".join(LOG), encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════
#  ALT-MEMORY STRESS TEST
# ═══════════════════════════════════════════════════════════════════════
def test_alt_memory():
    import alt_memory.dimension
    import alt_memory.mcp_server
    dim = alt_memory.dimension.Dimension(str(Path.home() / ".alt-memory"))
    dim.init()
    server = alt_memory.mcp_server.MCPServer(dim)

    def call(method, params=None):
        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})
        resp = server.handle_request(req)
        if resp is None:
            return None
        r = json.loads(resp)
        if "error" in r:
            raise RuntimeError(f"{method}: {r['error']}")
        return r.get("result")

    say("\n" + "=" * 60)
    say("ALT-MEMORY STRESS TEST")
    say("=" * 60)

    # --- Storage ---
    t0 = time.perf_counter()
    r = call("batch_add_entities", {"entities": ENTITIES})
    store_time = time.perf_counter() - t0
    say(f"\nStored 500 entities via batch_add_entities")
    say(f"  Time: {store_time:.2f}s")
    say(f"  Throughput: {500/store_time:.0f} entities/sec")

    # --- Get by ID ---
    t0 = time.perf_counter()
    list_r = call("list_entities", {"realm": "stress_test", "domain": "research", "limit": 500})
    list_time = time.perf_counter() - t0
    ids = [e["id"] for e in (list_r or [])]
    say(f"\nListed {len(ids)} entities in {list_time*1000:.1f}ms")

    # --- Random get ---
    if len(ids) >= 10:
        t0 = time.perf_counter()
        for eid in ids[:10]:
            call("get_entity", {"entity_id": eid})
        get_time = time.perf_counter() - t0
        say(f"Get 10 entities by ID: {get_time*1000:.1f}ms avg {(get_time/10)*1000:.1f}ms each")

    # --- Search recall ---
    say("\n--- Search Recall ---")
    correct = 0
    total_q = len(SEARCH_QUERIES)
    for query, expected_topic in SEARCH_QUERIES:
        t0 = time.perf_counter()
        results = call("search", {"query": query, "realm": "stress_test", "n_results": 5})
        qt = time.perf_counter() - t0
        # Check if any top-5 result has matching topic
        top_topics = [r.get("metadata", {}).get("topic", "") for r in (results or [])]
        hit = expected_topic in top_topics
        if hit:
            correct += 1
        rank = top_topics.index(expected_topic) + 1 if hit else "NR"
        say(f"  [{rank}/{len(top_topics)}] {query[:50]:50s} {qt*1000:5.1f}ms {hit}")

    p1 = correct / total_q
    say(f"\nPrecision@5: {correct}/{total_q} = {p1:.0%}")

    # --- Hybrid vs keyword vs vector speed ---
    say("\n--- Mode Comparison ---")
    for mode in ["hybrid", "vector", "keyword"]:
        t0 = time.perf_counter()
        for q, _ in SEARCH_QUERIES[:5]:
            call("search", {"query": q, "realm": "stress_test", "mode": mode, "n_results": 3})
        mt = time.perf_counter() - t0
        say(f"  {mode:8s}: {mt*1000:.0f}ms for 5 queries ({mt/5*1000:.0f}ms avg)")

    # --- Duplicate check (should find dups since stored same content) ---
    t0 = time.perf_counter()
    dup = call("check_duplicate", {"content": ENTITIES[0]["content"]})
    dup_time = time.perf_counter() - t0
    say(f"\nDuplicate check on existing content: {dup_time*1000:.1f}ms")
    say(f"  Found duplicate? {dup}")

    # --- KG stress ---
    say("\n--- KG Stress ---")
    t0 = time.perf_counter()
    for i in range(100):
        e = ENTITIES[i]
        call("kg_add", {"subject": f"Paper{i}", "predicate": "uses_topic", "object": e["metadata"]["topic"]})
    kg_time = time.perf_counter() - t0
    say(f"Added 100 KG triples: {kg_time:.2f}s ({100/kg_time:.0f} triples/sec)")

    t0 = time.perf_counter()
    for i in range(10):
        call("kg_query", {"entity": SEARCH_QUERIES[i][1]})
    kq_time = time.perf_counter() - t0
    say(f"10 KG queries: {kq_time*1000:.0f}ms avg {kq_time/10*1000:.1f}ms each")

    # --- Rebuild FTS ---
    t0 = time.perf_counter()
    call("rebuild_fts")
    fts_time = time.perf_counter() - t0
    say(f"\nRebuild FTS: {fts_time*1000:.1f}ms")

    return {
        "store_time": store_time,
        "store_throughput": 500 / store_time,
        "list_time": list_time,
        "search_recall": p1,
        "kg_write_speed": 100 / kg_time,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MEMPALACE STRESS TEST
# ═══════════════════════════════════════════════════════════════════════
def test_mempalace():
    # Import triggers stdout redirect; restore
    import mempalace.mcp_server as mcp
    mcp._restore_stdout()
    coll = mcp._get_collection(create=True)
    kg = mcp._get_kg()

    say("\n" + "=" * 60)
    say("MEMPALACE STRESS TEST")
    say("=" * 60)

    # --- Storage ---
    t0 = time.perf_counter()
    for i, e in enumerate(ENTITIES):
        mcp.tool_add_drawer(
            wing=e["realm"], room=e["domain"],
            content=e["content"],
        )
        if (i + 1) % 100 == 0:
            say(f"  Stored {i+1}/500...")
    store_time = time.perf_counter() - t0
    say(f"\nStored 500 entities via add_drawer")
    say(f"  Time: {store_time:.2f}s")
    say(f"  Throughput: {500/store_time:.0f} entities/sec")

    # --- Search recall ---
    say("\n--- Search Recall ---")
    correct = 0
    total_q = len(SEARCH_QUERIES)
    for query, expected_topic in SEARCH_QUERIES:
        t0 = time.perf_counter()
        results = mcp.tool_search(query=query, limit=5)
        qt = time.perf_counter() - t0
        matched = expected_topic.lower() in results.get("summary", "").lower() if isinstance(results, dict) else False
        top_texts = []
        for r in results.get("results", [] if isinstance(results, dict) else results)[:5]:
            t = r.get("text", r.get("content", "")) if isinstance(r, dict) else str(r)
            top_texts.append(t)
        hit = any(expected_topic in t for t in top_texts)
        if hit:
            correct += 1
        say(f"  {hit} {query[:50]:50s} {qt*1000:5.1f}ms")

    p1 = correct / total_q
    say(f"\nPrecision@5: {correct}/{total_q} = {p1:.0%}")

    # --- KG stress ---
    say("\n--- KG Stress ---")
    t0 = time.perf_counter()
    for i in range(100):
        e = ENTITIES[i]
        mcp.tool_kg_add(subject=f"Paper{i}", predicate="uses_topic", object=e["metadata"]["topic"])
    kg_time = time.perf_counter() - t0
    say(f"Added 100 KG triples: {kg_time:.2f}s ({100/kg_time:.0f} triples/sec)")

    t0 = time.perf_counter()
    for i in range(10):
        mcp.tool_kg_query(entity=SEARCH_QUERIES[i][1])
    kq_time = time.perf_counter() - t0
    say(f"10 KG queries: {kq_time*1000:.0f}ms avg {kq_time/10*1000:.1f}ms each")

    return {
        "store_time": store_time,
        "store_throughput": 500 / store_time,
        "search_recall": p1,
        "kg_write_speed": 100 / kg_time,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = {}

    if "--alt" in sys.argv or "--all" in sys.argv or len(sys.argv) == 1:
        results["alt_memory"] = test_alt_memory()

    if "--mp" in sys.argv or "--all" in sys.argv or len(sys.argv) == 1:
        results["mempalace"] = test_mempalace()

    say("\n" + "=" * 60)
    say("STRESS TEST COMPARISON")
    say("=" * 60)
    am = results.get("alt_memory", {})
    mp = results.get("mempalace", {})

    if am:
        say(f"\n  {'Metric':30s} {'alt-memory':>15s} {'MemPalace':>15s}")
        say(f"  {'-'*30} {'-'*15} {'-'*15}")
        for metric, key in [("Storage time", "store_time"),
                             ("Throughput (ents/s)", "store_throughput"),
                             ("Search Precision@5", "search_recall"),
                             ("KG write speed", "kg_write_speed")]:
            av = am.get(key, "N/A")
            mv = mp.get(key, "N/A")
            if isinstance(av, float):
                av = f"{av:.2f}" if "time" in key else f"{av:.0%}" if "recall" in key else f"{av:.0f}"
            if isinstance(mv, float):
                mv = f"{mv:.2f}" if "time" in key else f"{mv:.0%}" if "recall" in key else f"{mv:.0f}"
            say(f"  {metric:30s} {av:>15s} {mv:>15s}")

    log_results()
    say(f"\nResults saved to {LOG_FILE}")
