"""Benchmark MemPalace MCP tools for comparison with alt-memory."""
import time

# Import triggers stdout redirect; restore it immediately
import mempalace.mcp_server as mcp
mcp._restore_stdout()

# Force collection + KG init
coll = mcp._get_collection(create=False)
kg = mcp._get_kg()

LOG = []
def log(msg):
    LOG.append(msg)
    print(msg)

results = []

def bench(label, fn, *a, **kw):
    t0 = time.perf_counter()
    try:
        result = fn(*a, **kw)
        elapsed = time.perf_counter() - t0
        status = "OK"
        results.append({"tool": label, "status": status, "time_ms": round(elapsed * 1000, 1)})
        log(f"  [{status}] {label}: {elapsed*1000:.1f}ms")
        return result
    except Exception as e:
        elapsed = time.perf_counter() - t0
        status = f"ERROR: {e}"
        results.append({"tool": label, "status": status, "time_ms": round(elapsed * 1000, 1)})
        log(f"  [{status}] {label}")
        return None

log("=== MEMPALACE TOOL BENCHMARK ===\n")

# ── Status / Info ──
log("\n--- Status / Info ---")
bench("status", mcp.tool_status)
bench("list_wings", mcp.tool_list_wings)
bench("list_rooms", mcp.tool_list_rooms)  # no wing = all rooms
bench("get_taxonomy", mcp.tool_get_taxonomy)
bench("get_aaak_spec", mcp.tool_get_aaak_spec)

# ── CRUD ──
log("\n--- CRUD ---")
drawer_id = None
r = bench("add_drawer", mcp.tool_add_drawer, wing="bench_mp", room="alpha",
          content="Benchmark drawer: 中文 Español 日本語 ° — ✓ [unicode=°—✓]")
if r and isinstance(r, dict):
    drawer_id = r.get("drawer_id")

if drawer_id:
    bench("get_drawer", mcp.tool_get_drawer, drawer_id=drawer_id)
    bench("update_drawer", mcp.tool_update_drawer, drawer_id=drawer_id,
          content="Updated: 中文 Español 日本語 ° — ✓",
          wing="bench_mp", room="alpha")
    bench("get_drawer (after update)", mcp.tool_get_drawer, drawer_id=drawer_id)
    bench("delete_drawer", mcp.tool_delete_drawer, drawer_id=drawer_id)

# ── Search ──
log("\n--- Search ---")
bench("search", mcp.tool_search, query="Benchmark drawer")
bench("search (wing)", mcp.tool_search, query="drawer", wing="bench_mp")
bench("search (room)", mcp.tool_search, query="drawer", room="alpha")

# ── Duplicate Check ──
log("\n--- Duplicate Check ---")
bench("check_duplicate (new)", mcp.tool_check_duplicate, content="A completely novel unique string here")
bench("check_duplicate (dup)", mcp.tool_check_duplicate, content="Benchmark drawer: 中文 Español 日本語 ° — ✓")

# ── Knowledge Graph ──
log("\n--- Knowledge Graph ---")
bench("kg_add", mcp.tool_kg_add, subject="ConnectomeSeq", predicate="developed_by", object="BoxuanZhao")
bench("kg_add", mcp.tool_kg_add, subject="ConnectomeSeq", predicate="published_in", object="NatureMethods")
bench("kg_add", mcp.tool_kg_add, subject="ConnectomeSeq", predicate="funded_by", object="WuTsaiInstitute")
bench("kg_add", mcp.tool_kg_add, subject="ConnectomeSeq", predicate="funded_by", object="ElsaUPardeeFoundation")
bench("kg_query", mcp.tool_kg_query, entity="ConnectomeSeq")
bench("kg_query (predicate)", mcp.tool_kg_query, entity="ConnectomeSeq", direction="outgoing")
bench("kg_stats", mcp.tool_kg_stats)
bench("kg_timeline", mcp.tool_kg_timeline)
bench("kg_invalidate", mcp.tool_kg_invalidate, subject="ConnectomeSeq", predicate="funded_by", object="ElsaUPardeeFoundation")
bench("kg_query (after invalidate)", mcp.tool_kg_query, entity="ConnectomeSeq")

# ── Diary ──
log("\n--- Diary ---")
bench("diary_write", mcp.tool_diary_write, agent_name="benchmark_bot", entry="Test diary: 中文 ✓°", topic="testing")
bench("diary_read", mcp.tool_diary_read, agent_name="benchmark_bot", last_n=5)

# ── Tunnels / Graph ──
log("\n--- Tunnels / Graph ---")
bench("create_tunnel", lambda: mcp.tool_create_tunnel(
    source_wing="bench_mp", source_room="alpha",
    target_wing="benchmark", target_room="neuroscience",
    label="test tunnel"))
bench("list_tunnels", mcp.tool_list_tunnels)
bench("follow_tunnels", mcp.tool_follow_tunnels, wing="bench_mp", room="alpha")
bench("find_tunnels", mcp.tool_find_tunnels)
bench("graph_stats", mcp.tool_graph_stats)
bench("traverse", mcp.tool_traverse_graph, start_room="alpha", max_hops=2)

# ── Settings / Maintenance ──
log("\n--- Settings / Maintenance ---")
bench("hook_settings", mcp.tool_hook_settings)
bench("memories_filed_away", mcp.tool_memories_filed_away)
bench("reconnect", mcp.tool_reconnect)

# ── Error handling ──
log("\n--- Error Handling ---")
bench("add_drawer (bad args)", mcp.tool_add_drawer)
bench("get_drawer (not found)", mcp.tool_get_drawer, drawer_id="NONEXISTENT")

# ── Summary ──
log("\n" + "=" * 60)
log("MEMBENCH SUMMARY")
log("=" * 60)
total = len(results)
ok = sum(1 for r in results if r["status"].startswith("OK"))
err = sum(1 for r in results if r["status"].startswith("ERROR"))
times = [r["time_ms"] for r in results if r["time_ms"] >= 0]
fast = min(times) if times else 0
slow = max(times) if times else 0
mean = sum(times) / len(times) if times else 0
log(f"Total tools tested: {total}")
log(f"Passed: {ok}  |  Failed: {err}")
log(f"Fastest: {fast:.1f}ms  |  Slowest: {slow:.1f}ms  |  Average: {mean:.1f}ms")

log("\n--- Per-Tool Timing ---")
for r in results:
    log(f"  {r['tool']:35s} {r['status']:20s} {r['time_ms']:>8.1f}ms" if r['time_ms'] >= 0
        else f"  {r['tool']:35s} {r['status']:<20s}")
