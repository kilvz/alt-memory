"""Benchmark all alt-memory MCP tools for correctness and speed."""
import json
import time
from pathlib import Path

DIM_PATH = str(Path.home() / ".alt-memory")
PYTHON = r"C:\Python314\python.exe"
LOG = []

def log(msg):
    LOG.append(msg)
    print(msg)

class MCPClient:
    def __init__(self):
        import alt_memory.dimension
        self.dim = alt_memory.dimension.Dimension(DIM_PATH)
        self.dim.init()
        from alt_memory.mcp_server import MCPServer
        self.server = MCPServer(self.dim)

    def call(self, method, params=None):
        req = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        t0 = time.perf_counter()
        raw = json.dumps(req)
        resp = self.server.handle_request(raw)
        elapsed = time.perf_counter() - t0
        if resp is None:
            return None, elapsed
        result = json.loads(resp)
        if "error" in result:
            raise RuntimeError(f"{method}: {result['error']}")
        return result.get("result"), elapsed


def main():
    c = MCPClient()

    results = []

    def bench(name, method, params=None, expect_error=False):
        try:
            result, elapsed = c.call(method, params)
            status = "OK" if not expect_error else "UNEXPECTED_OK"
            if expect_error:
                status = "OK( expected error)"
            results.append({"tool": name, "status": status, "time_ms": round(elapsed * 1000, 1)})
            log(f"  [{status}] {name}: {elapsed*1000:.1f}ms")
            return result
        except Exception as e:
            status = f"ERROR: {e}" if not expect_error else "OK(expected)"
            results.append({"tool": name, "status": status, "time_ms": -1})
            log(f"  [{status}] {name}")
            return None

    log("=== ALT-MEMORY TOOL BENCHMARK ===\n")

    # ── Realm operations ──
    log("\n--- Realm / Domain ---")
    bench("create_realm", "create_realm", {"name": "bench_alpaca", "description": "test realm"})
    bench("create_domain", "create_domain", {"realm": "bench_alpaca", "name": "alpha", "description": "test domain"})
    bench("list_realms", "list_realms")
    bench("list_domains", "list_domains", {"realm": "bench_alpaca"})
    bench("get_status", "get_status")
    bench("get_taxonomy", "get_taxonomy")

    # ── Entity CRUD ──
    log("\n--- Entity CRUD ---")
    entity_id = None
    r = bench("add_entity", "add_entity", {
        "realm": "bench_alpaca", "domain": "alpha",
        "content": "Benchmark entity: 中文 Español 日本語 ° — ✓",
        "metadata": {"key": "val", "unicode": "°—✓"}
    })
    if r:
        entity_id = r.get("entity_id")

    if entity_id:
        bench("get_entity", "get_entity", {"entity_id": entity_id})
        bench("update_entity", "update_entity", {
            "entity_id": entity_id,
            "content": "Updated: 中文 Español 日本語 ° — ✓ updated",
            "metadata": {"key": "val2", "updated": True}
        })
        bench("get_entity (after update)", "get_entity", {"entity_id": entity_id})
        bench("delete_entity", "delete_entity", {"entity_id": entity_id})
        bench("get_entity (deleted)", "get_entity", {"entity_id": entity_id}, expect_error=True)

    # ── Batch operations ──
    log("\n--- Batch Operations ---")
    bench("batch_add_entities", "batch_add_entities", {
        "entities": [
            {"realm": "bench_alpaca", "domain": "alpha", "content": f"Batch entity {i}: ✓°中文" }
            for i in range(5)
        ]
    })
    bench("list_entities", "list_entities", {"realm": "bench_alpaca", "domain": "alpha"})

    # ── Search ──
    log("\n--- Search ---")
    bench("search_hybrid", "search", {"query": "Benchmark entity", "mode": "hybrid"})
    bench("search_vector", "search", {"query": "Benchmark entity", "mode": "vector"})
    bench("search_keyword", "search", {"query": "Benchmark entity", "mode": "keyword"})
    bench("search_realm_filtered", "search", {"query": "entity", "realm": "bench_alpaca"})
    bench("search_domain_filtered", "search", {"query": "entity", "domain": "alpha"})

    # ── Duplicate check ──
    log("\n--- Duplicate Check ---")
    bench("check_duplicate (new)", "check_duplicate", {"content": "A completely novel unique string here"})
    bench("check_duplicate (dup)", "check_duplicate", {"content": "Benchmark entity: 中文 Español 日本語 ° — ✓"})

    # ── Knowledge Graph ──
    log("\n--- Knowledge Graph ---")
    bench("kg_add", "kg_add", {"subject": "ConnectomeSeq", "predicate": "developed_by", "object": "BoxuanZhao"})
    bench("kg_add", "kg_add", {"subject": "ConnectomeSeq", "predicate": "published_in", "object": "NatureMethods"})
    bench("kg_add", "kg_add", {"subject": "ConnectomeSeq", "predicate": "funded_by", "object": "WuTsaiInstitute"})
    bench("kg_add", "kg_add", {"subject": "ConnectomeSeq", "predicate": "funded_by", "object": "ElsaUPardeeFoundation"})
    bench("kg_query", "kg_query", {"entity": "ConnectomeSeq"})
    bench("kg_query (predicate)", "kg_query", {"entity": "ConnectomeSeq", "predicate": "funded_by"})
    bench("kg_stats", "kg_stats")
    bench("kg_timeline", "kg_timeline", {"entity": "ConnectomeSeq"})
    bench("kg_invalidate", "kg_invalidate", {"subject": "ConnectomeSeq", "predicate": "funded_by", "object": "ElsaUPardeeFoundation"})
    # verify invalidation
    bench("kg_query (after invalidate)", "kg_query", {"entity": "ConnectomeSeq", "predicate": "funded_by"})

    # ── Agent Records ──
    log("\n--- Agent Records ---")
    bench("record_write", "record_write", {"agent": "benchmark_bot", "entry": "Test record entry: 中文 ✓°", "topic": "testing"})
    bench("record_read", "record_read", {"agent": "benchmark_bot", "last_n": 5})
    bench("list_agents", "list_agents")

    # ── AAAK Dialect ──
    log("\n--- AAAK Dialect ---")
    original = "SYS:mem=42|LANG:en,zh,es|ALC:core.tools.benchmarked|★★★"
    compressed = bench("aaak_compress", "aaak_compress", {"text": original, "max_len": 100})
    if compressed:
        decompressed = bench("aaak_decompress", "aaak_decompress", {"text": compressed})
    bench("aaak_parse", "aaak_parse", {"text": original})
    bench("get_aaak_spec", "get_aaak_spec")

    # ── People Map ──
    log("\n--- People Map ---")
    bench("get_people_map", "get_people_map")
    bench("set_people_map", "set_people_map", {"map": {"Bob": "Robert", "Rob": "Robert", "Bobby": "Robert"}})
    bench("get_people_map (after set)", "get_people_map")

    # ── Persona ──
    log("\n--- Persona ---")
    bench("get_persona", "get_persona")
    bench("set_persona", "set_persona", {"name": "benchmark_test"})
    bench("get_persona (after set)", "get_persona")
    bench("switch_persona", "switch_persona", {"name": "benchmark_test"})

    # ── Tunnels ──
    log("\n--- Tunnels ---")
    bench("create_tunnel", "create_tunnel", {
        "source_realm": "bench_alpaca", "source_domain": "alpha",
        "target_realm": "benchmark", "target_domain": "neuroscience",
        "label": "test tunnel: connecting brain mapping tools"
    })
    bench("find_tunnels", "find_tunnels", {"realm_a": "bench_alpaca", "realm_b": "benchmark"})
    bench("follow_tunnels", "follow_tunnels", {"realm": "bench_alpaca", "domain": "alpha"})
    bench("list_tunnels", "list_tunnels", {"realm": "bench_alpaca"})
    bench("graph_stats", "graph_stats")
    bench("traverse", "traverse", {"start_domain": "alpha", "max_hops": 2})

    # ── Export / Import ──
    log("\n--- Export / Import ---")
    exported = bench("export_collection", "export_collection", {"realm": "bench_alpaca"})
    if exported:
        bench("import_entities", "import_entities", {"entities": exported[:2]})

    # ── Sync (dry-run) ──
    log("\n--- Sync ---")
    bench("sync (dry-run)", "sync", {"project_dir": str(Path.home())})

    # ── Backend / Embedder (read-only, no destructive changes) ──
    log("\n--- Backend / Embedder ---")
    bench("get_backend", "get_backend")
    bench("get_default_embedder", "get_default_embedder")

    # ── Hook settings ──
    log("\n--- Hook Settings ---")
    bench("hook_settings", "hook_settings")

    # ── Memories filed away ──
    log("\n--- Memories Filed Away ---")
    bench("memories_filed_away", "memories_filed_away")

    # ── Rebuild FTS ──
    log("\n--- Rebuild FTS ---")
    bench("rebuild_fts", "rebuild_fts")

    # ── Reconnect ──
    log("\n--- Reconnect ---")
    bench("reconnect", "reconnect")

    # ── Delete domain/realm cleanup ──
    log("\n--- Cleanup ---")
    # need to delete tunnel first
    tunnels_result = c.call("list_tunnels", {"realm": "bench_alpaca"})
    if tunnels_result and tunnels_result[0]:
        for t in tunnels_result[0]:
            if isinstance(t, dict) and "entity_id" in t:
                bench("delete_tunnel", "delete_tunnel", {"tunnel_id": t["entity_id"]})
    bench("delete_domain", "delete_domain", {"realm": "bench_alpaca", "name": "alpha"})
    bench("delete_realm", "delete_realm", {"name": "bench_alpaca"})

    # ── Error handling ──
    log("\n--- Error Handling ---")
    bench("add_entity (missing realm)", "add_entity", {"realm": "NONEXISTENT_REALM_XXXX", "domain": "x", "content": "x"}, expect_error=True)
    bench("add_entity (bad params)", "add_entity", {}, expect_error=True)
    bench("unknown_method", "fly_me_to_the_moon", {}, expect_error=True)

    # ── Summary ──
    log("\n" + "=" * 60)
    log("BENCHMARK SUMMARY")
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
        log(f"  {r['tool']:40s} {r['status']:20s} {r['time_ms']:>8.1f}ms" if r['time_ms'] >= 0
            else f"  {r['tool']:40s} {r['status']:<20s}")

    return results


if __name__ == "__main__":
    main()
