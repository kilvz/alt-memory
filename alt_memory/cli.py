"""Alt Memory CLI — manage your memory dimension from the command line."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import alt_memory
from alt_memory.dimension import Dimension

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Alt Memory CLI")
    parser.add_argument("--dimension", default="~/.alt-memory",
                        help="Path to the dimension directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Multi-pass initialization with entity/domain detection")
    p_init.add_argument("dir", nargs="?", default=None, help="Project directory to initialize")
    p_init.add_argument("--version", type=int, default=2)
    p_init.add_argument("--lang", default=None, help="Comma-separated language codes (e.g. en,pt-br)")
    p_init.add_argument("--llm-provider", default=None, help="LLM provider (ollama, openai-compat, anthropic)")
    p_init.add_argument("--llm-model", default=None, help="LLM model name")

    sub.add_parser("status", help="Show dimension status")

    p_add = sub.add_parser("add", help="Add an entity")
    p_add.add_argument("--realm", "-w", required=True)
    p_add.add_argument("--domain", "-r", required=True)
    p_add.add_argument("--content", "-c", required=True)
    p_add.add_argument("--meta", "-m", default="{}")
    p_add.add_argument("--source", "-s", default="")

    p_srch = sub.add_parser("search", help="Search the dimension")
    p_srch.add_argument("query", nargs="?", default="")
    p_srch.add_argument("--realm", "-w")
    p_srch.add_argument("--domain", "-r")
    p_srch.add_argument("--limit", "-l", type=int, default=10)
    p_srch.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid")

    p_get = sub.add_parser("get", help="Get an entity by ID")
    p_get.add_argument("entity_id")

    p_list = sub.add_parser("list", help="List entities")
    p_list.add_argument("--realm", "-w")
    p_list.add_argument("--domain", "-r")
    p_list.add_argument("--limit", "-l", type=int, default=20)
    p_list.add_argument("--offset", "-o", type=int, default=0)

    p_realm = sub.add_parser("realms", help="List realms")
    p_realm.add_argument("--verbose", "-v", action="store_true")

    p_room = sub.add_parser("rooms", help="List domains")
    p_room.add_argument("--realm", "-w")

    p_del = sub.add_parser("delete", help="Delete an entity")
    p_del.add_argument("entity_id")

    p_kga = sub.add_parser("kg-add", help="Add KG fact")
    p_kga.add_argument("--subject", "-s", required=True)
    p_kga.add_argument("--predicate", "-p", required=True)
    p_kga.add_argument("--object", "-o", required=True)
    p_kga.add_argument("--source", default="")
    p_kga.add_argument("--valid-from")

    p_kgq = sub.add_parser("kg-query", help="Query KG")
    p_kgq.add_argument("entity", nargs="?", default="")
    p_kgq.add_argument("--predicate")
    p_kgq.add_argument("--as-of")
    p_kgq.add_argument("--direction", default="both",
                       choices=["outgoing", "incoming", "both"])
    p_kgq.add_argument("--all", "-a", action="store_true")

    p_kgi = sub.add_parser("kg-invalidate", help="Invalidate KG fact")
    p_kgi.add_argument("--subject", "-s", required=True)
    p_kgi.add_argument("--predicate", "-p", required=True)
    p_kgi.add_argument("--object", "-o", required=True)
    p_kgi.add_argument("--ended")

    p_kg_stats = sub.add_parser("kg-stats", help="KG statistics")

    p_record = sub.add_parser("record", help="Write record entry")
    p_record.add_argument("--agent", "-a", required=True)
    p_record.add_argument("--entry", "-e", required=True)
    p_record.add_argument("--topic", "-t", default="general")
    p_record.add_argument("--realm", "-w", default="")

    p_recordr = sub.add_parser("record-read", help="Read record entries")
    p_recordr.add_argument("--agent", "-a", required=True)
    p_recordr.add_argument("--last-n", type=int, default=10)

    p_record_ingest = sub.add_parser("record-ingest", help="Ingest daily summary files into the dimension")
    p_record_ingest.add_argument("--dir", required=True, help="Path to daily_summaries directory")
    p_record_ingest.add_argument("--realm", default="record")
    p_record_ingest.add_argument("--force", action="store_true")

    p_dedup = sub.add_parser("check-dup", help="Check for duplicate content")
    p_dedup.add_argument("content")
    p_dedup.add_argument("--threshold", type=float, default=0.9)

    p_rebuild = sub.add_parser("rebuild-fts", help="Rebuild FTS index")
    p_aaak = sub.add_parser("aaak", help="Compress text to AAAK")
    p_aaak.add_argument("text", nargs="?", default="")
    p_aaak.add_argument("--output-format", choices=["aaak", "json"],
                        default="aaak")

    p_mine = sub.add_parser("mine", help="Mine files into the dimension")
    p_mine.add_argument("file", help="File or directory to mine")
    p_mine.add_argument("--mode", choices=["projects", "convos", "extract"], default="projects",
                        help="Ingest mode: projects (default), convos, or extract")
    p_mine.add_argument("--realm", "-w", help="Realm/wing name")
    p_mine.add_argument("--domain", "-r", help="Domain name (projects mode)")
    p_mine.add_argument("--agent", default="alt-memory", help="Agent name (default: alt-memory)")
    p_mine.add_argument("--dry-run", action="store_true", help="Preview without filing")
    p_mine.add_argument("--no-gitignore", action="store_true",
                        help="Disable gitignore filtering when scanning directories")
    p_mine.add_argument("--redetect-origin", action="store_true",
                        help="Re-detect corpus origin even if already detected")
    p_mine.add_argument("--file-limit", type=int, default=None,
                        help="Maximum number of files to process")

    p_mcp = sub.add_parser("mcp", help="Run MCP server")
    p_mcp.add_argument("--host", default="127.0.0.1")
    p_mcp.add_argument("--port", type=int, default=8316)
    p_mcp.add_argument("--transport", choices=["stdio", "sse"], default="stdio")

    # -- Ported upstream commands --

    p_sweep = sub.add_parser("sweep", help="Sweep .jsonl files into dimension (message-granular mine)")
    p_sweep.add_argument("path", help="Path to .jsonl file or directory")

    p_sync = sub.add_parser("sync", help="Prune stale entities (gitignored/deleted sources)")
    p_sync.add_argument("--realm", "-w", help="Limit to realm")
    p_sync.add_argument("--project-dir", "-d", action="append", help="Project root directory")
    p_sync.add_argument("--dry-run", "-n", action="store_true", help="Preview only")
    p_sync.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run)")

    p_split = sub.add_parser("split", help="Split mega-files into per-session files")
    p_split.add_argument("--source", help="Source directory (default: ~/Desktop/transcripts)")
    p_split.add_argument("--output-dir", help="Output directory")
    p_split.add_argument("--file", help="Single file to split")
    p_split.add_argument("--min-sessions", type=int, default=2)
    p_split.add_argument("--dry-run", action="store_true")

    p_hook = sub.add_parser("hook", help="Run hook logic for Claude Code / Codex")
    hook_sub = p_hook.add_subparsers(dest="hook_command", required=True)
    p_hook_run = hook_sub.add_parser("run", help="Run a hook")
    p_hook_run.add_argument("--hook", required=True, choices=["session-start", "stop", "precompact"])
    p_hook_run.add_argument("--harness", required=True, choices=["claude-code", "codex"])

    p_instr = sub.add_parser("instructions", help="Output skill instructions")
    instr_sub = p_instr.add_subparsers(dest="instr_command", required=True)
    for instr_name in ("init", "search", "mine", "help", "status"):
        instr_sub.add_parser(instr_name, help=f"Instructions for {instr_name}")

    p_migrate = sub.add_parser("migrate", help="Schema migration and FAISS rebuild")
    p_migrate.add_argument("--dry-run", "-n", action="store_true",
                           help="Show pending migrations without applying")
    p_migrate.add_argument("--rebuild-faiss", "-f", action="store_true",
                           help="Rebuild FAISS index from SQLite data")
    p_migrate.add_argument("--status", "-s", action="store_true",
                           help="Show schema version and migration status")

    p_repair = sub.add_parser("repair", help="Repair utilities: integrity, VACUUM, FTS5 rebuild")
    p_repair.add_argument("--mode", choices=["status", "scan", "prune", "rebuild"], default=None,
                          help="Repair sub-mode")
    p_repair.add_argument("--integrity", action="store_true", help="Check SQLite integrity")
    p_repair.add_argument("--vacuum", action="store_true", help="Run VACUUM")
    p_repair.add_argument("--rebuild-fts", action="store_true", help="Rebuild FTS5 index")
    p_repair.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    p_repair_status = sub.add_parser("repair-status", help="Dimension health check")

    p_rebuild_sqlite = sub.add_parser("rebuild-from-sqlite", help="Rebuild FAISS from SQLite ground truth")

    p_wake = sub.add_parser("wake-up", help="Show L0+L1 wake-up context (dimension-scoped or agent records)")
    p_wake.add_argument("--agent", default=None, help="Agent name (omit for dimension-scoped wake-up)")
    p_wake.add_argument("--realm", "-w", default=None, help="Realm filter for dimension-scoped wake-up")
    p_wake.add_argument("--last-n", type=int, default=5, help="Entries per layer (agent mode only)")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    dim = Dimension(args.dimension)
    dim.init()

    try:
        if args.command == "init":
            dim_path = Path(args.dimension).expanduser().resolve()

            if args.dir:
                project_dir = str(Path(args.dir).expanduser().resolve())

                try:
                    from alt_memory.corpus_origin import detect_origin_heuristic
                    samples = []
                    proj = Path(project_dir)
                    if proj.is_dir():
                        from alt_memory.entity_detector import scan_for_detection
                        files = scan_for_detection(project_dir, max_files=30)
                        for fp in files:
                            try:
                                samples.append(fp.read_text("utf-8", "replace")[:100000])
                            except OSError:
                                continue
                    if samples:
                        origin = detect_origin_heuristic(samples)
                        if origin.likely_ai_dialogue:
                            platform = origin.primary_platform or "AI dialogue"
                            print(f"  Corpus origin: {platform}")
                except Exception:
                    pass

                try:
                    from alt_memory.project_scanner import discover_entities
                    detected = discover_entities(Path(project_dir))
                    total = (len(detected.get("people", [])) + len(detected.get("projects", [])))
                    if total > 0:
                        print(f"  Entities found: {total}")
                except Exception:
                    pass

                try:
                    from alt_memory.domain_detector_local import detect_domains_local
                    detect_domains_local(project_dir=project_dir, yes=True)
                except Exception:
                    pass

            print(f"Dimension initialized at {dim_path}")

        elif args.command == "status":
            s = dim.status()
            print(json.dumps(s, indent=2))

        elif args.command == "add":
            meta = json.loads(args.meta)
            did = dim.add_entity(args.realm, args.domain, args.content, meta, args.source)
            print(f"Added entity: {did}")

        elif args.command == "search":
            if not args.query:
                print("Enter search query:", file=sys.stderr)
                args.query = sys.stdin.read().strip()
            results = dim.search(args.query, n_results=args.limit,
                                     realm=args.realm, domain=args.domain, mode=args.mode)
            if not results:
                print("No results found.")
            else:
                for i, r in enumerate(results):
                    print(f"\n--- Result {i+1} (distance: {r.distance:.4f}) ---")
                    print(f"  ID:   {r.id}")
                    print(f"  Realm: {r.realm} / Domain: {r.domain}")
                    text = r.text[:300] + ("..." if len(r.text) > 300 else "")
                    print(f"  Text: {text}")

        elif args.command == "get":
            d = dim.get_entity(args.entity_id)
            if d:
                print(json.dumps(d, indent=2, default=str))
            else:
                print(f"Entity {args.entity_id} not found")

        elif args.command == "list":
            drawers = dim.list_entities(args.realm, args.domain, args.limit, args.offset)
            if not drawers:
                print("No entities found.")
            else:
                for d in drawers:
                    print(f"  {d['id']:20s} | {d['realm']:15s} / {d['domain']:20s} | {d['created_at']}")
                    print(f"  {d['content']}")
                    print()

        elif args.command == "realms":
            realms = dim.list_realms()
            if not realms:
                print("No realms.")
            else:
                for r in realms:
                    desc = f" - {r['description']}" if r['description'] and args.verbose else ""
                    print(f"  {r['name']:20s}  {r['entity_count']} entities{desc}")

        elif args.command == "rooms":
            rooms = dim.list_domains(args.realm)
            if not rooms:
                print("No domains.")
            else:
                for r in rooms:
                    print(f"  {r['realm']:15s} / {r['name']:20s}  {r['entity_count']} entities")

        elif args.command == "delete":
            ok = dim.delete_entity(args.entity_id)
            print("Deleted." if ok else "Not found.")

        elif args.command == "kg-add":
            fid = dim.kg.add(args.subject, args.predicate, args.object,
                                valid_from=args.valid_from, source=args.source)
            print(f"Added fact #{fid}")

        elif args.command == "kg-query":
            if args.all:
                facts = dim.kg.query(as_of=args.as_of)
            else:
                facts = dim.kg.query(entity=args.entity or None,
                                        predicate=args.predicate,
                                        as_of=args.as_of, direction=args.direction)
            if not facts:
                print("No facts found.")
            else:
                for f in facts:
                    valid = f" [{f['valid_from']} -> {f['valid_to'] or 'now'}]" if f['valid_from'] else ""
                    print(f"  {f['subject']} -- {f['predicate']} -- {f['object']}{valid}")

        elif args.command == "kg-invalidate":
            n = dim.kg.invalidate(args.subject, args.predicate, args.object, args.ended)
            print(f"Invalidated {n} fact(s).")

        elif args.command == "kg-stats":
            s = dim.kg.stats()
            print(json.dumps(s, indent=2))

        elif args.command == "record":
            wing = dim.record_write(args.agent, args.entry, args.topic, args.realm)
            print(f"Record entry written to realm: {wing}")

        elif args.command == "record-read":
            entries = dim.record_read(args.agent, args.last_n)
            if not entries:
                print("No record entries.")
            else:
                for e in entries:
                    print(f"\n[{e['created_at']}] topic={e['metadata'].get('topic', '?')}")
                    print(f"  {e['content'][:200]}")

        elif args.command == "record-ingest":
            from alt_memory.record_ingest import ingest_records
            result = ingest_records(args.dir, args.dimension, realm=args.realm, force=args.force)
            print(f"Record ingest: {result['days_updated']} days, {result['nodes_created']} nodes")

        elif args.command == "check-dup":
            dup = dim.check_duplicate(args.content, args.threshold)
            if dup:
                print(f"Similar content found (similarity={dup['similarity']:.4f}):")
                print(f"  ID:   {dup['id']}")
                print(f"  Text: {dup['text'][:200]}")
            else:
                print("No duplicate found.")

        elif args.command == "rebuild-fts":
            dim.rebuild_fts()
            print("FTS index rebuilt.")

        elif args.command == "aaak":
            from alt_memory.dialect import aaak_compress, aaak_parse_entry
            text = args.text if args.text else sys.stdin.read().strip()
            if args.output_format == "json":
                parsed = aaak_parse_entry(text)
                print(json.dumps(parsed, indent=2))
            else:
                compressed = aaak_compress(text)
                print(compressed)

        elif args.command == "mine":
            if args.mode == "convos":
                from alt_memory.convo_miner import mine_convos
                mine_convos(
                    convo_dir=args.file,
                    dim_path=args.dimension,
                    realm=args.realm,
                    agent=args.agent,
                    dry_run=args.dry_run,
                )
            elif args.mode == "extract":
                from alt_memory.format_miner import mine_formats
                mine_formats(
                    format_dir=args.file,
                    dim_path=args.dimension,
                    realm=args.realm,
                    agent=args.agent,
                    dry_run=args.dry_run,
                )
            else:
                filepath = Path(args.file)
                if filepath.is_dir():
                    from alt_memory.miner import batch_mine
                    result = batch_mine(
                        dim, str(filepath),
                        realm=args.realm or "files",
                        respect_gitignore=not args.no_gitignore,
                        file_limit=args.file_limit,
                    )
                    print(f"Mined {result['drawers_created']} items from {args.file}")
                else:
                    from alt_memory.miner import mine_file_into_dimension
                    count = mine_file_into_dimension(dim, args.file, args.realm, args.domain)
                    print(f"Mined {count} items from {args.file}")

        # ── Ported upstream command handlers ─────────────────────────

        elif args.command == "sweep":
            from alt_memory.sweeper import sweep, sweep_directory
            p = Path(args.path)
            if p.is_dir():
                result = sweep_directory(str(p), args.dimension)
                print(f"Swept directory: {result.get('files_succeeded', 0)} files, "
                      f"{result['total_added']} added, {result['total_skipped']} skipped")
            else:
                result = sweep(str(p), args.dimension, source_label=str(p))
                print(f"Swept {p.name}: {result['drawers_added']} added, "
                      f"{result['drawers_already_present']} existing, "
                      f"{result['drawers_skipped']} skipped")

        elif args.command == "sync":
            from alt_memory.sync import sync_dimension
            dry_run = not args.apply
            project_dirs = args.project_dir or None
            report = sync_dimension(
                args.dimension, project_dirs=project_dirs,
                realm=args.realm, dry_run=dry_run,
            )
            print(f"Sync {'(dry run)' if dry_run else ''}: "
                  f"{report['scanned']} scanned, "
                  f"{report['kept']} kept, "
                  f"{report['gitignored']} gitignored, "
                  f"{report['missing']} missing, "
                  f"{report['removed_entities']} removed")

        elif args.command == "split":
            from alt_memory.split_mega_files import split_file
            if args.file:
                written = split_file(args.file, args.output_dir, dry_run=args.dry_run)
                print(f"Split {args.file}: {len(written)} sessions")
            else:
                from alt_memory.split_mega_files import main as split_main
                # Rebuild argv for split_main's own argument parser
                old_argv = sys.argv
                split_argv = ["split"]
                if args.source:
                    split_argv.extend(["--source", args.source])
                if args.output_dir:
                    split_argv.extend(["--output-dir", args.output_dir])
                if args.dry_run:
                    split_argv.append("--dry-run")
                split_argv.extend(["--min-sessions", str(args.min_sessions)])
                sys.argv = split_argv
                try:
                    split_main()
                finally:
                    sys.argv = old_argv

        elif args.command == "hook":
            if args.hook_command == "run":
                from alt_memory.hooks_cli import run_hook
                run_hook(args.hook, args.harness)

        elif args.command == "instructions":
            from alt_memory.instructions_cli import run_instructions
            run_instructions(args.instr_command)

        elif args.command == "migrate":
            from alt_memory.migrate import migrate, rebuild_faiss, status as migrate_status
            base = str(Path(args.dimension).expanduser().resolve())

            if args.status:
                s = migrate_status(base)
                print(f"Dimension:   {s['path']}")
                print(f"Version:  {s['version']} (latest: {s['latest_version']})")
                print(f"Up to date: {s['up_to_date']}")
                print(f"Entities:  {s.get('entities', '?')}")
                print(f"Realms:    {s.get('realms', '?')}")
                print(f"Domains:    {s.get('domains', '?')}")
                print(f"Vectors:  {s.get('vectors', '?')}")
            elif args.rebuild_faiss:
                if args.dry_run:
                    print("DRY RUN: would rebuild FAISS index")
                else:
                    result = rebuild_faiss(base)
                    print(f"FAISS rebuild: {result['vectors_rebuilt']} vectors")
            else:
                result = migrate(base, dry_run=args.dry_run)
                if args.dry_run:
                    print(f"Pending migrations: {result['migrations_applied'] or 'none'}")
                else:
                    print(f"Version: {result['version_before']} -> {result['version_after']}")
                    for m in result['migrations_applied']:
                        print(f"  Applied: {m}")

        elif args.command == "repair":
            from alt_memory.repair_utils import (
                confirm_destructive_action, rebuild_fts5, run_vacuum,
                sqlite_integrity_errors,
            )
            db_path = str(Path(args.dimension).expanduser() / "dimension.db")

            if args.mode == "status":
                from alt_memory.repair import status as repair_status
                s = repair_status(args.dimension)
                print(json.dumps(s, indent=2))
            elif args.mode == "scan":
                from alt_memory.repair import scan_dimension
                good, bad = scan_dimension(args.dimension)
                print(f"Scan complete: {len(good)} good IDs, {len(bad)} bad IDs")
            elif args.mode == "prune":
                from alt_memory.repair import prune_corrupt
                prune_corrupt(args.dimension, confirm=args.yes)
            elif args.mode == "rebuild":
                from alt_memory.repair import rebuild_index
                n = rebuild_index(args.dimension)
                print(f"Rebuilt {n} vectors")
            else:
                if args.integrity:
                    errors = sqlite_integrity_errors(db_path)
                    if errors:
                        print(f"Integrity errors ({len(errors)}):")
                        for e in errors[:10]:
                            print(f"  - {e}")
                    else:
                        print("Integrity check passed.")
                if args.vacuum:
                    if confirm_destructive_action("VACUUM", db_path, assume_yes=args.yes):
                        run_vacuum(db_path)
                        print("VACUUM complete.")
                if args.rebuild_fts:
                    if confirm_destructive_action("FTS5 rebuild", db_path, assume_yes=args.yes):
                        rebuild_fts5(db_path, fts_table="entities_fts")
                        print("FTS5 index rebuilt.")
                if not any([args.integrity, args.vacuum, args.rebuild_fts]):
                    errors = sqlite_integrity_errors(db_path)
                    if errors:
                        print(f"Repair needed: {len(errors)} integrity issues")
                        for e in errors[:5]:
                            print(f"  - {e}")
                    else:
                        print("No repair needed — dimension is healthy.")

        elif args.command == "repair-status":
            from alt_memory.repair_utils import sqlite_entity_count, sqlite_integrity_errors
            db_path = str(Path(args.dimension).expanduser() / "dimension.db")
            count = sqlite_entity_count(db_path)
            errors = sqlite_integrity_errors(db_path)
            print(f"Dimension: {args.dimension}")
            print(f"Entities: {count or 0}")
            print(f"Integrity: {'PASS' if not errors else f'{len(errors)} issues'}")
            if errors:
                for e in errors[:5]:
                    print(f"  - {e}")

        elif args.command == "rebuild-from-sqlite":
            from alt_memory.repair import rebuild_from_sqlite
            n = rebuild_from_sqlite(args.dimension)
            print(f"Rebuilt {n} vectors from SQLite")

        elif args.command == "wake-up":
            from alt_memory.layers import MemoryStack
            stack = MemoryStack(dim)
            if args.agent:
                all_layers = stack.read_all(args.agent, last_n=args.last_n)
                for layer_num in sorted(all_layers):
                    layer_name = MemoryStack.LAYER_NAMES[layer_num]
                    entries = all_layers[layer_num]
                    print(f"\n--- {layer_name} ({len(entries)} entries) ---")
                    for e in entries:
                        print(f"  [{e['created_at']}] {e['content'][:200]}")
            else:
                text = stack.wake_up(realm=args.realm)
                tokens = len(text) // 4
                print(f"Wake-up text (~{tokens} tokens):")
                print("=" * 50)
                print(text)

        elif args.command == "mcp":
            from alt_memory.mcp_server import run_server
            run_server(dim, host=args.host, port=args.port, transport=args.transport)

    finally:
        dim.close()


if __name__ == "__main__":
    main()
