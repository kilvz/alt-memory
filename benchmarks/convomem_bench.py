#!/usr/bin/env python3
"""
Alt Memory x ConvoMem Benchmark

Uses alt-memory's embedder + raw FAISS (in-memory, no disk I/O)
for retrieval recall measurement on ConvoMem (75K+ QA pairs, 6 categories).

Downloads evidence files from HuggingFace on first run.
"""

import os
import sys
import json
import time
import argparse
import urllib.request
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np


HF_BASE = "https://huggingface.co/datasets/Salesforce/ConvoMem/resolve/main/core_benchmark/evidence_questions"

CATEGORIES = {
    "user_evidence": "User Facts",
    "assistant_facts_evidence": "Assistant Facts",
    "changing_evidence": "Changing Facts",
    "abstention_evidence": "Abstention",
    "preference_evidence": "Preferences",
    "implicit_connection_evidence": "Implicit Connections",
}


def _make_alt_embedder():
    from alt_memory.backends.embedder import get_embedder
    return get_embedder(model="sentence")


def _compress(texts):
    from alt_memory.dialect import aaak_compress
    return [aaak_compress(t) for t in texts]


def download_evidence_file(category, subpath, cache_dir):
    url = f"{HF_BASE}/{category}/{subpath}"
    cache_path = os.path.join(cache_dir, category, subpath.replace("/", "_"))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    print(f"    Downloading: {category}/{subpath}...")
    try:
        urllib.request.urlretrieve(url, cache_path)
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"    Failed to download {url}: {e}")
        return None


def discover_files(category, cache_dir):
    api_url = f"https://huggingface.co/api/datasets/Salesforce/ConvoMem/tree/main/core_benchmark/evidence_questions/{category}/1_evidence"
    cache_path = os.path.join(cache_dir, f"{category}_filelist.json")

    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    try:
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            files = json.loads(resp.read())
            paths = [f["path"].split(f"{category}/")[1] for f in files if f["path"].endswith(".json")]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(paths, f)
            return paths
    except Exception as e:
        print(f"    Failed to list files for {category}: {e}")
        return []


def load_evidence_items(categories, limit, cache_dir):
    all_items = []
    for category in categories:
        files = discover_files(category, cache_dir)
        if not files:
            print(f"  Skipping {category} — no files found")
            continue
        items_for_cat = []
        for fpath in files:
            if len(items_for_cat) >= limit:
                break
            data = download_evidence_file(category, fpath, cache_dir)
            if data and "evidence_items" in data:
                for item in data["evidence_items"]:
                    item["_category_key"] = category
                    items_for_cat.append(item)
        all_items.extend(items_for_cat[:limit])
        print(f"  {CATEGORIES.get(category, category)}: {len(items_for_cat[:limit])} items loaded")
    return all_items


def retrieve_for_item(item, embedder, dim, top_k=10, mode="raw"):
    conversations = item.get("conversations", [])
    question = item["question"]
    evidence_messages = item.get("message_evidences", [])
    evidence_texts = set(e["text"].strip().lower() for e in evidence_messages)

    corpus = []
    corpus_speakers = []
    for conv in conversations:
        for msg in conv.get("messages", []):
            corpus.append(msg["text"])
            corpus_speakers.append(msg["speaker"])

    if not corpus:
        return 0.0, {"error": "empty corpus"}

    docs = _compress(corpus) if mode == "aaak" else corpus

    embeddings = embedder.embed(docs)
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    query_vec = embedder.embed([question])
    query_vec = np.asarray(query_vec, dtype=np.float32)
    faiss.normalize_L2(query_vec)
    _, indices_arr = index.search(query_vec, min(top_k, len(corpus)))
    retrieved_indices = indices_arr[0].tolist()
    retrieved_texts = [corpus[i].strip().lower() for i in retrieved_indices if i != -1]

    found = 0
    for ev_text in evidence_texts:
        for ret_text in retrieved_texts:
            if ev_text in ret_text or ret_text in ev_text:
                found += 1
                break

    recall = found / len(evidence_texts) if evidence_texts else 1.0
    return recall, {
        "retrieved_count": len(retrieved_indices),
        "evidence_count": len(evidence_texts),
        "found": found,
    }


def run_benchmark(categories, limit_per_cat, top_k, mode, cache_dir, out_file):
    print(f"\n{'=' * 60}")
    print("  Alt Memory x ConvoMem Benchmark")
    print(f"{'=' * 60}")
    print(f"  Categories:  {len(categories)}")
    print(f"  Limit/cat:   {limit_per_cat}")
    print(f"  Top-k:       {top_k}")
    print(f"  Mode:        {mode}")
    print("-" * 60)
    print("\n  Loading data from HuggingFace...\n")

    items = load_evidence_items(categories, limit_per_cat, cache_dir)
    print(f"\n  Total items: {len(items)}")
    print("-" * 60 + "\n")

    embedder = _make_alt_embedder()
    _ = embedder.embed(["warmup"])
    dim = embedder.dimension

    all_recall = []
    per_category = defaultdict(list)
    results_log = []
    start_time = datetime.now()

    for i, item in enumerate(items):
        question = item["question"]
        answer = item.get("answer", "")
        cat_key = item.get("_category_key", "unknown")
        recall, details = retrieve_for_item(item, embedder, dim, top_k=top_k, mode=mode)
        all_recall.append(recall)
        per_category[cat_key].append(recall)

        results_log.append({
            "question": question,
            "answer": answer,
            "category": cat_key,
            "recall": recall,
            "details": details,
        })

        status = "HIT" if recall >= 1.0 else ("part" if recall > 0 else "miss")
        if (i + 1) % 20 == 0 or i == len(items) - 1:
            print(f"  [{i + 1:4}/{len(items)}] avg_recall={sum(all_recall) / len(all_recall):.3f}  last={status}")

    elapsed = (datetime.now() - start_time).total_seconds()
    avg_recall = sum(all_recall) / len(all_recall) if all_recall else 0

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — Alt Memory ({mode} mode, top-{top_k})")
    print(f"{'=' * 60}")
    print(f"  Time:        {elapsed:.1f}s ({elapsed / max(len(items), 1):.2f}s per item)")
    print(f"  Items:       {len(items)}")
    print(f"  Avg Recall:  {avg_recall:.3f}")

    print("\n  PER-CATEGORY RECALL:")
    for cat_key in sorted(per_category.keys()):
        vals = per_category[cat_key]
        avg = sum(vals) / len(vals)
        name = CATEGORIES.get(cat_key, cat_key)
        perfect = sum(1 for v in vals if v >= 1.0)
        print(f"    {name:25} R={avg:.3f}  perfect={perfect}/{len(vals)}")

    perfect_total = sum(1 for r in all_recall if r >= 1.0)
    zero_total = sum(1 for r in all_recall if r == 0)
    print("\n  DISTRIBUTION:")
    print(f"    Perfect (1.0):  {perfect_total:4} ({perfect_total / len(all_recall) * 100:.1f}%)")
    print(f"    Zero (0.0):     {zero_total:4} ({zero_total / len(all_recall) * 100:.1f}%)")
    print(f"\n{'=' * 60}\n")

    if out_file:
        with open(out_file, "w") as f:
            json.dump(results_log, f, indent=2)
        print(f"  Results saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alt Memory x ConvoMem Benchmark")
    parser.add_argument("--limit", type=int, default=100, help="Items per category (default: 100)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k retrieval (default: 10)")
    parser.add_argument(
        "--category", choices=list(CATEGORIES.keys()) + ["all"], default="all",
        help="Category to test (default: all)",
    )
    parser.add_argument(
        "--mode", choices=["raw", "aaak"], default="raw",
        help="Retrieval mode",
    )
    parser.add_argument("--cache-dir", default="/tmp/convomem_cache", help="Cache directory")
    parser.add_argument("--out", default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.category == "all":
        categories = list(CATEGORIES.keys())
    else:
        categories = [args.category]

    if not args.out:
        args.out = f"benchmarks/results_alt_convomem_{args.mode}_top{args.top_k}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    run_benchmark(categories, args.limit, args.top_k, args.mode, args.cache_dir, args.out)
