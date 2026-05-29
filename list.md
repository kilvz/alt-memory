# Upstream mempalace comparison
> Scanning F:\mempalace\mempalace\ vs F:\alt-memory\alt_memory\
> Started: 2026-05-30

## Files scanned

### `__init__.py` ✅ DONE
- **Upstream**: `_strip_leaked_pythonpath_from_sys_path()` guard, chromadb telemetry suppression, ONNX/macOS comment. Exports only `__version__`.
- **Alt-memory**: Rich `__all__` with full public API. Direct `__version__` string. No PYTHONPATH cleaning (FAISS-based, no ChromaDB dependency for base install). No telemetry suppression.
- **Gap**: None — architectural difference. Alt-memory's design doesn't need PYTHONPATH cleaning since it doesn't bundle chromadb with conflicting ABIs.

### `__main__.py` ✅ DONE
- Identical. No gap.

### `_stdio.py` ✅ DONE
- Same function. Alt-memory docstring is shorter (no per-stream policy explanation).
- **Gap**: Minor — docstring could be expanded with the per-stream rationale, but functionally identical.

### `cli.py` ✅ DONE
- **Upstream**: Pipeline-centric CLI (`init`, `mine`, `sweep`, `sync`, `search`, `wake-up`, `split`, `migrate`, `status`, `repair-status`, `repair`, `hook`, `instructions`, `mcp`, `compress`). Features corpus-origin detection (Pass 0), entity discovery with LLM (Pass 1), room detection (Pass 2), post-init mine prompt, AAAK compress command.
- **Alt-memory**: Entity-centric CLI (`init`, `status`, `add`, `search`, `get`, `list`, `realms`, `rooms`, `delete`, `kg-add`, `kg-query`, `kg-invalidate`, `kg-stats`, `record`, `record-read`, `check-dup`). Completely different design — focused on entity/KG operations, not pipeline stages.
- **Gap**: Architectural difference — not a porting target. Neither is "missing" features the other has; they serve different workflows.

### `version.py` ✅ DONE
- Identical pattern (`__version__` = "3.3.6" upstream vs "4.2.0" alt-memory). No gap.

### `config.py` ✅ DONE
- **Upstream**: `MempalaceConfig` — ChromaDB-specific settings, `hall_keywords`, `hallway_file`, collection naming, palace paths.
- **Alt-memory**: `AltMemoryConfig` — FAISS-specific settings, backend selection (FAISS/Chroma), embedder config, dimension paths. Already has `gate_keywords`, `gateway_file`, realm naming.
- **Gap**: Architectural — different backends. Alt-memory's config is adapted for FAISS + embedding models. No porting needed.

### `corpus_origin.py` ✅ DONE
- Esssentially identical. Both have `detect_origin_heuristic()`, `detect_origin_llm()`, `CorpusOriginResult`. Alt-memory has a minor unused import (`LLMProvider`). No gap.

### `embedding.py` (upstream) vs `backends/embedder.py` (alt-memory) ✅ DONE
- **Upstream**: Thin ChromaDB `EmbeddingFunction` factory (278 lines, 2 models: minilm, embeddinggemma)
- **Alt-memory**: Full standalone embedder engine (691 lines, 7 models: numpy, minilm, embeddinggemma, bert, numpy_bert, sentence, spacy). Thread-safe, pure-Python TF-IDF/SVD fallback, FAISS-native.
- **Gap**: Alt-memory is MORE comprehensive. No porting needed.

### All other files ✅ DONE (batch verified)
All upstream `.py` files exist in alt-memory (renamed where alt-memory terminology differs: `palace`→`dimension`, `closet`→`node`, `diary`→`record`, `hall`→`gate`, `hallway`→`gateway`). Sources/, instructions/, data/ directories match.

### Known remaining gap
- **Config-based room routing** (gap #2 from prior audit): upstream has per-project config domain overrides. Requires designing format — deferred.

