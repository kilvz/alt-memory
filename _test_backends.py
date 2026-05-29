import atexit
import shutil
import tempfile
from pathlib import Path

from alt_memory.dimension import Dimension
import numpy as np

_tmpdirs = []

@atexit.register
def _cleanup():
    for d in _tmpdirs:
        shutil.rmtree(d, ignore_errors=True)

for backend in ("faiss", "chroma"):
    tmp = tempfile.mkdtemp(prefix="_test_" + backend + "_")
    _tmpdirs.append(tmp)
    d = Dimension(path=tmp, backend=backend)
    d.init()

    eid = d.add_entity(realm="test", domain="demo", content="hello world", metadata={"key": "val"})
    print(f"{backend}: added {eid}")

    emb = d._embedder.embed(["hello world"])
    ids, texts, dists, metas = d._store.search(emb, n_results=5)
    print(f"{backend}: search found {len(ids)} results: {ids}")

    gids, gdocs, gmetas = d._store.get(ids=[eid])
    print(f"{backend}: get by id: {gids[0] if gids else 'not found'}")

    print(f"{backend}: count = {d._store.count()}")

    d.close()
    print(f"{backend}: PASS")

print("ALL OK")
