"""Pack task items -> WebDataset-style ``.tar`` shards (PLAN.md 8.5, P2).

Each item is written as one ``<key>.json`` member; members are rolled into
``shard-000000.tar``, ``shard-000001.tar`` … at an approximate size bound. Frames
are stored as **references** (``item["frames"]["t_sec"]`` + ``$FRAMES_ROOT`` layout
from P1), never pixels, so shards stay tiny and frame tier / resolution stays a
runtime knob in ``collate.py``.

Reader-agnostic: a plain ``tarfile`` walk or ``webdataset.WebDataset(<glob>)`` both
work. ``manifest.json`` records the item count, the shard list, the per-shard
counts, and the provenance block copied from the items.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

_SIZE_SLACK = 0.98  # start a new shard slightly before the nominal limit


def _key(item: dict, i: int) -> str:
    k = str(item.get("id") or f"item{i:08d}")
    return k.replace("/", "_").replace(" ", "_")


def pack(items, out_dir, shard_size_mb: int = 2000) -> list[Path]:
    """Write ``items`` to ``out_dir`` as ``shard-*.tar`` + ``manifest.json``.
    Returns the list of shard paths (also created: ``out_dir/manifest.json``)."""
    items = list(items)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("shard-*.tar"):
        stale.unlink()

    limit = int(shard_size_mb * 1024 * 1024 * _SIZE_SLACK)
    shard_paths: list[Path] = []
    per_shard: list[int] = []

    tf: tarfile.TarFile | None = None
    cur_bytes = 0
    cur_count = 0

    def _open_new() -> tarfile.TarFile:
        p = out_dir / f"shard-{len(shard_paths):06d}.tar"
        shard_paths.append(p)
        per_shard.append(0)
        return tarfile.open(p, "w")

    for i, item in enumerate(items):
        blob = json.dumps(item).encode()
        if tf is None or (cur_bytes + len(blob) > limit and cur_count > 0):
            if tf is not None:
                tf.close()
                per_shard[-1] = cur_count
            tf = _open_new()
            cur_bytes = 0
            cur_count = 0
        info = tarfile.TarInfo(name=f"{_key(item, i)}.json")
        info.size = len(blob)
        tf.addfile(info, io.BytesIO(blob))
        cur_bytes += len(blob) + 512
        cur_count += 1
    if tf is not None:
        tf.close()
        per_shard[-1] = cur_count

    manifest = {
        "n_items": len(items),
        "n_shards": len(shard_paths),
        "shards": [p.name for p in shard_paths],
        "items_per_shard": per_shard,
        "shard_size_mb": shard_size_mb,
        "provenance": items[0].get("provenance") if items else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return shard_paths


def iter_shard(path) -> "list[dict]":
    """Read one ``.tar`` shard back into a list of item dicts (test / debug)."""
    out = []
    with tarfile.open(path, "r") as tf:
        for m in tf.getmembers():
            if m.name.endswith(".json"):
                out.append(json.loads(tf.extractfile(m).read().decode()))
    return out
