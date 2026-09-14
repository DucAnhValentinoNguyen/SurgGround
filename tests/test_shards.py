from surgground.data.shards import iter_shard, pack


def _items(n):
    return [{"id": f"it{i:03d}", "task": "t1", "target": {"spans": [[0, 1]]}} for i in range(n)]


def test_pack_roundtrip(tmp_path):
    items = _items(25)
    shards = pack(items, tmp_path, shard_size_mb=2000)
    assert len(shards) == 1
    back = iter_shard(shards[0])
    assert {it["id"] for it in back} == {it["id"] for it in items}
    manifest = (tmp_path / "manifest.json").exists()
    assert manifest


def test_pack_splits_across_shards_by_size(tmp_path):
    items = _items(200)
    # force many tiny shards: ~ a few hundred bytes/item
    shards = pack(items, tmp_path, shard_size_mb=0.002)
    assert len(shards) > 1
    total = sum(len(iter_shard(s)) for s in shards)
    assert total == len(items)
