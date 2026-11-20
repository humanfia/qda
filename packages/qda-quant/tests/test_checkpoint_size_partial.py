"""A partly downloaded source is sized like a remote one, not read shard by shard."""

import json

from qda_quant.engine.checkpoint_size import _local_names


def test_a_snapshot_missing_an_indexed_shard_is_not_local(tmp_path):
    """An index naming two shards with one on disk: no header is read, the remote estimate is used."""
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 8)
    weight_map = {
        "a.weight": "model-00001-of-00002.safetensors",
        "b.weight": "model-00002-of-00002.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    assert _local_names(str(tmp_path), meta=None) is None


def test_a_directory_without_shards_is_not_local(tmp_path):
    """No safetensors at all is the remote case as before."""
    (tmp_path / "config.json").write_text("{}")
    assert _local_names(str(tmp_path), meta=None) is None
