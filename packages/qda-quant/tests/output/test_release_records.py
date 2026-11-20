from qda_quant.output.release import release_files


def test_an_older_builds_splice_record_is_left_out(tmp_path):
    """splice.json names the base and donor checkpoints' local paths, so a release never carries it."""
    for name in ("config.json", "model.safetensors", "build.json", "splice.json"):
        (tmp_path / name).write_text("{}")
    names = {p.name for p in release_files(tmp_path)}
    assert "splice.json" not in names and "build.json" not in names
    assert {"config.json", "model.safetensors"} <= names
