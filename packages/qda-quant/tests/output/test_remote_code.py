"""Exported auto-map targets retain their dependency closure without running source code."""

import json

import pytest
from qda_quant.output.remote_code import copy_remote_code
from qda_quant.step import Refused


def test_auto_map_closure_preserves_packages_and_does_not_execute(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "pkg").mkdir(parents=True)
    out.mkdir()
    files = {
        "processor.py": b"from .pkg import vision\nraise RuntimeError('do not execute')\n",
        "pkg/__init__.py": b"from .common import IMAGE_SIZE\n",
        "pkg/vision.py": b"from ..media import convert\nfrom .common import IMAGE_SIZE\n",
        "pkg/common.py": b"IMAGE_SIZE = 14\n",
        "media.py": b"from .processor import Processor\n",  # cycle
        "unreferenced.py": b"raise RuntimeError('unreferenced')\n",
    }
    for name, data in files.items():
        (src / name).write_bytes(data)
    config = {"text_config": {"auto_map": {"AutoProcessor": [None, "repo/name--processor.Processor"]}}}
    copy_remote_code(src, out, [config])
    expected = {k: v for k, v in files.items() if k != "unreferenced.py"}
    assert {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*.py")} == expected


@pytest.mark.parametrize("missing", ["processor.py", "vision.py"])
def test_missing_auto_map_target_or_import_refuses(tmp_path, missing):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (src / "processor.py").write_text("from .vision import ImageProcessor\n")
    (src / "vision.py").write_text("class ImageProcessor: pass\n")
    (src / missing).unlink()
    with pytest.raises(Refused, match=rf"required source module {missing} is missing"):
        copy_remote_code(src, out, [{"auto_map": {"AutoProcessor": "processor.Processor"}}])
    assert not list(out.iterdir())


@pytest.mark.parametrize(
    "asset", ["config.json", "preprocessor_config.json", "processor_config.json", "tokenizer_config.json"]
)
def test_writer_copies_auto_map_from_model_and_processor_assets(tmp_path, asset):
    from qda_quant import Target
    from qda_quant.arch import describe
    from qda_quant.output.export import Writer

    from ..tiny import model

    src = tmp_path / "src"
    m = model("qwen3")
    m.save_pretrained(src)
    cfg = json.loads((src / asset).read_text()) if (src / asset).exists() else {}
    cfg["auto_map"] = {"AutoProcessor": "processor.Processor", "AutoImageProcessor": "vision.Image"}
    (src / asset).write_text(json.dumps(cfg))
    files = {
        "processor.py": b"from .media import convert\n",
        "vision.py": b"from .media import convert\n",
        "media.py": b"def convert(x): return x\n",
    }
    for name, data in files.items():
        (src / name).write_bytes(data)
    writer = Writer(tmp_path / "out", src, describe(m, verify=False), "w4a16", Target())
    out = writer.finish(m.config, None, {})
    assert {p.name: p.read_bytes() for p in out.glob("*.py")} == files


def test_writer_drops_remote_model_code_of_a_type_transformers_implements(tmp_path):
    """A release that ships remote code for a native type (NemotronH) exports without it: QDA built the
    model with transformers' own classes, and a checkpoint carrying the `.py` is refused by the search
    judge. Another entry of the same `auto_map` (a processor's code) stays, with its module."""
    from qda_quant import Target
    from qda_quant.arch import describe
    from qda_quant.output.export import Writer

    from ..tiny import model

    src = tmp_path / "src"
    m = model("qwen3")
    m.save_pretrained(src)
    cfg = json.loads((src / "config.json").read_text())
    cfg["auto_map"] = {
        "AutoConfig": "configuration_q.QConfig",
        "AutoModelForCausalLM": "modeling_q.QForCausalLM",
        "AutoProcessor": "processor.Processor",
    }
    (src / "config.json").write_text(json.dumps(cfg))
    for name in ("configuration_q.py", "modeling_q.py", "processor.py"):
        (src / name).write_bytes(b"X = 1\n")
    out = Writer(tmp_path / "out", src, describe(m, verify=False), "w4a16", Target()).finish(
        m.config, None, {}
    )
    assert json.loads((out / "config.json").read_text())["auto_map"] == {
        "AutoProcessor": "processor.Processor"
    }
    assert sorted(p.name for p in out.glob("*.py")) == ["processor.py"]
