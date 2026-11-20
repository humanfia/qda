"""Calibration sources: chosen by key in a recipe, loaded through the ``qda.calib`` entry points."""

import json
import sys
import types
from pathlib import Path

import pytest
import yaml
from qda_quant import registry
from qda_quant.calib import sources
from qda_quant.calib.sources import FileSource, HFSource, MixSource, parse_source
from qda_quant.engine.calib import _tokenize, load_samples
from qda_quant.exit import ENVIRONMENT
from qda_quant.recipe import Recipe
from qda_quant.registry import Registry
from qda_quant.step import Refused

from ..engine_util import save_tiny_tokenizer
from ..goldens import check_golden, refusal

REPO = Path(__file__).resolve().parents[4]
SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def tok_dir(tmp_path):
    """A model directory holding only the tiny tokenizer; returns (dir, fingerprint)."""
    sha = save_tiny_tokenizer(tmp_path / "m")
    return tmp_path / "m", sha


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_recipe_names_its_source_by_key_and_round_trips():
    r = Recipe.from_yaml("steps: []\ncalib: {file: {path: c.jsonl, samples: 64}}\n")
    assert isinstance(r.calib, FileSource) and r.calib.samples == 64
    # the record fills in every default, so it names the run exactly
    assert r.to_dict()["calib"] == {
        "file": {
            "path": "c.jsonl",
            "samples": 64,
            "seqlen": 2048,
            "seed": 0,
            "chunks": False,
            "contamination_index": None,
        }
    }
    assert Recipe.from_yaml(r.to_yaml()).to_dict() == r.to_dict()


def test_overrides_reach_the_source_by_its_key():
    r = Recipe.from_dict(
        {"steps": [], "calib": {"file": {"path": "c.jsonl"}}}, overrides=["calib.file.samples=3"]
    )
    assert r.calib.samples == 3 and r.calib.path == "c.jsonl"


@pytest.mark.parametrize(
    "path",
    sorted([*(REPO / "bench").glob("*.yaml"), *(REPO / "configs").glob("*.yaml")]),
    ids=lambda p: p.name,
)
def test_shipped_configs_parse(path):
    from qda_quant.config import BuildConfig

    BuildConfig.from_file(path, ["model=M", "out=O"])  # benches leave the model and out to --set


# ---------------------------------------------------------------- a third-party source, installed for real


PLUGIN = """
from qda_quant.calib.sources import CalibSource, Rows


class Shout(CalibSource):
    name = "shout"
    text: str

    def read(self, model_dir):
        return Rows([{"text": w} for w in self.text.split()], None)


def register():
    from qda_quant.calib.sources import SOURCES
    SOURCES.register("shout", Shout)
"""


def test_a_third_party_source_loads_through_its_entry_point(tmp_path, monkeypatch, tok_dir):
    # a distribution on sys.path whose metadata declares `shout` in qda.calib, as `pip install` would leave it
    dist = tmp_path / "site" / "qda_shout-1.2.3.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: qda-shout\nVersion: 1.2.3\n")
    (dist / "entry_points.txt").write_text("[qda.plugins]\nshout = qda_shout:register\n")
    (tmp_path / "site" / "qda_shout.py").write_text(PLUGIN)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    fresh = Registry("calib")
    monkeypatch.setattr(sources, "SOURCES", fresh)
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    assert "shout" in fresh and fresh["shout"].__module__ == "qda_shout"
    assert fresh.provenance("shout") == ("qda-shout", "1.2.3")
    r = Recipe.from_dict(
        {"steps": [], "calib": {"shout": {"text": "w3 w4 w5 w6", "samples": 2, "seqlen": 2}}}
    )
    assert type(r.calib).__name__ == "Shout"
    got = load_samples(r.calib, tok_dir[0])
    assert [s.ids.tolist() for s in got] == [[3, 4], [5, 6]]


# ---------------------------------------------------------------- samples: (ids, prompt_len)


def test_text_and_ids_rows_score_from_the_first_position(tmp_path, tok_dir):
    m, sha = tok_dir
    text = FileSource(
        path=jsonl(tmp_path / "t.jsonl", [{"text": "w3 w4 w5"}, {"text": "w5 w6"}]), samples=2, seqlen=2
    )
    assert [(s.ids.tolist(), s.prompt_len) for s in load_samples(text, m)] == [([3, 4], 1), ([5, 6], 1)]
    ids = FileSource(
        path=jsonl(tmp_path / "i.jsonl", [{"tokenizer": sha}, {"input_ids": [7, 8, 9]}]), samples=1
    )
    assert [(s.ids.tolist(), s.prompt_len) for s in load_samples(ids, m)] == [([7, 8, 9], 1)]


def test_traces_keep_their_prompt_length(tmp_path, tok_dir):
    m, sha = tok_dir
    rows = [
        {"tokenizer": sha},
        {"prompt_ids": [1, 2, 3], "completion_ids": [4, 5]},
        {
            "prompt_ids": [1, 2, 3, 4],
            "completion_ids": [5, 6, 7],
        },  # cut to 5 tokens: one completion token left
        {"prompt_ids": [1, 2, 3, 4, 5], "completion_ids": [6]},  # nothing left after the cut: no sample
    ]
    c = FileSource(path=jsonl(tmp_path / "tr.jsonl", rows), samples=2, seqlen=5)
    got = [(s.ids.tolist(), s.prompt_len) for s in load_samples(c, m)]
    assert got == [([1, 2, 3, 4, 5], 3), ([1, 2, 3, 4, 5], 4)]


# ---------------------------------------------------------------- mix and hf


def test_mix_source_reads_what_mix_writes(tmp_path, tok_dir):
    from qda_quant.calib.mix import mix

    a = jsonl(tmp_path / "a.jsonl", [{"text": f"w{i} w{i + 1}"} for i in range(3, 20)])
    b = jsonl(tmp_path / "b.jsonl", [{"text": f"w{i}"} for i in range(30, 50)])
    src = parse_source(
        {
            "mix": {
                "parts": [{"path": a, "weight": "3"}, {"path": b}],
                "rows": 8,
                "out": str(tmp_path / "m.jsonl"),
            }
        }
    )
    assert isinstance(src, MixSource)
    want = mix([(a, 3), (b, 1)], 8, out=tmp_path / "ref.jsonl").read_text()
    assert [json.dumps(r) for r in src.read(tok_dir[0]).rows] == [
        json.dumps(json.loads(x)) for x in want.splitlines()
    ]


def test_slice_source_reads_one_named_slice_and_records_its_sha256(tmp_path, tok_dir):
    from qda_quant.calib import sha256_file
    from qda_quant.calib.sources import SliceSource

    pool = tmp_path / "pool"
    pool.mkdir()
    a = jsonl(pool / "if.jsonl", [{"text": f"w{i} w{i + 1}"} for i in range(3, 19, 2)])
    jsonl(pool / "chat.jsonl", [{"text": f"w{i} w{i + 1}"} for i in range(20, 36, 2)])
    r = Recipe.from_dict(
        {"steps": [], "calib": {"slice": {"pool": str(pool), "names": "if", "samples": 2, "seqlen": 2}}}
    )
    assert isinstance(r.calib, SliceSource)
    assert [s.ids.tolist() for s in load_samples(r.calib, tok_dir[0])] == [[3, 4], [5, 6]]
    assert r.to_dict()["calib"]["slice"]["files"] == []  # until plan time resolves the file
    r.calib.pin()
    assert r.to_dict()["calib"]["slice"]["files"] == [{"name": "if", "sha256": sha256_file(a)}]
    assert Recipe.from_yaml(r.to_yaml()).to_dict() == r.to_dict()  # the record reads back
    jsonl(Path(a), [{"text": "w9 w10"}])  # the pool file changed after plan time
    with pytest.raises(Refused, match="calib slice if: .* changed since the recipe pinned it"):
        load_samples(r.calib, tok_dir[0])


def test_slice_source_mixes_a_weighted_list_by_samples(tmp_path, tok_dir):
    pool = tmp_path / "pool"
    pool.mkdir()
    jsonl(pool / "a.jsonl", [{"text": f"w{i} w{i + 1}"} for i in range(3, 19, 2)])
    jsonl(pool / "b.jsonl", [{"text": f"w{i} w{i + 1}"} for i in range(20, 36, 2)])
    src = parse_source({"slice": {"pool": str(pool), "names": {"a": 3, "b": 1}, "samples": 4, "seqlen": 2}})
    assert sorted(s.ids.tolist() for s in load_samples(src, tok_dir[0])) == [
        [3, 4],
        [5, 6],
        [7, 8],
        [20, 21],
    ]


CHATS = [
    {"messages": [{"role": "user", "content": "w3 w4"}, {"role": "assistant", "content": "w5"}]},
    {"messages": [{"role": "user", "content": "w6"}]},
    {"messages": [{"role": "user", "content": "w7"}]},
]


@pytest.fixture
def downloaded(tmp_path, monkeypatch):
    """A dataset repository as ``hf download`` leaves it, and no network: no hub call, no ``datasets``."""
    import huggingface_hub

    monkeypatch.setitem(sys.modules, "datasets", None)  # importing it fails, as on a machine without it
    root = tmp_path / "snapshot"
    calls = []

    def snapshot_download(repo_id, *, repo_type, revision, local_files_only=False, allow_patterns=None):
        calls.append({"repo_id": repo_id, "repo_type": repo_type, "revision": revision})
        if not local_files_only:
            pytest.fail("downloaded a dataset that is already on disk")
        return str(root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return root, calls


def test_hf_source_reads_the_downloaded_files_and_records_its_revision(downloaded, tok_dir):
    """The files `hf download` wrote are the whole input: no `datasets`, no hub call, no network."""
    root, calls = downloaded
    (root / "data").mkdir(parents=True)
    jsonl(root / "data" / "train_sft-00000-of-00002.jsonl", CHATS[:2])
    jsonl(root / "data" / "train_sft-00001-of-00002.jsonl", CHATS[2:])
    jsonl(
        root / "data" / "test_sft-00000-of-00001.jsonl", [{"messages": [{"role": "user", "content": "w9"}]}]
    )
    r = Recipe.from_dict(
        {
            "steps": [],
            "calib": {"hf": {"dataset": "org/chat", "revision": SHA, "split": "train_sft", "rows": 3}},
        }
    )
    assert isinstance(r.calib, HFSource)
    got = r.calib.read(tok_dir[0])
    assert got.rows == CHATS  # the split's shards in order, the other split untouched
    assert calls == [{"repo_id": "org/chat", "repo_type": "dataset", "revision": SHA}]
    assert r.calib.model_copy(update={"rows": 2}).read(tok_dir[0]).rows == CHATS[:2]  # stops in shard 0
    record = yaml.safe_load(r.to_yaml())["calib"]["hf"]
    assert record["dataset"] == "org/chat" and record["revision"] == SHA
    # the engine renders the rows with the model's chat template, as it does for a file
    assert [
        s.ids.tolist()
        for s in load_samples(r.calib.model_copy(update={"samples": 1, "seqlen": 4}), tok_dir[0])
    ] == [[1, 3, 4, 2]]


def test_a_snapshot_without_the_split_reads_the_hub(tmp_path, monkeypatch, tok_dir):
    """`datasets.load_dataset` leaves only README.md in the hub cache; the rows then come from the hub."""
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "README.md").write_text("card")
    hub = tmp_path / "hub"
    (hub / "data").mkdir(parents=True)
    jsonl(hub / "data" / "train-00000-of-00001.jsonl", CHATS)
    _snapshot(monkeypatch, lambda *a, **k: str(snapshot))
    _hub(monkeypatch, lambda *a, **k: ["README.md", "data/train-00000-of-00001.jsonl"])
    fetched = []

    def download(repo_id, filename, *, repo_type, revision):
        fetched.append(filename)
        return str(hub / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    got = HFSource(dataset="org/chat", revision=SHA, split="train", column="messages").read(tok_dir[0])
    assert got.rows == CHATS
    assert fetched == ["data/train-00000-of-00001.jsonl"]


def test_a_snapshot_without_the_split_offline_names_the_download(tmp_path, monkeypatch):
    """No hub to fall back on: the refusal lists what the snapshot holds and says to download the dataset."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "README.md").write_text("card")
    _snapshot(monkeypatch, lambda *a, **k: str(snapshot))
    _hub(monkeypatch, _unreachable)
    with pytest.raises(Refused, match=r"no data file of split 'train' among \['README.md'\]") as refused:
        HFSource(dataset="org/chat", revision=SHA, split="train").read(tmp_path)
    assert refused.value.fix == f"hf download org/chat --repo-type dataset --revision {SHA}"


def test_hf_source_reads_one_subsets_parquet_shards(downloaded, tok_dir):
    """A dataset with configurations: only the files under the named subset's directory are read."""
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    root, _ = downloaded
    for subset, texts in (("ARC-Challenge", ["w3 w4", "w5"]), ("ARC-Easy", ["w9"])):
        (root / subset).mkdir(parents=True)
        pq.write_table(pa.table({"q": texts}), root / subset / "train-00000-of-00001.parquet")
    src = parse_source(
        {"hf": {"dataset": "org/arc", "revision": SHA, "subset": "ARC-Challenge", "column": "q"}}
    )
    assert src.read(tok_dir[0]).rows == [{"text": "w3 w4"}, {"text": "w5"}]


def test_hf_source_reads_a_repository_whose_names_say_no_split(downloaded, tok_dir):
    """One file, named after the dataset rather than a split (Dolly's layout): it is the train split."""
    root, _ = downloaded
    root.mkdir(parents=True)
    jsonl(root / "org-chat.jsonl", CHATS)
    assert parse_source({"hf": {"dataset": "org/chat", "revision": SHA}}).read(tok_dir[0]).rows == CHATS


class _ServingTemplate:
    """Stands in for a serving validator (mistral_common is not a dev dependency): refuses an assistant-last chat."""

    def __init__(self, tok):
        self.tok = tok

    def apply_chat_template(self, messages, tokenize=False, **kw):
        if messages[-1]["role"] == "assistant" and not kw.get("continue_final_message"):
            raise ValueError("Expected last role User or Tool for serving but got assistant")
        return self.tok.apply_chat_template(messages, tokenize=tokenize, **kw)


def test_an_assistant_last_chat_is_rendered_as_the_templates_continuation(tok_dir):
    """A row ending with assistant is training data: retry it as the template's continuation form."""
    from transformers import AutoTokenizer

    m, _ = tok_dir
    msgs = [{"role": "user", "content": "w3"}, {"role": "assistant", "content": "w4"}]
    tok = _ServingTemplate(AutoTokenizer.from_pretrained(str(m)))
    # the tokenizer's own ids: encoding the continuation text would turn control tags into plain text
    assert _tokenize({"messages": msgs}, tok) == tok.tok.apply_chat_template(
        msgs, tokenize=True, continue_final_message=True, return_dict=False
    )


def test_hf_source_maps_sharegpt_turns_to_chat_messages(downloaded, tok_dir):
    """A ShareGPT column (``conversations`` of ``{from, value}`` turns) reads as chat ``messages`` rows."""
    rows = [
        {"conversations": [{"from": "system", "value": "w7"}, {"from": "human", "value": "w3 w4"},
                           {"from": "gpt", "value": "w5"}]},
        {"conversations": [{"from": "user", "value": "w6"}, {"from": "assistant", "value": "w5"}]},
    ]  # fmt: skip
    root, _ = downloaded
    root.mkdir(parents=True)
    jsonl(root / "train.jsonl", rows)
    got = HFSource(dataset="org/chat", revision=SHA, column="conversations").read(tok_dir[0])
    assert got.rows == [
        {"messages": [{"role": "system", "content": "w7"}, {"role": "user", "content": "w3 w4"},
                      {"role": "assistant", "content": "w5"}]},
        {"messages": [{"role": "user", "content": "w6"}, {"role": "assistant", "content": "w5"}]},
    ]  # fmt: skip


def test_hf_a_branch_revision_resolves_to_its_commit(monkeypatch):
    import huggingface_hub

    asked = []

    class FakeApi:
        def dataset_info(self, repo, revision):
            asked.append((repo, revision))
            return types.SimpleNamespace(sha=SHA)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    src = parse_source({"hf": {"dataset": "org/chat", "revision": "v1.0"}})
    src.pin()
    assert asked == [("org/chat", "v1.0")]
    assert src.revision == SHA and src.to_dict()["hf"]["revision"] == SHA
    src.pin()  # a commit hash is left as it is, with no further hub call
    assert asked == [("org/chat", "v1.0")]


def test_a_mix_of_sources_pins_its_hf_part(monkeypatch):
    import huggingface_hub

    class FakeApi:
        def dataset_info(self, repo, revision):
            return types.SimpleNamespace(sha=SHA)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    src = parse_source({"mix": {"parts": [{"hf": {"dataset": "org/chat", "revision": "main"}, "weight": 1}]}})
    src.pin()
    assert src.to_dict()["mix"]["parts"][0]["hf"]["revision"] == SHA


def test_plan_pins_an_hf_branch_in_the_recipe(tmp_path, monkeypatch):
    """A branch is resolved while planning, so the recipe a build records names the commit."""
    import huggingface_hub
    import torch
    from qda_quant.engine.planning import compile_plan
    from qda_quant.format import Format
    from qda_quant.target import Target

    from ..engine_util import tiny_qwen3

    tiny_qwen3(layers=1, hidden=128, inter=256).save_pretrained(tmp_path / "m")

    class FakeApi:
        def dataset_info(self, repo, revision):
            return types.SimpleNamespace(sha=SHA)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    recipe = Recipe.from_dict(
        {
            "steps": [{"rtn": {}}],
            "calib": {"hf": {"dataset": "org/chat", "revision": "main", "samples": 1, "seqlen": 4}},
        }
    )
    compile_plan(str(tmp_path / "m"), recipe, Format.parse("w4a16"), Target(), [torch.device("cpu")])
    assert recipe.to_dict(full=True)["calib"]["hf"]["revision"] == SHA


def _calib_modules() -> list[str]:
    import pkgutil

    import qda_quant.calib

    return sorted(m.name for m in pkgutil.iter_modules(qda_quant.calib.__path__))


@pytest.mark.parametrize("module", _calib_modules())
def test_each_calib_module_imports_first_in_a_fresh_interpreter(module):
    """No calib module may need another one imported before it (the sources <-> source_files cycle).

    The facade re-exports the source classes from their modules; a module that imported a name back from the
    half-initialized facade failed whenever it was imported first (an agent's lazy import, the packaging
    test).
    """
    import subprocess

    code = (
        f"import qda_quant.calib.{module} as m; import qda_quant.calib.sources as s; assert s.SOURCES['file']"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


# ---------------------------------------------------------------- refusal wording: golden/errors_calib_sources.txt


def _snapshot(mp, download) -> None:
    import huggingface_hub

    mp.setattr(huggingface_hub, "snapshot_download", download)


def _hf(t, mp, files, **settings):
    """Read an `hf` source off a downloaded repository holding ``files`` (name -> rows, or raw bytes)."""
    root = t / "ds"
    root.mkdir(exist_ok=True)
    for name, rows in files.items():
        (root / name).write_bytes(rows) if isinstance(rows, bytes) else jsonl(root / name, rows)
    _snapshot(mp, lambda *a, **k: str(root))
    return HFSource(dataset="org/chat", revision=SHA, **settings).read(t)


def _hub(mp, answer) -> None:
    """Stand in for the hub API: both the revision lookup and the dataset's file list answer ``answer``."""
    import huggingface_hub

    mp.setattr(
        huggingface_hub, "HfApi", lambda: types.SimpleNamespace(dataset_info=answer, list_repo_files=answer)
    )


def _unreachable(*a, **k):
    import httpx2  # the installed hub client's transport errors are not OSErrors

    raise httpx2.ConnectError("Couldn't reach 'org/chat' on the Hub")


def _offline(*a, **k):
    raise ConnectionError("Couldn't reach 'org/chat' on the Hub (OfflineModeIsEnabled)")


def _file(t, m, rows):
    return load_samples(FileSource(path=jsonl(t / "r.jsonl", rows), samples=1), m)


def _mix(t, m, **mix):
    return parse_source({"mix": {"rows": 2, "out": str(t / "m.jsonl"), **mix}}).read(m)


def _slice(t, names):
    (t / "pool").mkdir(exist_ok=True)
    jsonl(t / "pool" / "a.jsonl", [{"text": "w3 w4"}])
    return parse_source({"slice": {"pool": str(t / "pool"), "names": names}})


TRACE = {"prompt_ids": [1], "completion_ids": [2, 3]}
HF = {"dataset": "org/chat", "revision": "main"}

# name -> render(tmp_path, model_dir, sha, monkeypatch); each one must refuse
REFUSALS = {
    "override names the source key": lambda t, m, sha, mp: Recipe.from_dict(
        {"steps": [], "calib": {"file": {"path": "c.jsonl"}}}, overrides=["calib.samples=3"]
    ),
    "unknown source setting": lambda t, m, sha, mp: Recipe.from_dict(
        {"steps": [], "calib": {"file": {"path": "c.jsonl", "sample": 3}}}
    ),
    "strict source settings": lambda t, m, sha, mp: parse_source({"file": {"path": "c.jsonl", "sample": 4}}),
    "old calib shape": lambda t, m, sha, mp: parse_source({"path": "c.jsonl"}),
    "two sources": lambda t, m, sha, mp: parse_source({"file": {"path": "a"}, "mix": {}}),
    "unknown source": lambda t, m, sha, mp: parse_source({"nope": {}}),
    "missing calib file": lambda t, m, sha, mp: FileSource(path=str(t / "default-calib.jsonl")).read(t),
    "trace without fingerprint": lambda t, m, sha, mp: _file(t, m, [TRACE]),
    "ids mixed with traces": lambda t, m, sha, mp: _file(
        t, m, [{"tokenizer": sha}, TRACE, {"input_ids": [1, 2]}]
    ),
    "ids mixed with text": lambda t, m, sha, mp: _file(t, m, [TRACE, {"text": "w3"}]),
    "mix part missing": lambda t, m, sha, mp: _mix(t, m, parts=[{"path": str(t / "missing.jsonl")}]),
    "mix heldout missing": lambda t, m, sha, mp: _mix(
        t,
        m,
        parts=[{"path": jsonl(t / "a.jsonl", [{"text": "w3 w4"}] * 4)}],
        heldout=str(t / "missing.jsonl"),
    ),
    "slice unknown name": lambda t, m, sha, mp: _slice(t, "nope").read(t),
    "slice weight not positive": lambda t, m, sha, mp: _slice(t, {"a": 0}),
    # path containment: a slice name never leaves its pool
    "slice name with a path": lambda t, m, sha, mp: _slice(t, "../a"),
    "hf without revision": lambda t, m, sha, mp: parse_source({"hf": {"dataset": "org/chat"}}),
    "hf without pyarrow": lambda t, m, sha, mp: (
        mp.setitem(sys.modules, "pyarrow.parquet", None),  # the import fails as if it were not installed
        _hf(t, mp, {"train.parquet": b""}),
    ),
    "hf split without a file": lambda t, m, sha, mp: (
        _hub(mp, _unreachable),  # offline: the snapshot is all there is
        _hf(t, mp, {"train.jsonl": [{"messages": []}]}, split="val"),
    )[1],
    "hf column missing": lambda t, m, sha, mp: _hf(t, mp, {"train.jsonl": [{"other": "w3"}]}),
    "hf unknown column shape": lambda t, m, sha, mp: _hf(t, mp, {"train.jsonl": [{"messages": 3}]}),
    "hf unknown sharegpt speaker": lambda t, m, sha, mp: _hf(
        t,
        mp,
        {"train.jsonl": [{"c": [{"from": "human", "value": "w3"}, {"from": "bot", "value": "w4"}]}]},
        column="c",
    ),
    "hf offline": lambda t, m, sha, mp: (
        _snapshot(mp, _offline),
        _hub(mp, _offline),
        HFSource(dataset="org/chat", revision=SHA).read(m),
    ),
    "hf revision unreachable": lambda t, m, sha, mp: (_hub(mp, _unreachable), parse_source({"hf": HF}).pin()),
    "hf revision without sha": lambda t, m, sha, mp: (
        _hub(mp, lambda *a, **k: types.SimpleNamespace(sha=None)),
        parse_source({"hf": HF}).pin(),
    ),
    "hf revision that is a branch": lambda t, m, sha, mp: (
        _hub(mp, lambda *a, **k: types.SimpleNamespace(sha="main")),
        parse_source({"hf": HF}).pin(),
    ),
}


def test_a_calibration_source_that_needs_a_package_is_the_machine(tmp_path, tok_dir, monkeypatch):
    """A package this machine does not have is status 3: the fix is installing it, not another command."""
    m, sha = tok_dir
    with pytest.raises(Refused) as refused:
        REFUSALS["hf without pyarrow"](tmp_path, m, sha, monkeypatch)
    assert refused.value.code == ENVIRONMENT


@pytest.mark.parametrize("name", list(REFUSALS))
def test_refusals_keep_their_wording(tmp_path, tok_dir, monkeypatch, name):
    """Each calibration-source refusal fires, with the exact text in ``golden/errors_calib_sources.txt``."""
    m, sha = tok_dir
    text = refusal(lambda: REFUSALS[name](tmp_path, m, sha, monkeypatch), ValueError)
    check_golden("calib_sources", name, text.replace(str(tmp_path), "<tmp>"), list(REFUSALS))
