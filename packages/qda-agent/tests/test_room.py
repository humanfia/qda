"""`qda_agent.room`: one statvfs per directory, and the cache and temp disk lines doctor and prepare share."""

import os
import tempfile
from pathlib import Path

import pytest
from qda_agent import room as R
from qda_agent.room import Place, cache_room, existing, places, room, room_notes, room_report


def test_a_path_not_made_yet_is_measured_where_it_will_land(tmp_path: Path) -> None:
    later = tmp_path / "run" / "judge" / "frozen"
    assert existing(later) == tmp_path
    r = room(later)
    assert r.path == tmp_path
    assert r.free_bytes > 0 and r.total_bytes >= r.free_bytes


def test_an_existing_directory_is_read_in_place(tmp_path: Path) -> None:
    assert room(tmp_path).path == tmp_path


@pytest.fixture
def only(monkeypatch):
    """`only(*places)`: the places `cache_room` checks, on a local filesystem unless a test says otherwise."""
    monkeypatch.setattr(R, "_mount_table", lambda: [(Path("/"), "xfs")])

    def set_places(*found: Place) -> None:
        monkeypatch.setattr(R, "places", lambda: list(found))

    return set_places


def test_a_disk_short_of_bytes_is_a_miss_that_names_its_variable(tmp_path, fake_room, only) -> None:
    fake_room(500_000_000, 10**6)
    only(Place("HF_HOME", tmp_path, "the Hub downloads"))
    (line,) = cache_room()
    assert not line.ok and line.name == "disk room"
    assert line.detail.startswith("HF_HOME is on / with 0.5 GB / 1,000,000 inodes free")
    assert "the Hub downloads needs at least 1.1 GB" in line.detail
    assert line.fix == "export HF_HOME=<a larger disk>"


@pytest.mark.parametrize(
    ("free_inodes", "total_inodes", "reason"),
    [(9_000, 10**9, "below the 10,000-inode floor"), (40_000, 10**6, "below 5% of the filesystem's total")],
)
def test_a_disk_short_of_inodes_is_a_miss(
    tmp_path, fake_room, only, free_inodes, total_inodes, reason
) -> None:
    fake_room(1 << 40, free_inodes, total_inodes)
    only(Place("TMPDIR", tmp_path, "the temporary files"))
    (line,) = cache_room()
    assert not line.ok
    assert f"{free_inodes:,} free inodes are {reason}" in line.detail
    assert line.fix == "export TMPDIR=<a larger disk>"


def test_a_filesystem_that_reports_no_inodes_is_judged_on_bytes(tmp_path, fake_room, only) -> None:
    fake_room(1 << 40, 0, 0)
    only(Place("TMPDIR", tmp_path, "the temporary files"))
    (line,) = cache_room()
    assert line.ok


def test_one_line_per_filesystem_and_its_floors_add_up(tmp_path, fake_room, only) -> None:
    """A cache and the temp dir on one disk write into the same free space: one line, one verdict."""
    (tmp_path / "cache").mkdir()
    (tmp_path / "tmp").mkdir()
    fake_room(1_500_000_000, 10**6)
    only(
        Place("QDA_CACHE", tmp_path / "cache", "the caches"),
        Place("TMPDIR", tmp_path / "tmp", "the temporary files"),
    )
    (line,) = cache_room()
    assert not line.ok  # two floors of 1 GiB, 1.5 GB free
    assert "QDA_CACHE and TMPDIR are on " in line.detail
    assert "the caches and the temporary files need at least 2.1 GB" in line.detail
    assert line.fix == "export QDA_CACHE=<a larger disk>; export TMPDIR=<a larger disk>"


def test_a_roomy_filesystem_is_an_ok_line(tmp_path, only) -> None:
    only(Place("QDA_CACHE", tmp_path, "the caches"))
    (line,) = cache_room()
    assert line.ok and "QDA_CACHE on /: " in line.detail


def test_each_place_is_read_from_the_code_that_defines_it(monkeypatch, tmp_path) -> None:
    from huggingface_hub import constants
    from qda_quant.io import qda_cache

    monkeypatch.setenv("QDA_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("TMPDIR", raising=False)
    found = {p.name: p for p in places()}
    assert found["QDA_CACHE"].path == qda_cache() and not found["QDA_CACHE"].defaulted
    assert found["HF_HOME"].path == Path(constants.HF_HOME)
    assert found["TMPDIR"].path == Path(tempfile.gettempdir()) and found["TMPDIR"].defaulted


def test_the_engine_cache_place_is_the_one_qda_compiles_into(monkeypatch, tmp_path) -> None:
    """The disk the report names is the one the engines fill, not vLLM's own default."""
    from qda_quant.output.vllm_env import cache_root

    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    found = {p.name: p for p in places()}
    assert found["VLLM_CACHE_ROOT"].path == cache_root() / "vllm" and found["VLLM_CACHE_ROOT"].defaulted


def test_a_named_engine_cache_is_measured_where_it_is_named(monkeypatch, tmp_path) -> None:
    """VLLM_CACHE_ROOT set is the directory the engines compile into, so it is the one measured."""
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm"))
    found = {p.name: p for p in places()}
    assert found["VLLM_CACHE_ROOT"].path == tmp_path / "vllm" and not found["VLLM_CACHE_ROOT"].defaulted


def test_mountinfo_gives_each_mount_its_filesystem_type() -> None:
    line = "872 740 0:70 / /mnt/u rw,relatime shared:9 master:7 - nfs4 server:/export rw,vers=4.2\n"
    assert R._mountinfo_rows(line) == [(Path("/mnt/u"), "nfs4")]
    spaced = r"60 1 0:5 / /mnt/a\040b rw - ext4 /dev/sda1 rw"
    assert R._mountinfo_rows(spaced) == [(Path("/mnt/a b"), "ext4")]


def test_a_path_lands_on_its_longest_mount_point(tmp_path) -> None:
    root = Path(os.path.realpath(tmp_path))  # a mount point is where the data is, through any link
    table = [(Path("/"), "xfs"), (root / "home", "nfs4"), (root / "home" / "u" / "fast", "ext4")]
    assert R._mount(root / "home" / "u" / "x", table) == (root / "home", "nfs4")
    assert R._mount(root / "home" / "u" / "fast" / "x", table) == (root / "home" / "u" / "fast", "ext4")
    assert R._mount(root / "home" / "u" / "x", []) is None


def test_mountinfo_that_cannot_be_read_is_no_table(monkeypatch) -> None:
    def missing(_self):
        raise OSError

    monkeypatch.setattr(Path, "read_text", missing)
    assert R._mount_table() == []


def test_a_defaulted_cache_on_a_network_home_is_a_note_not_a_miss(tmp_path, monkeypatch, only) -> None:
    """statvfs sees the whole network filesystem, not a user's quota: name the variable in a note (``--report``
    and prepare's warning), never a MISS, since nothing can prove the quota short."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(R, "_mount_table", lambda: [(Path("/"), "nfs4")])
    only(Place("HF_HOME", tmp_path / ".cache" / "huggingface", "the Hub downloads", defaulted=True))
    (ok,) = cache_room()
    assert ok.ok
    (note,) = room_notes()
    assert note.line().startswith("disk room: ") and "HF_HOME" in note.detail and "nfs4" in note.detail
    assert "fix: export HF_HOME=<a local disk>" in note.detail
    assert [i.name for i in room_report()] == ["disk room", "disk room"], "the disk and its note"


@pytest.mark.parametrize(
    ("fstype", "defaulted"), [("nfs4", False), ("xfs", True), ("fuse.fuse-overlayfs", True)]
)
def test_an_explicit_cache_or_a_local_filesystem_gets_no_quota_note(
    tmp_path, monkeypatch, only, fstype, defaulted
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    only(Place("HF_HOME", tmp_path / "hf", "the Hub downloads", defaulted=defaulted))
    monkeypatch.setattr(R, "_mount_table", lambda: [(Path("/"), fstype)])
    (line,) = cache_room()
    assert line.ok
