"""The hidden-path mask policy has one definition; both callers must keep using it.

`sched.sandbox` is the one definition of the policy: it runs standalone (`python -I <file>`), so it cannot
import `isolation.wall`, and the wall imports it. A re-fork on either side would otherwise leave the other
quietly enforcing the old policy. Each test compares one rule by behaviour, through the smallest seam the
two modules expose.

Coverage contract: each case table holds one case per branch the rule has today, and a new branch on
either side must add a case here. That is what keeps a behaviour-only comparison from silently missing
a rule.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

fence = pytest.importorskip("hmz.coganchor.fence")

from qda_agent.config import RunConfig  # noqa: E402
from qda_agent.isolation import SANDBOX, wall  # noqa: E402
from qda_agent.sched import sandbox  # noqa: E402


class _Captured(Exception):
    """Stops the wall once it has named its hidden spellings, before it builds the turn's argv."""


def _private_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A workdir with a hidden directory, a hidden file and a readable sibling."""
    work = tmp_path / "work"
    hidden_dir = work / ".agent"
    hidden_file = work / ".token"
    hidden_dir.mkdir(parents=True)
    hidden_file.write_text("SYNTHETIC PRIVATE\n")
    (work / "public.txt").write_text("public control\n")
    return work, hidden_dir, hidden_file


def _mounts(work: Path) -> list[tuple[str, str, str | None]]:
    """The mounts both rules answer on: the host root, and the workdir bound writable."""
    return [("bind", "/", "/"), ("bind", str(work), str(work))]


def _fence(work: Path):
    """The fence `isolation` draws for a turn that runs in `work`."""
    return fence.Fence.of(
        local=str(SANDBOX.local),
        user=str(SANDBOX.user),
        system=str(SANDBOX.system),
        online=False,
        workdir=work,
        home=Path.home(),
        read=(),
        hosts=["api.anthropic.com"],
    )


def test_hidden_covers_agrees_with_the_wall(tmp_path):
    """A hide that is, or contains, a needed path is returned by both; every other shape is None."""
    work, hidden_dir, hidden_file = _private_tree(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(work, target_is_directory=True)
    cases = [
        ([hidden_dir], [work], None),  # a child of a needed path covers nothing
        ([work], [hidden_dir], (work, hidden_dir)),  # a parent takes the needed child
        ([hidden_dir], [hidden_dir], (hidden_dir, hidden_dir)),  # the path itself
        ([alias / ".agent"], [hidden_dir], (alias / ".agent", hidden_dir)),  # compared resolved
        ([hidden_file], [work], None),  # a sibling covers nothing
        ([tmp_path / "absent"], [work], None),  # an absent hide covers nothing
    ]
    for hide, needed, expected in cases:
        job = sandbox.hidden_covers(hide, needed)
        other = wall.hidden_covers(hide, needed)
        assert job == other, (hide, needed)
        assert other == expected, (hide, needed)


def test_the_wall_expands_hidden_spellings_like_the_job(tmp_path, monkeypatch):
    """The wall names the same spellings `hidden_paths(hide)` does.

    The config spelling, the home's real spelling, and each of those resolved.
    """
    home = tmp_path / "real-home"
    (home / "private").mkdir(parents=True)
    (home / "private" / "heldout.txt").write_text("SYNTHETIC PRIVATE\n")
    alias = tmp_path / "home-alias"
    alias.symlink_to(home, target_is_directory=True)
    monkeypatch.setenv("HOME", str(alias))  # the config's spelling is through the link
    run = tmp_path / "run"
    work = run / "work"
    work.mkdir(parents=True)
    cfg = RunConfig(
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        sandbox={"name": "bwrap", "hide": [str(alias / "private")], "pid": "private"},
        contamination={"name": "none"},  # this test does not exercise the check; ngram is main's default
    )
    seen: dict[str, list[Path]] = {}

    def capture(read, write, hidden, **kwargs):
        seen["hidden"] = list(hidden)
        raise _Captured

    monkeypatch.setattr(wall, "bwrap_argv", capture)  # the wall's own turn argv is not built here
    with pytest.raises(_Captured), wall.gpu_wall(cfg):
        fence.wrapper(_fence(work))
    got = [str(p) for p in seen["hidden"]]
    assert got == [str(p) for p in sandbox.hidden_paths(cfg.sandbox.hide)]
    assert got == [str(alias / "private"), str(home / "private")]  # the link, then its target


def test_the_wall_selects_the_masks_the_job_selects(tmp_path):
    """For every spelling a mount shows there, both select it as a mask destination."""
    work, hidden_dir, hidden_file = _private_tree(tmp_path)
    mounts = _mounts(work)
    for hide in ([hidden_dir], [hidden_file], [hidden_dir, hidden_file]):
        job = [view for view, _ in sandbox.mask_after(mounts, hide)]
        assert [view for view, _ in wall.mask_after(mounts, hide)] == job  # `hide` holds no bind-source case
        assert job == hide


def test_both_sides_mask_a_hidden_object_at_a_relocated_bind(tmp_path):
    """A bind that re-exposes a hidden object elsewhere is masked there on both sides.

    The job's own label and the wall's (`--bind` / `--ro-bind`) all select the destination, and the bind
    source that is exactly the hidden tree counts as covering it.
    """
    work, hidden_dir, hidden_file = _private_tree(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for kind in ("bind", "--bind", "--ro-bind"):
        mounts = [*_mounts(work), (kind, str(elsewhere), str(work))]
        job = [view for view, _ in sandbox.mask_after(mounts, [hidden_file])]
        assert job == [hidden_file, elsewhere / ".token"]  # both destinations the bind exposes
        assert [view for view, _ in wall.mask_after(mounts, [hidden_file])] == job
        mounts = [*_mounts(work), (kind, str(elsewhere), str(hidden_dir))]  # the source is the hidden tree
        job = [view for view, _ in sandbox.mask_after(mounts, [hidden_dir])]
        assert job == [hidden_dir, elsewhere]  # the relocated bind destination itself
        assert [view for view, _ in wall.mask_after(mounts, [hidden_dir])] == job


def test_an_absent_hide_is_never_masked(tmp_path):
    """A hide missing on the host needs no mask, even under a read-only bind.

    Naming the absent destination would make bwrap `mkdir` a mount point inside the read-only bind, which
    fails at startup with `Can't mkdir ...: Read-only file system`.
    """
    work, _, _ = _private_tree(tmp_path)
    absent = work / "absent"
    mounts = [("tmpfs", "/", None), ("--ro-bind", str(work), str(work))]
    assert sandbox.mask_after(mounts, [absent]) == []
    assert wall.mask_after(mounts, [absent]) == []
    argv = wall.bwrap_argv([str(work)], [], [absent], minimal=True, pid="private")
    assert str(work) in argv  # the read-only bind whose source holds the absent hide is mounted
    assert str(absent) not in argv  # and no mask point inside it is named


def test_a_later_bind_that_replaces_the_hidden_tree_needs_no_mask(tmp_path):
    """A bind mounted over the hidden tree decides what shows there; the earlier bind's destination is gone.

    That destination is inside the later read-only bind, so a mask there fails bwrap at startup with
    `Can't mkdir ...: Read-only file system`.
    """
    etc, other = tmp_path / "etc", tmp_path / "other"
    hidden = etc / "ssl" / "certs"
    hidden.mkdir(parents=True)
    other.mkdir()
    mounts = [
        ("tmpfs", "/", None),
        ("--ro-bind", str(etc), str(etc)),
        ("--ro-bind", str(etc / "ssl"), str(other)),  # replaces the hidden tree
    ]
    assert sandbox.mask_after(mounts[:2], [hidden]) == [(hidden, hidden)]  # without the replacement
    assert sandbox.mask_after(mounts, [hidden]) == []
    assert wall.mask_after(mounts, [hidden]) == []
    argv = wall.bwrap_argv([str(etc)], [], [hidden], binds={etc / "ssl": other}, minimal=True, pid="private")
    assert str(hidden) not in argv


def _mask_argv(argv: list[str], hidden: list[Path]) -> list[str]:
    """The mask entries of a bwrap argv (`--tmpfs <hidden>` / `--ro-bind /dev/null <hidden>`)."""
    spellings = {str(p) for p in sandbox.hidden_paths(hidden)}
    out: list[str] = []
    for i, arg in enumerate(argv):
        if (
            arg == "--ro-bind"
            and i + 2 < len(argv)
            and argv[i + 1] == "/dev/null"
            and argv[i + 2] in spellings
        ):
            out += argv[i : i + 3]
        elif arg == "--tmpfs" and i + 1 < len(argv) and argv[i + 1] in spellings:
            out += argv[i : i + 2]
    return out


def test_the_wall_emits_the_job_s_mask_arguments(tmp_path, monkeypatch):
    """Parent-first, a directory by an empty tmpfs and a file by `/dev/null`: the same argv entries."""
    work, hidden_dir, hidden_file = _private_tree(tmp_path)
    nested = work / "tree" / "inner"
    nested.mkdir(parents=True)
    (nested / "notes.txt").write_text("SYNTHETIC PRIVATE\n")
    hidden = [hidden_dir, hidden_file, nested, nested / "notes.txt"]
    monkeypatch.setattr(sandbox, "_ns", ("fresh", "private"))  # the form probe, not bwrap: none is run
    job = sandbox.bwrap_argv(["/bin/true"], str(work), writable=[work], hide=hidden, pid="private")
    turn = wall.bwrap_argv([], [str(work)], hidden, pid="private")
    assert _mask_argv(job, hidden) == _mask_argv(turn, hidden)


def test_the_wall_records_the_same_hidden_identities(tmp_path):
    """Each existing spelling with the host's `(device, inode)`, the last duplicate skipped.

    A link's target is stat'ed, and an absent path is left out.
    """
    work, hidden_dir, hidden_file = _private_tree(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(hidden_file)
    paths = [hidden_dir, hidden_file, link, tmp_path / "absent", hidden_file]
    assert wall._hidden_specs(paths) == sandbox.hidden_specs([(p, p) for p in paths])


def _guard_says_visible(path: Path, dev: int, ino: int) -> bool:
    """What `_HIDDEN_GUARD` decides for one path: 70 is the refusal, 0 is its `exec` of the command."""
    argv = [
        sys.executable,
        "-I",
        "-S",
        "-c",
        sandbox._HIDDEN_GUARD,
        json.dumps([(str(path), dev, ino)]),
        sys.executable,
        "-c",
        "pass",
    ]
    got = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert got.returncode in (0, 70), got
    return got.returncode == 70


def _checker_says_visible(path: Path, dev: int, ino: int) -> bool:
    """What `_CHECKER` decides for one path: it reports the `(device, inode)` still there."""
    argv = [sys.executable, "-I", "-S", "-c", wall._CHECKER, json.dumps([(str(path), dev, ino)])]
    got = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert got.returncode == 0, got
    return json.loads(got.stdout or "[]") != []


def test_the_two_hidden_checkers_classify_a_path_the_same_way(tmp_path):
    """Both stat each path and report the host's own `(device, inode)`; a link is followed, absent is hidden.

    The programs cannot be compared as text: the refusal (exit 70) and the report (JSON) are deliberately
    different, so the shared loop is compared by its answer on every branch.
    """
    work, hidden_dir, hidden_file = _private_tree(tmp_path)
    copy = tmp_path / "copy.txt"
    copy.write_text("SYNTHETIC PRIVATE\n")
    link = tmp_path / "link.txt"
    link.symlink_to(hidden_file)
    host = hidden_file.stat()
    cases = [
        (hidden_file, True),  # the host's own object
        (copy, False),  # same content, another inode
        (link, True),  # os.stat follows the link
        (tmp_path / "absent", False),  # OSError is hidden
        (hidden_dir, False),  # another object
    ]
    for path, expected in cases:
        guard = _guard_says_visible(path, host.st_dev, host.st_ino)
        checker = _checker_says_visible(path, host.st_dev, host.st_ino)
        assert guard == checker == expected, path
