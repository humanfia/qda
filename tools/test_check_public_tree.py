"""Tests for the release tree checker."""

import subprocess

import pytest

from tools.check_public_tree import check_tree, find_matches, load_patterns, public_path


def test_forbidden_terms_are_detected_and_clean_text_passes(tmp_path):
    """Detect a planted forbidden name in a tracked file and accept clean text."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    readme = tmp_path / "README.md"
    readme.write_text("A qda-publish checkout", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    assert check_tree(tmp_path)
    assert find_matches("Prior findings req-23 and res-24")
    assert find_matches("/home/user/config")
    assert not find_matches("/home/<user>/config")
    # `/Users/<name>` is the same leak on macOS; the rule matched only `/home/` until 2026-11-08.
    assert find_matches("/Users/user/config")
    assert not find_matches("/Users/<user>/config")
    assert not public_path("packages/qda-agent/tests/test_cli.py")

    readme.write_text("A clean public tree with qda-publisher names", encoding="utf-8")
    assert not check_tree(tmp_path)


@pytest.mark.parametrize("directory", ["packages", "docs"])
@pytest.mark.parametrize(
    "role",
    [
        *(f"{prefix}-m{track}-7" for prefix in ("w", "v") for track in range(1, 6)),
        *(
            f"{prefix}-{track}{suffix}"
            for prefix in ("w", "v")
            for track in "abcdefghijkl"
            for suffix in ("", "24")
        ),
        *(f"mgr-m{track}" for track in range(1, 6)),
        "s-objective",
        "s-allocation",
    ],
)
def test_team_identifiers_are_rejected_in_tracked_public_files(tmp_path, directory, role):
    """Reject internal identifiers in comments without rejecting mathematical terms."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    path = tmp_path / directory / "note.md"
    path.parent.mkdir()
    path.write_text(f"# Regression observed by {role}\n", encoding="utf-8")
    subprocess.run(["git", "add", str(path.relative_to(tmp_path))], cwd=tmp_path, check=True)
    assert any(f"{directory}/note.md:1:" in violation for violation in check_tree(tmp_path))
    path.write_text("The v-norm controls the vector magnitude.\n", encoding="utf-8")
    assert not check_tree(tmp_path)


# The first two lines follow bench/awq-kl-cell.sh and bench/awq-shard-cell.sh before they were parameterised
# (tree 6ac72b2f), with the account and the cluster renamed. They are what the check has to catch on a
# public path.
PREFIX_LINES = (
    "export HF_HOME=/home/user/pool/hf_cache HF_HUB_OFFLINE=1\n",
    "CAL=/scratch/011/user/qda-team-runs/calib/calib128.messages.jsonl\n",
    "OUT=/Users/user/qda-team-runs/out\n",
)


@pytest.mark.parametrize("line", [PREFIX_LINES[1], PREFIX_LINES[2]])
def test_internal_scratch_and_macos_home_paths_fire_on_a_public_path(tmp_path, line):
    """A team run tree or a macOS home path fails the check on its own, with no role name to trip over."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    path = tmp_path / "bench" / "cell.sh"
    path.parent.mkdir()
    path.write_text(line, encoding="utf-8")
    subprocess.run(["git", "add", "bench/cell.sh"], cwd=tmp_path, check=True)
    violations = check_tree(tmp_path)
    assert any("bench/cell.sh:1:" in violation for violation in violations), violations
    assert not find_matches(line.replace("qda-team-runs/", "runs/").replace("/Users/", "/Users/<"))


def test_the_three_pre_fix_lines_each_name_the_rule_that_catches_them(tmp_path):
    """Each leak is caught by the rule meant for it, so a green result cannot stand on an unrelated match."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    path = tmp_path / "bench" / "cell.sh"
    path.parent.mkdir()
    path.write_text("".join(PREFIX_LINES), encoding="utf-8")
    subprocess.run(["git", "add", "bench/cell.sh"], cwd=tmp_path, check=True)
    rules = {violation.split(": ", 1)[1].split(":")[0] for violation in check_tree(tmp_path)}
    assert rules == {"home path", "team run tree"}, rules


def test_extra_patterns_add_rules_from_a_file_outside_the_tree(tmp_path):
    """A name the repository may not spell is checked from a file: tab-separated, any case, comments skipped."""
    patterns = tmp_path / "private.txt"
    patterns.write_text("# names kept out of the tree\n\ncode name\t\\bzorblat\\b\n", encoding="utf-8")
    extra = load_patterns(patterns)
    assert [rule for _, rule, _ in find_matches("the Zorblat build", extra)] == ["code name"]
    assert not find_matches("the Zorblat build") and not find_matches("zorblats", extra)

    tree = tmp_path / "tree"
    tree.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=tree, check=True)
    (tree / "README.md").write_text("Results of /home/user and ZORBLAT\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tree, check=True)
    assert check_tree(tree) == ["README.md:1: home path: Results of /home/user and ZORBLAT"]
    assert check_tree(tree, extra)[1:] == ["README.md:1: code name: Results of /home/user and ZORBLAT"]

    patterns.write_text("no tab on this line\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"private\.txt:1: expected"):
        load_patterns(patterns)


@pytest.mark.parametrize("stamp", ["13:22" + "Z", "2026-11-11T05:28" + "Z", "2026-11-09T05:13:00" + "Z"])
def test_a_clock_stamp_is_rejected_and_a_zoneless_time_passes(stamp):
    """A stamp says when a note was written, not what it was; the parts are split so this file passes too."""
    assert [rule for _, rule, _ in find_matches(f"seen at {stamp}")] == ["clock stamp"]
    assert not find_matches("the stage waited from 05:13:00 to 05:22:00")


@pytest.mark.parametrize(
    "line",
    [
        "the fix a start refusal names ({number})",
        "({number}) looks like this",
        "a number at the end {number}.",
        "{number}: the refusal it came from",
    ],
)
def test_an_issue_reference_is_rejected(line):
    """A tracker number names a page a reader outside the project cannot open; the sentence must stand alone."""
    number = "#" + "620"
    assert [rule for _, rule, _ in find_matches(line.format(number=number))] == ["issue reference"]


@pytest.mark.parametrize(
    "line",
    [
        '<path fill="#1f77b4" stroke="#fff">',
        '<rect fill="#000"/>',
        "the palette's #123abc shade",
        "see [the rules](#section)",
        "# " + "123 is a heading, not a reference",
        "an entity: &#123;",
        "a marker, url(#a)",
    ],
)
def test_a_colour_an_anchor_a_heading_and_an_entity_are_not_references(line):
    """Each is a number after `#` that carries no tracker meaning, so the rule has to let it through."""
    assert not find_matches(line)


def test_an_issue_reference_fails_the_tree_check_on_a_public_path(tmp_path):
    """The rule runs over tracked public files, so a reference in one fails the release check."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    path = tmp_path / "docs" / "note.md"
    path.parent.mkdir()
    number = "#" + "498"
    path.write_text(f"the start refuses before it creates anything ({number})\n", encoding="utf-8")
    subprocess.run(["git", "add", "docs/note.md"], cwd=tmp_path, check=True)
    rules = {violation.split(": ", 1)[1].split(":")[0] for violation in check_tree(tmp_path)}
    assert rules == {"issue reference"}, rules
    path.write_text('filled with "#fff" and a #123abc shade\n', encoding="utf-8")
    assert not check_tree(tmp_path)
