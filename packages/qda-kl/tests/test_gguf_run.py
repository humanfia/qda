"""The GGUF harness (`qda_kl.gguf.run`): what it accepts as a build of the pinned llama.cpp."""

import pytest
from qda_kl.gguf.run import built_at_pin
from qda_quant.gguf.llama_cpp import PINNED


def stamp(tmp_path, commit):
    path = tmp_path / "build-info.cpp"
    path.write_text(f'int LLAMA_BUILD_NUMBER = 12118;\nchar const * LLAMA_COMMIT = "{commit}";\n')
    return path


@pytest.mark.parametrize("digits", [7, 9, 12, 40])
def test_a_build_of_the_pin_is_accepted_however_git_abbreviated_it(tmp_path, digits):
    """git abbreviates by repository size: a full clone of llama.cpp stamps 9 digits, a shallow one 7."""
    assert built_at_pin(stamp(tmp_path, PINNED[:digits]))


@pytest.mark.parametrize("commit", [PINNED[:6], "0000000aa", PINNED[:8] + "0"])
def test_another_or_too_short_a_commit_is_refused(tmp_path, commit):
    assert not built_at_pin(stamp(tmp_path, commit))


def test_an_old_config_with_references_gets_one_line(tmp_path, monkeypatch):
    """The harness's removed KL: `references:` exits with one line naming `qda-kl score`, not a traceback."""
    import sys

    from qda_kl.gguf import run

    config = tmp_path / "old.yaml"
    config.write_text("model: m.gguf\nreferences: [ref]\n")
    monkeypatch.setattr(sys, "argv", ["run", str(config)])
    with pytest.raises(SystemExit) as got:
        run.main()
    assert str(got.value).count("\n") == 0 and "`qda-kl score`" in str(got.value)
    assert "`qda decode-gguf`" in str(got.value)
