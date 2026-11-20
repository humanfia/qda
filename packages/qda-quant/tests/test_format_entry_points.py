"""Format plugins must not change the meaning of an existing config spelling."""

import pytest
from qda_quant import Format, Refused
from qda_quant.format import FORMATS

from .test_plugins import fake_plugins


class DuplicateCanonicalFormat(Format):
    """A plugin claiming the built-in format's canonical name under its own registry key."""

    @property
    def name(self):
        return "w4a16-asym-g128"


def test_format_entry_point_cannot_shadow_an_existing_spelling(monkeypatch):
    """An alias outside the preset keys can still collide with the built-in format grammar."""
    spelling = "w4a16-sym-g128"
    assert Format.parse(spelling) == Format(4, 128, symmetric=True)
    monkeypatch.setattr(FORMATS, "_entries", dict(FORMATS._entries))
    fake_plugins(monkeypatch, lambda: FORMATS.register(spelling, Format(8)))
    with pytest.raises(ValueError):
        Format.parse(spelling)


@pytest.mark.parametrize("entry_point", [False, True])
def test_format_entry_point_cannot_change_meaning_when_recorded(monkeypatch, entry_point):
    """A canonical-name collision is refused at registration or discovery, before entering resolved.yaml."""
    assert Format.parse("w4a16-asym-g128") == Format(4, symmetric=False)
    monkeypatch.setattr(FORMATS, "_entries", dict(FORMATS._entries))
    fake_plugins(monkeypatch, lambda: FORMATS.register("zz_external8", DuplicateCanonicalFormat(8)))
    with pytest.raises(Refused, match="canonical name"):
        if entry_point:
            Format.parse("zz_external8")
        else:
            FORMATS.register("zz_external8", DuplicateCanonicalFormat(8))


def test_case_folded_entry_point_cannot_flip_an_existing_format(monkeypatch):
    """A symmetric vendor-style W4A16 alias must not replace QDA's asymmetric w4a16 preset."""
    before = Format.parse("w4a16")
    assert before == Format(4, 128, symmetric=False)
    monkeypatch.setattr(FORMATS, "_entries", dict(FORMATS._entries))
    fake_plugins(monkeypatch, lambda: FORMATS.register("W4A16", Format(4)))
    try:
        after = Format.parse("w4a16")
    except ValueError:
        return  # Refusing the clashing entry point is safe.
    assert after == before
