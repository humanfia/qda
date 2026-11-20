"""Package extensions keep the default sensitivity scorer's arguments intact."""

from qda_kl import extensions, score
from qda_quant.gguf import read


def test_default_sensitivity_scorer_forwards_the_read(monkeypatch):
    monkeypatch.setattr(read, "DEFAULT_SCORER", None)
    result = object()
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(score, "score", fake)
    extensions.register()
    scorer = read._default_scorer()
    assert scorer("checkpoint", "reference", subset="held", tp=4, check=True) is result
    assert calls == [(("checkpoint", "reference"), {"subset": "held", "tp": 4, "check": True})]
