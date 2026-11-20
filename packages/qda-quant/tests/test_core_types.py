import re
from dataclasses import dataclass

import pytest
from qda_quant import Format, Recipe, Refused, Step, Target
from qda_quant.format import FORMATS
from qda_quant.step import REGISTRY


@dataclass(frozen=True)
class Wide8(Format):
    """A third party's spelling for 8-bit symmetric codes, which the built-in target serves."""

    @property
    def name(self) -> str:
        return f"int8-g{self.group_size}"

    @classmethod
    def from_name(cls, name: str) -> Format | None:
        m = re.fullmatch(r"int8-g(\d+)", name)
        return cls(8, int(m[1])) if m else None


@dataclass(frozen=True)
class Fp8Shape(Format):
    """A non-integer format's shape only: no levels, grid or kernel behind it yet."""

    weight_type: str = "fp8_e4m3"

    @property
    def name(self) -> str:
        return f"fp8-e4m3-g{self.group_size}"


@dataclass(frozen=True)
class MixedCase(Format):
    """A third party's canonical spelling and registry alias are mixed-case."""

    @property
    def name(self) -> str:
        return f"Probe{self.weight_bits}-g{self.group_size}"


def test_format_presets_and_parse():
    assert Format.parse("w4a16").name == "w4a16-asym-g128"
    assert Format.parse("w3a16").name == "w3a16-sym-g128"
    w4a8 = Format.parse("w4a8")
    assert w4a8.act == "fp8_token" and w4a8.symmetric
    assert Format.parse("w4a16-sym-g64") == Format(4, 64, True)
    assert (Format(4).qmin, Format(4).qmax) == (-8, 7)
    assert (Format(4, symmetric=False).qmin, Format(4, symmetric=False).qmax) == (0, 15)
    with pytest.raises(Refused):
        Format.parse("w9a99")


def test_the_removed_w8a8_name_is_refused_with_the_two_formats_that_replace_it():
    """Bare `w8a8` was a preset name before 0.1.0; as a generic spelling it would parse to int8 weights with FP8
    activations and be refused by a kernel check that never names the fix. Other `wNaM` spellings are unchanged."""
    for spelled in ("w8a8", " W8A8 "):
        with pytest.raises(Refused) as refused:
            Format.parse(spelled)
        assert str(refused.value) == f"format {spelled!r}: QDA 0.1.0 has no format of that name"
        assert refused.value.fix == "use w8a8-fp8 (FP8 W8A8) or w8a8i (INT8 W8A8)"
    assert Format.parse("w8a8-fp8").name == "w8a8-fp8" and Format.parse("w8a8i").name == "w8a8i-sym-pc"
    assert Format.parse("w8a8-sym-g128").name == "w8a8-sym-g128"  # a full spelling still parses

    from qda_quant.config import BuildConfig

    # a config is refused with the same two lines: the key's message, then the validator's own fix
    with pytest.raises(Refused) as refused:
        BuildConfig.from_dict({"model": "Qwen/Qwen3-0.6B", "format": "w8a8", "out": "o"}, name="w8a8.yaml")
    assert "\n  format: format 'w8a8': QDA 0.1.0 has no format of that name" in str(refused.value)
    assert refused.value.fix == "use w8a8-fp8 (FP8 W8A8) or w8a8i (INT8 W8A8)"


def test_a_third_party_format_registers_and_plans():
    int8 = Wide8(8, 128)
    FORMATS.register(int8.name, int8)
    assert Format.parse("int8-g128") is int8
    assert Format.parse("int8-g64") == Wide8(8, 64)
    assert Target().plan(int8, "dense").kernel == "HummingLinearKernel"

    fp8 = Fp8Shape(8, 128)
    FORMATS.register(fp8.name, fp8)
    assert Format.parse("fp8-e4m3-g128") is fp8
    with pytest.raises(Refused, match="integer"):
        Target().plan(fp8, "dense")


def test_a_registered_format_parses_by_key_and_canonical_name_at_any_case():
    fmt = MixedCase(8, 128)
    FORMATS.register("probe8", fmt)
    assert Format.parse("probe8") is fmt
    assert Format.parse(fmt.name) is fmt
    assert Format.parse(fmt.name.lower()) is fmt
    alias = MixedCase(8, 64)
    FORMATS.register("Probe8Alias", alias)
    assert Format.parse("PROBE8ALIAS") is alias


@pytest.mark.parametrize(
    "spelling,bits,group,symmetric",
    [
        ("w4a16", 4, 128, False),
        ("w4a16-g64", 4, 64, False),
        ("w4a16-g128", 4, 128, False),
        ("w4a16-asym-g32", 4, 32, False),
        ("w4a16-sym", 4, 128, True),
        ("w4a16-sym-g64", 4, 64, True),
        ("w4a8-g128", 4, 128, True),
        ("w4a8-asym-g128", 4, 128, False),
        ("w3a16-g64", 3, 64, True),
        ("w3a16-g128", 3, 128, True),
        ("w3a16-sym-g128", 3, 128, True),
        ("W4A16", 4, 128, False),
        ("w4A16-Sym-G64", 4, 64, True),
        (" w4a16-sym ", 4, 128, True),
        ("w4a8", 4, 128, True),
        ("w8a8i", 8, -1, True),
        ("w8a8i-sym-pc", 8, -1, True),
    ],
)
def test_format_spellings(spelling, bits, group, symmetric):
    fmt = Format.parse(spelling)
    assert (fmt.weight_bits, fmt.group_size, fmt.symmetric) == (bits, group, symmetric)
    assert Format.parse(fmt.name) == fmt


def test_w8a8i_is_int8_activations_and_refuses_weights_vllm_cannot_serve():
    """Its own family spelling, so the INT8 family never answers to the FP8 spelling."""
    fmt = Format.parse("w8a8i")
    assert fmt == Format.parse("w8a8i-sym-pc") == Format.parse("w8a8i-sym")
    assert fmt.act == "int8_token" and fmt.name == "w8a8i-sym-pc"
    # W8A8 has no user preset: the canonical name is the entry point
    fp8 = Format.parse("w8a8-fp8")
    assert (fp8.act, fp8.weight_type) == ("fp8_token", "fp8") and fp8.name == "w8a8-fp8"
    for spelling in ("w8a8i-asym", "w8a8i-asym-pc", "w8a8i-g128", "w8a8i-sym-g64"):
        with pytest.raises(Refused, match="symmetric per-channel or per-tensor"):
            Format.parse(spelling)


def test_target_parse_names_the_card_tables():
    assert Target.parse("h200").arch == "sm90"
    assert Target.parse("sm90@0.33.0").vllm == "0.33.0"
    assert (Target.parse("b200").arch, Target.parse("gb300").arch) == ("sm100", "sm103")
    with pytest.raises(Refused):
        Target.parse("blackwell")


def test_recipe_yaml_roundtrip_and_overrides():
    class Dummy(Step):
        damp: float = 0.01
        actorder: str = "static"

    REGISTRY.register("dummy", Dummy)
    text = "steps:\n  - dummy: {damp: 0.02}\ncalib: {file: {path: c.jsonl, samples: 64}}\n"
    r = Recipe.from_yaml(text, overrides=["steps.0.damp=0.003"])
    assert r.steps[0].damp == 0.003 and r.calib.samples == 64
    assert Recipe.from_yaml(r.to_yaml()).to_dict() == r.to_dict()
