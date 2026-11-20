"""`qda_quant.guards`: one ordered list per command, walked by the command (`first_problem`) and by doctor."""

import pytest
from qda_quant.exit import ENVIRONMENT, USAGE
from qda_quant.guards import BUILD, SERVE, Guard, Problem, all_problems, first_problem

ALWAYS = frozenset({BUILD, SERVE})


def _guard(name, problem=None, *, commands=ALWAYS, applies=None, advance=True, calls=None):
    def check(ctx):
        if calls is not None:
            calls.append(name)
        return problem

    extra = {} if applies is None else {"applies": applies}
    return Guard(
        name, frozenset(commands), check if advance else None, f"{name}: at the command", advance, **extra
    )


def test_first_problem_stops_at_the_first_guard_that_refuses():
    calls: list[str] = []
    guards = (
        _guard("a", calls=calls),
        _guard("b", Problem("b is wrong", "fix b", USAGE), calls=calls),
        _guard("c", Problem("c is wrong"), calls=calls),
    )
    assert first_problem(guards, BUILD, None) == Problem("b is wrong", "fix b", USAGE)
    assert calls == ["a", "b"], "the command stops where it refuses"


def test_first_problem_runs_only_its_commands_applying_advance_guards():
    calls: list[str] = []
    guards = (
        _guard("serve only", Problem("no"), commands={SERVE}, calls=calls),
        _guard("does not apply", Problem("no"), applies=lambda ctx: "not here", calls=calls),
        _guard("at load", advance=False),
        _guard("held", calls=calls),
    )
    assert first_problem(guards, BUILD, None) is None
    assert calls == ["held"]


def test_all_problems_walks_every_guard_and_says_what_each_found():
    guards = (
        _guard("a"),
        _guard("b", Problem("b is wrong")),
        _guard("c", Problem("c is wrong")),
        _guard("skipped", Problem("never run"), applies=lambda ctx: "no cards"),
        _guard("at load", advance=False),
        _guard("serve only", Problem("no"), commands={SERVE}),
    )
    found = all_problems(guards, {BUILD}, None)
    assert [(v.guard.name, v.problem, v.skipped, v.advance) for v in found] == [
        ("a", None, None, True),
        ("b", Problem("b is wrong"), None, True),
        ("c", Problem("c is wrong"), None, True),
        ("skipped", None, "no cards", True),
        ("at load", None, None, False),
    ]


def test_a_check_that_raises_is_its_guards_problem_in_doctor_and_an_exception_in_the_command():
    def boom(ctx):
        raise OSError("disk gone")

    guards = (Guard("boom", ALWAYS, boom, "at the command"), _guard("after", Problem("after")))
    (crashed, after) = all_problems(guards, {BUILD}, None)
    assert crashed.problem is not None and crashed.problem.text == "the check raised OSError: disk gone"
    assert crashed.problem.code == ENVIRONMENT and after.problem == Problem("after")
    with pytest.raises(OSError, match="disk gone"):
        first_problem(guards, BUILD, None)


def test_an_advance_guard_needs_a_check():
    with pytest.raises(ValueError, match="needs a check"):
        Guard("x", ALWAYS, None, "somewhere")


def test_a_problem_prints_the_two_line_refusal():
    assert Problem("disk full", "free some space").refusal() == "error: disk full\nfix:   free some space\n"
    assert Problem("disk full").refusal() == "error: disk full\n"
