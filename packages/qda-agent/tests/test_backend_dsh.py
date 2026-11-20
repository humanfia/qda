"""The `dsh` backend: DeepSeek Harness, which Humanize drives through its Python SDK.

It has no program on `PATH` and no sign-in file. Humanize reads its key on the host, from the account in the
spec or from `DEEPSEEK_API_KEY`, so the run gives it a home of its own and mounts nothing of `~/.dsh`. Its
endpoint is asked for the smallest turn there is before a run starts (and by `qda doctor --search`), because an
endpoint refuses a model name it does not serve only once a turn is being taken.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

pytest.importorskip("hmz.coganchor.fence")

import testkit  # noqa: E402
from hmz.sdk import fakes  # noqa: E402
from qda_agent import dsh, isolation  # noqa: E402
from qda_agent.agents import BACKENDS, Claude, Codex, Dsh  # noqa: E402
from qda_agent.config import RunConfig  # noqa: E402
from qda_agent.flow import AgentTurnRefused, qda_search  # noqa: E402
from qda_agent.flow import check_agent_turn as run_check  # noqa: E402
from qda_agent.isolation import CLI_CONFIG, cli_dirs, wall  # noqa: E402
from tests_support import Endpoint, agent  # noqa: E402

MODEL = "deepseek-v4.1-flash-1017"
#: What the official DeepSeek API serves against a key from platform.deepseek.com, as it lists it.
OFFICIAL_MODEL = "deepseek-flash"


def config(run, spec, *fallbacks, level="strict"):
    return RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        agent=agent(spec, *fallbacks),
        isolation={"level": level},
    )


def mapped(run, **backend):
    """A run whose `agent.backend` is the mapping given, for a config that names no model of its own."""
    return RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        agent={"backend": {"name": "dsh", **backend}},
        isolation={"level": "strict"},
    )


def test_dsh_is_a_backend_with_a_humanize_spec(tmp_path, monkeypatch):
    """`name: dsh` builds Humanize's `dsh[@ACCOUNT]/MODEL[:EFFORT]`; it names SDK modules, not a program."""
    assert "dsh" in BACKENDS and BACKENDS["dsh"] is Dsh
    cfg = config(tmp_path, f"dsh@gateway/{MODEL}:high")
    assert isinstance(cfg.agent.backend, Dsh)
    assert cfg.agent.spec == f"dsh@gateway/{MODEL}:high"
    assert cfg.agent.place == f"dsh@gateway/{MODEL}"
    assert cfg.agent.specs == [f"dsh@gateway/{MODEL}:high"]
    assert cfg.agent.backend.cli == "dsh"
    assert cfg.agent.backend.modules == ("deepseek_harness", "deepseek_harness_runtime")
    monkeypatch.setenv(
        "DEEPSEEK_BASE_URL", "https://gateway.invalid/v1"
    )  # no account: the base URL says where
    assert config(tmp_path, f"dsh/{MODEL}").agent.spec == f"dsh/{MODEL}"  # no account, no effort
    assert config(tmp_path, "claude/m").agent.backend.modules == ()  # a CLI is found on PATH


def test_dsh_refuses_a_sign_in_dir(tmp_path):
    """The harness has no sign-in file, so a run cannot name a dir to take one from."""
    section = {"backend": {"name": "dsh", "model": MODEL, "sign_in_dir": str(tmp_path)}}
    with pytest.raises(ValueError, match="dsh has no sign-in file"):
        RunConfig(model="m", format="w4a16", budget_gpu_hours=1, cards=[0], run_dir=tmp_path, agent=section)
    with pytest.raises(ValueError, match="'dsh' has no sign-in file"):
        cli_dirs.sign_in_file("dsh")
    assert cli_dirs.sign_in_file("claude") == ".credentials.json"


def test_dsh_accepts_the_official_deepseek_api(tmp_path, monkeypatch):
    """The official DeepSeek API is the SDK's default endpoint and the product default: a bare `dsh` spec
    loads, and `DEEPSEEK_BASE_URL` may name the official API (the docs' shape) or a gateway."""
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    config(tmp_path, f"dsh/{MODEL}")  # no base URL: the SDK's own default
    for base in ("https://api.deepseek.com", "https://api.deepseek.com/v1", "https://gateway.invalid/v1"):
        monkeypatch.setenv("DEEPSEEK_BASE_URL", base)
        config(tmp_path, f"dsh/{MODEL}")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    config(tmp_path, f"dsh@gateway/{MODEL}")  # the account names the endpoint


def test_dsh_model_default_follows_the_endpoint(tmp_path, monkeypatch):
    """A `dsh` config that names no model runs the one its endpoint serves: `deepseek-flash` against the
    official API, and a gateway's own name against a gateway (the name the shipped config names for one)."""
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    assert mapped(tmp_path).agent.backend.model == OFFICIAL_MODEL  # no endpoint named: the SDK's default
    for base in ("https://api.deepseek.com", "https://api.deepseek.com/v1"):
        monkeypatch.setenv("DEEPSEEK_BASE_URL", base)
        assert mapped(tmp_path).agent.backend.model == OFFICIAL_MODEL  # the same API under /v1
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gateway.invalid/v1")
    assert mapped(tmp_path).agent.backend.model == MODEL
    # a model the config names is kept, whichever endpoint it is
    assert config(tmp_path, f"dsh/{OFFICIAL_MODEL}").agent.backend.model == OFFICIAL_MODEL


def test_dsh_model_default_follows_a_named_accounts_endpoint(tmp_path, monkeypatch):
    """An account's endpoint is its own: the run's ambient `DEEPSEEK_BASE_URL` is not what the account's key
    goes to (Humanize removes it for the turn), so it does not decide the account's default model either."""
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gateway.invalid/v1")  # the run's own: not the account's
    from hmz.coganchor import providers

    providers.add("dsh", "official", way="key", env={"DEEPSEEK_API_KEY": "k"})
    assert mapped(tmp_path, account="official").agent.backend.model == OFFICIAL_MODEL  # names no endpoint
    providers.add("dsh", "gateway", way="openai-gateway", env={"DEEPSEEK_BASE_URL": "https://gw.invalid/v1"})
    assert mapped(tmp_path, account="gateway").agent.backend.model == MODEL


def test_the_endpoint_is_asked_for_the_smallest_turn(endpoint, tmp_path):
    """The turn asked for is one message and one token, at the endpoint the run's key goes to: an endpoint
    answers `max_tokens: 1` with a token, which is what makes this check cost nothing worth counting."""
    cfg = config(tmp_path / "run", f"dsh/{MODEL}")
    assert cfg.agent.backend.turn_problem() is None
    assert cfg.agent.backend.turn_asked() == f"{endpoint.url} takes {MODEL}"
    (asked,) = endpoint.asked
    assert asked["path"] == "/chat/completions"
    assert asked["headers"]["Authorization"] == "Bearer a-test-key"
    assert asked["json"]["model"] == MODEL and asked["json"]["max_tokens"] == 1
    assert [m["role"] for m in asked["json"]["messages"]] == ["user"]


def test_the_endpoints_refusal_is_the_runs_refusal(endpoint, tmp_path):
    """A model the endpoint does not serve is refused before the clock, in the endpoint's own words and with
    the names it does serve -- which is what a run would otherwise meet minutes into its start."""
    endpoint.answers = (
        400,
        {
            "error": {
                "message": "The supported API model names are deepseek-flash, deepseek-v4-pro, but you passed "
                f"{MODEL}. (request_id: 6b89e650)"
            }
        },
    )
    cfg = config(tmp_path / "run", f"dsh/{MODEL}")
    found = cfg.agent.backend.turn_problem()
    assert not found.transient  # it answered, and it will answer again: not a retry's
    why = found.said
    assert why.startswith(f"{endpoint.url} refused a turn at {MODEL}")
    assert "The supported API model names are deepseek-flash, deepseek-v4-pro" in why
    # the run refuses on that same check: one function, one answer
    with pytest.raises(AgentTurnRefused, match="refused a turn at") as refused:
        run_check(cfg)
    assert str(refused.value) == f"dsh: {' '.join(why.split())}"


def test_an_endpoint_that_does_not_answer_is_a_retrys_fact_not_the_configs(tmp_path, monkeypatch):
    """An endpoint that will not answer is not a fact about the config: `agent.retries` and a fallback are
    what take that turn, so the run starts and the check is recorded rather than refused."""
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://127.0.0.1:9/v1")  # nothing listens on port 9
    monkeypatch.setenv("DEEPSEEK_API_KEY", "a-test-key")
    monkeypatch.setattr(Dsh, "modules", ())
    cfg = config(tmp_path / "run", f"dsh/{MODEL}")
    found = cfg.agent.backend.turn_problem()
    assert found is not None and found.transient and "would not take a turn" in found.said
    why = run_check(cfg)  # the run's own start check: recorded, and the run goes on
    assert why == f"dsh: {found.said} (a retry takes the turn)"


def test_an_endpoint_that_is_failing_is_a_retrys_fact_too(endpoint, tmp_path):
    """A throttled or failing endpoint (429, 5xx) answers, but not that this run may not take a turn."""
    endpoint.answers = (503, {"error": {"message": "the gateway is unwell"}})
    cfg = config(tmp_path / "run", f"dsh/{MODEL}")
    found = cfg.agent.backend.turn_problem()
    assert found is not None and found.transient and "the gateway is unwell" in found.said
    assert "the gateway is unwell" in run_check(cfg)  # recorded, and the run starts
    endpoint.answers = (429, {"error": {"message": "slow down"}})
    assert cfg.agent.backend.turn_problem().transient


def test_a_refusal_a_fallback_can_take_does_not_refuse_the_run(endpoint, tmp_path):
    """`agent.fallbacks` is a place that can take the turn: Humanize walks there after its retries, so a
    primary endpoint that refuses the run's model does not cost the run."""
    endpoint.answers = (400, {"error": {"message": "The supported API model names are deepseek-flash"}})
    cfg = config(tmp_path / "run", f"dsh/{MODEL}", "claude/claude-opus-5-5")
    why = run_check(cfg)
    assert (
        why
        == f"dsh: {endpoint.url} refused a turn at {MODEL}: The supported API model names are deepseek-flash (claude takes the turn)"
    )


def test_the_probe_speaks_the_protocol_the_accounts_way_names(endpoint, tmp_path, monkeypatch):
    """A turn under a gateway account runs on one of pi-ai's routes, so the probe asks in that protocol: the
    account's own `DSH_GATEWAY_API` for `openai-gateway`, and the way itself for the two that speak a
    vendor's API. The stock adapter -- no account, or the key way -- speaks DeepSeek's chat completions."""
    from hmz.coganchor import providers

    for account, env, path, header in (
        ("responses", {"DSH_GATEWAY_API": "openai-responses"}, "/responses", "Authorization"),
        ("completions", {"DSH_GATEWAY_API": "openai-completions"}, "/chat/completions", "Authorization"),
        ("default", {}, "/chat/completions", "Authorization"),  # what the way fixes it to
        ("messages", {}, "/v1/messages", "x-api-key"),
        ("gemini", {}, f"/models/{MODEL}:streamGenerateContent", "x-goog-api-key"),
    ):
        way = {
            "responses": "openai-gateway",
            "completions": "openai-gateway",
            "default": "openai-gateway",
            "messages": "anthropic-gateway",
            "gemini": "gemini-gateway",
        }[account]
        providers.add(
            "dsh",
            account,
            way=way,
            env={"DEEPSEEK_API_KEY": "a-test-key", "DEEPSEEK_BASE_URL": endpoint.url, **env},
        )
        cfg = mapped(tmp_path / "run", account=account, model=MODEL)
        assert cfg.agent.backend.turn_problem() is None
        asked = endpoint.asked[-1]
        assert asked["path"] == path
        sent = {name.lower(): value for name, value in asked["headers"].items()}  # http.server title-cases
        assert sent[header.lower()] == ("Bearer a-test-key" if header == "Authorization" else "a-test-key")
    # the run's own environment, with no account: the stock adapter's chat completions
    assert dsh.protocol() == dsh.STOCK


def test_the_responses_probe_asks_for_what_that_api_takes(endpoint, tmp_path):
    """The Responses API refuses fewer than 16 output tokens, so its smallest turn is 16 rather than one:
    asking for one there would reject an account that works."""
    from hmz.coganchor import providers

    providers.add(
        "dsh",
        "responses",
        way="openai-gateway",
        env={
            "DEEPSEEK_API_KEY": "a-test-key",
            "DEEPSEEK_BASE_URL": endpoint.url,
            "DSH_GATEWAY_API": "openai-responses",
        },
    )
    cfg = mapped(tmp_path / "run", account="responses", model=MODEL)
    assert cfg.agent.backend.turn_problem() is None
    assert endpoint.asked[-1]["json"]["max_output_tokens"] == dsh.RESPONSES_MIN_TOKENS >= 16


def test_a_body_that_stops_half_way_through_is_a_transient_failure(endpoint, tmp_path):
    """A gateway that claims a length and closes before sending it stopped answering, which is a retry's and a
    fallback's: it must not raise out of the check (the crash the review found), and the status line stands in
    for the body it did not send."""
    endpoint.answers = (503, b"busy")
    endpoint.length = 100  # claims more than it sends, then closes: http.client calls that IncompleteRead
    cfg = config(tmp_path / "run", f"dsh/{MODEL}", "claude/claude-opus-5-5")
    found = cfg.agent.backend.turn_problem()  # no exception out of the check
    assert found is not None and found.transient
    # the body it claimed is not there whole, so the status line is what the report has to say
    assert "Service Unavailable" in found.said and "busy" not in found.said
    why = run_check(cfg)  # the run's own start check, with a fallback: recorded, not refused
    assert why is not None and "(a retry takes the turn)" in why  # the retry comes before the fallback


def test_a_chat_route_asks_for_its_output_limit_in_its_own_field(endpoint, tmp_path):
    """`max_tokens` is a 400 to the reasoning models behind an OpenAI-shaped gateway, which pi-ai's own route
    answers by sending `max_completion_tokens` (the runtime's `maxTokensField`); the stock DeepSeek adapter
    sends `max_tokens`, its own dialect."""
    from hmz.coganchor import providers

    for account, way, extra, limit in (
        ("completions", "openai-gateway", {"DSH_GATEWAY_API": "openai-completions"}, "max_completion_tokens"),
        ("stock", "key", {}, "max_tokens"),
    ):
        providers.add(
            "dsh",
            account,
            way=way,
            env={"DEEPSEEK_API_KEY": "a-test-key", "DEEPSEEK_BASE_URL": endpoint.url, **extra},
        )
        cfg = mapped(tmp_path / "run", account=account, model=MODEL)
        assert cfg.agent.backend.turn_problem() is None
        asked = endpoint.asked[-1]["json"]
        assert asked[limit] == 1
        if limit == "max_completion_tokens":
            assert "max_tokens" not in asked  # the field the gateway's models refuse


def test_a_redirect_to_another_origin_is_not_followed_with_the_key(endpoint, tmp_path):
    """urllib re-sends every header it was given wherever a redirect points, the credential among them: an
    endpoint answering `302 https://somewhere-else` would hand the run's key to a machine the config never
    named. The redirect stops here, and the other origin is never asked."""
    with Endpoint() as other:
        endpoint.answers = (302, {"error": {"message": "moved"}})
        endpoint.location = f"{other.url}/v1/chat/completions"
        cfg = config(tmp_path / "run", f"dsh/{MODEL}")
        found = cfg.agent.backend.turn_problem()
        assert found is not None and not found.transient  # the endpoint answered, and will again
        assert f"{endpoint.url} answered 302 to {other.url}" in found.said
        assert other.asked == []  # the key never reached the origin the redirect named
        # the same origin is followed: a path corrected is the endpoint this run named
        endpoint.answers = (200, {"choices": [{}]})
        endpoint.location = "/v1/chat/completions"
        assert cfg.agent.backend.turn_problem() is None
        assert endpoint.asked[-1]["headers"]["Authorization"] == "Bearer a-test-key"


def test_a_same_origin_307_goes_again_with_its_method_and_body(endpoint, tmp_path):
    """307 and 308 mean "the same request at another URL", which the runtime's own client follows with its POST
    and its body and urllib refuses to make of a POST at all: a probe that stopped there would refuse a run
    whose turn the runtime would have sent on."""
    endpoint.answers = [(307, {}), (200, {"choices": [{}]})]
    endpoint.location = "/v1/chat/completions"  # the same origin: a redirect this run's key may follow
    cfg = config(tmp_path / "run", f"dsh/{MODEL}")
    assert cfg.agent.backend.turn_problem() is None  # the redirect was followed, and the turn taken
    assert [a["path"] for a in endpoint.asked] == ["/chat/completions", "/v1/chat/completions"]
    assert (
        endpoint.asked[-1]["json"]["model"] == MODEL
    )  # the body went again, as a POST (a GET is not handled)


def test_a_transports_own_words_are_redacted_too(endpoint, tmp_path, monkeypatch):
    """A malformed status line can carry the credential (`Invalid API key: <key>`): the finding redacts and cuts
    the transport's own words like an endpoint's, so neither doctor's line nor flow.jsonl carries the key."""
    import http.client

    class Malformed:
        def open(self, *args: object, **kwargs: object) -> object:
            raise http.client.BadStatusLine("Invalid API key: a-test-key")

    monkeypatch.setattr(dsh, "_OPENING", Malformed())
    found = config(tmp_path / "run", f"dsh/{MODEL}").agent.backend.turn_problem()
    assert found is not None and found.transient
    assert "a-test-key" not in found.said and "<redacted>" in found.said


def test_a_json_shape_this_has_no_field_for_is_redacted_after_decoding(endpoint, tmp_path):
    """A body whose shape this module does not know (no `error.message`, no `message`) is reported from the JSON
    it read, not from the body's own bytes: a credential the body spelled escaped is the credential again once
    it is read, and a report of the raw body would put it back."""
    escaped = "".join(f"\\u{ord(c):04x}" for c in "a-test-key")  # the key, as an endpoint could spell it
    endpoint.answers = (401, '{"detail": "invalid key: ' + escaped + '"}')
    why = config(tmp_path / "run", f"dsh/{MODEL}").agent.backend.turn_problem().said
    assert escaped not in why and "a-test-key" not in why
    assert "<redacted>" in why and "detail" in why  # the endpoint's own words, with the key taken out


def test_a_turn_the_endpoint_takes_is_reported_as_passing(endpoint, tmp_path):
    """The whole check: the run's own start check passes, the backend says what was asked, and a backend
    whose first turn is its own check says there is nothing to ask."""
    cfg = config(tmp_path / "run", f"dsh/{MODEL}")
    assert run_check(cfg) is None  # the run's own start check: no refusal, nothing to record
    assert cfg.agent.backend.turn_asked() == f"{endpoint.url} takes {MODEL}"
    assert Claude().turn_problem() is None  # a CLI takes a turn only by taking one
    assert Claude().turn_asked() == "nothing to ask before its first turn"


def test_the_sdk_the_run_is_driven_through_is_asked_first(tmp_path, monkeypatch):
    """A run without the SDK refuses before asking any endpoint: Humanize drives the harness in this
    interpreter, and the refusal carries Humanize's own install line."""
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "a-test-key")
    monkeypatch.setattr(Dsh, "modules", ("qda_no_such_sdk_module",))
    cfg = config(tmp_path / "run", f"dsh/{MODEL}")
    found = cfg.agent.backend.turn_problem()
    why = found.said
    assert why is not None and "qda_no_such_sdk_module not installed" in why and "pip install" in why
    with pytest.raises(AgentTurnRefused, match="qda_no_such_sdk_module"):
        run_check(cfg)


def test_an_account_nothing_answers_to_is_refused_before_the_first_turn(tmp_path, monkeypatch):
    """`agent.backend.account` names the account a run's key comes from; a name nothing answers to is a
    refusal here rather than a run that starts and dies (nothing else in a run reads the account)."""
    monkeypatch.setattr(Dsh, "modules", ())
    cfg = config(tmp_path / "run", f"dsh@nosuchaccount/{MODEL}")
    found = cfg.agent.backend.turn_problem()
    assert found.said == (
        "no Humanize account called 'nosuchaccount' (agent.backend.account); make the account, or set "
        "DEEPSEEK_API_KEY and drop the account"
    )
    assert not found.transient  # a name that is not there will not appear on a retry


def test_no_credential_is_the_sign_in_checks_fact(tmp_path, monkeypatch):
    """With nothing to ask with, this check says nothing: `agent sign-in` owns where the credential is, and
    the run refuses that before any turn (a 401 from an endpoint would be the wrong line to report it on)."""
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(Dsh, "modules", ())
    assert config(tmp_path / "run", f"dsh/{MODEL}").agent.backend.turn_problem() is None


def test_no_credential_reaches_a_report(endpoint, tmp_path):
    """An endpoint that echoed the credential back would otherwise put a key in a report or a run's log."""
    endpoint.answers = (401, {"error": {"message": "bad key: Bearer a-test-key"}})
    why = config(tmp_path / "run", f"dsh/{MODEL}").agent.backend.turn_problem().said
    assert "a-test-key" not in why and "<redacted>" in why


def test_a_credential_escaped_in_the_body_is_redacted_too(endpoint, tmp_path):
    """A body may carry the credential escaped (`\u0061-test-key`), which reading it as JSON turns back into
    the credential itself: redacting the body alone would put the key in the report anyway."""
    escaped = "".join(f"\\u{ord(c):04x}" for c in "a-test-key")  # the key, as an endpoint could spell it
    endpoint.answers = (401, '{"error": {"message": "bad key: ' + escaped + '"}}')
    why = config(tmp_path / "run", f"dsh/{MODEL}").agent.backend.turn_problem().said
    assert "a-test-key" not in why and "<redacted>" in why


def test_dsh_gets_a_home_of_the_run_and_never_a_file_of_the_users(tmp_path, monkeypatch):
    """`DSH_HOME` is `<run_dir>/cli/dsh` for the length of the run, holding an empty settings file only: the
    user's `~/.dsh` credentials are neither linked nor mounted, with an account or with the token variable."""
    home = tmp_path / "home"
    (home / ".dsh").mkdir(parents=True)
    (home / ".dsh" / ".credentials.yaml").write_text("DEEPSEEK_API_KEY: user\n")
    (home / ".dsh" / "settings.yaml").write_text("llm-deepseek: {baseURL: https://elsewhere.invalid}\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("DSH_HOME", str(home / ".dsh"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    run = tmp_path / "run"
    cfg = config(run, f"dsh@gateway/{MODEL}")
    assert isolation.cli_config_dirs(cfg) == {"dsh": run / "cli" / "dsh"}
    assert isolation.live_dirs(cfg) == {"dsh": run / "live" / "dsh"}
    assert isolation.sign_in_accounts(cfg) == {"dsh": None}
    with isolation.private_cli_config(cfg) as dirs:
        assert dirs == {"dsh": run / "cli" / "dsh"}
        assert os.environ["DSH_HOME"] == str(dirs["dsh"])
        assert [p.name for p in dirs["dsh"].iterdir()] == ["settings.yaml"]
        assert (dirs["dsh"] / "settings.yaml").read_text() == ""  # not the user's endpoint
        assert list((run / "live" / "dsh").iterdir()) == []
        assert not (run / isolation.SIGNIN).exists()  # nothing for the wall to bind
    assert os.environ["DSH_HOME"] == str(home / ".dsh")  # restored
    # the token variable in the run's environment, and no account
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gateway.invalid/v1")
    cfg = config(run, f"dsh/{MODEL}")
    assert isolation.sign_in_accounts(cfg) == {"dsh": None}
    with isolation.private_cli_config(cfg) as dirs:
        assert [p.name for p in dirs["dsh"].iterdir()] == ["settings.yaml"]
    isolation.check_sign_in(cfg, datetime.now(UTC) + timedelta(hours=1))  # no expiry to hold it to


def test_dsh_without_an_account_or_a_key_is_refused_at_the_start(tmp_path, monkeypatch):
    """Neither an account nor `DEEPSEEK_API_KEY`: refused before any turn, naming both ways in."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DSH_HOME", raising=False)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gateway.invalid/v1")
    cfg = config(tmp_path / "run", f"dsh/{MODEL}")
    with (
        pytest.raises(
            isolation.SignInRefused, match="Humanize account or a token variable .DEEPSEEK_API_KEY."
        ),
        isolation.private_cli_config(cfg),
    ):
        pass
    assert "DSH_HOME" not in os.environ  # nothing of the refused run is left in the environment


def test_at_none_the_dsh_home_is_the_runs_own_too(tmp_path, monkeypatch):
    """No wall at none, but the runtime still reads `$DSH_HOME/settings.yaml`, and a saved
    `llm-deepseek.baseURL` outranks the run's environment: the run's dir holds an empty one, so the user's own
    (the official API, say) is not the endpoint the run uses."""
    home = tmp_path / "home"
    (home / ".dsh").mkdir(parents=True)
    (home / ".dsh" / "settings.yaml").write_text("llm-deepseek: {baseURL: https://api.deepseek.com/v1}\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("DSH_HOME", str(home / ".dsh"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gateway.invalid/v1")
    cfg = config(tmp_path / "run", f"dsh/{MODEL}", level="none")
    with isolation.private_cli_config(cfg) as dirs:
        assert dirs == {"dsh": tmp_path / "run" / "cli" / "dsh"}
        assert os.environ["DSH_HOME"] == str(dirs["dsh"])
        assert (dirs["dsh"] / "settings.yaml").read_text() == ""  # not the user's official endpoint
    assert os.environ["DSH_HOME"] == str(home / ".dsh")  # restored
    with wall.gpu_wall(cfg):  # the CLI's own entry: nothing starts, but the run's dir is what dsh reads
        assert os.environ["DSH_HOME"] == str(tmp_path / "run" / "cli" / "dsh")
    assert os.environ["DSH_HOME"] == str(home / ".dsh")


def test_the_wall_names_no_install_tree_for_dsh(tmp_path):
    """The wall knows `dsh` (it is not refused as an unknown CLI) and binds nothing of the home back for it;
    a fallback CLI still brings its own install trees."""
    cfg = config(tmp_path, f"dsh@gateway/{MODEL}")
    assert isolation.installs(cfg) == ()
    assert isolation.INSTALLS["dsh"] == ()
    both = config(tmp_path, f"dsh@gateway/{MODEL}", "codex/gpt-6.1-sol:low")
    assert "node_modules/@openai/codex" in isolation.installs(both)
    assert isolation.cli_config_dirs(both) == {
        "dsh": tmp_path / "cli" / "dsh",
        "codex": tmp_path / "cli" / "codex",
    }


def test_the_config_table_moves_the_dsh_home_and_names_its_key_variable():
    var, settings, sign_in, tokens, expiry = CLI_CONFIG["dsh"]
    assert (var, settings, sign_in, tokens, expiry) == (
        "DSH_HOME",
        "settings.yaml",
        None,
        ("DEEPSEEK_API_KEY",),
        None,
    )


def test_the_wall_binds_the_hosts_scratch_directory_for_dsh_only(tmp_path):
    """The harness is started from a patch its driver writes into the fence's scratch dir on the host, so the
    wall binds that dir for a dsh run; any other run keeps the empty one the wall's `/tmp` gives it."""
    scratch = tmp_path / "hmz-fence-x"
    scratch.mkdir()
    dsh = config(tmp_path / "run", f"dsh@gateway/{MODEL}")
    assert isolation.HOST_SCRATCH == ("dsh",)
    assert wall.host_scratch(dsh, str(scratch)) == [str(scratch)]
    assert wall.host_scratch(dsh, "") == []  # Humanize named none: `hmz internal fence` makes its own
    assert wall.host_scratch(dsh, str(tmp_path / "gone")) == []
    assert wall.host_scratch(config(tmp_path / "run", "claude/m"), str(scratch)) == []
    fallback = config(tmp_path / "run", "claude/m", f"dsh@gateway/{MODEL}")
    assert wall.host_scratch(fallback, str(scratch)) == [str(scratch)]


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="no bubblewrap on PATH")
def test_a_scratch_directory_is_bound_writable_over_the_private_temp_dirs(tmp_path):
    """What `host_scratch` adds reaches bwrap as a writable bind of the host's directory at its own path."""
    scratch = tmp_path / "hmz-fence-x"
    scratch.mkdir()
    argv = wall.bwrap_argv([], [str(scratch)], [], minimal=True)
    at = argv.index(str(scratch))
    assert argv[at - 1 : at + 2] == ["--bind", str(scratch), str(scratch)]


def test_only_a_dsh_session_takes_one_instruction():
    assert Dsh.resumes is False
    assert Claude.resumes and Codex.resumes


async def test_every_goal_goes_to_a_new_dsh_session(tmp_path, monkeypatch, endpoint):
    """A harness session that has taken a turn refuses the next one as unrecoverable, which would end the
    agent phase at the first idle return: the goal is sent again to a new session, until the deadline."""
    (tmp_path / "base" / "gptq").mkdir(parents=True)
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": "tiny/model",
                "contamination": "none",
                "format": "w4a16",
                "budget_gpu_hours": 0.4 / 3600,
                "cards": [0],
                "run_dir": "run",
                "sandbox": "none",
                "host": "testkit:fake_host",
                "baselines": ["base/gptq"],
                "drain_timeout_s": 5,
                "judge": {"eval_max_tokens": 32768},
                "commands": None,
                "agent": {"resend_wait_s": 0.02, "backend": {"name": "dsh", "model": MODEL, "effort": None}},
            }
        )
    )
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.05))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": path})
    assert out["ended"] == "deadline" and out["failed"] == 0
    assert len(worker.prompts) >= 3  # idle -> sent again, until the deadline
    assert len(worker.sessions) == len(worker.prompts)  # and never twice to one session
    events = [json.loads(line) for line in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()]
    restarts = [e for e in events if e["ev"] == "restart"]
    assert len(restarts) == len(worker.prompts) - 1
    assert all("takes one instruction" in e["why"] for e in restarts)


async def test_a_resuming_backend_spawns_as_before(tmp_path, monkeypatch):
    """A backend whose session takes another instruction keeps it: one spawn for every goal the flow sends,
    never a new session per goal (the `restart` the one-turn backends need does not happen)."""
    (tmp_path / "base" / "gptq").mkdir(parents=True)
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": "tiny/model",
                "contamination": "none",
                "format": "w4a16",
                "budget_gpu_hours": 0.4 / 3600,
                "cards": [0],
                "run_dir": "run",
                "sandbox": "none",
                "host": "testkit:fake_host",
                "baselines": ["base/gptq"],
                "drain_timeout_s": 5,
                "judge": {"eval_max_tokens": 32768},
                "commands": None,
                "agent": {
                    "resend_wait_s": 0.02,
                    "backend": {"name": "claude", "model": "claude-opus-5-5", "effort": "high"},
                },
            }
        )
    )
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker(idle_s=0.05))
    out = await fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": path})
    assert out["ended"] == "deadline" and out["failed"] == 0
    assert len(worker.prompts) >= 3  # idle -> sent again, until the deadline
    assert len(worker.sessions) == 1  # the same session takes every goal
    events = [json.loads(line) for line in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()]
    assert not [e for e in events if e["ev"] == "restart" and "takes one instruction" in e.get("why", "")]


def _fake_host_run(
    tmp_path, monkeypatch, endpoint, seconds: float = 600, host: str = "testkit:fake_host"
) -> Path:
    """A ten-minute dsh run on the fake host, with candidate 0 at `base/gptq`: its config's path.

    `endpoint` is the stub the run's own `agent turn` check asks: a run that cannot ask one refuses to start.
    """
    (tmp_path / "base" / "gptq").mkdir(parents=True)
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": "tiny/model",
                "contamination": "none",
                "format": "w4a16",
                "budget_gpu_hours": seconds / 3600,  # ten minutes: the deadline never ends these runs
                "cards": [0],
                "run_dir": "run",
                "sandbox": "none",
                "host": host,
                "baselines": ["base/gptq"],
                "drain_timeout_s": 5,
                "judge": {"eval_max_tokens": 32768},
                "commands": None,
                "agent": {"resend_wait_s": 0.02, "backend": {"name": "dsh", "model": MODEL, "effort": None}},
            }
        )
    )

    return path


async def test_an_agent_that_never_ran_is_said_in_final_json_and_the_end_report(
    tmp_path, monkeypatch, endpoint
):
    """The live failure: the first turn said "Not logged in"; FINAL.json said "no valid hand-in; candidate 0
    as-is", as if the agent had searched and handed in nothing."""
    from hmz.flows import HarnessRefused

    path = _fake_host_run(tmp_path, monkeypatch, endpoint)

    refused = HarnessRefused("claude: Not logged in · Please run /login\nmore detail")
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([refused], idle_s=0.05))
    out = await asyncio.wait_for(
        fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": path}), timeout=60
    )
    run = tmp_path / "run"
    final = json.loads((run / "FINAL.json").read_text())
    why = (
        "the agent never ran (HarnessRefused: the agent CLI refused the run (sign-in or permission); "
        f"details in {run}/flow.jsonl); candidate 0 as-is"
    )
    assert final["agent_ran"] is False and final["reason"] == why and final["baseline"].endswith("base/gptq")
    assert "Not logged in" not in final["reason"]  # the CLI's own words stay in flow.jsonl
    assert out["agent_ran"] is False and out["ended"] == "fatal: HarnessRefused"
    report = (run / "END-REPORT.md").read_text()
    assert f"**The agent never ran.** {why}." in report and "no valid hand-in" not in report


async def test_an_agent_that_ran_has_no_agent_ran_field(tmp_path, monkeypatch, endpoint):
    path = _fake_host_run(tmp_path, monkeypatch, endpoint)
    config = yaml.safe_load(path.read_text())
    config["budget_gpu_hours"] = 0.4 / 3600  # the deadline ends it after the first turn
    path.write_text(yaml.safe_dump(config))
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([lambda: "looked around"], idle_s=0.05))
    out = await asyncio.wait_for(
        fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": path}), timeout=60
    )
    final = json.loads((tmp_path / "run" / "FINAL.json").read_text())
    assert (
        out["agent_ran"] is True
        and "agent_ran" not in final
        and final["reason"] == "no valid hand-in; candidate 0 as-is"
    )


def _home_tmpdir(tmp_path, monkeypatch):
    """A cluster layout: `$TMPDIR` under the (fake) home, with the user's own state beside it."""
    home = tmp_path / "home"
    temp = home / "tmp"
    temp.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr(tempfile, "tempdir", None)  # read `$TMPDIR` again
    assert tempfile.gettempdir() == str(temp)
    for state in (".claude/projects/x.jsonl", ".claude.json", ".dsh/key", "tmp/not-this-turn/x"):
        (home / state).parent.mkdir(parents=True, exist_ok=True)
        (home / state).write_text("the user's own")
    return home, temp


def _per_turn(temp):
    """Humanize's per-turn dirs in `temp`: the fence scratch (the dsh patch inside), a hook dir, a session home."""
    made = {}
    for key, prefix in (("scratch", "hmz-fence-"), ("hook", "humanize-hook-"), ("homes", "hmz-homes-")):
        made[key] = Path(tempfile.mkdtemp(prefix=prefix, dir=temp))
    (made["scratch"] / "hmz-dsh-x").mkdir()
    (made["scratch"] / "hmz-dsh-x" / "humanize.patch.yml").write_text("patch: {}\n")
    (made["hook"] / "hook.sock").write_text("")
    made["session"] = made["homes"] / "hmz-home-y"
    made["session"].mkdir()
    return made


def _walled_argv(cfg, monkeypatch, f, hooks):
    """The bwrap argv (up to `--`) the wall builds for one turn of fence `f`, the hook dirs recorded."""
    gates = wall._Gates()
    gates.dirs = list(hooks)
    monkeypatch.setattr(wall, "_hook_gates", lambda: gates)
    monkeypatch.setattr(wall, "pid_form", lambda _: "private")
    monkeypatch.setattr(wall, "private_bwrap_argv", lambda argv, _: argv)  # the argv, not its settings file
    hook = isolation.wall_hook()
    with isolation.gpu_wall(cfg):
        argv = hook.wrapper(f)
    return argv[: argv.index("--")]


def _wall_cfg(tmp_path, spec=f"dsh@gateway/{MODEL}", **sandbox):
    run = tmp_path / "run"
    (run / "work").mkdir(parents=True, exist_ok=True)
    return RunConfig(
        contamination="none",
        model="m",
        format="w4a16",
        budget_gpu_hours=1,
        cards=[0],
        run_dir=run,
        agent=agent(spec),
        isolation={"level": "strict"},
        sandbox={"name": "bwrap", **sandbox},
    )


def test_a_tmpdir_under_the_home_binds_this_turns_dirs_and_nothing_else(tmp_path, monkeypatch):
    """With `TMPDIR=~/tmp` the wall hid Humanize's per-turn dirs with the rest of the home, so the dsh
    runtime could not read its patch (ENOENT 160 s in) and the claude CLI's hook never reached the judge (an
    empty tools.jsonl). The scratch, the hook dir and the session home this turn made are bound back by their
    exact paths; the user's state beside them (`~/.claude*`, `~/.dsh`, another dir in `$TMPDIR`) stays hidden."""
    from hmz.coganchor import fence as hmz_fence

    home, temp = _home_tmpdir(tmp_path, monkeypatch)
    made = _per_turn(temp)
    stale = temp / "hmz-fence-not-recorded"  # a name like Humanize's, but not this turn's: never searched for
    stale.mkdir()
    cfg = _wall_cfg(tmp_path)
    f = hmz_fence.Fence(
        read=(str(cfg.workdir), str(home / ".claude"), str(home / ".dsh")),
        write=(
            str(cfg.workdir),
            str(made["session"]),
            str(temp / "not-this-turn"),
            str(home / ".claude.json"),
        ),
        online=False,
        hosts=("api.deepseek.com",),
        tmp=str(made["scratch"]),
    )
    argv = _walled_argv(cfg, monkeypatch, f, [made["hook"]])
    binds = {(argv[i], argv[i + 2]) for i in range(len(argv) - 2) if argv[i] in ("--bind", "--ro-bind")}
    real = Path(os.path.realpath(home))  # the wall binds by the home's real path

    def at(p):
        return str(real / Path(p).relative_to(home))

    assert ("--bind", at(made["scratch"])) in binds  # writable: the runtime unpacks beside its patch
    assert ("--ro-bind", at(made["hook"])) in binds  # read-only: reachable, not replaceable
    assert ("--ro-bind", at(made["session"])) in binds
    tmpfs = {(argv[i], argv[i + 1]) for i in range(len(argv) - 1)}
    assert ("--tmpfs", str(real)) in tmpfs  # the home itself: hidden
    shown = [dst for _, dst in binds]
    for hidden in (".claude", ".claude.json", ".dsh", "tmp/not-this-turn", "tmp/hmz-fence-not-recorded"):
        assert not any(Path(d).is_relative_to(real / hidden) for d in shown), hidden
    assert str(real / "tmp") not in shown  # never the whole $TMPDIR
    assert isolation.tmpdir_refusal(cfg) is None  # binding is possible: nothing refused at the start


def test_a_hook_dir_under_the_home_is_bound_for_claude_too(tmp_path, monkeypatch):
    """The second symptom of a `TMPDIR` under the home: the claude CLI's PreToolUse hook socket lies in `$TMPDIR` too. A non-dsh run
    binds no scratch (its `/tmp` is the wall's), but the hook dir the wall recorded comes back read-only."""
    from hmz.coganchor import fence as hmz_fence

    home, temp = _home_tmpdir(tmp_path, monkeypatch)
    made = _per_turn(temp)
    cfg = _wall_cfg(tmp_path, spec="claude/m")
    f = hmz_fence.Fence(
        read=(str(cfg.workdir),),
        write=(str(cfg.workdir),),
        online=False,
        hosts=("api.anthropic.com",),
        tmp=str(made["scratch"]),
    )
    argv = _walled_argv(cfg, monkeypatch, f, [made["hook"]])
    real = Path(os.path.realpath(home))
    hook = str(real / made["hook"].relative_to(home))
    assert any(argv[i : i + 3] == ["--ro-bind", hook, hook] for i in range(len(argv)))
    assert str(real / made["scratch"].relative_to(home)) not in argv  # claude's scratch is the wall's own


def test_a_per_turn_dir_that_is_not_this_users_own_is_refused_naming_tmpdir(tmp_path, monkeypatch):
    """A recorded dir is bound back only as a directory of this user's alone in `$TMPDIR`: anything else is a
    one-line refusal naming TMPDIR, never a silent drop (and never a bind of what it points at)."""
    from hmz.coganchor import fence as hmz_fence

    home, temp = _home_tmpdir(tmp_path, monkeypatch)
    made = _per_turn(temp)
    made["scratch"].chmod(0o777)
    cfg = _wall_cfg(tmp_path)
    f = hmz_fence.Fence(
        read=(str(cfg.workdir),),
        write=(str(cfg.workdir),),
        online=False,
        hosts=("api.deepseek.com",),
        tmp=str(made["scratch"]),
    )
    with pytest.raises(isolation.WallRefused, match="TMPDIR=") as err:
        _walled_argv(cfg, monkeypatch, f, [])
    assert "writable by others" in str(err.value) and "\n" not in str(err.value)
    made["scratch"].chmod(0o700)
    link = temp / "hmz-fence-link"
    link.symlink_to(home / ".dsh")  # a recorded path that is a link into the user's state
    f = dataclasses.replace(f, tmp=str(link))
    with pytest.raises(isolation.WallRefused, match="not a directory"):
        _walled_argv(cfg, monkeypatch, f, [])


def test_only_a_tmpdir_the_wall_cannot_show_is_refused_at_start(tmp_path, monkeypatch):
    """Doctor and the strict start refuse a `$TMPDIR` only where binding is impossible: under a path the wall
    masks last (`sandbox.hide`). Under the home, under the run's other dirs or in `/tmp`, it is bound."""
    home, temp = _home_tmpdir(tmp_path, monkeypatch)
    assert isolation.tmpdir_refusal(_wall_cfg(tmp_path)) is None  # under the home: bound, not refused
    assert isolation.tmpdir_refusal(_wall_cfg(tmp_path, hide=[str(home / "private")])) is None  # a sibling
    cfg = _wall_cfg(tmp_path, hide=[str(home)])
    why = isolation.tmpdir_refusal(cfg)
    assert why is not None and why.startswith(f"TMPDIR={temp} lies under {home}") and "\n" not in why
    assert "set TMPDIR to a directory of your own" in why
    with pytest.raises(isolation.WallRefused, match="TMPDIR="), isolation.gpu_wall(cfg):
        pass  # refused when the wall is put up, before any turn
    # `isolation.check`, which `qda doctor --search` and the strict preflight run, says the same one line. Its run
    # dir may not lie in a temp dir, so this one is in the package's (ignored) runs/.
    runs = Path(__file__).parents[1] / "runs" / f"t400-{os.getpid()}"
    monkeypatch.setattr(isolation, "_private_socket_unwritable", lambda _: None)  # its own subject, elsewhere
    try:
        with pytest.raises(ValueError, match="TMPDIR=") as err:
            isolation.check(_wall_cfg(runs, hide=[str(home)]))
        assert str(err.value) == why
        isolation.check(_wall_cfg(runs))  # binding possible: no refusal
    finally:
        shutil.rmtree(runs, ignore_errors=True)


class _AgentJobs:
    """The scheduler of a host, as the flow asks it: the jobs it holds, by origin."""

    def __init__(self, agent: int) -> None:
        self.agent = agent

    def list(self, origin=None):
        return [object()] * self.agent if origin == "agent" else []


async def test_an_agent_with_jobs_ran_though_its_tool_log_is_empty(tmp_path, monkeypatch, endpoint):
    """The second symptom of a `TMPDIR` under the home: the agent submitted `origin: agent` jobs, but the hook that writes tools.jsonl
    could not reach the judge, and no turn returned (the turn ended in an error here). The scheduler's jobs say
    the agent ran: FINAL.json keeps the rule's own reason, and it and END-REPORT warn that the tool log is empty."""
    from hmz.flows import HarnessRefused

    class Busy(testkit.FakeHost):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.sched = _AgentJobs(4)

    monkeypatch.setattr(testkit, "busy_host", lambda *a, **kw: Busy(*a, **kw), raising=False)
    path = _fake_host_run(tmp_path, monkeypatch, endpoint, host="testkit:busy_host")
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([HarnessRefused("the session dropped")]))
    out = await asyncio.wait_for(
        fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": path}), timeout=60
    )
    run = tmp_path / "run"
    final = json.loads((run / "FINAL.json").read_text())
    assert out["agent_ran"] is True and out["tools"] == 0 and "agent_ran" not in final
    assert final["reason"] == "no valid hand-in; candidate 0 as-is"  # the rule's own reason, as before
    [warning] = final["warnings"]
    assert warning.startswith("tool log empty: the agent worked (4 agent job(s)") and "hook" in warning
    report = (run / "END-REPORT.md").read_text()
    assert f"**Warning:** {warning}." in report and "never ran" not in report


async def test_a_turn_cut_by_the_deadline_ran_though_nothing_was_logged(tmp_path, monkeypatch, endpoint):
    """One long turn that runs to the GPU deadline has no idle return; with no tool logged it read as "the
    agent never ran". The flow saw it running (turn seconds), so it ran; nothing else says the tool log lost
    calls, so there is no warning."""
    path = _fake_host_run(tmp_path, monkeypatch, endpoint, seconds=0.5)

    async def endless():
        await asyncio.sleep(3600)
        return "never"

    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([endless]))
    out = await asyncio.wait_for(
        fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": path}), timeout=60
    )
    final = json.loads((tmp_path / "run" / "FINAL.json").read_text())
    events = [json.loads(line) for line in (tmp_path / "run" / "flow.jsonl").read_text().splitlines()]
    assert out["ended"] == "deadline" and out["tools"] == 0 and not [e for e in events if e["ev"] == "idle"]
    assert out["agent_ran"] is True and "agent_ran" not in final and "warnings" not in final


def test_agent_ran_counts_the_facts_the_flow_and_the_judge_own():
    """Any one of them is enough; an empty tool log beside jobs or written tokens with no returned turn is warned."""
    from qda_agent.flow import _agent_ran

    class Counts:
        def __init__(self, **counts):
            self.counts = counts

    none, idle, tool = Counts(), Counts(idle=1), Counts(tool=3)
    assert _agent_ran(none, none, jobs=0, seen_s=0.0, written=0) == (False, None)  # the dsh ENOENT run
    assert _agent_ran(idle, none, jobs=0, seen_s=0.0, written=0) == (True, None)  # it answered, no tools
    assert _agent_ran(none, tool, jobs=0, seen_s=0.0, written=0) == (True, None)
    assert _agent_ran(none, none, jobs=0, seen_s=1692.0, written=0) == (True, None)
    for kw in ({"jobs": 4}, {"written": 9000}):
        ran, warning = _agent_ran(none, none, **{"jobs": 0, "seen_s": 1692.0, "written": 0, **kw})
        assert ran and warning is not None and warning.startswith("tool log empty")
    # the log has its calls: no warning
    assert _agent_ran(none, tool, jobs=4, seen_s=1692.0, written=9000) == (True, None)


async def test_a_stop_signal_ends_the_turn_in_flight_and_sends_no_new_one(tmp_path, monkeypatch, endpoint):
    """A driver that swallows the interrupt and reports a failed turn would be sent a new one, turn after
    turn; the first stop signal (`host.STOPPING`) ends the turn in flight, sends no other, and closes the run."""
    from qda_agent.host import STOPPING

    monkeypatch.setattr("qda_agent.flow.STOP_POLL_S", 0.05)
    path = _fake_host_run(tmp_path, monkeypatch, endpoint)

    async def endless() -> str:
        STOPPING.set()  # the signal arrives while the turn runs
        await asyncio.sleep(3600)
        return "never"

    STOPPING.clear()
    worker = fakes.FakeAgentDriver(reply=testkit.ScriptedWorker([endless, endless, endless], idle_s=0.05))
    started = asyncio.get_running_loop().time()
    try:
        out = await asyncio.wait_for(
            fakes.run_fake(qda_search, agents={"worker": worker}, params={"config": path}), timeout=60
        )
    finally:
        STOPPING.clear()
    assert out["ended"] == "stopped" and len(worker.prompts) == 1  # one turn, ended; none sent after it
    assert asyncio.get_running_loop().time() - started < 30
