"""Where a `dsh` run's key goes, what that endpoint serves, and whether it takes the model the run names.

DeepSeek Harness is driven through its Python SDK, and the endpoint a turn goes to is the account's own, or
`DEEPSEEK_BASE_URL`, or the SDK's default, the official DeepSeek API. Each endpoint serves its own model
names -- the official API `deepseek-flash`, a gateway whatever that gateway calls the model -- and refuses a
name it does not serve only once a turn is already being taken, minutes into a run's start. So a run whose
config names no model takes the name its own endpoint serves, and the smallest turn there is is asked for
before the clock starts (`agents.Backend.turn_problem`), which is also what `qda doctor --search` asks.

The endpoint is asked rather than Humanize's catalogue: with no endpoint named, that catalogue answers with the
advisory names the SDK ships (`deepseek-v4-flash`), which are not the names the official API serves. And it is
asked in the protocol the run's own turn would be sent in: an account made by a gateway way runs on one of
pi-ai's routes, which is the account's `DSH_GATEWAY_API` for `openai-gateway` and the way itself for the two
that speak a vendor's own API (`PROTOCOLS`).
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import IO, TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from qda_agent.agents import TurnProblem

#: Where the official DeepSeek API is: the SDK's own default endpoint, and the one a run that names none uses.
OFFICIAL_BASE = "https://api.deepseek.com"
#: Its host, which a base URL names with or without a version path under it (`/v1` is the same API).
OFFICIAL_HOST = "deepseek.com"
#: The model name the official API serves its chat turns under, a name it refuses any other against.
OFFICIAL_MODEL = "deepseek-flash"
#: The name a gateway account's model was verified under in this repository (`docs/search-config.md`); a
#: gateway serves its own names, and this is the one the shipped config names for one.
GATEWAY_MODEL = "deepseek-v4.1-flash-1017"

#: The protocol a turn runs in, as `hmz.coganchor.agents.dsh` tells its runtime to send it. The stock adapter
#: is DeepSeek's own dialect of chat completions, and every gateway way is one of pi-ai's routes.
STOCK = "deepseek"
OPENAI_COMPLETIONS = "openai-completions"
OPENAI_RESPONSES = "openai-responses"
ANTHROPIC_MESSAGES = "anthropic-messages"
GOOGLE = "google"
#: Which protocol the ways that do not ask their account speak: `openai-gateway` reads `GATEWAY_API`
#: instead, and every other way (the key way, and one a local gateway holds) runs on the stock adapter.
GATEWAY_API = "DSH_GATEWAY_API"
PROTOCOLS = {"anthropic-gateway": ANTHROPIC_MESSAGES, "gemini-gateway": GOOGLE}

#: How long an endpoint is given to take the smallest turn there is. A model that thinks before it answers
#: still answers one output token, and this is one request rather than a turn of a run.
TIMEOUT_S = 60.0
#: The most of an endpoint's own refusal that is repeated back, so one long body cannot fill a report.
SAID_MAX = 400
#: The statuses an endpoint answers with that a retry takes: it is busy, throttled or too slow rather than
#: saying this run may not take a turn at all.
TRANSIENT_STATUS = frozenset({408, 429})
#: The version the Messages API refuses to answer without, as pi-ai's route sends it.
ANTHROPIC_VERSION = "2023-06-01"
#: The fewest output tokens the Responses API takes: it refuses a request that asks for fewer, so the
#: smallest turn there is costs this many rather than one.
RESPONSES_MIN_TOKENS = 16


def run_env(account: str | None = None) -> dict[str, str]:
    """The variables the endpoint is resolved from: a named account's own, or the run's environment.

    Humanize runs a turn of a named account with that account's variables and without the run's ambient
    ones, so an account's endpoint is read from the account and never from the environment beside it (a
    `DEEPSEEK_BASE_URL` in a shell profile does not route an account's key).
    """
    environ = dict(os.environ)
    if not account:
        return environ
    from hmz.coganchor import backends, providers

    held = providers.find("dsh", account)
    if (profile := backends.named("dsh")) is not None:
        for name in profile.hushes() - set(held.env if held is not None else {}):
            environ.pop(name, None)
    if held is not None:
        environ.update(held.env)
    return environ


def base_url(account: str | None = None) -> str:
    """Where the key of a `dsh` run goes: `DEEPSEEK_BASE_URL` as it is spelled, else the official API."""
    return run_env(account).get("DEEPSEEK_BASE_URL", "").strip() or OFFICIAL_BASE


def is_official(base: str) -> bool:
    """Whether a base URL is the official DeepSeek API, by its host."""
    host = (urlsplit(base).hostname or "").lower()
    return host == OFFICIAL_HOST or host.endswith(f".{OFFICIAL_HOST}")


def default_model(account: str | None = None) -> str:
    """The model a `dsh` run takes when its config names none: the one its own endpoint serves."""
    return OFFICIAL_MODEL if is_official(base_url(account)) else GATEWAY_MODEL


def protocol(account: str | None = None) -> str:
    """Which protocol a turn under this account speaks, as Humanize composes that turn's runtime.

    No account, and every account but a gateway one, runs on the stock DeepSeek adapter, which speaks
    DeepSeek's own dialect of chat completions. A gateway account runs on the pi-ai route its way names: the
    account's own `DSH_GATEWAY_API` for `openai-gateway`, and the way itself for the two whose endpoint
    speaks a vendor's API rather than one to be chosen.
    """
    if account is None:
        return STOCK
    from hmz.coganchor import providers

    held = providers.find("dsh", account)
    if held is None:
        return STOCK
    if held.way == "openai-gateway":
        return held.env.get(GATEWAY_API, "").strip() or OPENAI_COMPLETIONS
    return PROTOCOLS.get(held.way, STOCK)


def missing_account(account: str) -> str | None:
    """Why no turn can go to the account `account`, or None when there is such an account.

    A name nothing answers to is refused here rather than at the first turn: nothing else in a run reads
    the account's own variables, and a typo in one would otherwise be a run that dies after its start.
    """
    from hmz.coganchor import providers

    if providers.find("dsh", account) is None:
        return (
            f"no Humanize account called {account!r} (agent.backend.account); make the account, or set "
            "DEEPSEEK_API_KEY and drop the account"
        )
    return None


def turn_problem(model: str, account: str | None = None) -> TurnProblem | None:
    """Why the endpoint this run's key goes to will not take a turn at `model`, or None when it takes one.

    The endpoint is asked for the smallest turn there is, in the protocol this account's own turn is sent in
    (`protocol`), because that is the only way it says what it serves: it answers an unknown model with its
    own refusal and the names it does take. What it answers with is told apart: a refusal (4xx) is this run's
    config said back to it, while an endpoint that did not answer, or one that is throttled or failing (5xx,
    429), is for `agent.retries` and Humanize's own retry policy rather than a fact about the config. A run
    with no credential at all says nothing here: that is `agent sign-in`'s fact, and a run refuses it before
    any turn.
    """
    if account is not None and (why := missing_account(account)) is not None:
        return _found(why)
    environ = run_env(account)
    key = environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None
    base = environ.get("DEEPSEEK_BASE_URL", "").strip() or OFFICIAL_BASE
    return _ask(base, key, model, protocol(account))


def _found(said: str, transient: bool = False) -> TurnProblem:
    """One finding, made here rather than at the module's top: `agents` imports this module, not the other way."""
    from qda_agent.agents import TurnProblem

    return TurnProblem(said, transient)


class _SameOrigin(urllib.request.HTTPRedirectHandler):
    """A redirect this run's key may follow: the same origin, and never down to plain HTTP.

    urllib sends the headers it was given again wherever it is sent, and one of them is the credential
    (`redirect_request` drops only the content headers), so an endpoint answering `302 https://somewhere-else`
    would hand the run's key to a machine the config never named. A path corrected, or `http` upgraded to
    `https`, is the same origin and is followed; anywhere else stops here.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        moved = urllib.parse.urlsplit(newurl)
        if moved.netloc != req.host or (req.type == "https" and moved.scheme != "https"):
            return None  # not a place this run's key may go
        if code in (307, 308):
            # "the same request, at another URL", which urllib refuses to make of a POST: a probe that stopped
            # here would refuse a run whose own turn the runtime sends on. The method and the body go again.
            return urllib.request.Request(
                newurl, data=req.data, headers=dict(req.headers), origin_req_host=req.origin_req_host
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


#: What asks, with that in place of urllib's own -- which follows anywhere and takes the headers with it.
_OPENING = urllib.request.build_opener(_SameOrigin)


def _ask(base: str, key: str, model: str, spoken: str) -> TurnProblem | None:
    """Take the smallest turn at `base` and return its answer, or None when it took the turn.

    The credential goes in the header this protocol takes it in and is never repeated into what comes back:
    an endpoint that echoed it -- escaped or not -- would otherwise put a key in a run's log.
    """
    url, headers, body = _turn(base, key, model, spoken)
    request = urllib.request.Request(  # noqa: S310 -- the endpoint this run's own config names
        url, data=json.dumps(body).encode(), method="POST", headers=headers
    )
    try:
        with _OPENING.open(request, timeout=TIMEOUT_S) as answer:  # noqa: S310 -- the run's own endpoint
            answer.read(1)
    except urllib.error.HTTPError as e:
        if 300 <= e.code < 400:
            # one this run's key may not follow: it answered, and not with a turn (a retry lands on it again)
            return _found(f"{base} answered {e.code} to {_line(_redact(_location(e), key))}", False)
        said = _body(e, key)
        # 5xx, and the throttles: the endpoint failing, which is what a retry is for, not a refusal
        transient = e.code >= 500 or e.code in TRANSIENT_STATUS
        return _found(f"{base} refused a turn at {model}: {said}", transient)
    except (OSError, ValueError, http.client.HTTPException) as e:
        # its host does not answer, it is not HTTP, it answered nonsense, or the answer stopped half way. Its
        # own words, redacted and cut like an endpoint's: a status line it echoed a key into is still a key.
        said = _line(_redact(str(e), key))
        return _found(f"{base} would not take a turn: {said}", True)
    return None


def _turn(base: str, key: str, model: str, spoken: str) -> tuple[str, dict[str, str], dict]:
    """The URL, headers and body of the smallest turn in one protocol, at `base` as the account spells it.

    Each mirrors the request `hmz.coganchor.agents.dsh` composes for that way, down to the path under the
    base URL: the runtime appends `/chat/completions` or `/responses` to an OpenAI-shaped base, the Messages
    API is reached under its `/v1`, and Gemini's own API under `/models/<model>`.
    """
    said = base.rstrip("/")
    message = [{"role": "user", "content": "."}]
    bearer = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "qda-agent"}
    if spoken == OPENAI_RESPONSES:
        body = {"model": model, "input": [{"role": "user", "content": "."}]}
        body["max_output_tokens"] = RESPONSES_MIN_TOKENS  # its own floor: fewer is refused
        return f"{said}/responses", {**bearer, "Authorization": f"Bearer {key}"}, body
    if spoken == ANTHROPIC_MESSAGES:
        headers = {**bearer, "x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
        return f"{said}/v1/messages", headers, {"model": model, "max_tokens": 1, "messages": message}
    if spoken == GOOGLE:
        body = {"contents": [{"parts": [{"text": "."}]}], "generationConfig": {"maxOutputTokens": 1}}
        return f"{said}/models/{model}:streamGenerateContent", {**bearer, "x-goog-api-key": key}, body
    # The field a chat-completions route names its output limit in: pi-ai's own route defaults to
    # `max_completion_tokens` (the runtime's `maxTokensField`), which is what the reasoning models behind
    # such a gateway take -- `max_tokens` is a 400 to them -- while the stock DeepSeek adapter sends
    # `max_tokens`, its own dialect.
    limit = "max_completion_tokens" if spoken == OPENAI_COMPLETIONS else "max_tokens"
    body = {"model": model, "messages": message, limit: 1, "stream": False}
    return f"{said}/chat/completions", {**bearer, "Authorization": f"Bearer {key}"}, body


def _location(e: urllib.error.HTTPError) -> str:
    """Where a redirect this run did not follow points, or a word for one that names nowhere."""
    return str(e.headers.get("Location") or "nowhere") if e.headers is not None else "nowhere"


def _body(e: urllib.error.HTTPError, secret: str) -> str:
    """What a refused turn's body says, or the status line where the body is not there whole.

    A gateway that claims a length and closes before it has sent it raises `IncompleteRead` out of `read()`,
    which is the transport failing rather than an answer: the status line stands in for it, so the finding is
    reported as the transient failure it is.
    """
    try:
        return _said(e.read(), secret)
    except (OSError, http.client.HTTPException) as bad:
        return _line(_redact(str(e.reason or bad), secret))


def _said(body: bytes, secret: str) -> str:
    """One line of an endpoint's refusal, with the credential taken back out of it.

    Redacted after the body is read, not in the body: one may carry the credential escaped (`\\u0073k-...`),
    which reading it as JSON turns back into the credential itself.
    """
    text = body.decode("utf-8", "replace")
    try:
        said = json.loads(text)
    except ValueError:
        return _line(_redact(text, secret))
    if isinstance(said, dict):
        error = said.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return _line(_redact(error["message"], secret))
        if isinstance(said.get("message"), str):
            return _line(_redact(said["message"], secret))
    # A shape this has no field for: its words are taken from the JSON it read rather than from the body's
    # bytes, so a credential the body spelled escaped is the credential again and is redacted with the rest.
    return _line(_redact(json.dumps(said, ensure_ascii=False), secret))


def _redact(text: str, secret: str) -> str:
    """The credential out of one line of an endpoint's own words, and never the other way round."""
    return text.replace(secret, "<redacted>") if secret else text


def _line(text: str) -> str:
    """Whitespace-collapsed and cut to length, so a refusal is one line of a report."""
    return " ".join(text.split())[:SAID_MAX]
