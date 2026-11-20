"""Whether this node can reach the agent's model API the way the run's turns will.

A node with no route to the API ran its search to the deadline with zero tool calls and exit 0. This asks, before
the run starts, the one question a turn's first request asks: can a TCP connection be opened to the API's host,
by the route the turn takes? Nothing else is sent: no TLS, no request, no credential, nothing billable.

The host is Humanize's for the backend's CLI (`hmz.coganchor.backends.named(cli).hosts`), or the gateway the
environment points it at (`Profile.endpoint`, e.g. `ANTHROPIC_BASE_URL`), in the environment the flow gives the
CLI (`flow._endpoint_env` removes the endpoint variables a sign-in by file or account must not follow). The route
is the run's: a turn inside the bwrap wall goes out through Humanize's fence proxy on this host
(`isolation.hostnet`), which opens its upstream connection directly (hmz 0.2.2, `hmz/coganchor/fence/proxy.py`;
read in the code, not observed on a proxy-only node), so it is probed by a direct connection; any other turn is
the CLI's own process, which uses ``HTTPS_PROXY`` unless ``NO_PROXY`` covers the host (as `urllib.request` reads
them).
"""

from __future__ import annotations

import os
import socket
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

from qda_quant.exit import ENVIRONMENT
from qda_quant.guards import SEARCH_LAUNCH, Guard, Problem

if TYPE_CHECKING:
    from qda_agent.config import RunConfig

__all__ = ["AGENT_GUARDS", "api_endpoint", "api_route_problem", "walled"]

#: How long the probe waits for a connection or a proxy's answer, in seconds.
WAIT_S = 10.0


def walled(cfg: RunConfig) -> bool:
    """Whether the run's turns run inside the bwrap wall, whose only way out is the fence proxy (`isolation.wall`)."""
    level = getattr(getattr(cfg, "isolation", None), "level", None)
    return cfg.sandbox.name == "bwrap" and level != "none"


def api_endpoint(cfg: RunConfig) -> tuple[str, int] | None:
    """The host and port a turn of the run's agent CLI sends its first request to; None for no CLI or none known."""
    from hmz.coganchor.backends import named

    from qda_agent import flow

    cli = cfg.agent.backend.cli
    profile = named(cli) if cli else None
    if profile is None or not profile.hosts:
        return None
    with flow._endpoint_env(cfg):  # the environment the flow gives the CLI
        gateway = os.environ.get(profile.endpoint, "") if profile.endpoint else ""
    if gateway:
        url = urllib.parse.urlsplit(gateway if "://" in gateway else f"https://{gateway}")
        if url.hostname:
            return url.hostname, url.port or (80 if url.scheme == "http" else 443)
    return profile.hosts[0], 443


def _proxy(host: str) -> str | None:
    """The proxy the CLI's own process sends `host` through: ``HTTPS_PROXY`` unless ``NO_PROXY`` covers it."""
    proxies = urllib.request.getproxies_environment()
    proxy = proxies.get("https")
    # `proxy_bypass_environment` is the stdlib's NO_PROXY rule; typeshed does not list it
    bypass = getattr(urllib.request, "proxy_bypass_environment")  # noqa: B009
    if not proxy or bypass(host, proxies):
        return None
    return proxy


def api_route_problem(host: str, port: int, *, direct: bool) -> str | None:
    """Why `host:port` cannot be reached by that route, or None: one TCP connect, or one ``CONNECT`` to the proxy."""
    proxy = None if direct else _proxy(host)
    if proxy is None:
        try:
            socket.create_connection((host, port), timeout=WAIT_S).close()
        except OSError as e:
            how = "by a direct connection" if direct else "by a direct connection (HTTPS_PROXY is unset)"
            return f"no route to {host}:{port} from this node {how}: {e.strerror or type(e).__name__}"
        return None
    url = urllib.parse.urlsplit(proxy if "://" in proxy else f"http://{proxy}")
    where = f"{url.hostname}:{url.port or 80}"
    try:
        with socket.create_connection((url.hostname or "", url.port or 80), timeout=WAIT_S) as s:
            s.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
            status = s.recv(256).split(b"\r\n", 1)[0].decode(errors="replace")
    except OSError as e:
        return f"the HTTPS_PROXY at {where} cannot be reached: {e.strerror or type(e).__name__}"
    code = status.split()[1] if len(status.split()) > 1 else ""
    if (
        code == "200" or code == "407"
    ):  # 407: the proxy is there and asks for credentials, which this never sends
        return None
    return f"the HTTPS_PROXY at {where} does not connect to {host}:{port}: {status or 'no answer'}"


def _api_route(cfg: RunConfig) -> Problem | None:
    found = api_endpoint(cfg)
    assert found is not None  # `applies`
    host, port = found
    direct = walled(cfg)
    if (why := api_route_problem(host, port, direct=direct)) is None:
        return None
    fix = (
        f"run on a node with a direct route to {host}:{port}"
        if direct
        else "set HTTPS_PROXY (and NO_PROXY) to a proxy that reaches it, or run on a node that can"
    )
    return Problem(why, fix, ENVIRONMENT)


def _no_api(cfg: RunConfig) -> str | None:
    return None if api_endpoint(cfg) is not None else "no agent CLI with a known API host"


def _reached(cfg: RunConfig) -> str:
    found = api_endpoint(cfg)
    route = (
        "by a direct connection (the wall's turns)"
        if walled(cfg)
        else ("through HTTPS_PROXY" if found and _proxy(found[0]) else "by a direct connection")
    )
    return f"{found[0]}:{found[1]} {route}" if found else ""


#: The agent's own preconditions: walked by `qda-agent run` before it creates anything, when the run drives a real
#: CLI (`cli.execute` without a supplied worker), and by `qda doctor --search`.
AGENT_GUARDS: tuple[Guard[RunConfig], ...] = (
    Guard(
        "agent API reachable",
        frozenset({SEARCH_LAUNCH}),
        _api_route,
        "qda-agent run, before it creates anything (cli.execute); a turn's first request otherwise",
        applies=_no_api,
        held=_reached,
    ),
)
