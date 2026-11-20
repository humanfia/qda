"""The host's half of the wall's network: a proxy per allowlist, and the socket a turn reaches it through.

The wall's turns run with `--unshare-net` (`wall`): inside there is loopback and nothing else. The model
API is still the only thing that leaves the machine, through Humanize's `Proxy`, which passes only the
hosts of the turn's fence -- but the proxy is on the host, and the one way to it is a unix socket the wall
binds in read-only. `live_state` starts a relay inside that carries a loopback connection to the socket; this
module accepts on the socket and carries it to the proxy.

A socket and its proxy belong to one allowlist (`Fence.hosts`), made on first use: a turn of one backend
cannot reach another backend's hosts, and an allowlist is served once however many turns name it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import socket
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from hmz.coganchor.fence import Proxy

from qda_agent import rpc

if TYPE_CHECKING:
    from collections.abc import Sequence

_log = logging.getLogger(__name__)

#: How much one side of a relayed connection reads before it hands the bytes to the other.
_CHUNK = 64 * 1024


def _pump(src: socket.socket, dst: socket.socket) -> None:
    """Carry what `src` sends to `dst` until it closes, then half-close `dst`."""
    try:
        while data := src.recv(_CHUNK):
            dst.sendall(data)
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            dst.shutdown(socket.SHUT_WR)


class NetBridge:
    """The host's proxies and sockets for one wall: one unix socket per allowlist, bound into the sandbox."""

    def __init__(self) -> None:
        # In the private base, as the RPC socket's directory is: short enough for a unix socket, and under
        # a temp dir or $XDG_RUNTIME_DIR, which the wall masks -- so the agent cannot reach or rename it.
        self.base = Path(os.path.realpath(tempfile.mkdtemp(prefix="qda-net-", dir=rpc.private_base())))
        self._paths: dict[tuple[str, ...], Path] = {}
        self._proxies: list[Proxy] = []
        self._listeners: list[socket.socket] = []
        self._lock = threading.Lock()

    def socket(self, hosts: Sequence[str]) -> Path:
        """The unix socket that carries connections to a proxy passing `hosts`, made on first use.

        Raises:
            ValueError: The socket's path does not fit a unix socket (a very long private base).
        """
        key = tuple(dict.fromkeys(hosts))
        with self._lock:
            if (path := self._paths.get(key)) is not None:
                return path
            path = self.base / f"p{len(self._paths)}.sock"
            if not rpc.fits(path):
                raise ValueError(
                    f"the wall's proxy socket {path} is too long for a unix socket ({rpc.SUN_PATH_MAX} "
                    "bytes): set $XDG_RUNTIME_DIR to a shorter directory"
                )
            proxy = Proxy(key)
            proxy.start()
            srv = socket.socket(socket.AF_UNIX)
            try:
                srv.bind(str(path))
                srv.listen(128)
            except OSError:
                srv.close()
                proxy.stop()
                raise
            self._proxies.append(proxy)
            self._listeners.append(srv)
            self._paths[key] = path
            threading.Thread(target=self._serve, args=(srv, proxy), name="qda-wall-net", daemon=True).start()
            return path

    def _serve(self, srv: socket.socket, proxy: Proxy) -> None:
        """Accept what the sandbox's relay sends, one thread per connection."""
        while True:
            try:
                client, _ = srv.accept()
            except OSError:
                return  # stopped: `stop` closed the listening socket
            threading.Thread(target=self._carry, args=(client, proxy), daemon=True).start()

    def _carry(self, client: socket.socket, proxy: Proxy) -> None:
        """Carry one connection from the sandbox to `proxy` on the host's loopback."""
        try:
            upstream = socket.create_connection(("127.0.0.1", proxy.port))
        except OSError:
            client.close()
            return
        with client, upstream:
            back = threading.Thread(target=_pump, args=(client, upstream), daemon=True)
            back.start()
            _pump(upstream, client)
            back.join()

    def stop(self) -> None:
        """Stop accepting, stop the proxies, and remove the sockets' directory."""
        for srv in self._listeners:
            with contextlib.suppress(OSError):
                srv.close()
        for proxy in self._proxies:
            proxy.stop()
        shutil.rmtree(self.base, ignore_errors=True)
