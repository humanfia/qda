"""Wait for a local server to be ready."""

from __future__ import annotations

import time
from typing import Any


def wait_ready(base_url: str, process: Any = None, timeout: float = 900.0) -> None:
    """Wait for a served model's health endpoint, preserving its process and timeout errors."""
    import requests

    deadline = time.monotonic() + timeout
    url = base_url.rstrip("/").removesuffix("/v1/completions") + "/health"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"served model: server exited with {process.returncode} before it was healthy")
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2.0)
    raise RuntimeError(f"served model: {url} was not healthy within {timeout:.0f} s")
