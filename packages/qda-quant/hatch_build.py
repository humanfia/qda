"""Build qda-quant's linux wheel with the shipped C quantizer.

The library is one plain C shared object, `qda_quant/gguf/_quant.so`, so it has no CPython ABI: the runtime
loads it through ctypes exactly as it loads a user-built `libggml-base`. There is no new Python build dependency
and no network: the hook calls the system C compiler, so `pip wheel --no-index --find-links <wheelhouse>` builds
offline. Without a compiler, or off linux, the wheel stays pure Python and the numpy port is the fallback.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

#: The wheel path of the compiled library; `qda_quant.gguf.kernel` looks for it beside this module.
LIB = "qda_quant/gguf/_quant.so"
SOURCE = "src/qda_quant/gguf/_quant.c"
#: Fixed so the bytes match the numpy reference on every platform: no native tuning (`GGML_NATIVE=OFF`), no FMA
#: contraction (aarch64 has FMA by default, and the reference is explicit multiply-then-add), and `NDEBUG` as in
#: llama.cpp's release build (the reference's asserts are debug-only; the overflow cases the golden bytes pin
#: are undefined in C and must not abort).
FLAGS = ("-O3", "-DNDEBUG", "-fPIC", "-fno-fast-math", "-ffp-contract=off", "-shared")


class CustomBuildHook(BuildHookInterface):
    """Compile `_quant.c` into the wheel when the target and toolchain allow it."""

    def initialize(self, _version: str, build_data: dict[str, Any]) -> None:
        """Compile the library and hand it to the wheel, or leave the wheel pure Python."""
        if self.target_name != "wheel" or sys.platform != "linux":
            return
        source = Path(self.root) / SOURCE
        if not source.is_file():
            return
        cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if cc is None:
            self.app.display_warning(f"{SOURCE}: no C compiler; the wheel ships the numpy fallback only")
            return
        out = Path(tempfile.mkdtemp(prefix="qda-quant-c-")) / "_quant.so"
        cmd = [
            cc,
            *FLAGS,
            f'-DQDA_VERSION="{self.metadata.version}"',
            "-o",
            str(out),
            str(source),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"{SOURCE}: the C quantizer did not build: {e.stderr.strip()}") from e
        build_data["force_include"][str(out)] = LIB
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-linux_{platform.machine().lower()}"
