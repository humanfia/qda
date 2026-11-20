"""Argument parsing for the host side of the four commands: errors and help become replies, not exits."""

from __future__ import annotations

import argparse
from typing import NoReturn


class Usage(Exception):
    """A parser wants to exit: `status` and the text it would have printed."""

    def __init__(self, status: int, text: str) -> None:
        super().__init__(text)
        self.status, self.text = status, text


class Parser(argparse.ArgumentParser):
    """An `ArgumentParser` that raises `Usage` instead of printing and exiting (it runs in a server thread)."""

    def print_help(self, file: object = None) -> NoReturn:  # noqa: ARG002
        """Raise the help text as a successful reply."""
        raise Usage(0, self.format_help())

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        """Raise instead of exiting."""
        raise Usage(status, message or "")

    def error(self, message: str) -> NoReturn:
        """Raise the usage and the error as a refusal."""
        raise Usage(2, f"{self.format_usage()}{self.prog}: error: {message}\n")


UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0}


def duration(text: str) -> float:
    """A positive duration as seconds: `90s`, `20m`, `1.5h`, or a bare number of minutes."""
    unit = text[-1:] if text[-1:] in UNITS else "m"
    try:
        v = float(text[:-1] if text[-1:] in UNITS else text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: a duration like 90s, 20m or 1.5h") from None
    if not v > 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v * UNITS[unit]


def names(text: str) -> list[str]:
    """A comma-separated list of job names."""
    return [x for x in text.split(",") if x]
