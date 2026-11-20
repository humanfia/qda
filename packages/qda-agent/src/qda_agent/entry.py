"""Start the CLI with foreground signal handling before its configuration imports."""

import signal
import threading


def main() -> int:
    """Run `qda-agent`, accepting Ctrl-C even when started in the background."""
    if (
        threading.current_thread() is threading.main_thread()
        and signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    ):
        signal.signal(signal.SIGINT, signal.default_int_handler)
    from qda_agent.cli import main as cli_main

    return cli_main()
