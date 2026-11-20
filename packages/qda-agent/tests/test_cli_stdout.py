"""Every QDA command line flushes stdout per line (`qda_quant.io.line_buffered_stdout`), so `> log` shows progress."""

import inspect

from qda_agent import cli as agent_cli
from qda_kl import cli as kl_cli
from qda_quant import cli as quant_cli


def test_every_cli_sets_it_first():
    for mod in (quant_cli, kl_cli, agent_cli):
        assert "line_buffered_stdout()" in inspect.getsource(mod.main), mod.__name__
