"""A plain `pip install` has no vLLM, datasets, lm-eval or llm-compressor: importing the packages must not need them."""

import subprocess
import sys


def test_every_module_imports_without_a_heavy_optional_dependency():
    """Import every module of the three packages and check that the heavy extras stayed out.

    Each is imported only by the feature that needs it (`selfgen`, `hf`, `qda eval`, the llm-compressor
    comparison); a module-level import would make a plain install of the package fail.
    """
    code = (
        "import importlib, pkgutil, sys\n"
        "for pkg in ('qda_quant', 'qda_kl', 'qda_agent'):\n"
        "    module = importlib.import_module(pkg)\n"
        "    for info in pkgutil.walk_packages(module.__path__, pkg + '.'):\n"
        "        if 'vendor' not in info.name and 'third_party' not in info.name:\n"
        "            importlib.import_module(info.name)\n"
        "heavy = {'vllm', 'datasets', 'lm_eval', 'llmcompressor'}\n"
        "print(sorted({name.split('.')[0] for name in sys.modules} & heavy))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]", done.stdout
