"""Write the kernels a vLLM worker binds to ``$QDA_KERNEL_PROBE/<pid>.json`` once its model has loaded.

Put this directory on ``PYTHONPATH`` of a ``vllm serve`` command: the command line itself stays as printed.
`qda_kl.vllm_hooks.bound_kernels` reads the model, the same reading ``qda-kl kl`` records in ``score.json``.
"""

import importlib.abc
import importlib.util
import json
import os
import sys

_WORKER = "vllm.v1.worker.gpu_worker"


def _patch(module):
    load = module.Worker.load_model

    def load_model(self, *a, **k):
        out = load(self, *a, **k)
        from qda_kl.vllm_hooks import bound_kernels

        path = os.path.join(os.environ["QDA_KERNEL_PROBE"], f"{os.getpid()}.json")
        with open(path, "w") as f:
            json.dump(bound_kernels(self.get_model()), f)
        return out

    module.Worker.load_model = load_model


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _WORKER:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("QDA_KERNEL_PROBE"):
    sys.meta_path.insert(0, _Finder())
