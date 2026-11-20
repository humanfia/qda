"""Host-side checks of a checkpoint before it is scored or handed in.

Paper §4.1, admissible recipes: a checkpoint must keep the public baselines' quantized modules, bit-width
and format and run on their kernels. So a checkpoint the judge serves:

- holds regular files only, no hidden files or directories, no code or pickle; weights only as top-level `*.safetensors` shards that parse
  as safetensors and agree with the shard index; every other file is `config.json`, the index, an inert
  file (`INERT`), or the original model's (anchor's) own file at the same path, byte for byte;
- has the anchor's tokenizer files, chat templates, processor files and generation config, byte for byte
  (`SAME_AS_ANCHOR`; `additional_chat_templates/` holds exactly the anchor's files);
- has a `config.json` without `auto_map`, its architecture equal to the anchor's apart from `CONFIG_FREE`
  (the quantization config and the save's dtype and bookkeeping): the same architecture;
- keeps the serving engine's format facts: every quantized module's format is in `judge.format_menu`
  (`DEFAULT_MENU` is the W4 tier's: bf16, W8A16, W4A16; a W3 search adds W3A16), q, k and v of one
  attention share a format, gate and up share a format, a routed expert bank shares a format (group size
  and symmetry included: the engine fuses each unit into one kernel), and a layer has one activation
  format (`format_problem`). A module's format is the one its quantization config declares for it; group
  size and symmetry are free between whole units, so a per-layer mix of the menu's formats is admitted.
  No reference checkpoint's tensor inventory or quantization config is compared.

Not enforced (the historical search's policy, not the format): the old judge refused a checkpoint whose
records say a GSQ optimisation made it in-round; the release judge does not.

A baseline as-is is recognized by `tensor_fingerprint`: the loaded tensors' names, dtypes, shapes and
bytes, whatever the shard names, the sharding or the header metadata, unscored tensors left out. So that
one weight moved by one ulp is not another checkpoint, `near_copy` compares the quantized tensors' packed
codes, scales and zero points with every baseline's again: a candidate whose codes are all a baseline's,
or whose weights are within `NEAR_COPY` of it, is that baseline, not the agent's result.

The checks are split by subject, each module holding the code it had here: `tensors` (shards, index,
inventory, fingerprint, near copies), `formats` (the serving engine's format facts) and `model_files` (the
file policy, the anchor's files and trusted model directories). This module re-exports them, so a caller
keeps importing the same names from here.
"""

from __future__ import annotations

from ..freeze import INDEX as INDEX  # re-exported: callers read `checks.INDEX`
from ..freeze import trusted_model_files
from .formats import (
    DEFAULT_MENU,
    HEAD_16BIT,
    HEAD_16BIT_TYPES,
    format_key,
    format_problem,
    map_head_format,
    map_menu_problem,
    stored_formats,
)
from .model_files import INERT as INERT  # re-exported: docs and callers read `checks.INERT`
from .model_files import anchor_problem, files_problem, trusted_files_problem
from .tensors import MAX_HEADER as MAX_HEADER
from .tensors import NEAR_COPY as NEAR_COPY  # re-exported: docs read `checks.NEAR_COPY`
from .tensors import UNSCORED as UNSCORED  # re-exported: callers read `checks.UNSCORED`
from .tensors import (
    CheckpointError,
    inventory,
    loaded,
    module_tensors,
    near_copy,
    tensor_fingerprint,
)
from .tensors import _norms as _norms
from .tensors import _norms_slow as _norms_slow

__all__ = [
    "DEFAULT_MENU",
    "HEAD_16BIT",
    "HEAD_16BIT_TYPES",
    "CheckpointError",
    "anchor_problem",
    "files_problem",
    "format_key",
    "format_problem",
    "map_head_format",
    "map_menu_problem",
    "inventory",
    "loaded",
    "module_tensors",
    "near_copy",
    "stored_formats",
    "tensor_fingerprint",
    "trusted_files_problem",
    "trusted_model_files",
]
