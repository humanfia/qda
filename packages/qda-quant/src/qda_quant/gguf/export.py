"""The ``gguf`` exporter: a ``q4k`` build, with the default Q4_K_M mix or explicit tensor types.

llama.cpp's own converter makes a bf16 template of the source checkpoint (``convert_hf_to_gguf.py --outtype bf16``),
which fixes every KV pair, tensor name and the row permutes it applies; a template the user already has can be
given instead (``template``). The exporter then fills each tensor:

* a tensor the Q4_K_M rule (`rule`) stores as Q4_K, and that the build quantized: the build's codes and scales,
  encoded as Q4_K records, rows in the template's order;
* any other quantized tensor (Q6_K, or Q4_K the build kept in bf16 such as ``token_embd``): llama.cpp's own
  quantizer (`kernel`) applied to the template's values, as ``llama-quantize`` does;
* a tensor a step changed (AWQ folded its scales into a norm, an offline rotation rewrote a weight): the
  build's value, in the converter's own basis for that tensor -- derived per tensor from the template against
  the source (the same values, a constant offset such as Qwen's ``norm.weight + 1``, or a row permutation),
  and refused by name when it is none of those. Model-level tensors (the embedding, the final norm, the head)
  and declared source blocks (an MTP head) are included;
* everything else: the template's bytes.

With the default mix, RTN is byte-identical to ``llama-quantize`` on the template. With GPTQ, Q4_K tensors carry the
solved codes while Q6_K tensors are llama.cpp's round-to-nearest of the original weights.
``tensor_types`` overrides the default mix. Q4_K tensors use solved codes when present; all other types use the
source template's values.

Rows: the build's codes are in the source checkpoint's layout and the template's rows may be permuted (llama's
q/k permute). Q4_K blocks run along a row, so a row's record does not depend on its position; each template row is
matched to the source row with the same bf16 bytes. Identical rows have identical records, so which of them a row
is matched to does not change a byte; matches are assigned in order, which keeps the result stable.
"""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from pydantic import Field

from ..output.exporters import Exporter
from ..step import Refused
from . import file, rule
from . import kernel as kern
from .candidates import mix_formats
from .columns import template_columns
from .iq4 import IQ4NL_FORMAT, IQ4NL_GRID, IQ4XS_FORMAT, IQ4XS_GRID
from .llama_cpp import ENV, PINNED, checkout_command, commit_of, locate
from .packing import q4k_bytes as q4k_bytes
from .packing import q6k_bytes as q6k_bytes
from .packing import row_order as row_order
from .q3k import Q3K_FORMAT, Q3K_GRID
from .q4k import IMATRIX, Q4K_FORMAT, Q4K_GRID
from .q5k import Q5K_FORMAT, Q5K_GRID
from .q6k import Q6K_FORMAT, Q6K_GRID
from .tensor_types import TensorTypeRule, dropped_head_hint, resolve_types
from .writer import OUT_NAME as OUT_NAME
from .writer import GGUFWriter as GGUFWriter
from .writer import make_template as make_template

if TYPE_CHECKING:
    from ..engine.planning import Plan
    from ..recipe import Recipe


#: Files llama.cpp's CMake writes for the shared ``ggml-base``, on each platform.
_GGML_BASE = ("libggml-base.so*", "libggml-base.*.dylib", "ggml-base.dll")


def libggml_base(root: str | Path) -> Path | None:
    """The built ``libggml-base`` in `root`, its ``build*/bin`` or its ``build*/lib``, or None.

    A build tree's own directory holds the library (``build/bin``); passed a checkout, the first build tree in
    sorted order is read, so a checkout with several is read the same way every time. llama.cpp's CMake puts
    shared libraries in ``build/bin``; a build tree made with another output layout has them in ``build/lib``.
    """
    root = Path(root)
    dirs = [root, *(build / d for build in sorted(root.glob("build*")) for d in ("bin", "lib"))]
    for d in dirs:
        for pattern in _GGML_BASE:
            found = sorted(d.glob(pattern))
            if found:
                return found[0]
    return None


def build_kernel_command(root: str | Path) -> str:
    """The one command that builds ``libggml-base`` (shared) from a llama.cpp checkout."""
    root = Path(root)
    return (
        f"cmake -S {root} -B {root / 'build'} -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON "
        f"&& cmake --build {root / 'build'} --target ggml-base -j"
    )


def template_refusal(template: str | Path | None) -> str | None:
    """Why a template the user named cannot be used, or None."""
    if template is None:
        return None
    path = Path(template)
    if not path.is_file():
        return f"template {path} is not a file"
    return None


class GGUF(Exporter):
    """A llama.cpp Q4_K_M file, byte-identical to ``llama-quantize`` when the method is plain rounding.

    Attributes:
        llama_cpp: a llama.cpp checkout; its ``convert_hf_to_gguf.py`` makes the bf16 template, and its commit is
            recorded (parity is pinned to one). None takes ``$QDA_LLAMA_CPP``, and neither is needed when
            `template` is given.
        workers: threads of llama.cpp's quantizers; None takes every core with llama.cpp's kernel, as ``llama-quantize``
            does, and up to 32 with the Python port. The bytes do not depend on it.
        template: a bf16 GGUF of the source the user already has (``convert_hf_to_gguf.py --outtype bf16``);
            given, the converter is not run. It must be the file the converter would write.
        libggml: a built ``libggml-base``, or a directory that holds one (its build directory, or the llama.cpp
            checkout), to quantize with llama.cpp's own kernel; None reads the one the ``llama_cpp`` checkout
            built, if any. Without one the build uses the Python port, which writes the same bytes several times
            slower.
        layer_pass: run the layer pass even for a recipe that only rounds (`writes_alone`). The slow reference
            path: it loads the model and writes the same bytes.
        tensor_types: ordered rules ``{pattern, type}`` using GGUF tensor names or ``re:`` patterns matched
            from the start. Last match wins. Empty keeps the default Q4_K_M mix. Types other than Q4_K
            use the source template's values, not the build's solved codes.
        groups: a derived-group cache (`qda_quant.gguf.groups`) written beside the config, naming the model's
            solve units and the GGUF tensors each one stores. `type_map` expands through it at config time.
        type_map: a candidate's ``{unit id: ggml type}`` over the solve units in `groups`. It expands to
            `tensor_types` (one exact-name rule per stored tensor) and to ``format_overrides`` (a unit stored
            as Q6_K or Q5_K must have been solved in that format), which the config then carries as if it had
            written them; both tables are recorded in ``resolved.yaml``. Refused when `groups` is missing, when
            `tensor_types` is also given, or when a unit or a type does not validate.
        imatrix: an importance matrix from ``llama-imatrix``, measured on this build's template, to fit the
            tensors written from it (the Q6_K promotions and any fallback). The build's own codes are
            unaffected. It needs `libggml`; missing entries use the plain fit. Q8_0 fitting ignores weights,
            as in llama.cpp, and this is recorded in `build.json`.
        mtp: write the multi-token-prediction (MTP/NextN) head: the template's head blocks when it has them,
            the converter's default. False leaves them out of the file, as ``convert_hf_to_gguf.py --no-mtp``
            does; a build that changed one of their tensors is refused rather than quietly dropping the change.
    """

    #: A GGUF names a type per tensor, so it can hold units the build stored in another format -- but only
    #: the Q6_K the mix already gives them, which `_format_refusal` and `_add` check.
    supports_formats: ClassVar[bool] = True
    #: llama.cpp reads a routed bank's gate, up and down as three tensors, so down may be solved in its own format.
    splits_banks: ClassVar[bool] = True
    #: Served by llama.cpp, whatever card it runs on, so this output names its own target and a config that
    #: names a different one is refused rather than having it quietly ignored.
    target_name: ClassVar[str] = "llama_cpp"

    llama_cpp: str | None = None
    workers: int | None = Field(default=None, ge=1)
    template: str | None = None
    libggml: str | None = None
    layer_pass: bool = False
    tensor_types: list[TensorTypeRule] = Field(default_factory=list)
    groups: str | None = None
    type_map: dict[str, str] = Field(default_factory=dict)
    imatrix: str | None = None
    mtp: bool = True

    def _template_view(self) -> file.Template | None:
        """The template this build reads: without the MTP head when ``mtp: false``, as the writer writes it.

        Every type decision has to read the same view. The Q4_K_M rule schedules some promotions (Q6_K on the
        first and last ``attn_v``, Q5_K on some ``ffn_down``) from the block count, so a head the file does not
        hold would move every later block's type by one and drop the solved codes of tensors the file stores in
        another type.
        """
        if self.template is None:
            return None
        tp = file.Template(self.template)
        return tp if self.mtp else tp.without_mtp()

    def stored_format_rules(self, graph: Any) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
        """The mix's own solved formats: a unit the Q4_K_M rule stores Q6_K or Q5_K is solved in that format.

        The mix -- not the recipe -- decides each tensor's stored type, and a Q6_K tensor writes Q6_K codes
        only. Deriving the pairs from the template before the solve is what keeps a plain recipe's codes
        instead of having the export drop them. Nothing is derived without a template: the mix's types are
        the template's tensor types, and a build that converts its own template has none to read yet.
        """
        if self.template is None or not Path(self.template).is_file():
            return [], {}
        view = self._template_view()
        if view is None:  # unreachable: the template above is a readable file
            return [], {}
        return mix_formats(view, graph, self.tensor_types)

    def writes_alone(self, plan: Plan, recipe: Recipe) -> bool:  # noqa: ARG002
        """True for the one recipe whose file is a single ``q4k`` fit of each weight: plain ``rtn``, one step.

        `GGUFWriter.finish` quantizes every tensor from the template once. That equals the layer pass only when the
        recipe does the same: one round-to-nearest step with the default ``minmax`` clip. A second step (even another
        ``rtn``) re-quantizes the first step's dequantized weights, and the bytes differ; any other method searches
        or solves. Those take the layer pass.

        Fused expert banks use the same per-expert row layout in both paths.

        A build that stores units in another format (``format_overrides``) always takes the layer pass: the
        shortcut quantizes every tensor from the template, which is what those overrides exist to replace, and
        it would skip the checks `_add` makes on each one.
        """
        from ..steps.rtn import RTN

        return (
            not self.layer_pass
            # a recipe's override codes are written by the layer pass, which this skips; a format the mix
            # itself requires is the same fit this path takes from the template
            and not (plan.formats.keys() - getattr(plan, "derived_units", frozenset()))
            and len(recipe.steps) == 1
            and type(step := recipe.steps[0]) is RTN
            and step.clip == "minmax"
        )

    def solve_settings(self) -> dict[str, dict[str, Any]]:
        """Settings without what decides only the stored file: types, template, imatrix, workers, mtp.

        A tensor's stored type picks the encoder of the codes the solve already made (or the template fit that
        replaces them); the solve formats come from ``format_overrides``, which the layer cache keys by layer.
        """
        kept = self.model_dump(exclude={"tensor_types", "template", "imatrix", "workers", "mtp"})
        return {self.name: kept}

    def layer_context(self, layer: torch.nn.Module, graph: Any):  # noqa: ARG002
        """Calibrate and solve GDN output projections in the converter's column order."""
        return template_columns(layer)

    def _library(self) -> Path | None:
        """The ``libggml-base`` this build would load: the one `libggml` names, else the checkout's own, or None.

        Without a checkout there is no library to discover.
        """
        if self.libggml:
            named = Path(self.libggml)
            return libggml_base(named) if named.is_dir() else named
        root = locate(self.llama_cpp)
        return libggml_base(root) if root is not None else None

    def _needs_weighted(self, recipe: Recipe | None) -> bool:
        """True when a step fits its scales with ``clip: imatrix``, llama.cpp's weighted fit with no Python port."""
        from ..step import QuantStep

        return recipe is not None and any(
            isinstance(s, QuantStep) and s.clip_search() is IMATRIX for s in recipe.steps
        )

    def prepare(self, recipe: Recipe | None = None) -> str | None:
        """Load the kernel this build quantizes with; return the one line `qda plan` and the build log print.

        The kernel is process-wide (`qda_quant.gguf.kernel.use_ggml`), so it must be set before the first layer.
        An explicit `libggml` wins; otherwise the C quantizer shipped in the wheel is used, then one discovered
        in the checkout, then the Python port. A plain-only shipped kernel (``_quant.so`` without
        ``quantize_q4_K``) yields to the checkout's when `recipe` needs the weighted ``clip: imatrix`` fit, which
        has no Python port; every other recipe keeps the shipped kernel. A library that cannot be loaded is
        refused when it was named; a discovered or shipped one is skipped, and the build falls back to the Python
        port. A library built from another commit than the converter's is used, and said so. A converter at
        another commit than the pinned one is used, and said so.

        Raises:
            Refused: the library `libggml` names is not there, does not load, or is not llama.cpp's quantizer.
        """
        note = self._prepare(recipe)
        if note is None:
            return None
        return note + self._converter_note()

    def _converter_note(self) -> str:
        """A sentence when the checkout is not at the pinned commit, else nothing."""
        root = locate(self.llama_cpp)
        if root is None or not (root / "convert_hf_to_gguf.py").is_file():
            return ""
        try:
            commit = commit_of(root)
        except Refused:
            return ""
        if commit == PINNED:
            return ""
        return (
            f"; the converter is llama.cpp {commit[:9]}, not the pinned {PINNED[:9]}, so this file may differ "
            "from the parity and quality numbers"
        )

    def _prepare(self, recipe: Recipe | None) -> str | None:
        """`prepare` without the converter note."""
        root = locate(self.llama_cpp)
        if not self.libggml:  # a linux wheel ships the C quantizer; an explicit setting still wins
            shipped = kern.use_bundled()
            usable = kern.has_weighted() or not self._needs_weighted(recipe) or self._library() is None
            if shipped != "numpy" and usable:
                return f"gguf: quantizing with the shipped kernel, {shipped}"
        path = self._library()
        if path is None and self.libggml:
            fix = (
                f"build llama.cpp's kernel with: {build_kernel_command(root)}"
                if root is not None
                else "name a llama.cpp checkout with `llama_cpp` and build its kernel"
            )
            raise Refused(f"gguf: no libggml-base in {self.libggml}; {fix}")
        if path is None:
            kern.use_ggml(None)
            if root is None:
                return (
                    "gguf: no libggml-base (no llama.cpp checkout); quantizing with the Python port, several "
                    "times slower. Set `libggml` to llama.cpp's built libggml-base to use its kernel"
                )
            return (
                f"gguf: no libggml-base found under {root} or its build*/; quantizing with the Python port, "
                f"several times slower. Build llama.cpp's kernel with: {build_kernel_command(root)}"
            )
        if not path.is_file():
            raise Refused(f"gguf: {path}: not a file")
        try:
            desc = kern.use_ggml(path)
        except ValueError as e:
            if self.libggml:
                fix = (
                    f"build llama.cpp's kernel with: {build_kernel_command(root)}"
                    if root is not None
                    else "build llama.cpp's kernel"
                )
                raise Refused(f"gguf: {e}; {fix}") from e
            kern.use_ggml(None)
            hint = f" Build llama.cpp's kernel with: {build_kernel_command(root)}" if root is not None else ""
            return f"gguf: {e}; quantizing with the Python port, several times slower.{hint}"
        note = f"gguf: quantizing with llama.cpp's own kernel, {desc} at {path}"
        commit = commit_of(root) if root is not None else None
        built = kern.ggml_commit()
        if built and commit and not commit.startswith(built):
            note += (
                f"; it was built from llama.cpp {built}, but the converter is at {commit[:9]}, so its bytes "
                "may differ from this converter's llama-quantize"
            )
        return note

    def refusal(self, plan: Plan, recipe: Recipe) -> str | None:
        """Why this build cannot be written, or None: a K-quant grid, layer or model steps, the default ``ignore``.

        A readable llama.cpp checkout is required when the converter must write the template (no ``template``
        given). Clip ``imatrix`` also needs llama.cpp's library, which is its only implementation.

        The file's tensor types follow llama.cpp's Q4_K_M rule (`rule`), which does not read ``ignore``: a module
        the config keeps float would be written quantized from the template without a word.

        ``gguf`` is an optional extra: a config naming this exporter in an environment without it plans every
        other build and is refused here, by name, instead of failing at import.
        """
        try:
            import gguf  # noqa: F401
        except ImportError:
            return file.NEEDS_GGUF

        from ..arch import DEFAULT_KEEP
        from ..solve.grid import DEFAULT_GRID
        from ..step import QuantStep

        if self.type_map and not self.tensor_types:
            # a build config expands `type_map` through its derived-group cache before this exporter exists;
            # an engine caller passing the exporter dict itself would otherwise get the default mix silently
            return (
                "type_map was not expanded: a candidate's unit map expands from its derived-group cache when "
                "a build config is read; run the build through BuildConfig (a config file), not the engine's "
                "exporter dict"
            )

        if plan.keep != DEFAULT_KEEP:
            added, dropped = sorted(plan.keep - DEFAULT_KEEP), sorted(DEFAULT_KEEP - plan.keep)
            what = ", ".join([*(repr(p) for p in added), *(f"without {r!r}" for r in dropped)])
            return (
                f"ignore ({what}): a Q4_K_M file's tensor types follow llama.cpp's rule, which does not read "
                f"ignore; use the default {sorted(DEFAULT_KEEP)}, or the compressed_tensors exporter"
            )

        if why := self._format_refusal(plan):
            return why
        native_types = {r.type for r in self.tensor_types} & {"Q5_K", *map(file.type_name, rule.NATIVE)}
        if any(f.name == Q5K_FORMAT for f in (plan.fmt, *plan.formats.values())):
            native_types.add("Q5_K")
        if native_types:
            types = ", ".join(sorted(native_types))
            # the library the build loads (`_library`: the named one, else the checkout's), not the setting alone
            if (library := self._library()) is None:
                return (
                    f"{types} requires native ggml_quantize_chunk; set exporter.gguf.libggml to a built "
                    "libggml-base, or exporter.gguf.llama_cpp to a llama.cpp checkout with its kernel built"
                )
            try:
                _ = ctypes.CDLL(str(library.resolve())).ggml_quantize_chunk
            except (OSError, AttributeError) as e:
                return f"{types} requires native ggml_quantize_chunk in {library}: {e}"
        for s in recipe.steps:
            if isinstance(s, QuantStep) and (s.grid or DEFAULT_GRID) not in (
                Q4K_GRID.name,
                Q5K_GRID.name,
                Q6K_GRID.name,
                IQ4NL_GRID.name,
                IQ4XS_GRID.name,
                Q3K_GRID.name,
            ):
                return (
                    f"{s.name}: grid {s.grid or DEFAULT_GRID!r}; a GGUF build is written from grids "
                    f"{Q4K_GRID.name!r}, {Q5K_GRID.name!r}, {Q6K_GRID.name!r}, {IQ4NL_GRID.name!r}, "
                    f"{IQ4XS_GRID.name!r} and {Q3K_GRID.name!r}"
                )
            if isinstance(s, QuantStep) and s.clip_search() is IMATRIX and self._library() is None:
                root = locate(self.llama_cpp)
                fix = (
                    f"build it with: {build_kernel_command(root)}"
                    if root is not None
                    else f"name a llama.cpp checkout with `llama_cpp` or ${ENV} and build its kernel"
                )
                return f"{s.name}: clip {IMATRIX.name!r} is llama.cpp's own fit and needs its library; {fix}"
        if plan.staged or any(kind == "checkpoint" for kind, _ in plan.segments):
            return (
                "checkpoint steps change weights the llama.cpp template holds in its own packing; only layer "
                "and model steps are written from the build"
            )
        if self.imatrix is not None and (
            residual := [s.name for s in recipe.steps if getattr(s, "changes_stream", False)]
        ):
            return (
                f"imatrix {self.imatrix!r}: the matrix is measured on the source's activations, and "
                f"{', '.join(residual)} changes the activations the weights read, so no tensor fitted with it "
                "would weigh what the file actually computes"
            )
        if self.template is None:  # the converter writes the template, so a checkout must be readable
            root = locate(self.llama_cpp)
            if root is None:
                return (
                    "no llama.cpp checkout: llama.cpp's convert_hf_to_gguf.py writes the bf16 template; set "
                    f"`llama_cpp` or ${ENV} ({checkout_command()}), or give one with `template`"
                )
            try:
                commit_of(root)
            except Refused as e:
                return str(e)
        if why := template_refusal(self.template):
            return why
        if self.template is not None and self.tensor_types:
            view = self._template_view()
            if view is None:  # unreachable: the template above is a readable file
                return None
            try:
                resolve_types(view, self.tensor_types)
            except Refused as e:
                if self.mtp:
                    return str(e)
                return dropped_head_hint(str(e), self.tensor_types, file.Template(self.template), view)
        return None

    def _layout(self, plan: Plan) -> file.Layout | None:  # noqa: ARG002
        """Where this build's tensors would land, or None without a readable template.

        Every tensor's type, and so every offset, follows from the template's shapes and the mix; nothing a
        step decides moves a byte. ``format_overrides`` choose whether a tensor holds solved codes or
        llama.cpp's round-to-nearest, not which type stores it, so they do not change the layout -- only
        `tensor_types` rules do. Refusals stay with `refusal`: a size is not the place to report them.

        This resolves types through the same `resolve_types` call the writer fixes its own layout with, so
        the two cannot disagree. `plan` is not read yet, and is taken so that it cannot start to: when a
        format override is allowed to promote a stored type, both sides must resolve through one function
        given the same inputs, and `plan.formats` and `plan.graph` are then already in reach here.
        """
        if self.template is None:
            return None  # the converter has not run, so there is no template whose layout to read
        try:
            tp = self._template_view()
            if tp is None:  # unreachable: the template above is not None
                return None
            return file.Layout(tp, resolve_types(tp, self.tensor_types))
        except (Refused, OSError, ValueError, KeyError, struct.error):
            return None

    def output_bytes(self, plan: Plan, recipe: Recipe) -> int | None:  # noqa: ARG002
        """The exact size of the Q4_K_M file, or None without a template to read."""
        layout = self._layout(plan)
        return None if layout is None else layout.total

    def weight_bytes(self, plan: Plan, recipe: Recipe) -> int | None:  # noqa: ARG002
        """The exact stored bytes of the file's tensors, without its header, or None without a template.

        A GGUF carries its vocabulary and metadata in the same file as its weights (megabytes of it on a large
        model), so a budget stated in weight bytes is not a budget on the file.
        """
        layout = self._layout(plan)
        return None if layout is None else layout.payload

    def builds(self, plan: Plan, recipe: Recipe) -> tuple[str, ...]:  # noqa: ARG002
        """``("template",)`` when this build would run llama.cpp's converter for its own bf16 template.

        A search that prepares one template and reuses it reads this to see which candidates would instead
        convert the whole source checkpoint again, per candidate.
        """
        return () if self.template is not None else ("template",)

    def _format_refusal(self, plan: Plan) -> str | None:
        """Why this build's ``format_overrides`` cannot be written, or None.

        A GGUF file names a type per tensor, so a build may store some tensors in another format -- but only
        a format the file can hold solved codes in, which is Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL and Q3_K.
        Whether each overridden module is one the mix actually promotes to Q6_K needs the template's tensor
        types, so `_add` checks that per tensor and names the module; this refuses the whole build early for a
        format that can never work.
        """
        solved = {Q4K_FORMAT, Q5K_FORMAT, Q6K_FORMAT, IQ4XS_FORMAT, IQ4NL_FORMAT, Q3K_FORMAT}
        wrong = sorted({f.name for f in plan.formats.values()} - solved)
        if wrong:
            return (
                f"format_overrides store modules in {wrong[0]}; a Q4_K_M file writes solved codes in "
                + ", ".join([Q4K_FORMAT, Q5K_FORMAT, Q6K_FORMAT, IQ4XS_FORMAT, IQ4NL_FORMAT, Q3K_FORMAT])
                + " only"
            )
        return None

    def writer(  # noqa: ANN001
        self,
        out_dir,
        source,
        graph,
        fmt,
        target,
        *,
        dtype=None,  # noqa: ARG002
        formats=None,  # noqa: ARG002
    ) -> GGUFWriter:
        """`GGUFWriter` (llama.cpp's converter sets the file's own dtypes, so the build `dtype` is unused).

        `formats` is the unit -> format map for a build with ``format_overrides``; the writer decides each
        tensor's type from the template and the mix, so it reads the format off each qparams entry instead.
        """
        return GGUFWriter(
            out_dir,
            source,
            graph,
            fmt,
            target,
            llama_cpp=self.llama_cpp,
            workers=self.workers,
            template=self.template,
            tensor_types=self.tensor_types,
            imatrix=self.imatrix,
            mtp=self.mtp,
        )

    def provenance(self) -> dict[str, Any]:
        """The converter's llama.cpp commit, the pinned one, and the kernel the build quantized with."""
        root = locate(self.llama_cpp)
        commit = None
        if root is not None and (root / "convert_hf_to_gguf.py").is_file():
            try:
                commit = commit_of(root)
            except Refused:
                commit = None
        return {
            "llama_cpp_commit": commit,
            "llama_cpp_pinned": PINNED,
            "kernel": kern.ggml_backend() or "numpy",
        }
