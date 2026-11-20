"""Render one ``qda eval`` run: its results table, paired comparison and KL report.

``qda eval table OUT [--format md|html] [--baseline KEY]`` reads ``OUT/resolved.yaml`` and the run's result files,
merging each suite's parts, and prints one table of the run's metrics. ``qda eval paired OUT --baseline KEY
[--resamples N] [--seed N]`` prints, per benchmark task, each checkpoint's mean over seeds and its paired
per-question difference from the baseline with a bootstrap 95 % interval, the flip rates and the cap hits.

The ``kl`` suite writes one :class:`~qda_kl.tally.SeqKL` per sequence (``detail.per_seq``); the table derives
``kl/<group>``, ``kl/<domain>``, ``kl/<split>`` and the weighted ``kl/total`` from those rows, each with a 95 %
percentile bootstrap interval, and a ``guard`` column that fails where a checkpoint's ``kl/easy`` exceeds the
baseline's by more than ``report.guard_threshold``.

The run's ``report`` and ``paired`` blocks in ``resolved.yaml`` shape these renderings, never a score, so editing
them never reruns a suite.
"""

from __future__ import annotations

import argparse
import html
import math
import random
import statistics
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from qda_quant.exit import refuse
from qda_quant.recipe import error_lines
from qda_quant.step import Refused

from qda_kl.eval.suites import SuiteResult
from qda_kl.tally import SeqKL

#: The group the easy-group guard watches: the decisions' hard/easy split names it.
EASY = "easy"

#: The derived column that weights the groups; no corpus label may use this name.
TOTAL = "total"

#: The run's resolved config and the directory that holds one subdirectory per checkpoint.
RESOLVED = "resolved.yaml"
RESULTS = "results"


class Report(BaseModel):
    """How the eval table summarizes the KL suite's sequences.

    Attributes:
        weights: group -> weight of ``kl/total``; a group a score lacks is left out and the others renormalized.
        resamples: bootstrap resamples over sequences for the 95 % intervals.
        seed: the bootstrap's seed; the same sequences and seed give the same intervals.
        guard_threshold: a checkpoint's easy-group KL fails the guard when it exceeds the baseline's by more
            than this many nats.
    """

    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float] = Field(default_factory=lambda: {"hard": 0.9, "easy": 0.1})
    resamples: int = Field(default=1000, ge=1)
    seed: int = 17
    guard_threshold: float = 0.0

    @field_validator("weights")
    @classmethod
    def _weights(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("give at least one group")
        negative = sorted(k for k, w in v.items() if w < 0)
        if negative:
            raise ValueError(f"weights {negative} are negative")
        if sum(v.values()) <= 0:
            raise ValueError("give at least one positive weight")
        return v


def parse_report(raw: object, where: str) -> Report:
    """Read a resolved config's ``report`` block (None: the defaults), naming ``where`` in a refusal."""
    if raw is None:
        return Report()
    try:
        return Report.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"{where}: " + "; ".join(error_lines(e))) from None


class Paired(BaseModel):
    """How ``qda eval paired`` draws its confidence intervals.

    Attributes:
        resamples: question resamples behind every 95 % interval.
        seed: RNG seed of those resamples; the same questions and seed give the same intervals.
    """

    model_config = ConfigDict(extra="forbid")

    resamples: int = Field(default=1000, ge=1)
    seed: int = 17


def parse_paired(raw: object, where: str) -> Paired:
    """Read a config's ``paired`` block (None: the defaults), naming ``where`` in a refusal."""
    if raw is None:
        return Paired()
    try:
        return Paired.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"{where}: " + "; ".join(error_lines(e))) from None


def derive(result: SuiteResult, report: Report) -> SuiteResult:
    """Return ``result`` with the report's KL columns and their 95 % intervals added; other results unchanged."""
    seqs = sequences(result)
    if not seqs:
        return result
    values = values_of(seqs, report.weights)
    return result.model_copy(
        update={
            "metrics": {**result.metrics, **values},
            "higher_is_better": {**result.higher_is_better, **dict.fromkeys(values, False)},
            "percent": {**result.percent, **dict.fromkeys(values, False)},
            "detail": {**result.detail, "ci": bootstrap(seqs, report.weights, report.resamples, report.seed)},
        }
    )


def sequences(result: SuiteResult) -> list[SeqKL]:
    """The result's answer sequences and labelled text sequences, in record order.

    The suite's plain ``kl`` stays answer-only; a web-text sequence joins the group, domain, split and total
    columns when the corpus labelled it (an unlabelled one would reach no column and only enlarge the bootstrap's
    draw). Returns none when the result carries no sequence records.

    Raises:
        ValueError: a ``per_seq`` entry is a mapping but not a sequence record.
    """
    raw = result.detail.get("per_seq")
    if not isinstance(raw, dict):
        return []
    out = []
    for value in raw.values():
        if not isinstance(value, dict):
            return []  # a per-sequence KL from before records carried their row's labels
        try:
            out.append(SeqKL.model_validate(value))
        except ValidationError:
            raise ValueError(
                f"{result.checkpoint}/{result.suite}: a per_seq entry is not a sequence record; rerun the suite"
            ) from None
    return [s for s in out if s.kind == "answer" or s.group or s.domain or s.split]


def values_of(seqs: Iterable[SeqKL], weights: dict[str, float]) -> dict[str, float]:
    """The derived columns: groups in ``weights``' order then by name, the total, then domains and splits sorted.

    Raises:
        ValueError: a group and a domain or split share a name, a domain and a split share a name, or any label is
            ``total`` — each pair would collide in one ``kl/<name>`` column.
    """
    groups = _means(seqs, lambda s: s.group)
    domains = _means(seqs, lambda s: s.domain)
    splits = _means(seqs, lambda s: s.split)
    if reserved := sorted(name for name in {*groups, *domains, *splits} if name == TOTAL):
        raise ValueError(
            f"a group, domain or split is named {reserved[0]!r}, which the total column uses; rename it"
        )
    clash = sorted(set(groups) & (set(domains) | set(splits)))
    if clash:
        raise ValueError(f"group and domain/split share the name(s) {clash}; rename one in the corpus")
    clash = sorted(set(domains) & set(splits))
    if clash:
        raise ValueError(f"domain and split share the name(s) {clash}; rename one in the corpus")
    out = {
        f"kl/{g}": groups[g] for g in [g for g in weights if g in groups] + sorted(set(groups) - set(weights))
    }
    used = [(w, groups[g]) for g, w in weights.items() if g in groups]
    if mass := sum(w for w, _ in used):
        out[f"kl/{TOTAL}"] = sum(w * kl for w, kl in used) / mass
    out |= {f"kl/{d}": v for d, v in sorted(domains.items())}
    out |= {f"kl/{s}": v for s, v in sorted(splits.items())}
    return out


def bootstrap(
    seqs: list[SeqKL], weights: dict[str, float], resamples: int, seed: int
) -> dict[str, list[float]]:
    """A 95 % percentile interval per derived column, from ``resamples`` draws of whole sequences.

    A column no drawn sequence belongs to is skipped for that draw; every column keeps at least as many values as
    draws it appears in.
    """
    rng = random.Random(seed)
    drawn: dict[str, list[float]] = {metric: [] for metric in values_of(seqs, weights)}
    for _ in range(resamples):
        sample = [seqs[rng.randrange(len(seqs))] for _ in seqs]
        for metric, value in values_of(sample, weights).items():
            drawn[metric].append(value)
    return {
        metric: [_quantile(sorted(values), 0.025), _quantile(sorted(values), 0.975)]
        for metric, values in drawn.items()
        if values
    }


def guard(easy: float | None, baseline: float | None, threshold: float) -> bool | None:
    """Whether ``easy`` exceeds ``baseline`` by more than ``threshold``; None when either is unknown."""
    if easy is None or baseline is None:
        return None
    return easy > baseline + threshold


def _means(seqs: Iterable[SeqKL], key: Callable[[SeqKL], str]) -> dict[str, float]:
    """Token-weighted mean per-token KL per non-empty key, in first-appearance order."""
    acc: dict[str, list[float]] = {}
    for s in seqs:
        if k := key(s):
            a = acc.setdefault(k, [0.0, 0.0])
            a[0] += s.kl * s.tokens
            a[1] += s.tokens
    return {k: v[0] / v[1] for k, v in acc.items()}


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile of sorted ``values`` (NumPy's default definition)."""
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


@dataclass(frozen=True)
class EvalTable:
    """One eval run's checkpoint order and the results found for it.

    Attributes:
        checkpoints: Every checkpoint key of ``resolved.yaml``, in config order.
        results: The result files found for those checkpoints, in checkpoint and suite order.
        report: The run's report settings; the defaults when ``resolved.yaml`` has none.
        paired: The run's paired-comparison settings; the defaults when ``resolved.yaml`` has none.
    """

    checkpoints: list[str]
    results: list[SuiteResult]
    report: Report = field(default_factory=Report)
    paired: Paired = field(default_factory=Paired)


_STYLE = """<style>
.qda-eval { border-collapse: collapse; font-variant-numeric: tabular-nums; }
.qda-eval th, .qda-eval td { border: 1px solid #ccc; padding: 0.3em 0.7em; }
.qda-eval th { background: #f5f5f5; }
.qda-eval td { text-align: right; }
.qda-eval td:first-child, .qda-eval th:first-child { text-align: left; }
</style>"""


def load(out: Path | str) -> EvalTable:
    """Read one eval run: its checkpoints in config order and every result found for them.

    Rows are the keys of ``checkpoints`` in ``out/resolved.yaml``, in config order; a checkpoint without a
    result directory still gets a row. Suites follow ``suites`` in the config; results for other suites follow,
    sorted by name. A suite's parts (``<suite>.<label>.json``) are merged into one result per suite, and a result
    with sequence records gains the report's derived KL columns (:func:`derive`).

    Args:
        out: The run directory that holds ``resolved.yaml`` and ``results``.

    Returns:
        The run's checkpoints and results, grouped by checkpoint and by suite in config order.

    Raises:
        ValueError: ``out`` is not an eval run, or a result file is not a :class:`SuiteResult`.
    """
    root = Path(out)
    resolved = root / RESOLVED
    if not resolved.is_file():
        raise Refused(f"{root}: not a qda eval run (no {RESOLVED})")
    try:
        config = yaml.safe_load(resolved.read_text())
    except yaml.YAMLError as why:
        raise Refused(f"{resolved}: not valid YAML: {why}") from None
    if not isinstance(config, dict):
        raise Refused(f"{resolved}: a qda eval resolved config is a YAML mapping")
    checkpoints = config.get("checkpoints")
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise Refused(f"{resolved}: no checkpoints")
    suites = _suite_names(config.get("suites"), resolved)
    report = parse_report(config.get("report"), f"{resolved} report")
    paired = parse_paired(config.get("paired"), f"{resolved} paired")

    results: list[SuiteResult] = []
    for name in checkpoints:
        directory = root / RESULTS / name
        if directory.is_dir():
            results.extend(derive(result, report) for result in _read_dir(directory, suites))
    _refuse_mixed_references(results)
    return EvalTable(list(checkpoints), results, report, paired)


def _refuse_mixed_references(results: list[SuiteResult]) -> None:
    """Refuse a table whose measurements of one suite were made against different references.

    Results with no measurement (no metrics) are skipped: a failed row measured nothing, so it records no reference
    and must keep its exact error instead of making the measurements next to it unrenderable. A partly measured
    row, whose merge kept a failed part's error beside the good parts' metrics and reference, is still checked.

    Raises:
        ValueError: one suite's measurements name different references, or mix a known reference with unknown
            provenance. Recompute unknown results before comparing them with frozen-reference results.
    """
    seen: dict[str, dict[str, list[str]]] = {}
    subsets: dict[str, set[str]] = {}
    for result in results:
        if not result.metrics:
            continue
        sha = result.detail.get("reference_sha256")
        sha = sha if isinstance(sha, str) else ""
        seen.setdefault(result.suite, {}).setdefault(sha, []).append(result.checkpoint)
        subsets.setdefault(result.suite, set()).add(str(result.detail.get("subset") or ""))
    for suite, hashes in seen.items():
        if len(subsets[suite]) > 1:
            raise ValueError(f"suite {suite}: results come from different reference subsets")
        if len(hashes) > 1:
            found = "; ".join(f"{', '.join(names)}: {sha[:16] or 'unknown'}" for sha, names in hashes.items())
            raise ValueError(
                f"suite {suite}: results come from different references ({found}); "
                "point the runs at one shared `refs` directory and recompute results with unknown provenance"
            )


def table_to_markdown(table: EvalTable, baseline: str | None = None) -> str:
    """Render the results as one GitHub-flavoured Markdown table.

    The best value of each column is bold, following that metric's ``higher_is_better``. A metric is a fraction
    shown as a percent with one decimal, unless its result's ``percent`` says otherwise (KL); then it keeps four
    significant digits. A metric's standard error in ``detail["stderr"]`` follows the value as ``±``; a bootstrap
    interval in ``detail["ci"]`` follows it as ``[low, high]``. With a baseline and an easy-group KL, a final
    ``guard`` column reads ``ok`` or ``FAIL`` (no baseline or no easy group: ``-``).

    Args:
        table: The run returned by :func:`load`.
        baseline: The checkpoint key that defines 100 % recovery; no recovery column when None.

    Returns:
        The table, without a trailing newline.

    Raises:
        ValueError: ``baseline`` is not a checkpoint in ``table``.
    """
    if not table.results:
        return ""
    headers, rows = _table(table, baseline)
    lines = [
        "| " + " | ".join(_escape_md(header) for header in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for name, cells in rows:
        rendered = [f"**{_escape_md(text)}**" if best else _escape_md(text) for text, best in cells]
        lines.append("| " + " | ".join([_escape_md(name), *rendered]) + " |")
    return "\n".join(lines)


def table_to_html(table: EvalTable, baseline: str | None = None) -> str:
    """Render the results as one self-contained HTML table (no JavaScript, no external CSS).

    Values and bolding follow :func:`table_to_markdown`.

    Args:
        table: The run returned by :func:`load`.
        baseline: The checkpoint key that defines 100 % recovery; no recovery column when None.

    Returns:
        A ``<style>`` block followed by one ``<table>`` element.

    Raises:
        ValueError: ``baseline`` is not a checkpoint in ``table``.
    """
    if not table.results:
        return ""
    headers, rows = _table(table, baseline)
    out = [_STYLE, '<table class="qda-eval">', "<thead>", "<tr>"]
    out.extend(f"<th>{html.escape(header)}</th>" for header in headers)
    out.extend(["</tr>", "</thead>", "<tbody>"])
    for name, cells in rows:
        out.append("<tr>")
        out.append(f"<td>{html.escape(name)}</td>")
        for text, best in cells:
            body = html.escape(text)
            out.append(f"<td><strong>{body}</strong></td>" if best else f"<td>{body}</td>")
        out.append("</tr>")
    out.extend(["</tbody>", "</table>"])
    return "\n".join(out)


def table_main(out: Path | str, *, fmt: str = "md", baseline: str | None = None) -> int:
    """Print the table for one eval run.

    Args:
        out: The run directory that holds ``resolved.yaml`` and ``results``.
        fmt: ``html`` for an HTML table; anything else renders Markdown.
        baseline: The checkpoint key that defines 100 % recovery; no recovery column when None.

    Returns:
        0 on success, 2 when the run directory, a result file or the baseline is refused.
    """
    try:
        run = load(out)
        if not run.results:
            raise Refused(f"{out}: no results under {RESULTS}")
        render = table_to_html if fmt == "html" else table_to_markdown
        table = render(run, baseline)
    except Refused as why:
        return refuse(str(why), why.fix or None, code=why.code)
    except ValueError as why:
        return refuse(str(why))
    print(table)
    return 0


def _suite_names(suites: Any, path: Path) -> list[str]:
    """The suite names of ``suites`` in config order, one ``{name: settings}`` per entry."""
    names: list[str] = []
    for entry in suites or []:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise Refused(f"{path}: each suite is one {{name: settings}} mapping")
        names.append(next(iter(entry)))
    return names


def _read_dir(directory: Path, suites: Iterable[str]) -> list[SuiteResult]:
    """Read the named suites in order, then every other suite sorted by name, merging each suite's parts."""
    files = sorted(directory.glob("*.json"), key=lambda path: path.name)
    names = list(dict.fromkeys(suites))
    names += sorted({path.stem.partition(".")[0] for path in files} - set(names))
    out: list[SuiteResult] = []
    for name in names:
        found = [path for path in files if path.stem == name or path.stem.startswith(f"{name}.")]
        if found:
            out.append(_merge([_read_result(path) for path in found]))
    return out


def _merge(results: list[SuiteResult]) -> SuiteResult:
    """One suite's result files (``<suite>.json`` and its parts) as one result.

    Metrics, their directions and how they are shown are unioned; a ``detail`` value that is a mapping
    (``stderr``, a task's numbers) is unioned one level deep; ``wall_s`` is the sum over the parts; a failed
    part's error is kept.
    """
    _refuse_mixed_references(results)
    first = results[0]
    metrics: dict[str, float] = {}
    higher: dict[str, bool] = {}
    percent: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    errors = [result.error for result in results if result.error is not None]
    for result in results:
        metrics.update(result.metrics)
        higher.update(result.higher_is_better)
        percent.update(result.percent)
        for key, value in result.detail.items():
            if isinstance(value, dict) and isinstance(detail.get(key), dict):
                detail[key] = {**detail[key], **value}
            else:
                detail[key] = value
    return first.model_copy(
        update={
            "metrics": metrics,
            "higher_is_better": higher,
            "percent": percent,
            "detail": detail,
            "wall_s": sum(result.wall_s for result in results),
            "error": "; ".join(errors) if errors else None,
        }
    )


def _read_result(path: Path) -> SuiteResult:
    try:
        return SuiteResult.model_validate_json(path.read_text())
    except ValueError as why:
        raise Refused(f"{path}: not a qda eval result: {why}") from None


def _table(
    table: EvalTable, baseline: str | None
) -> tuple[list[str], list[tuple[str, list[tuple[str, bool]]]]]:
    """The header labels and, per checkpoint, each cell's text and whether it is the column's best."""
    checkpoints = table.checkpoints
    columns, labels = _layout(table.results)
    if baseline is not None and baseline not in checkpoints:
        raise Refused(f"unknown baseline {baseline!r}; checkpoints: {', '.join(checkpoints)}")
    by_key = {(result.checkpoint, result.suite): result for result in table.results}
    errors = _errors(table.results)
    first = _first(by_key, columns, checkpoints)
    directions = {column: result.higher_is_better[column[1]] for column, result in first.items()}
    best = {}
    for suite, metric in columns:
        values = [_value(by_key, checkpoint, suite, metric) for checkpoint in checkpoints]
        best[(suite, metric)] = _best(values, directions[(suite, metric)])
    benchmarks = [column for column in columns if directions[column]]
    recovery = (
        _recovery(by_key, checkpoints, benchmarks, baseline) if baseline is not None and benchmarks else None
    )
    best_recovery = _best(recovery.values(), True) if recovery is not None else None
    guards = _guard(by_key, checkpoints, columns, baseline, table.report.guard_threshold)

    rows = []
    for checkpoint in checkpoints:
        cells = [] if errors is None else [(errors.get(checkpoint, "-"), False)]
        for suite, metric in columns:
            value = _value(by_key, checkpoint, suite, metric)
            if value is None:
                cells.append(("-", False))
            else:
                text = _text(
                    by_key[(checkpoint, suite)].percent.get(metric, True),
                    value,
                    _stderr(by_key, checkpoint, suite, metric),
                    _ci(by_key, checkpoint, suite, metric),
                )
                cells.append((text, value == best[(suite, metric)]))
        if recovery is not None:
            value = recovery[checkpoint]
            cells.append(("-", False) if value is None else (f"{value * 100:.1f}%", value == best_recovery))
        if guards is not None:
            flag = guards[checkpoint]
            cells.append(("-", False) if flag is None else ("FAIL" if flag else "ok", False))
        rows.append((checkpoint, cells))
    headers = [
        "checkpoint",
        *(["error"] if errors is not None else []),
        *labels,
        *(["recovery %"] if recovery is not None else []),
        *(["guard"] if guards is not None else []),
    ]
    return headers, rows


def _errors(results: list[SuiteResult]) -> dict[str, str] | None:
    """Each checkpoint's failed suites' exact errors, or None when no result failed.

    One failed suite prints its error alone; more than one are joined as ``suite: error``.
    """
    failed: dict[str, list[tuple[str, str]]] = {}
    for result in results:
        if result.error is not None:
            failed.setdefault(result.checkpoint, []).append((result.suite, result.error))
    if not failed:
        return None
    out = {}
    for checkpoint, errors in failed.items():
        out[checkpoint] = (
            errors[0][1] if len(errors) == 1 else "; ".join(f"{suite}: {error}" for suite, error in errors)
        )
    return out


def _layout(results: list[SuiteResult]) -> tuple[list[tuple[str, str]], list[str]]:
    """``(suite, metric)`` columns in suite order and their labels.

    A label is the metric name, or ``suite/metric`` when two suites share the name.
    """
    suites = list(dict.fromkeys(result.suite for result in results))
    metrics: dict[str, list[str]] = {suite: [] for suite in suites}
    for result in results:
        for metric in result.metrics:
            if metric not in metrics[result.suite]:
                metrics[result.suite].append(metric)
    columns = [(suite, metric) for suite in suites for metric in metrics[suite]]
    seen = Counter(metric for _, metric in columns)
    labels = [f"{suite}/{metric}" if seen[metric] > 1 else metric for suite, metric in columns]
    return columns, labels


def _first(
    by_key: dict[tuple[str, str], SuiteResult],
    columns: list[tuple[str, str]],
    checkpoints: list[str],
) -> dict[tuple[str, str], SuiteResult]:
    """Each column's first result that has the metric, which gives its direction."""
    out: dict[tuple[str, str], SuiteResult] = {}
    for suite, metric in columns:
        for checkpoint in checkpoints:
            result = by_key.get((checkpoint, suite))
            if result is not None and metric in result.metrics:
                if metric not in result.higher_is_better:
                    raise Refused(f"{checkpoint}/{suite}: no direction for metric {metric!r}")
                out[(suite, metric)] = result
                break
        else:
            raise Refused(f"no result defines metric {metric!r} of suite {suite!r}")
    return out


def _value(
    by_key: dict[tuple[str, str], SuiteResult], checkpoint: str, suite: str, metric: str
) -> float | None:
    result = by_key.get((checkpoint, suite))
    return None if result is None else result.metrics.get(metric)


def _stderr(
    by_key: dict[tuple[str, str], SuiteResult], checkpoint: str, suite: str, metric: str
) -> float | None:
    result = by_key.get((checkpoint, suite))
    stderr = None if result is None else result.detail.get("stderr")
    value = stderr.get(metric) if isinstance(stderr, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _ci(
    by_key: dict[tuple[str, str], SuiteResult], checkpoint: str, suite: str, metric: str
) -> tuple[float, float] | None:
    result = by_key.get((checkpoint, suite))
    ci = None if result is None else result.detail.get("ci")
    value = ci.get(metric) if isinstance(ci, dict) else None
    if isinstance(value, list) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
        return float(value[0]), float(value[1])
    return None


def _best(values: Iterable[float | None], higher_is_better: bool) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present) if higher_is_better else min(present)


def _recovery(
    by_key: dict[tuple[str, str], SuiteResult],
    checkpoints: list[str],
    benchmarks: list[tuple[str, str]],
    baseline: str,
) -> dict[str, float | None]:
    """Each checkpoint's mean benchmark ratio to ``baseline``.

    A metric missing on either side is skipped, and so is one on which the baseline scores 0 (no ratio exists).
    """
    out: dict[str, float | None] = {}
    for checkpoint in checkpoints:
        ratios = []
        for suite, metric in benchmarks:
            base = _value(by_key, baseline, suite, metric)
            value = _value(by_key, checkpoint, suite, metric)
            if base is not None and value is not None and base != 0:
                ratios.append(value / base)
        out[checkpoint] = sum(ratios) / len(ratios) if ratios else None
    return out


def _guard(
    by_key: dict[tuple[str, str], SuiteResult],
    checkpoints: list[str],
    columns: list[tuple[str, str]],
    baseline: str | None,
    threshold: float,
) -> dict[str, bool | None] | None:
    """Each checkpoint's easy-group guard, or None without a baseline or an easy-group metric.

    A checkpoint fails when its ``kl/easy`` exceeds the baseline's by more than ``threshold``; a cell is None when
    either value is missing.
    """
    easy = next(((suite, metric) for suite, metric in columns if metric == f"kl/{EASY}"), None)
    if baseline is None or easy is None:
        return None
    base = _value(by_key, baseline, *easy)
    return {
        checkpoint: guard(_value(by_key, checkpoint, *easy), base, threshold) for checkpoint in checkpoints
    }


def _number(percent: bool, value: float) -> str:
    return f"{value * 100:.1f}%" if percent else f"{value:.4g}"


def _text(percent: bool, value: float, stderr: float | None, ci: tuple[float, float] | None) -> str:
    """A cell's text: the value and, when known, its standard error or its bootstrap interval."""
    text = _number(percent, value)
    if ci is not None:
        return f"{text} [{_number(percent, ci[0])}, {_number(percent, ci[1])}]"
    return text if stderr is None else f"{text} ±{_number(percent, stderr)}"


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|")


def paired_to_markdown(
    table: EvalTable, baseline: str, resamples: int | None = None, seed: int | None = None
) -> str:
    """Render the bootstrap recipe and one table per benchmark task as Markdown.

    Args:
        table: The run returned by :func:`load`.
        baseline: The checkpoint every other row is paired against.
        resamples: Question resamples behind every interval; None uses the run's recorded setting.
        seed: RNG seed of those resamples; None uses the run's recorded setting.

    Returns:
        The output, without a trailing newline, or ``""`` when no task can be paired.

    Raises:
        ValueError: ``baseline`` is unknown, ``resamples`` is not positive, or a row is refused.
    """
    resamples = table.paired.resamples if resamples is None else resamples
    seed = table.paired.seed if seed is None else seed
    if not table.results:
        return ""
    if baseline not in table.checkpoints:
        raise ValueError(f"unknown baseline {baseline!r}; checkpoints: {', '.join(table.checkpoints)}")
    if resamples < 1:
        raise ValueError(f"resamples {resamples}: expected a positive count")
    ci = f"delta vs {baseline} [95% CI]"
    header = ["checkpoint", "mean", ci, "right->wrong", "wrong->right", "cap hits"]
    columns, labels = _layout(table.results)
    tasks = [
        (suite, metric, label)
        for (suite, metric), label in zip(columns, labels, strict=True)
        if any(isinstance(r.detail.get(metric), dict) for r in table.results if r.suite == suite)
    ]
    sections = []
    for suite, metric, label in tasks:
        title, rows, notes = _benchmark(table, suite, metric, label, baseline, resamples, seed)
        lines = [
            f"### {title}",
            "",
            "| " + " | ".join(map(_escape_md, header)) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        lines += ["| " + " | ".join(_escape_md(cell) for cell in row) + " |" for row in rows]
        lines += [line for note in notes for line in ("", f"note: {note}")]
        sections.append("\n".join(lines))
    if not sections:
        return ""
    recipe = f"bootstrap 95% CI: {resamples} resamples of the questions, RNG seed {seed}."
    return "\n\n".join([recipe, *sections])


def paired_main(argv: list[str] | None = None) -> int:
    """Print the paired comparison for one eval run; 0 on success, 2 when something is refused."""
    parser = argparse.ArgumentParser(
        prog="qda eval paired",
        description="Paired per-question comparison of an eval run's checkpoints against one baseline.",
    )
    parser.add_argument("out", type=Path, help="the eval run directory")
    parser.add_argument("--baseline", required=True, help="checkpoint to pair every row against")
    parser.add_argument(
        "--resamples", type=int, help="question resamples (default: the run's recorded paired.resamples)"
    )
    parser.add_argument(
        "--seed", type=int, help="RNG seed of the resamples (default: the run's recorded paired.seed)"
    )
    args = parser.parse_args(argv)
    try:
        run = load(args.out)
        if not run.results:
            raise ValueError(f"{args.out}: no results under {RESULTS}")
        text = paired_to_markdown(run, args.baseline, args.resamples, args.seed)
    except ValueError as why:
        return refuse(str(why))
    print(text)
    return 0


def _benchmark(
    table: EvalTable, suite: str, metric: str, label: str, baseline: str, resamples: int, seed: int
) -> tuple[str, list[list[str]], list[str]]:
    """One benchmark's heading, the cells of every checkpoint row, and the notes under its table.

    Raises:
        ValueError: a score is not finite, or a row answers different questions, prompts or seeds.
    """
    by_key = {(result.checkpoint, result.suite): result for result in table.results}
    base_result = by_key.get((baseline, suite))
    base = _scores(base_result, metric, f"{label}: {baseline}")
    base_hashes = None if base is None else _prompt_hashes(base_result, metric, base)
    base_means = {} if base is None else {question: statistics.fmean(v) for question, v in base.items()}
    base_seeds = None if base is None else _seeds(base_result, metric, base, f"{label}: {baseline}")
    questions = sorted(base_means)
    draws = (
        np.random.default_rng(seed).integers(0, len(questions), size=(resamples, len(questions)))
        if questions
        else None
    )

    rows: list[list[str]] = []
    notes: list[str] = []
    if base is not None and base_seeds is None:
        notes.append(f"{baseline}: does not record its seeds; paired columns are '-'.")
    for checkpoint in table.checkpoints:
        result = by_key.get((checkpoint, suite))
        scores = _scores(result, metric, f"{label}: {checkpoint}")
        mean = None if result is None else result.metrics.get(metric)
        percent = True if result is None else result.percent.get(metric, True)
        if mean is not None and not math.isfinite(mean):
            raise ValueError(f"{label}: {checkpoint}: metric {metric!r} is not finite: {mean!r}")
        detail = None if result is None else result.detail.get(metric)
        caps = detail.get("cap_hits") if isinstance(detail, dict) else None
        cap_hits = sum(caps) if isinstance(caps, list) and all(type(cap) is int for cap in caps) else None
        diff = flips = None
        if scores is not None:
            mean = statistics.fmean(value for values in scores.values() for value in values)
            hashes = _prompt_hashes(result, metric, scores)
            if base is not None and draws is not None and checkpoint != baseline:
                _agree(scores, base, checkpoint, baseline, label)
                count, base_count = len(next(iter(scores.values()))), len(next(iter(base.values())))
                if count != base_count:
                    raise ValueError(
                        f"{label}: {checkpoint} has {count} seeds, {baseline} has {base_count}; "
                        f"rerun {checkpoint} with the same repeats"
                    )
                seeds = _seeds(result, metric, scores, f"{label}: {checkpoint}")
                if seeds is None:
                    notes.append(f"{checkpoint}: does not record its seeds; paired columns are '-'.")
                elif base_seeds is not None and seeds != base_seeds:
                    raise ValueError(
                        f"{label}: {checkpoint} has seeds {seeds}, {baseline} has seeds {base_seeds}; "
                        "rerun both with the same seed and repeats"
                    )
                if (
                    seeds is not None
                    and base_seeds is not None
                    and hashes is not None
                    and base_hashes is not None
                ):
                    for question in questions:
                        if hashes[question] != base_hashes[question]:
                            raise ValueError(
                                f"{label}: {checkpoint} and {baseline} have different prompt_hashes "
                                f"for question {question!r}; rerun both with the same rendered prompts"
                            )
                    means = {question: statistics.fmean(scores[question]) for question in questions}
                    diff = _difference(means, base_means, questions, draws)
                    flips = _flips(scores, base, questions)
            if checkpoint != baseline and hashes is None:
                notes.append(
                    f"{checkpoint}: no complete prompt_hashes in the result; paired columns are '-'."
                )
        elif checkpoint != baseline:
            why = "no result" if result is None else "no per_question in the result"
            notes.append(f"{checkpoint}: {why}; paired columns are '-'.")
        rows.append(
            [
                checkpoint,
                "-" if mean is None else _number(percent, mean),
                "-"
                if diff is None
                else f"{diff[0] * 100:+.1f}pp [{diff[1] * 100:+.1f}, {diff[2] * 100:+.1f}]",
                _flip_text(flips, 0),
                _flip_text(flips, 1),
                "-" if cap_hits is None else str(cap_hits),
            ]
        )

    if base is None:
        why = "no result" if base_result is None else "no per_question in the result"
        notes.insert(0, f"{baseline}: {why}; paired columns are '-'.")
    elif base_hashes is None:
        notes.insert(0, f"{baseline}: no complete prompt_hashes in the result; paired columns are '-'.")
    summary = f" ({len(questions)} questions x {len(base[questions[0]])} seeds)" if questions and base else ""
    return label + summary, rows, notes


def _scores(result: SuiteResult | None, metric: str, where: str) -> dict[str, list[float]] | None:
    """``per_question`` as ``{question: [score per seed]}``, or None when the result has none.

    Raises:
        ValueError: the mapping is malformed, a score is not finite, or the questions have different counts.
    """
    detail = None if result is None else result.detail.get(metric)
    per_question = detail.get("per_question") if isinstance(detail, dict) else None
    if per_question is None:
        return None
    if not isinstance(per_question, dict) or not per_question:
        raise ValueError(f"{where}: per_question is not a {{question: [score per seed]}} mapping")
    for question, values in per_question.items():
        if not (isinstance(values, list) and values and all(isinstance(v, (int, float)) for v in values)):
            raise ValueError(f"{where}: per_question[{question!r}] is not a non-empty list of scores")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{where}: per_question[{question!r}] holds a non-finite score: {values!r}")
    scores = {str(question): [float(value) for value in values] for question, values in per_question.items()}
    counts = {len(values) for values in scores.values()}
    if len(counts) != 1:
        raise ValueError(
            f"{where}: per_question has {sorted(counts)} scores per question; expected one count"
        )
    return scores


def _prompt_hashes(
    result: SuiteResult | None, metric: str, scores: dict[str, list[float]]
) -> dict[str, str] | None:
    """Recorded prompt hashes for every scored question, or None when provenance is incomplete."""
    detail = None if result is None else result.detail.get(metric)
    hashes = detail.get("prompt_hashes") if isinstance(detail, dict) else None
    if not isinstance(hashes, dict) or set(hashes) != set(scores):
        return None
    if not all(isinstance(value, str) and value for value in hashes.values()):
        return None
    return hashes


def _seeds(
    result: SuiteResult | None, metric: str, scores: dict[str, list[float]], where: str
) -> list[int] | None:
    """The seed ids behind ``scores``: the recorded seed plus one per repeat, or None when it records none.

    The built-in harness records the seed at the result's top level and the repeats under the task;
    LiveCodeBench records the repeats under ``detail.repeats``. A seed id counts only when both are recorded: a
    score count alone cannot certify which seed produced which score.

    Raises:
        ValueError: the recorded repeats disagree with the number of scores per question.
    """
    detail = {} if result is None else result.detail
    task = detail.get(metric)
    task = task if isinstance(task, dict) else {}
    seed = detail.get("seed", task.get("seed"))
    repeats = task.get("repeats", detail.get("repeats"))
    count = len(next(iter(scores.values())))
    if type(seed) is not int or type(repeats) is not int:
        return None
    if repeats != count:
        raise ValueError(
            f"{where}: repeats={repeats} but per_question has {count} scores per question; the result disagrees "
            "with itself"
        )
    return list(range(seed, seed + repeats))


def _difference(
    means: dict[str, float], base_means: dict[str, float], questions: list[str], draws: np.ndarray
) -> tuple[float, float, float]:
    """The paired difference's mean and the 2.5th and 97.5th percentiles of its question resamples."""
    diffs = np.array([means[question] - base_means[question] for question in questions], dtype=float)
    low, high = np.percentile(diffs[draws].mean(axis=1), (2.5, 97.5))
    return float(diffs.mean()), float(low), float(high)


def _flips(
    scores: dict[str, list[float]], base: dict[str, list[float]], questions: list[str]
) -> tuple[int, int, int] | None:
    """Right-to-wrong and wrong-to-right counts over (question, seed) pairs, or None if a score is not 0/1."""
    right_wrong = wrong_right = pairs = 0
    for question in questions:
        for base_score, score in zip(base[question], scores[question], strict=True):
            if base_score not in (0, 1) or score not in (0, 1):
                return None
            pairs += 1
            if base_score == 1 and score == 0:
                right_wrong += 1
            elif base_score == 0 and score == 1:
                wrong_right += 1
    return right_wrong, wrong_right, pairs


def _flip_text(flips: tuple[int, int, int] | None, index: int) -> str:
    """One flip rate with its count and the pairs it is out of, or ``-``."""
    if flips is None:
        return "-"
    return f"{100 * flips[index] / flips[2]:.1f}% ({flips[index]}/{flips[2]})"


def _agree(
    scores: dict[str, list[float]], base: dict[str, list[float]], checkpoint: str, baseline: str, label: str
) -> None:
    """Refuse two rows that do not answer exactly the same questions.

    Raises:
        ValueError: the question sets differ, naming a few of each side's own questions and the fix.
    """
    parts = []
    for side, ids in (
        (baseline, sorted(set(base) - set(scores))),
        (checkpoint, sorted(set(scores) - set(base))),
    ):
        if ids:
            shown = ", ".join(ids[:3]) + (", ..." if len(ids) > 3 else "")
            parts.append(f"{len(ids)} only in {side} ({shown})")
    if parts:
        raise ValueError(
            f"{label}: {checkpoint} and {baseline} do not share the same questions: {'; '.join(parts)}; "
            f"rerun both with the same suite settings"
        )
