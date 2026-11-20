"""Verify and merge reference shards without running a model."""

from pathlib import Path

from qda_quant.output.vllm_env import CACHE_ENV

from qda_kl.reference import dropped_prompts, read_prompts
from qda_kl.store import Manifest, Reference, ReferenceWriter, _sha256


def _difference(left, right, path="") -> str | None:
    """Return the first differing dotted field, including missing settings."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(left.keys() | right.keys()):
            field = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                return field
            if difference := _difference(left[key], right[key], field):
                return difference
        return None
    return path if left != right else None


def merge(
    out: str | Path,
    shards: list[str | Path],
    *,
    corpus: str | Path | None = None,
    subsets: dict[str, list[str]] | None = None,
) -> Manifest:
    """Merge verified shards with identical settings, optionally enforcing corpus coverage and order.

    Shard-specific build facts are retained under their content hashes. Multiple answer draws belong to
    one prompt; a prompt must belong to exactly one shard. Missing or extra corpus prompts are refused, but a
    prompt whose every answer the builder dropped as too short (its draws are in a shard's ``dropped_short``)
    counts as covered: a build of the whole corpus would have dropped it too. The merged ``dropped_short`` is
    every shard's, in corpus order. An answer's window cap (``capped_answers``) is per shard as well: the merged
    build holds the union, and no ``capped_answers`` when no shard capped an answer. So is the engine's derived
    ``max_num_seqs``, which follows the shard's answer count; a ``serve_args`` value must still agree.
    """
    if not shards:
        raise ValueError("at least one reference shard is required")
    refs = [Reference(path, verify=True) for path in shards]
    facts = {
        "prompts",
        "prompts_sha256",
        "written_utc",
        "seconds",
        "dropped_short",
        "capped_answers",
        "max_num_seqs",  # derived from the shard's own answer count (qda_kl.serve.batch_args)
        "shards",
    }
    settings = []
    for ref in refs:
        metadata = ref.manifest.model_dump(
            exclude={"content_sha256", "sequences", "positions", "tail_mass", "files"}
        )
        metadata["build"] = {k: v for k, v in metadata["build"].items() if k not in facts}
        if "serve_env" in metadata["build"]:  # keep the shared keys; each shard's cache paths stay its own
            env = metadata["build"]["serve_env"]
            metadata["build"]["serve_env"] = {k: v for k, v in env.items() if k not in CACHE_ENV}
        settings.append(metadata)
        if difference := _difference(settings[0], metadata):
            raise ValueError(f"{ref.path}: shard settings differ at {difference}")
    rows, owners = {}, {}
    for ref in refs:
        for sequence in ref.sequences:
            prompt_id = sequence.prompt_id
            if prompt_id in owners and owners[prompt_id] is not ref:
                raise ValueError(f"duplicate prompt across shards: {prompt_id}")
            owners[prompt_id] = ref
            rows.setdefault(prompt_id, []).append((ref, sequence))
    order = list(rows)
    build = settings[0]["build"]
    build["shards"] = {
        ref.content_sha256: {k: v for k, v in ref.manifest.build.items() if k in facts} for ref in refs
    }
    dropped = {draw for ref in refs for draw in ref.manifest.build.get("dropped_short", [])}
    capped = {
        key: cap for ref in refs for key, cap in (ref.manifest.build.get("capped_answers") or {}).items()
    }
    if capped:  # an uncapped merge records none, keeping the manifest and identity it had
        build["capped_answers"] = capped
    answers = int(build.get("answers", 1))
    if corpus is not None:
        order = [p.id for p in read_prompts(corpus)]
        if len(order) != len(set(order)):
            raise ValueError("duplicate prompt in corpus")
        covered = set(rows) | {p for ref in refs for p in dropped_prompts(ref.manifest.build)}
        missing = sorted(set(order) - covered)
        extra = sorted(rows.keys() - set(order))
        if missing or extra:
            raise ValueError(f"corpus coverage differs: missing={missing}, extra={extra}")
        build.update(prompts=str(corpus), prompts_sha256=_sha256(Path(corpus)))
        build["dropped_short"] = [f"{p}#{d}" for p in order for d in range(answers) if f"{p}#{d}" in dropped]
    else:
        build["dropped_short"] = sorted(dropped)
    first = refs[0].manifest
    subsets = first.subsets if subsets is None else subsets
    for name, ids in subsets.items():
        unknown = sorted(set(ids) - rows.keys())
        if not name or not ids or len(ids) != len(set(ids)) or unknown:
            raise ValueError(f"subset {name!r}: empty name/ids, duplicate ids, or unknown ids {unknown}")
    # the merged reference holds the anchor tokenizer its shards hold, so it is self-contained as they are
    writer = ReferenceWriter(
        Path(out),
        first.k,
        first.vocab,
        model=first.model,
        build=build,
        subsets=subsets,
        tokenizer=refs[0].tokenizer,
    )
    for prompt_id in order:
        for ref, s in rows.get(prompt_id, []):  # a prompt the builder dropped has no sequence to write
            labels = s.model_dump(include={"group", "split", "kind", "finish"})
            writer.add(s.id, s.domain, ref.tokens(s), s.start, ref.topk(s), **labels)
    return writer.close()
