from __future__ import annotations

import hashlib
import json
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer
from llm_lab.core.tokenization.sp16k_tokenizer_artifact import (
    load_docs_one_per_non_empty_line,
    load_tokenizer_from_artifact_dir as load_tokenizer_from_artifact_dir_v2,
)

# SP16K bridge START: sp16k pretokenized shard pipeline
PRETOK_FORMAT_VERSION = "tinyllama_p15_pretok_uint16_v1"
SUPPORTED_DTYPE = "uint16"
MAX_UINT16 = 65535

NORMALIZATION_POLICY_MARKER = "tinyllama_p15_norm_v1_identity_preserve_docs"
EOT_POLICY_MARKER = "tinyllama_p15_eot_v1_append_after_every_doc_including_final"

ROOT_MANIFEST_FILENAME = "tinyllama_p15_shards_manifest.json"

REQUIRED_SPLIT_MANIFEST_KEYS = {
    "contract_version",
    "split_seed",
    "total_docs",
    "train_doc_indices",
    "val_doc_indices",
    "docs_sha256",
    "tokenizer_hash",
}


@dataclass(frozen=True)
class Sp16kShardBuildConfig:
    max_tokens_per_shard: int
    format_version: str = PRETOK_FORMAT_VERSION


@dataclass(frozen=True)
class Sp16kDocSpan:
    global_doc_index: int
    start_token: int
    end_token: int


@dataclass(frozen=True)
class Sp16kShardChunk:
    split: Literal["train", "val"]
    shard_index: int
    token_ids: list[int]
    first_doc_global_index: int
    last_doc_global_index: int
    document_count: int


@dataclass(frozen=True)
class Sp16kShardSidecar:
    format_version: str
    filename: str
    split: Literal["train", "val"]
    dtype: str
    shard_index: int
    token_count: int
    document_count: int
    tokenizer_hash: str
    split_manifest_sha256: str
    sha256: str
    first_doc_global_index: int
    last_doc_global_index: int
    prep_config_ref: str


@dataclass(frozen=True)
class Sp16kRootManifest:
    format_version: str
    tokenizer_provenance: dict[str, Any]
    split_provenance: dict[str, Any]
    shard_sizing_config: dict[str, Any]
    normalization_policy_marker: str
    eot_policy_marker: str
    shard_inventory: list[dict[str, Any]]
    total_docs_per_split: dict[str, int]
    total_tokens_per_split: dict[str, int]


def _canonical_json_bytes(value: object) -> bytes:
    # Stable serialization gives deterministic hashes and manifest refs.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_hex_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_docs_sha256(docs: Sequence[str]) -> str:
    return sha256_hex_bytes(_canonical_json_bytes(list(docs)))


def read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_split_manifest(path: Path) -> Dict[str, Any]:
    manifest = read_json(path)
    validate_split_manifest_structure(manifest)
    return manifest


def validate_split_manifest_structure(split_manifest: Dict[str, Any]) -> None:
    missing = REQUIRED_SPLIT_MANIFEST_KEYS - set(split_manifest.keys())
    if missing:
        raise ValueError(f"split manifest missing required keys: {sorted(missing)}")

    if not isinstance(split_manifest["total_docs"], int) or split_manifest["total_docs"] < 0:
        raise ValueError("split manifest total_docs must be a non-negative int")

    for k in ("train_doc_indices", "val_doc_indices"):
        indices = split_manifest[k]
        if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
            raise ValueError(f"split manifest {k} must be list[int]")

    for k in ("docs_sha256", "tokenizer_hash"):
        if not isinstance(split_manifest[k], str) or not split_manifest[k]:
            raise ValueError(f"split manifest {k} must be non-empty str")


def validate_split_membership_bounds(split_manifest: Dict[str, Any], doc_count: int) -> None:
    seen: set[int] = set()
    for split_key in ("train_doc_indices", "val_doc_indices"):
        indices = split_manifest[split_key]
        for idx in indices:
            if idx < 0 or idx >= doc_count:
                raise ValueError(f"split index {idx} out of bounds for doc_count={doc_count}")
            if idx in seen:
                raise ValueError(f"split index {idx} appears in more than one split")
            seen.add(idx)


def validate_strict_provenance(
    split_manifest: Dict[str, Any],
    *,
    tokenizer_hash: str,
    docs_sha256: str,
    doc_count: int,
) -> None:
    # Strict provenance checks required by HSU-2 plan.
    if split_manifest["tokenizer_hash"] != tokenizer_hash:
        raise ValueError("tokenizer hash mismatch between split manifest and tokenizer artifact")
    if split_manifest["docs_sha256"] != docs_sha256:
        raise ValueError("docs_sha256 mismatch between split manifest and current input docs")
    if split_manifest["total_docs"] != doc_count:
        raise ValueError("total_docs mismatch between split manifest and current input docs")
    validate_split_membership_bounds(split_manifest, doc_count)


def load_tokenizer_from_artifact_dir(tokenizer_artifact_dir: Path) -> tuple[SubwordTokenizer, str]:
    # HSU-1D reclosure: backend-aware tokenizer artifact loading delegated to artifact module.
    return load_tokenizer_from_artifact_dir_v2(tokenizer_artifact_dir)


def deterministic_shard_basename(split: Literal["train", "val"], shard_index: int) -> str:
    return f"{split}_{shard_index:06d}"


def build_prep_config_ref(payload: Dict[str, Any]) -> str:
    return sha256_hex_bytes(_canonical_json_bytes(payload))


def chunk_split_token_stream(
    *,
    split: Literal["train", "val"],
    token_stream: Sequence[int],
    doc_spans: Sequence[Sp16kDocSpan],
    max_tokens_per_shard: int,
) -> list[Sp16kShardChunk]:
    if max_tokens_per_shard <= 0:
        raise ValueError("max_tokens_per_shard must be > 0")
    if not token_stream:
        return []

    chunks: list[Sp16kShardChunk] = []
    token_cursor = 0
    shard_index = 0

    while token_cursor < len(token_stream):
        next_cursor = min(token_cursor + max_tokens_per_shard, len(token_stream))
        shard_tokens = list(token_stream[token_cursor:next_cursor])

        overlapping_docs = [
            s.global_doc_index
            for s in doc_spans
            if s.end_token > token_cursor and s.start_token < next_cursor
        ]
        if not overlapping_docs:
            raise ValueError("doc spans do not cover chunk token range; invalid span metadata")

        chunks.append(
            Sp16kShardChunk(
                split=split,
                shard_index=shard_index,
                token_ids=shard_tokens,
                first_doc_global_index=min(overlapping_docs),
                last_doc_global_index=max(overlapping_docs),
                document_count=len(set(overlapping_docs)),
            )
        )

        token_cursor = next_cursor
        shard_index += 1

    return chunks


def write_uint16_tokens(path: Path, token_ids: Sequence[int]) -> None:
    # Explicit overflow refusal instead of silent cast.
    for idx, token_id in enumerate(token_ids):
        if token_id < 0 or token_id > MAX_UINT16:
            raise ValueError(
                f"token id {token_id} at position {idx} cannot be represented as uint16"
            )

    arr = array("H", token_ids)
    if arr.itemsize != 2:
        raise RuntimeError("unexpected uint16 itemsize")
    path.write_bytes(arr.tobytes())


def read_uint16_tokens(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 2 != 0:
        raise ValueError(f"invalid uint16 shard size (odd byte count): {path}")
    arr = array("H")
    arr.frombytes(raw)
    return arr.tolist()


def write_shard_sidecar(path: Path, sidecar: Sp16kShardSidecar) -> None:
    write_json(path, asdict(sidecar))


def write_root_manifest(path: Path, root_manifest: Sp16kRootManifest) -> None:
    write_json(path, asdict(root_manifest))


def spot_decode_shard(
    shard_path: Path,
    *,
    tokenizer: SubwordTokenizer,
    eot_token_id: int,
    max_docs: int = 5,
) -> list[dict[str, Any]]:
    token_ids = read_uint16_tokens(shard_path)
    docs: list[list[int]] = []
    current: list[int] = []

    for token_id in token_ids:
        if token_id == eot_token_id:
            docs.append(current)
            current = []
            if len(docs) >= max_docs:
                break
        else:
            current.append(token_id)

    if current and len(docs) < max_docs:
        docs.append(current)

    out: list[dict[str, Any]] = []
    for idx, doc_ids in enumerate(docs):
        out.append(
            {
                "doc_index": idx,
                "token_count": len(doc_ids),
                "text": tokenizer.decode(doc_ids),
            }
        )
    return out


def _apply_normalization_policy(docs: Sequence[str]) -> list[str]:
    # v1 policy: identity normalization with preserved order and preserved boundaries.
    # Empty-doc handling is performed upstream by one-non-empty-line doc loader.
    return [str(doc) for doc in docs]


def _resolve_split_membership_semantics(
    split_manifest: Dict[str, Any],
    *,
    normalized_doc_count: int,
) -> dict[Literal["train", "val"], list[int]]:
    train = list(split_manifest["train_doc_indices"])
    val = list(split_manifest["val_doc_indices"])

    # Fail fast if normalization changed cardinality vs split manifest expectations.
    if split_manifest["total_docs"] != normalized_doc_count:
        raise ValueError(
            "normalized_doc_count mismatch with split manifest total_docs; "
            "normalization policy must preserve doc indexing for this contract"
        )

    seen: set[int] = set()
    for split_name, indices in (("train", train), ("val", val)):
        for idx in indices:
            if idx < 0 or idx >= normalized_doc_count:
                raise ValueError(
                    f"{split_name} split index {idx} out of bounds for "
                    f"normalized_doc_count={normalized_doc_count}"
                )
            if idx in seen:
                raise ValueError(f"split index {idx} appears in more than one split")
            seen.add(idx)

    if len(seen) != normalized_doc_count:
        raise ValueError(
            "split membership must cover all normalized docs exactly once for sp16k"
        )

    return {"train": train, "val": val}


def _apply_eot_insertion_policy(
    encoded_docs: Sequence[tuple[int, list[int]]],
    *,
    eot_id: int,
) -> tuple[list[int], list[Sp16kDocSpan]]:
    stream: list[int] = []
    spans: list[Sp16kDocSpan] = []

    # v1 policy: append EOT after every doc, including the final doc.
    for global_doc_index, encoded_ids in encoded_docs:
        start = len(stream)
        stream.extend(encoded_ids)
        stream.append(eot_id)
        end = len(stream)
        spans.append(
            Sp16kDocSpan(
                global_doc_index=global_doc_index,
                start_token=start,
                end_token=end,
            )
        )

    return stream, spans


def _summarize_split_counts(
    *,
    split_docs: Sequence[str],
    split_token_stream: Sequence[int],
    doc_spans: Sequence[Sp16kDocSpan],
) -> tuple[int, int]:
    # v1 semantics:
    # - document_count is the number of split docs post-normalization.
    # - token_count is the full flattened stream length, including EOT tokens.
    doc_count = len(split_docs)
    token_count = len(split_token_stream)

    if len(doc_spans) != doc_count:
        raise ValueError(
            f"doc span count mismatch: expected {doc_count}, got {len(doc_spans)}"
        )

    prev_end = 0
    covered_tokens = 0
    for i, span in enumerate(doc_spans):
        if span.start_token != prev_end:
            raise ValueError(
                f"non-contiguous doc spans at index {i}: start={span.start_token}, expected={prev_end}"
            )
        if span.end_token < span.start_token:
            raise ValueError(
                f"invalid doc span at index {i}: end={span.end_token} < start={span.start_token}"
            )
        covered_tokens += span.end_token - span.start_token
        prev_end = span.end_token

    if prev_end != token_count or covered_tokens != token_count:
        raise ValueError(
            "doc spans do not exactly cover split token stream under current counting semantics"
        )

    return doc_count, token_count


def build_sp16k_pretokenized_shards(
    *,
    input_file: Path,
    tokenizer_artifact_dir: Path,
    split_manifest_path: Path,
    output_dir: Path,
    max_tokens_per_shard: int,
) -> Path:
    docs = load_docs_one_per_non_empty_line(input_file)
    docs_sha256 = compute_docs_sha256(docs)

    split_manifest = read_split_manifest(split_manifest_path)
    tokenizer, tokenizer_hash = load_tokenizer_from_artifact_dir(tokenizer_artifact_dir)

    validate_strict_provenance(
        split_manifest,
        tokenizer_hash=tokenizer_hash,
        docs_sha256=docs_sha256,
        doc_count=len(docs),
    )

    normalized_docs = _apply_normalization_policy(docs)
    split_membership = _resolve_split_membership_semantics(
        split_manifest,
        normalized_doc_count=len(normalized_docs),
    )

    eot_id = tokenizer.token_to_id("<|endoftext|>")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_sha256 = sha256_hex_file(split_manifest_path)

    prep_config_payload = {
        "format_version": PRETOK_FORMAT_VERSION,
        "max_tokens_per_shard": max_tokens_per_shard,
        "normalization_policy_marker": NORMALIZATION_POLICY_MARKER,
        "eot_policy_marker": EOT_POLICY_MARKER,
    }
    prep_config_ref = build_prep_config_ref(prep_config_payload)

    shard_inventory: list[dict[str, Any]] = []
    total_docs_per_split: dict[str, int] = {"train": 0, "val": 0}
    total_tokens_per_split: dict[str, int] = {"train": 0, "val": 0}

    for split in ("train", "val"):
        split_literal = split  # for readability in dataclass construction
        split_indices = split_membership[split_literal]
        encoded_docs = [(idx, tokenizer.encode(normalized_docs[idx])) for idx in split_indices]
        split_token_stream, doc_spans = _apply_eot_insertion_policy(encoded_docs, eot_id=eot_id)

        doc_count, token_count = _summarize_split_counts(
            split_docs=[normalized_docs[idx] for idx in split_indices],
            split_token_stream=split_token_stream,
            doc_spans=doc_spans,
        )
        total_docs_per_split[split_literal] = doc_count
        total_tokens_per_split[split_literal] = token_count

        split_dir = output_dir / split_literal
        split_dir.mkdir(parents=True, exist_ok=True)

        chunks = chunk_split_token_stream(
            split=split_literal,
            token_stream=split_token_stream,
            doc_spans=doc_spans,
            max_tokens_per_shard=max_tokens_per_shard,
        )

        for chunk in chunks:
            basename = deterministic_shard_basename(split_literal, chunk.shard_index)
            shard_filename = f"{basename}.bin"
            sidecar_filename = f"{basename}.json"

            shard_path = split_dir / shard_filename
            sidecar_path = split_dir / sidecar_filename

            write_uint16_tokens(shard_path, chunk.token_ids)
            shard_sha = sha256_hex_file(shard_path)

            sidecar = Sp16kShardSidecar(
                format_version=PRETOK_FORMAT_VERSION,
                filename=shard_filename,
                split=split_literal,
                dtype=SUPPORTED_DTYPE,
                shard_index=chunk.shard_index,
                token_count=len(chunk.token_ids),
                document_count=chunk.document_count,
                tokenizer_hash=tokenizer_hash,
                split_manifest_sha256=split_manifest_sha256,
                sha256=shard_sha,
                first_doc_global_index=chunk.first_doc_global_index,
                last_doc_global_index=chunk.last_doc_global_index,
                prep_config_ref=prep_config_ref,
            )
            write_shard_sidecar(sidecar_path, sidecar)

            shard_inventory.append(
                {
                    "split": split_literal,
                    "bin": str((Path(split_literal) / shard_filename).as_posix()),
                    "sidecar": str((Path(split_literal) / sidecar_filename).as_posix()),
                }
            )

    root_manifest = Sp16kRootManifest(
        format_version=PRETOK_FORMAT_VERSION,
        tokenizer_provenance={
            "tokenizer_artifact_dir": str(tokenizer_artifact_dir),
            "tokenizer_hash": tokenizer_hash,
        },
        split_provenance={
            "split_manifest_path": str(split_manifest_path),
            "split_manifest_sha256": split_manifest_sha256,
            "docs_sha256": docs_sha256,
            "total_docs": len(docs),
            "split_seed": split_manifest.get("split_seed"),
        },
        shard_sizing_config={"max_tokens_per_shard": max_tokens_per_shard},
        normalization_policy_marker=NORMALIZATION_POLICY_MARKER,
        eot_policy_marker=EOT_POLICY_MARKER,
        shard_inventory=shard_inventory,
        total_docs_per_split=total_docs_per_split,
        total_tokens_per_split=total_tokens_per_split,
    )

    root_manifest_path = output_dir / ROOT_MANIFEST_FILENAME
    write_root_manifest(root_manifest_path, root_manifest)
    return root_manifest_path


# SP16K bridge END: sp16k pretokenized shard pipeline
