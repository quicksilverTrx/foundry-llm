from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Sequence

from llm_lab.core.tokenization.subword_tokenizer import (
    RESERVED_SPECIAL_TOKENS,
    SubwordTokenizer,
    SubwordTokenizerConfig,
    get_pretokenizer_spec,
)

# TinyLlama bridge START: tinyllama_p15 tokenizer artifact pipeline
TOKENIZER_CONTRACT_VERSION_V1 = "tinyllama_p15_tokenizer_artifact_v1"
TOKENIZER_CONTRACT_VERSION_V2 = "tinyllama_p15_tokenizer_artifact_v2"
TOKENIZER_CONTRACT_VERSION = TOKENIZER_CONTRACT_VERSION_V2
DOC_SHAPE_POLICY = "one_non_empty_line_per_doc"
SPLIT_ALGORITHM = "tinyllama_p15_sha256(seed|index|doc_utf8) < train_ratio"
HASH_SCOPE = "tinyllama_p15_behavioral_core_plus_backend_payload_v2"

TOKENIZER_ARTIFACT_FILENAMES: Dict[str, str] = {
    "vocab": "vocab.json",
    "merges": "merges.txt",
    "sentencepiece_model": "sentencepiece.model",
    "external_id_map": "external_id_map.json",
    "sentencepiece_meta": "sentencepiece_meta.json",
    "config": "tokenizer_config.json",
    "stats": "tokenizer_stats.json",
    "reserved_tokens": "reserved_tokens.json",
    "split_manifest": "split_manifest.json",
    "tokenizer_hash": "tokenizer_hash.txt",
}

SplitScoreFn = Callable[[int, int, str], float]


@dataclass(frozen=True)
class DeterministicSplit:
    train_docs: List[str]
    val_docs: List[str]
    train_doc_indices: List[int]
    val_doc_indices: List[int]
    docs_sha256: str
    assignment_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split_score(seed: int, index: int, doc: str) -> float:
    payload = f"{seed}\n{index}\n{doc}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)


def load_docs_one_per_non_empty_line(input_file: Path) -> List[str]:
    docs = [line.strip() for line in input_file.read_text(encoding="utf-8").splitlines()]
    return [doc for doc in docs if doc]


def deterministic_split_docs(
    docs: Sequence[str],
    train_ratio: float,
    split_seed: int,
    *,
    split_score_fn: SplitScoreFn | None = None,
) -> DeterministicSplit:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not docs:
        raise ValueError("no documents found after applying one-non-empty-line-per-doc policy")

    score_fn = split_score_fn or _split_score
    train_docs: List[str] = []
    val_docs: List[str] = []
    train_doc_indices: List[int] = []
    val_doc_indices: List[int] = []
    assignments: List[int] = []

    for idx, doc in enumerate(docs):
        score = score_fn(split_seed, idx, doc)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"split score must be in [0, 1], got {score}")
        if score < train_ratio:
            train_docs.append(doc)
            train_doc_indices.append(idx)
            assignments.append(1)
        else:
            val_docs.append(doc)
            val_doc_indices.append(idx)
            assignments.append(0)

    docs_payload = _canonical_json_bytes(list(docs))
    assignment_payload = _canonical_json_bytes(assignments)
    return DeterministicSplit(
        train_docs=train_docs,
        val_docs=val_docs,
        train_doc_indices=train_doc_indices,
        val_doc_indices=val_doc_indices,
        docs_sha256=_sha256_hex(docs_payload),
        assignment_sha256=_sha256_hex(assignment_payload),
    )


def _validate_reserved_token_abi(tokenizer: SubwordTokenizer) -> None:
    for token, expected_id in RESERVED_SPECIAL_TOKENS.items():
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            raise ValueError(
                f"reserved token ABI mismatch for {token}: expected {expected_id}, got {actual_id}"
            )
    reserved_count = len(RESERVED_SPECIAL_TOKENS)
    for token, token_id in tokenizer.stoi.items():
        if token not in RESERVED_SPECIAL_TOKENS and token_id < reserved_count:
            raise ValueError(
                f"non-reserved token {token!r} uses reserved ID range [0, {reserved_count - 1}]"
            )


def _build_tokenizer_hash_payload(tokenizer: SubwordTokenizer) -> Dict[str, object]:
    backend_components = list(tokenizer.iter_backend_hash_components())

    def _pick_component(key: str) -> object:
        for component in backend_components:
            if key in component:
                return component[key]
        raise ValueError(f"missing backend hash component: {key}")

    payload = {
        "hash_scope": HASH_SCOPE,
        "hash_contract_version": "tinyllama_p15_tokenizer_hash_v2",
        "backend_family": tokenizer.backend_family,
        "reserved_tokens": RESERVED_SPECIAL_TOKENS,
        "pretokenizer_spec": get_pretokenizer_spec(),
    }

    if tokenizer.backend_family == "legacy_bpe":
        payload["backend_behavior"] = {
            "vocab_symbols_in_id_order": _pick_component("vocab_symbols_in_id_order"),
            "merges_in_rank_order": _pick_component("merges_in_rank_order"),
        }
        return payload

    if tokenizer.backend_family == "sentencepiece":
        sp_fingerprint = _pick_component("sentencepiece_behavior_fingerprint")
        external_map_payload = _pick_component("external_id_map")
        sentencepiece_meta = _pick_component("sentencepiece_meta")
        external_to_internal = external_map_payload["external_to_internal"]
        external_to_internal = {
            k: external_to_internal[k] for k in sorted(external_to_internal, key=lambda s: int(s))
        }
        payload["backend_behavior"] = {
            "pieces_in_id_order": sp_fingerprint["pieces_in_id_order"],
            "piece_scores_in_id_order": sp_fingerprint["piece_scores_in_id_order"],
            "external_to_internal": external_to_internal,
            "internal_model_type": sentencepiece_meta.get("internal_model_type"),
            "normalization_rule_name": sentencepiece_meta.get("normalization_rule_name"),
        }
        return payload

    raise ValueError(f"unsupported backend family for tokenizer hash: {tokenizer.backend_family}")


def compute_tokenizer_hash(tokenizer: SubwordTokenizer) -> str:
    return _sha256_hex(_canonical_json_bytes(_build_tokenizer_hash_payload(tokenizer)))


def _count_tokens(tokenizer: SubwordTokenizer, docs: Sequence[str]) -> tuple[int, int]:
    token_count = 0
    unencodable_doc_count = 0
    for doc in docs:
        try:
            token_count += len(tokenizer.encode(doc))
        except KeyError:
            unencodable_doc_count += 1
    return token_count, unencodable_doc_count


def load_tokenizer_from_artifact_dir(tokenizer_artifact_dir: Path) -> tuple[SubwordTokenizer, str]:
    tok_dir = Path(tokenizer_artifact_dir)
    tok_hash_path = tok_dir / TOKENIZER_ARTIFACT_FILENAMES["tokenizer_hash"]
    if not tok_hash_path.exists():
        raise FileNotFoundError(f"required tokenizer artifact file missing: {tok_hash_path}")
    tokenizer_hash = tok_hash_path.read_text(encoding="utf-8").strip()
    if not tokenizer_hash:
        raise ValueError("tokenizer_hash.txt is empty")
    tokenizer = SubwordTokenizer.load(tok_dir)
    return tokenizer, tokenizer_hash


def _tokenizer_payload_keys_for_backend(backend_family: str) -> List[str]:
    if backend_family == "legacy_bpe":
        return ["vocab", "merges"]
    if backend_family == "sentencepiece":
        return ["sentencepiece_model", "external_id_map", "sentencepiece_meta"]
    raise ValueError(f"Unsupported backend_family: {backend_family}")


def build_tokenizer_artifact_from_docs(
    docs: Sequence[str],
    output_dir: Path,
    *,
    vocab_size: int,
    model_type: Literal["bpe", "legacy_bpe", "sentencepiece"] = "sentencepiece",
    train_ratio: float = 0.98,
    split_seed: int = 0,
    split_score_fn: SplitScoreFn | None = None,
) -> Dict[str, Path]:
    split = deterministic_split_docs(
        docs, train_ratio=train_ratio, split_seed=split_seed, split_score_fn=split_score_fn
    )
    if not split.train_docs:
        raise ValueError("train split is empty; adjust train_ratio, split_seed, or corpus")

    config = SubwordTokenizerConfig(vocab_size=vocab_size, model_type=model_type)
    tokenizer = SubwordTokenizer.train_from_iterator(split.train_docs, config=config)
    _validate_reserved_token_abi(tokenizer)

    output_dir.mkdir(parents=True, exist_ok=True)
    emitted_payload_paths = tokenizer.export_backend_payload(output_dir)
    tokenizer_hash = compute_tokenizer_hash(tokenizer)

    train_char_count = sum(len(doc) for doc in split.train_docs)
    val_char_count = sum(len(doc) for doc in split.val_docs)
    train_token_count, train_unencodable = _count_tokens(tokenizer, split.train_docs)
    val_token_count, val_unencodable = _count_tokens(tokenizer, split.val_docs)

    tokenizer_config = {
        "contract_version": TOKENIZER_CONTRACT_VERSION,
        "model_type": model_type,
        "backend_family": tokenizer.backend_family,
        "requested_vocab_size": int(vocab_size),
        "actual_vocab_size": len(tokenizer.stoi),
        "split": {"train_ratio": train_ratio, "split_seed": split_seed, "algorithm": SPLIT_ALGORITHM},
        "doc_shape_policy": DOC_SHAPE_POLICY,
        "hash_scope": HASH_SCOPE,
        "backend_payload_files": _tokenizer_payload_keys_for_backend(tokenizer.backend_family),
    }
    tokenizer_stats = {
        "contract_version": TOKENIZER_CONTRACT_VERSION,
        "backend_family": tokenizer.backend_family,
        "document_counts": {
            "total": len(docs),
            "train": len(split.train_docs),
            "val": len(split.val_docs),
        },
        "character_counts": {"train": train_char_count, "val": val_char_count},
        "token_counts": {"train": train_token_count, "val": val_token_count},
        "unencodable_doc_count": {"train": train_unencodable, "val": val_unencodable},
        "merge_count": len(tokenizer.merges),
        "reserved_token_count": len(RESERVED_SPECIAL_TOKENS),
    }
    split_manifest = {
        "contract_version": TOKENIZER_CONTRACT_VERSION,
        "algorithm": SPLIT_ALGORITHM,
        "train_ratio": train_ratio,
        "split_seed": split_seed,
        "total_docs": len(docs),
        "train_doc_count": len(split.train_docs),
        "val_doc_count": len(split.val_docs),
        "train_doc_indices": split.train_doc_indices,
        "val_doc_indices": split.val_doc_indices,
        "docs_sha256": split.docs_sha256,
        "assignment_sha256": split.assignment_sha256,
        "tokenizer_hash": tokenizer_hash,
    }

    (output_dir / TOKENIZER_ARTIFACT_FILENAMES["config"]).write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / TOKENIZER_ARTIFACT_FILENAMES["stats"]).write_text(
        json.dumps(tokenizer_stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / TOKENIZER_ARTIFACT_FILENAMES["reserved_tokens"]).write_text(
        json.dumps(RESERVED_SPECIAL_TOKENS, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / TOKENIZER_ARTIFACT_FILENAMES["split_manifest"]).write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / TOKENIZER_ARTIFACT_FILENAMES["tokenizer_hash"]).write_text(
        tokenizer_hash + "\n", encoding="utf-8"
    )

    output_paths: Dict[str, Path] = {
        "config": output_dir / TOKENIZER_ARTIFACT_FILENAMES["config"],
        "stats": output_dir / TOKENIZER_ARTIFACT_FILENAMES["stats"],
        "reserved_tokens": output_dir / TOKENIZER_ARTIFACT_FILENAMES["reserved_tokens"],
        "split_manifest": output_dir / TOKENIZER_ARTIFACT_FILENAMES["split_manifest"],
        "tokenizer_hash": output_dir / TOKENIZER_ARTIFACT_FILENAMES["tokenizer_hash"],
    }
    output_paths.update(emitted_payload_paths)
    return output_paths


def build_tokenizer_artifact_from_file(
    input_file: Path,
    output_dir: Path,
    *,
    vocab_size: int,
    model_type: Literal["bpe", "legacy_bpe", "sentencepiece"] = "sentencepiece",
    train_ratio: float = 0.98,
    split_seed: int = 0,
) -> Dict[str, Path]:
    docs = load_docs_one_per_non_empty_line(input_file)
    return build_tokenizer_artifact_from_docs(
        docs=docs,
        output_dir=output_dir,
        vocab_size=vocab_size,
        model_type=model_type,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )


def build_tokenizer_artifact_from_train_file(
    input_file: Path,
    output_dir: Path,
    *,
    vocab_size: int,
    model_type: Literal["bpe", "legacy_bpe", "sentencepiece"] = "sentencepiece",
) -> Dict[str, Path]:
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if model_type != "sentencepiece":
        raise ValueError("build_tokenizer_artifact_from_train_file currently supports model_type='sentencepiece' only")

    doc_count = 0
    char_count = 0
    hasher = hashlib.sha256()
    with input_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            doc = raw_line.strip()
            if not doc:
                continue
            doc_count += 1
            char_count += len(doc)
            hasher.update(doc.encode("utf-8"))
            hasher.update(b"\n")
    if doc_count <= 0:
        raise ValueError(f"no documents found in tokenizer training file: {input_path}")
    docs_sha256 = hasher.hexdigest()

    config = SubwordTokenizerConfig(vocab_size=vocab_size, model_type=model_type)
    tokenizer = SubwordTokenizer.train_from_file(input_path, config=config)
    _validate_reserved_token_abi(tokenizer)

    output_dir.mkdir(parents=True, exist_ok=True)
    emitted_payload_paths = tokenizer.export_backend_payload(output_dir)
    tokenizer_hash = compute_tokenizer_hash(tokenizer)

    tokenizer_config = {
        "contract_version": TOKENIZER_CONTRACT_VERSION,
        "model_type": model_type,
        "backend_family": tokenizer.backend_family,
        "requested_vocab_size": int(vocab_size),
        "actual_vocab_size": len(tokenizer.stoi),
        "split": {"train_ratio": 1.0, "split_seed": 0, "algorithm": "tinyllama_p15_train_only_file_v1"},
        "doc_shape_policy": DOC_SHAPE_POLICY,
        "hash_scope": HASH_SCOPE,
        "backend_payload_files": _tokenizer_payload_keys_for_backend(tokenizer.backend_family),
        "training_input_file": str(input_path),
    }
    tokenizer_stats = {
        "contract_version": TOKENIZER_CONTRACT_VERSION,
        "backend_family": tokenizer.backend_family,
        "document_counts": {
            "total": doc_count,
            "train": doc_count,
            "val": 0,
        },
        "character_counts": {"train": char_count, "val": 0},
        "token_counts": {"train": 0, "val": 0},
        "unencodable_doc_count": {"train": 0, "val": 0},
        "merge_count": len(tokenizer.merges),
        "reserved_token_count": len(RESERVED_SPECIAL_TOKENS),
        "note": "Token counts omitted in train-only file mode for scalability.",
    }
    split_manifest = {
        "contract_version": TOKENIZER_CONTRACT_VERSION,
        "algorithm": "tinyllama_p15_train_only_file_v1",
        "train_ratio": 1.0,
        "split_seed": 0,
        "total_docs": doc_count,
        "train_doc_count": doc_count,
        "val_doc_count": 0,
        "train_doc_indices": [],
        "val_doc_indices": [],
        "docs_sha256": docs_sha256,
        "assignment_sha256": _sha256_hex(_canonical_json_bytes({"train_only": True, "total_docs": doc_count})),
        "tokenizer_hash": tokenizer_hash,
        "indices_omitted_for_scalability": True,
    }

    (output_dir / TOKENIZER_ARTIFACT_FILENAMES["config"]).write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / TOKENIZER_ARTIFACT_FILENAMES["stats"]).write_text(
        json.dumps(tokenizer_stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / TOKENIZER_ARTIFACT_FILENAMES["reserved_tokens"]).write_text(
        json.dumps(RESERVED_SPECIAL_TOKENS, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / TOKENIZER_ARTIFACT_FILENAMES["split_manifest"]).write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / TOKENIZER_ARTIFACT_FILENAMES["tokenizer_hash"]).write_text(
        tokenizer_hash + "\n", encoding="utf-8"
    )

    output_paths: Dict[str, Path] = {
        "config": output_dir / TOKENIZER_ARTIFACT_FILENAMES["config"],
        "stats": output_dir / TOKENIZER_ARTIFACT_FILENAMES["stats"],
        "reserved_tokens": output_dir / TOKENIZER_ARTIFACT_FILENAMES["reserved_tokens"],
        "split_manifest": output_dir / TOKENIZER_ARTIFACT_FILENAMES["split_manifest"],
        "tokenizer_hash": output_dir / TOKENIZER_ARTIFACT_FILENAMES["tokenizer_hash"],
    }
    output_paths.update(emitted_payload_paths)
    return output_paths


# TinyLlama bridge END: tinyllama_p15 tokenizer artifact pipeline
