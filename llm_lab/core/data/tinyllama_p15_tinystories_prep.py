from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal

from llm_lab.core.data.pretok_dataset_contract import (
    TinyLlamaP15PretokDatasetConfig,
    load_runtime_manifest_view,
)
from llm_lab.core.data.pretok_shards import (
    REQUIRED_SPLIT_MANIFEST_KEYS,
    ROOT_MANIFEST_FILENAME,
    validate_split_manifest_structure,
)
from llm_lab.core.tokenization.tinyllama_p15_tokenizer_artifact import (
    TOKENIZER_CONTRACT_VERSION,
    load_tokenizer_from_artifact_dir,
)

RuntimeSplit = Literal["train", "val"]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _iter_non_empty_docs(input_file: Path) -> Iterable[str]:
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            doc = line.strip()
            if doc:
                yield doc


def _stream_json_array_sha256(items: Iterable[str | int]) -> tuple[int, str]:
    hasher = hashlib.sha256()
    count = 0
    hasher.update(b"[")
    first = True
    for item in items:
        if not first:
            hasher.update(b",")
        hasher.update(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        first = False
        count += 1
    hasher.update(b"]")
    return count, hasher.hexdigest()


def docs_sha256_for_file(input_file: Path) -> tuple[int, str]:
    return _stream_json_array_sha256(_iter_non_empty_docs(input_file))


def _assignment_sha_for_fixed_split(*, total_docs: int, split: RuntimeSplit) -> str:
    value = 1 if split == "train" else 0
    _, digest = _stream_json_array_sha256(value for _ in range(total_docs))
    return digest


def build_fixed_split_manifest(
    *,
    input_file: Path,
    split: RuntimeSplit,
    tokenizer_hash: str,
    split_seed: int = 0,
) -> dict[str, Any]:
    total_docs, docs_sha = docs_sha256_for_file(input_file)
    if split == "train":
        train_doc_indices = list(range(total_docs))
        val_doc_indices: list[int] = []
    else:
        train_doc_indices = []
        val_doc_indices = list(range(total_docs))

    manifest = {
        "contract_version": TOKENIZER_CONTRACT_VERSION,
        "algorithm": f"tinyllama_p15_fixed_all_to_{split}_v1",
        "train_ratio": 1.0 if split == "train" else 0.0,
        "split_seed": int(split_seed),
        "total_docs": total_docs,
        "train_doc_count": len(train_doc_indices),
        "val_doc_count": len(val_doc_indices),
        "train_doc_indices": train_doc_indices,
        "val_doc_indices": val_doc_indices,
        "docs_sha256": docs_sha,
        "assignment_sha256": _assignment_sha_for_fixed_split(total_docs=total_docs, split=split),
        "tokenizer_hash": str(tokenizer_hash),
    }
    # Keep explicit guard in case upstream required keys evolve.
    missing = REQUIRED_SPLIT_MANIFEST_KEYS - set(manifest.keys())
    if missing:
        raise ValueError(f"generated split manifest is missing required keys: {sorted(missing)}")
    validate_split_manifest_structure(manifest)
    return manifest


def write_fixed_split_manifest(
    *,
    input_file: Path,
    split: RuntimeSplit,
    tokenizer_hash: str,
    output_path: Path,
    split_seed: int = 0,
) -> Path:
    manifest = build_fixed_split_manifest(
        input_file=input_file,
        split=split,
        tokenizer_hash=tokenizer_hash,
        split_seed=split_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _read_root_manifest_tokenizer_hash(root_dir: Path) -> str:
    root_manifest_path = root_dir / ROOT_MANIFEST_FILENAME
    payload = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {root_manifest_path}")
    tok_prov = payload.get("tokenizer_provenance")
    if not isinstance(tok_prov, dict):
        raise ValueError(f"tokenizer_provenance missing in {root_manifest_path}")
    tok_hash = tok_prov.get("tokenizer_hash")
    if not isinstance(tok_hash, str) or not tok_hash:
        raise ValueError(f"tokenizer_provenance.tokenizer_hash missing in {root_manifest_path}")
    return tok_hash


def _resolve_from_config_dir(raw: str, *, config_dir: Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (config_dir / p).resolve()
    return p


def preflight_p15_train_config(config_path: Path) -> dict[str, Any]:
    cfg_path = Path(config_path).expanduser().resolve()
    config_dir = cfg_path.parent
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {cfg_path}")

    model_cfg = dict(payload.get("model", {}))
    tokenizer_cfg = dict(payload.get("tokenizer", {}))
    data_cfg = dict(payload.get("data", {}))

    tokenizer_artifact_dir = _resolve_from_config_dir(str(tokenizer_cfg["artifact_dir"]), config_dir=config_dir)
    train_root_dir = _resolve_from_config_dir(str(data_cfg["train_root_dir"]), config_dir=config_dir)
    val_root_dir = _resolve_from_config_dir(str(data_cfg["val_root_dir"]), config_dir=config_dir)
    train_split = str(data_cfg.get("train_split", "train"))
    val_split = str(data_cfg.get("val_split", "val"))

    tokenizer, tokenizer_hash = load_tokenizer_from_artifact_dir(tokenizer_artifact_dir)
    tokenizer_vocab_size = len(tokenizer.stoi)
    model_vocab_size = int(model_cfg["vocab_size"])
    if tokenizer_vocab_size != model_vocab_size:
        raise ValueError(
            "tokenizer vocab size does not match model vocab size "
            f"(tokenizer={tokenizer_vocab_size}, model={model_vocab_size})"
        )

    cfg_tokenizer_hash = str(tokenizer_cfg.get("tokenizer_hash", "")).strip()
    if cfg_tokenizer_hash and cfg_tokenizer_hash != tokenizer_hash:
        raise ValueError(
            "tokenizer hash mismatch between config and artifact "
            f"(config={cfg_tokenizer_hash}, artifact={tokenizer_hash})"
        )

    block_size = int(model_cfg["block_size"])
    train_view = load_runtime_manifest_view(
        TinyLlamaP15PretokDatasetConfig(
            root_dir=train_root_dir,
            split=train_split,  # type: ignore[arg-type]
            block_size=block_size,
            base_seed=int(data_cfg.get("base_seed", 0)),
        )
    )
    val_view = load_runtime_manifest_view(
        TinyLlamaP15PretokDatasetConfig(
            root_dir=val_root_dir,
            split=val_split,  # type: ignore[arg-type]
            block_size=block_size,
            base_seed=int(data_cfg.get("base_seed", 0)),
        )
    )

    train_root_hash = _read_root_manifest_tokenizer_hash(train_root_dir)
    val_root_hash = _read_root_manifest_tokenizer_hash(val_root_dir)
    expected_hashes = [tokenizer_hash, train_root_hash, val_root_hash]
    if cfg_tokenizer_hash:
        expected_hashes.append(cfg_tokenizer_hash)
    if len(set(expected_hashes)) != 1:
        raise ValueError(
            "tokenizer hash mismatch across config/artifact/manifests: "
            f"config={cfg_tokenizer_hash or None}, artifact={tokenizer_hash}, "
            f"train_manifest={train_root_hash}, val_manifest={val_root_hash}"
        )

    return {
        "config_path": str(cfg_path),
        "resolved_paths": {
            "tokenizer_artifact_dir": str(tokenizer_artifact_dir),
            "train_root_dir": str(train_root_dir),
            "val_root_dir": str(val_root_dir),
        },
        "tokenizer": {
            "backend_family": tokenizer.backend_family,
            "tokenizer_hash": tokenizer_hash,
            "config_tokenizer_hash": cfg_tokenizer_hash or None,
            "vocab_size": tokenizer_vocab_size,
        },
        "model": {
            "vocab_size": model_vocab_size,
            "block_size": block_size,
        },
        "eligible_shards": {
            "train": len(train_view.eligible_shards),
            "val": len(val_view.eligible_shards),
        },
        "samples_per_pass": {
            "train": train_view.total_samples_per_pass,
            "val": val_view.total_samples_per_pass,
        },
    }
