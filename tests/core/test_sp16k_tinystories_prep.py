from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_lab.core.data.pretok_shards import build_sp16k_pretokenized_shards
from llm_lab.core.data.sp16k_tinystories_prep import (
    build_fixed_split_manifest,
    preflight_sp16k_train_config,
    write_fixed_split_manifest,
)
from llm_lab.core.tokenization.sp16k_tokenizer_artifact import build_tokenizer_artifact_from_docs, load_tokenizer_from_artifact_dir


def _docs_train() -> list[str]:
    return [
        "alpha beta gamma",
        "gamma beta alpha",
        "delta epsilon zeta",
        "eta theta iota",
    ]


def _docs_val() -> list[str]:
    return [
        "one two three",
        "three two one",
        "tiny validation sample",
    ]


def _write_docs(path: Path, docs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(docs) + "\n", encoding="utf-8")


def _build_tokenizer(train_docs: list[str], out_dir: Path) -> tuple[Path, str, int]:
    build_tokenizer_artifact_from_docs(
        docs=train_docs,
        output_dir=out_dir,
        vocab_size=128,
        model_type="sentencepiece",
        train_ratio=0.98,
        split_seed=0,
    )
    tok, tok_hash = load_tokenizer_from_artifact_dir(out_dir)
    return out_dir, tok_hash, len(tok.stoi)


def test_build_fixed_split_manifest_assigns_all_docs_to_target_split(tmp_path: Path) -> None:
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"
    _write_docs(train_file, _docs_train())
    _write_docs(val_file, _docs_val())

    train_manifest = build_fixed_split_manifest(
        input_file=train_file,
        split="train",
        tokenizer_hash="tokhash",
        split_seed=7,
    )
    assert train_manifest["train_doc_count"] == len(_docs_train())
    assert train_manifest["val_doc_count"] == 0
    assert train_manifest["train_doc_indices"] == list(range(len(_docs_train())))
    assert train_manifest["val_doc_indices"] == []

    val_manifest = build_fixed_split_manifest(
        input_file=val_file,
        split="val",
        tokenizer_hash="tokhash",
        split_seed=7,
    )
    assert val_manifest["train_doc_count"] == 0
    assert val_manifest["val_doc_count"] == len(_docs_val())
    assert val_manifest["train_doc_indices"] == []
    assert val_manifest["val_doc_indices"] == list(range(len(_docs_val())))


def test_pretok_train_val_separate_roots_and_preflight_pass(tmp_path: Path) -> None:
    train_file = tmp_path / "data" / "train.txt"
    val_file = tmp_path / "data" / "valid.txt"
    _write_docs(train_file, _docs_train())
    _write_docs(val_file, _docs_val())

    tokenizer_dir, tokenizer_hash, vocab_size = _build_tokenizer(_docs_train(), tmp_path / "tokenizer")

    manifests_dir = tmp_path / "manifests"
    train_manifest = write_fixed_split_manifest(
        input_file=train_file,
        split="train",
        tokenizer_hash=tokenizer_hash,
        output_path=manifests_dir / "train_split_manifest.json",
    )
    val_manifest = write_fixed_split_manifest(
        input_file=val_file,
        split="val",
        tokenizer_hash=tokenizer_hash,
        output_path=manifests_dir / "val_split_manifest.json",
    )

    shards_train = tmp_path / "artifacts" / "shards_train"
    shards_val = tmp_path / "artifacts" / "shards_val"
    build_sp16k_pretokenized_shards(
        input_file=train_file,
        tokenizer_artifact_dir=tokenizer_dir,
        split_manifest_path=train_manifest,
        output_dir=shards_train,
        max_tokens_per_shard=1024,
    )
    build_sp16k_pretokenized_shards(
        input_file=val_file,
        tokenizer_artifact_dir=tokenizer_dir,
        split_manifest_path=val_manifest,
        output_dir=shards_val,
        max_tokens_per_shard=1024,
    )

    cfg = {
        "model": {
            "arch_family": "nanollama",
            "vocab_size": vocab_size,
            "block_size": 4,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "num_kv_heads": 1,
            "d_ff": 64,
            "dropout": 0.0,
            "norm_type": "rmsnorm",
            "mlp_type": "swiglu",
            "attention_type": "gqa",
            "pos_encoding_type": "rope",
        },
        "tokenizer": {
            "backend_family": "sentencepiece",
            "artifact_dir": str(tokenizer_dir),
            "tokenizer_hash": tokenizer_hash,
        },
        "data": {
            "train_root_dir": str(shards_train),
            "val_root_dir": str(shards_val),
            "train_split": "train",
            "val_split": "val",
            "base_seed": 0,
        },
        "train": {"dtype": "fp32", "lr": 0.01, "eval_every_n_steps": 2},
        "run": {"output_root": str(tmp_path / "runs"), "run_name": "x"},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    report = preflight_sp16k_train_config(cfg_path)
    assert report["tokenizer"]["tokenizer_hash"] == tokenizer_hash
    assert report["eligible_shards"]["train"] >= 1
    assert report["eligible_shards"]["val"] >= 1


def test_preflight_fails_on_vocab_hash_and_no_eligible_shards(tmp_path: Path) -> None:
    train_file = tmp_path / "data" / "train.txt"
    val_file = tmp_path / "data" / "valid.txt"
    _write_docs(train_file, _docs_train())
    _write_docs(val_file, _docs_val())

    tokenizer_dir, tokenizer_hash, vocab_size = _build_tokenizer(_docs_train(), tmp_path / "tokenizer")

    manifests_dir = tmp_path / "manifests"
    train_manifest = write_fixed_split_manifest(
        input_file=train_file,
        split="train",
        tokenizer_hash=tokenizer_hash,
        output_path=manifests_dir / "train_split_manifest.json",
    )
    val_manifest = write_fixed_split_manifest(
        input_file=val_file,
        split="val",
        tokenizer_hash=tokenizer_hash,
        output_path=manifests_dir / "val_split_manifest.json",
    )

    shards_train = tmp_path / "artifacts" / "shards_train"
    shards_val = tmp_path / "artifacts" / "shards_val"
    build_sp16k_pretokenized_shards(
        input_file=train_file,
        tokenizer_artifact_dir=tokenizer_dir,
        split_manifest_path=train_manifest,
        output_dir=shards_train,
        max_tokens_per_shard=1024,
    )
    build_sp16k_pretokenized_shards(
        input_file=val_file,
        tokenizer_artifact_dir=tokenizer_dir,
        split_manifest_path=val_manifest,
        output_dir=shards_val,
        max_tokens_per_shard=1024,
    )

    base_cfg = {
        "model": {
            "arch_family": "nanollama",
            "vocab_size": vocab_size,
            "block_size": 4,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "num_kv_heads": 1,
            "d_ff": 64,
            "dropout": 0.0,
            "norm_type": "rmsnorm",
            "mlp_type": "swiglu",
            "attention_type": "gqa",
            "pos_encoding_type": "rope",
        },
        "tokenizer": {
            "backend_family": "sentencepiece",
            "artifact_dir": str(tokenizer_dir),
            "tokenizer_hash": tokenizer_hash,
        },
        "data": {
            "train_root_dir": str(shards_train),
            "val_root_dir": str(shards_val),
            "train_split": "train",
            "val_split": "val",
            "base_seed": 0,
        },
        "train": {"dtype": "fp32", "lr": 0.01, "eval_every_n_steps": 2},
        "run": {"output_root": str(tmp_path / "runs"), "run_name": "x"},
    }

    bad_vocab = dict(base_cfg)
    bad_vocab["model"] = dict(base_cfg["model"])
    bad_vocab["model"]["vocab_size"] = vocab_size + 1
    bad_vocab_path = tmp_path / "bad_vocab.json"
    bad_vocab_path.write_text(json.dumps(bad_vocab, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="vocab size"):
        preflight_sp16k_train_config(bad_vocab_path)

    bad_hash = dict(base_cfg)
    bad_hash["tokenizer"] = dict(base_cfg["tokenizer"])
    bad_hash["tokenizer"]["tokenizer_hash"] = "bad_hash"
    bad_hash_path = tmp_path / "bad_hash.json"
    bad_hash_path.write_text(json.dumps(bad_hash, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        preflight_sp16k_train_config(bad_hash_path)

    no_eligible = dict(base_cfg)
    no_eligible["model"] = dict(base_cfg["model"])
    no_eligible["model"]["block_size"] = 10000
    no_eligible_path = tmp_path / "no_eligible.json"
    no_eligible_path.write_text(json.dumps(no_eligible, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="no eligible shards"):
        preflight_sp16k_train_config(no_eligible_path)
