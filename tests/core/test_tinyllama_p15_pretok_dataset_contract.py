from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llm_lab.core.data.pretok_dataset_contract import (
    TinyLlamaP15PretokDatasetConfig,
    build_epoch_sampler_plan,
    epoch_shard_permutation,
    last_valid_offset,
    load_runtime_manifest_view,
    offset_permutation,
    valid_offset_count,
)
from llm_lab.core.data.pretok_shards import PRETOK_FORMAT_VERSION, ROOT_MANIFEST_FILENAME, write_uint16_tokens


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_sidecar(path: Path, *, split: str, token_count: int, dtype: str = "uint16") -> None:
    _write_json(
        path,
        {
            "format_version": PRETOK_FORMAT_VERSION,
            "filename": path.with_suffix(".bin").name,
            "split": split,
            "dtype": dtype,
            "shard_index": 0,
            "token_count": token_count,
            "document_count": max(1, token_count),
            "tokenizer_hash": "tokhash",
            "split_manifest_sha256": "splithash",
            "sha256": "shardhash",
            "first_doc_global_index": 0,
            "last_doc_global_index": 0,
            "prep_config_ref": "prep",
        },
    )


def test_valid_offset_formula_edges() -> None:
    b = 8
    assert valid_offset_count(token_count=8, block_size=b) == 0
    assert last_valid_offset(token_count=8, block_size=b) is None

    assert valid_offset_count(token_count=9, block_size=b) == 1
    assert last_valid_offset(token_count=9, block_size=b) == 0

    assert valid_offset_count(token_count=12, block_size=b) == 4
    assert last_valid_offset(token_count=12, block_size=b) == 3


def test_manifest_view_filters_split_skips_short_and_keeps_inventory_order(tmp_path: Path) -> None:
    root = tmp_path

    # train shard 0: eligible (N=9, B=8 => 1 sample)
    train0_bin = root / "train" / "train_000000.bin"
    train0_sidecar = root / "train" / "train_000000.json"
    train0_bin.parent.mkdir(parents=True, exist_ok=True)
    write_uint16_tokens(train0_bin, list(range(9)))
    _write_sidecar(train0_sidecar, split="train", token_count=9)

    # val shard: ignored for train split
    val0_bin = root / "val" / "val_000000.bin"
    val0_sidecar = root / "val" / "val_000000.json"
    val0_bin.parent.mkdir(parents=True, exist_ok=True)
    write_uint16_tokens(val0_bin, list(range(15)))
    _write_sidecar(val0_sidecar, split="val", token_count=15)

    # train shard 1: eligible (N=12, B=8 => 4 samples)
    train1_bin = root / "train" / "train_000001.bin"
    train1_sidecar = root / "train" / "train_000001.json"
    train1_bin.parent.mkdir(parents=True, exist_ok=True)
    write_uint16_tokens(train1_bin, list(range(12)))
    _write_sidecar(train1_sidecar, split="train", token_count=12)

    # train shard 2: short (N=8, B=8 => 0 samples)
    train2_bin = root / "train" / "train_000002.bin"
    train2_sidecar = root / "train" / "train_000002.json"
    train2_bin.parent.mkdir(parents=True, exist_ok=True)
    write_uint16_tokens(train2_bin, list(range(8)))
    _write_sidecar(train2_sidecar, split="train", token_count=8)

    _write_json(
        root / ROOT_MANIFEST_FILENAME,
        {
            "format_version": PRETOK_FORMAT_VERSION,
            "tokenizer_provenance": {},
            "split_provenance": {},
            "shard_sizing_config": {},
            "normalization_policy_marker": "x",
            "eot_policy_marker": "y",
            "shard_inventory": [
                {"split": "train", "bin": "train/train_000000.bin", "sidecar": "train/train_000000.json"},
                {"split": "val", "bin": "val/val_000000.bin", "sidecar": "val/val_000000.json"},
                {"split": "train", "bin": "train/train_000001.bin", "sidecar": "train/train_000001.json"},
                {"split": "train", "bin": "train/train_000002.bin", "sidecar": "train/train_000002.json"},
            ],
            "total_docs_per_split": {"train": 0, "val": 0},
            "total_tokens_per_split": {"train": 0, "val": 0},
        },
    )

    view = load_runtime_manifest_view(
        TinyLlamaP15PretokDatasetConfig(root_dir=root, split="train", block_size=8)
    )

    assert [s.inventory_index for s in view.eligible_shards] == [0, 2]
    assert [s.valid_offsets for s in view.eligible_shards] == [1, 4]
    assert view.short_shard_stats.short_shards_skipped == 1
    assert view.short_shard_stats.short_tokens_skipped == 8
    assert view.total_samples_per_pass == 5


def test_manifest_view_fails_on_uint16_size_mismatch(tmp_path: Path) -> None:
    root = tmp_path
    bin_path = root / "train" / "train_000000.bin"
    sidecar_path = root / "train" / "train_000000.json"
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    # Write 8 tokens, claim 9 in sidecar.
    write_uint16_tokens(bin_path, list(range(8)))
    _write_sidecar(sidecar_path, split="train", token_count=9)

    _write_json(
        root / ROOT_MANIFEST_FILENAME,
        {
            "format_version": PRETOK_FORMAT_VERSION,
            "shard_inventory": [
                {"split": "train", "bin": "train/train_000000.bin", "sidecar": "train/train_000000.json"}
            ],
        },
    )

    with pytest.raises(ValueError, match="byte size mismatch"):
        load_runtime_manifest_view(
            TinyLlamaP15PretokDatasetConfig(root_dir=root, split="train", block_size=8)
        )


def test_manifest_view_fails_if_no_eligible_shards(tmp_path: Path) -> None:
    root = tmp_path
    bin_path = root / "train" / "train_000000.bin"
    sidecar_path = root / "train" / "train_000000.json"
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    # Exactly block_size tokens => short under frozen rule.
    write_uint16_tokens(bin_path, list(range(8)))
    _write_sidecar(sidecar_path, split="train", token_count=8)

    _write_json(
        root / ROOT_MANIFEST_FILENAME,
        {
            "format_version": PRETOK_FORMAT_VERSION,
            "shard_inventory": [
                {"split": "train", "bin": "train/train_000000.bin", "sidecar": "train/train_000000.json"}
            ],
        },
    )

    with pytest.raises(ValueError, match="no eligible shards"):
        load_runtime_manifest_view(
            TinyLlamaP15PretokDatasetConfig(root_dir=root, split="train", block_size=8)
        )


def test_epoch_permutation_and_plan_are_seeded_and_reproducible() -> None:
    p1 = epoch_shard_permutation(num_shards=4, base_seed=42, epoch=0)
    p2 = epoch_shard_permutation(num_shards=4, base_seed=42, epoch=0)
    p3 = epoch_shard_permutation(num_shards=4, base_seed=42, epoch=1)

    assert p1 == p2
    assert sorted(p1) == [0, 1, 2, 3]
    assert p3 != p1

    shard0 = type("_Shard", (), {"valid_offsets": 1})()
    shard1 = type("_Shard", (), {"valid_offsets": 3})()
    shard2 = type("_Shard", (), {"valid_offsets": 2})()
    view = type("_View", (), {"eligible_shards": [shard0, shard1, shard2]})()
    plan_a = build_epoch_sampler_plan(manifest_view=view, base_seed=7, epoch=3)
    plan_b = build_epoch_sampler_plan(manifest_view=view, base_seed=7, epoch=3)
    plan_c = build_epoch_sampler_plan(manifest_view=view, base_seed=7, epoch=4)

    assert plan_a.shard_order == plan_b.shard_order
    assert plan_c.shard_order != plan_a.shard_order

    pairs_a = plan_a.iter_ordered_pairs()
    pairs_b = plan_b.iter_ordered_pairs()
    assert pairs_a == pairs_b
    assert [idx for idx, _ in pairs_a] == plan_a.shard_order
    for shard_idx, offsets in pairs_a:
        assert sorted(offsets) == list(range(plan_a.valid_offsets_per_shard[shard_idx]))


def test_offset_permutation_is_without_replacement() -> None:
    rng = np.random.Generator(np.random.PCG64(123))
    offsets = offset_permutation(valid_offsets=6, rng=rng)
    assert sorted(offsets) == [0, 1, 2, 3, 4, 5]
