from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from llm_lab.core.data.pretok_dataset import TinyLlamaP15PretokIterableDataset
from llm_lab.core.data.pretok_dataset_contract import build_epoch_sampler_plan
from llm_lab.core.data.pretok_shards import PRETOK_FORMAT_VERSION, ROOT_MANIFEST_FILENAME, write_uint16_tokens
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.train.trainer import Trainer, TrainerConfig


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_sidecar(path: Path, *, split: str, shard_index: int, token_count: int) -> None:
    _write_json(
        path,
        {
            "format_version": PRETOK_FORMAT_VERSION,
            "filename": path.with_suffix(".bin").name,
            "split": split,
            "dtype": "uint16",
            "shard_index": shard_index,
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


def _add_shard(
    *,
    root: Path,
    inventory: list[dict[str, str]],
    split: str,
    shard_index: int,
    token_ids: list[int],
) -> None:
    bin_path = root / split / f"{split}_{shard_index:06d}.bin"
    sidecar_path = root / split / f"{split}_{shard_index:06d}.json"
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    write_uint16_tokens(bin_path, token_ids)
    _write_sidecar(sidecar_path, split=split, shard_index=shard_index, token_count=len(token_ids))
    inventory.append(
        {
            "split": split,
            "bin": str(Path(split) / bin_path.name),
            "sidecar": str(Path(split) / sidecar_path.name),
        }
    )


def _write_root_manifest(root: Path, inventory: list[dict[str, str]]) -> None:
    _write_json(
        root / ROOT_MANIFEST_FILENAME,
        {
            "format_version": PRETOK_FORMAT_VERSION,
            "tokenizer_provenance": {},
            "split_provenance": {},
            "shard_sizing_config": {},
            "normalization_policy_marker": "x",
            "eot_policy_marker": "y",
            "shard_inventory": inventory,
            "total_docs_per_split": {"train": 0, "val": 0},
            "total_tokens_per_split": {"train": 0, "val": 0},
        },
    )


def _build_boundary_tree(root: Path) -> None:
    inventory: list[dict[str, str]] = []
    # Inventory order is authoritative and intentionally non-trivial.
    _add_shard(root=root, inventory=inventory, split="train", shard_index=1, token_ids=[20, 21, 22, 23, 24, 25, 26])
    _add_shard(root=root, inventory=inventory, split="val", shard_index=0, token_ids=[90, 91, 92, 93, 94, 95])
    _add_shard(root=root, inventory=inventory, split="train", shard_index=0, token_ids=[10, 11, 12, 13, 14])  # N=B+1
    _add_shard(root=root, inventory=inventory, split="train", shard_index=2, token_ids=[30, 31, 32, 33])  # short
    _write_root_manifest(root, inventory)


def _build_three_shard_tree(root: Path) -> None:
    inventory: list[dict[str, str]] = []
    _add_shard(root=root, inventory=inventory, split="train", shard_index=0, token_ids=[1, 2, 3, 4, 5, 6])       # valid=2
    _add_shard(root=root, inventory=inventory, split="train", shard_index=1, token_ids=[7, 8, 9, 10, 11, 12, 13])  # valid=3
    _add_shard(root=root, inventory=inventory, split="train", shard_index=2, token_ids=[14, 15, 16, 17, 18, 19, 20, 21])  # valid=4
    _write_root_manifest(root, inventory)


def test_shapes_shift_boundaries_and_exact_len(tmp_path: Path) -> None:
    root = tmp_path / "shards"
    _build_boundary_tree(root)
    block_size = 4
    ds = TinyLlamaP15PretokIterableDataset(root_dir=root, split="train", block_size=block_size, base_seed=7)

    # Short shard accounting and exact pass cardinality.
    assert ds.short_shards_skipped == 1
    assert ds.short_tokens_skipped == 4
    assert ds.eligible_shard_count == 2
    assert len(ds) == ds.total_samples_per_pass == 4

    sample_count = sum(1 for _ in ds)
    assert sample_count == len(ds)

    x, y = next(iter(ds))
    assert x.shape == (block_size,)
    assert y.shape == (block_size,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert torch.equal(y[:-1], x[1:])

    refs = ds.sample_refs_for_current_epoch()
    # Eligible shard 0 (train_000001): N=7, B=4 -> valid offsets {0,1,2}; 3 is invalid.
    offsets0 = [o for shard_idx, o in refs if shard_idx == 0]
    assert set(offsets0) == {0, 1, 2}
    assert 3 not in offsets0
    # Eligible shard 1 (train_000000): N=B+1 -> exactly one sample at offset 0.
    offsets1 = [o for shard_idx, o in refs if shard_idx == 1]
    assert offsets1 == [0]


def test_all_short_shards_fail_construction(tmp_path: Path) -> None:
    root = tmp_path / "all_short"
    inventory: list[dict[str, str]] = []
    _add_shard(root=root, inventory=inventory, split="train", shard_index=0, token_ids=[1, 2, 3, 4])  # B=4 -> short
    _write_root_manifest(root, inventory)

    with pytest.raises(ValueError, match="no eligible shards"):
        TinyLlamaP15PretokIterableDataset(root_dir=root, split="train", block_size=4, base_seed=0)


def test_epoch_semantics_default_set_zero_repeatability_and_reorder(tmp_path: Path) -> None:
    root = tmp_path / "epoch_tree"
    _build_three_shard_tree(root)
    ds = TinyLlamaP15PretokIterableDataset(root_dir=root, split="train", block_size=4, base_seed=0)

    refs_default = ds.sample_refs_for_current_epoch()
    ds.set_epoch(0)
    refs_zero = ds.sample_refs_for_current_epoch()
    assert refs_default == refs_zero

    ds.set_epoch(1)
    refs_one_a = ds.sample_refs_for_current_epoch()
    refs_one_b = ds.sample_refs_for_current_epoch()
    assert refs_one_a == refs_one_b
    assert refs_one_a != refs_zero


def test_multi_shard_plan_matches_frozen_epoch_and_offset_permutations(tmp_path: Path) -> None:
    root = tmp_path / "plan_match"
    _build_three_shard_tree(root)
    ds = TinyLlamaP15PretokIterableDataset(root_dir=root, split="train", block_size=4, base_seed=0)
    ds.set_epoch(1)

    plan = build_epoch_sampler_plan(
        manifest_view=ds._manifest_view,  # intentional in-test verification of frozen plan wiring
        base_seed=ds.config.base_seed,
        epoch=ds.epoch,
    )
    expected: list[tuple[int, int]] = []
    for shard_idx, offsets in plan.iter_ordered_pairs():
        for offset in offsets:
            expected.append((shard_idx, offset))

    assert ds.sample_refs_for_current_epoch() == expected


def test_manifest_inventory_is_authoritative_not_filesystem_glob(tmp_path: Path) -> None:
    root = tmp_path / "manifest_authority"
    _build_boundary_tree(root)

    # Extra files not listed in root manifest must be ignored.
    extra_bin = root / "train" / "train_999999.bin"
    extra_sidecar = root / "train" / "train_999999.json"
    write_uint16_tokens(extra_bin, [100, 101, 102, 103, 104, 105, 106, 107, 108])
    _write_sidecar(extra_sidecar, split="train", shard_index=999999, token_count=9)

    ds = TinyLlamaP15PretokIterableDataset(root_dir=root, split="train", block_size=4, base_seed=7)
    assert len(ds) == 4  # unchanged from inventory-only eligible shards
    assert {shard_idx for shard_idx, _ in ds.sample_refs_for_current_epoch()} == {0, 1}


def test_iter_fails_fast_when_used_with_worker_processes(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "worker_guard"
    _build_boundary_tree(root)
    ds = TinyLlamaP15PretokIterableDataset(root_dir=root, split="train", block_size=4, base_seed=7)

    monkeypatch.setattr("llm_lab.core.data.pretok_dataset.get_worker_info", lambda: object())
    with pytest.raises(RuntimeError, match="single-worker mode only"):
        next(iter(ds))


def test_dataloader_one_batch_smoke_and_forward_loss(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    _build_boundary_tree(root)
    ds = TinyLlamaP15PretokIterableDataset(root_dir=root, split="train", block_size=4, base_seed=7)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    xb, yb = next(iter(loader))
    assert xb.shape == (2, 4)
    assert yb.shape == (2, 4)
    assert xb.dtype == torch.long
    assert yb.dtype == torch.long

    cfg = MiniGPTConfig(
        vocab_size=128,
        d_model=16,
        n_layers=1,
        n_heads=1,
        d_ff=32,
        block_size=4,
        dropout=0.0,
    )
    model = MiniGPT(cfg)
    logits, _ = model(xb)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_short_trainer_smoke_without_api_changes(tmp_path: Path) -> None:
    root = tmp_path / "trainer_smoke"
    _build_three_shard_tree(root)
    ds = TinyLlamaP15PretokIterableDataset(root_dir=root, split="train", block_size=4, base_seed=0)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    cfg = MiniGPTConfig(
        vocab_size=128,
        d_model=16,
        n_layers=1,
        n_heads=1,
        d_ff=32,
        block_size=4,
        dropout=0.0,
    )
    model = MiniGPT(cfg)
    trainer_cfg = TrainerConfig(
        device="cpu",
        lr=1e-3,
        max_grad_norm=1.0,
        log_every_n_steps=1,
        max_steps=1,
        num_epochs=1,
    )
    trainer = Trainer(model, loader, None, trainer_cfg)
    loss = trainer.train_epoch(epoch_index=0)
    assert isinstance(loss, float)
