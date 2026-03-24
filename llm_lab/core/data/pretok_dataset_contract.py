from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from llm_lab.core.data.pretok_shards import PRETOK_FORMAT_VERSION, ROOT_MANIFEST_FILENAME

RuntimeSplit = Literal["train", "val"]
RUNTIME_SUPPORTED_DTYPE = "uint16"
REQUIRED_ROOT_ENTRY_KEYS = frozenset({"split", "bin", "sidecar"})
REQUIRED_SIDECAR_KEYS = frozenset(
    {"format_version", "split", "dtype", "token_count", "filename", "tokenizer_hash"}
)


@dataclass(frozen=True)
class TinyLlamaP15PretokDatasetConfig:
    root_dir: Path
    split: RuntimeSplit
    block_size: int
    base_seed: int = 0


@dataclass(frozen=True)
class TinyLlamaP15RuntimeShardRef:
    inventory_index: int
    bin_path: Path
    sidecar_path: Path
    token_count: int
    valid_offsets: int


@dataclass(frozen=True)
class TinyLlamaP15ShortShardStats:
    short_shards_skipped: int
    short_tokens_skipped: int


@dataclass(frozen=True)
class TinyLlamaP15RuntimeManifestView:
    split: RuntimeSplit
    block_size: int
    eligible_shards: list[TinyLlamaP15RuntimeShardRef]
    short_shard_stats: TinyLlamaP15ShortShardStats
    total_samples_per_pass: int


@dataclass(frozen=True)
class TinyLlamaP15EpochSamplerPlan:
    base_seed: int
    epoch: int
    valid_offsets_per_shard: list[int]
    shard_order: list[int]

    def iter_ordered_pairs(self) -> list[tuple[int, list[int]]]:
        num_shards = len(self.valid_offsets_per_shard)
        for shard_idx in self.shard_order:
            if shard_idx < 0 or shard_idx >= num_shards:
                raise ValueError(
                    f"shard_order contains out-of-range index {shard_idx} for {num_shards} shards"
                )

        rng = build_epoch_rng(self.base_seed, self.epoch)
        out: list[tuple[int, list[int]]] = []
        for shard_idx in self.shard_order:
            valid_offsets = self.valid_offsets_per_shard[shard_idx]
            offsets = offset_permutation(valid_offsets=valid_offsets, rng=rng)
            out.append((shard_idx, offsets))
        return out


def valid_offset_count(token_count: int, block_size: int) -> int:
    """Returns max(0, N - B) where N=token_count, B=block_size."""
    _validate_non_negative_int("token_count", token_count)
    _validate_positive_int("block_size", block_size)
    return max(0, token_count - block_size)


def last_valid_offset(token_count: int, block_size: int) -> int | None:
    """Returns N-B-1 if N >= B+1 else None."""
    count = valid_offset_count(token_count, block_size)
    if count == 0:
        return None
    return token_count - block_size - 1


def is_eligible_shard(token_count: int, block_size: int) -> bool:
    return valid_offset_count(token_count, block_size) > 0


def build_epoch_rng(base_seed: int, epoch: int) -> np.random.Generator:
    _validate_non_negative_int("epoch", epoch)
    return np.random.Generator(np.random.PCG64(int(base_seed) + int(epoch)))


def epoch_shard_permutation(*, num_shards: int, base_seed: int, epoch: int) -> list[int]:
    _validate_non_negative_int("num_shards", num_shards)
    rng = build_epoch_rng(base_seed, epoch)
    if num_shards == 0:
        return []
    return [int(x) for x in rng.permutation(num_shards)]


def offset_permutation(*, valid_offsets: int, rng: np.random.Generator) -> list[int]:
    _validate_non_negative_int("valid_offsets", valid_offsets)
    if valid_offsets == 0:
        return []
    return [int(x) for x in rng.permutation(valid_offsets)]


def load_runtime_manifest_view(config: TinyLlamaP15PretokDatasetConfig) -> TinyLlamaP15RuntimeManifestView:
    _validate_positive_int("block_size", config.block_size)

    root_manifest_path = config.root_dir / ROOT_MANIFEST_FILENAME
    root = _read_json_object(root_manifest_path)

    root_format = root.get("format_version")
    if root_format != PRETOK_FORMAT_VERSION:
        raise ValueError(
            f"root manifest format_version mismatch: expected {PRETOK_FORMAT_VERSION}, got {root_format!r}"
        )

    inventory = root.get("shard_inventory")
    if not isinstance(inventory, list):
        raise ValueError("root manifest shard_inventory must be a list")

    eligible: list[TinyLlamaP15RuntimeShardRef] = []
    short_shards = 0
    short_tokens = 0

    for inventory_index, entry in enumerate(inventory):
        if not isinstance(entry, dict):
            raise ValueError(f"shard_inventory[{inventory_index}] must be an object")

        missing = REQUIRED_ROOT_ENTRY_KEYS - set(entry.keys())
        if missing:
            raise ValueError(f"shard_inventory[{inventory_index}] missing required keys: {sorted(missing)}")

        entry_split = entry["split"]
        if entry_split != config.split:
            continue

        bin_path = config.root_dir / Path(str(entry["bin"]))
        sidecar_path = config.root_dir / Path(str(entry["sidecar"]))
        sidecar = _read_json_object(sidecar_path)

        missing_sidecar = REQUIRED_SIDECAR_KEYS - set(sidecar.keys())
        if missing_sidecar:
            raise ValueError(
                f"sidecar missing required keys at {sidecar_path}: {sorted(missing_sidecar)}"
            )

        sidecar_split = sidecar["split"]
        if sidecar_split != config.split:
            raise ValueError(
                f"split mismatch for sidecar {sidecar_path}: expected {config.split}, got {sidecar_split!r}"
            )

        sidecar_format = sidecar["format_version"]
        if sidecar_format != PRETOK_FORMAT_VERSION:
            raise ValueError(
                f"sidecar format_version mismatch for {sidecar_path}: "
                f"expected {PRETOK_FORMAT_VERSION}, got {sidecar_format!r}"
            )

        sidecar_dtype = sidecar["dtype"]
        if sidecar_dtype != RUNTIME_SUPPORTED_DTYPE:
            raise ValueError(
                f"sidecar dtype mismatch for {sidecar_path}: "
                f"expected {RUNTIME_SUPPORTED_DTYPE}, got {sidecar_dtype!r}"
            )

        token_count = sidecar["token_count"]
        if not isinstance(token_count, int) or token_count < 0:
            raise ValueError(f"sidecar token_count must be non-negative int at {sidecar_path}")

        _validate_bin_size(bin_path, token_count=token_count)

        offsets = valid_offset_count(token_count, config.block_size)
        if offsets == 0:
            short_shards += 1
            short_tokens += token_count
            continue

        eligible.append(
            TinyLlamaP15RuntimeShardRef(
                inventory_index=inventory_index,
                bin_path=bin_path,
                sidecar_path=sidecar_path,
                token_count=token_count,
                valid_offsets=offsets,
            )
        )

    if not eligible:
        raise ValueError(
            "no eligible shards for split after short-shard filtering; "
            f"split={config.split} block_size={config.block_size}"
        )

    total_samples = sum(shard.valid_offsets for shard in eligible)
    return TinyLlamaP15RuntimeManifestView(
        split=config.split,
        block_size=config.block_size,
        eligible_shards=eligible,
        short_shard_stats=TinyLlamaP15ShortShardStats(
            short_shards_skipped=short_shards,
            short_tokens_skipped=short_tokens,
        ),
        total_samples_per_pass=total_samples,
    )


def build_epoch_sampler_plan(
    *,
    manifest_view: TinyLlamaP15RuntimeManifestView,
    base_seed: int,
    epoch: int,
) -> TinyLlamaP15EpochSamplerPlan:
    valid_offsets_per_shard = [int(shard.valid_offsets) for shard in manifest_view.eligible_shards]
    order = epoch_shard_permutation(
        num_shards=len(valid_offsets_per_shard),
        base_seed=base_seed,
        epoch=epoch,
    )
    return TinyLlamaP15EpochSamplerPlan(
        base_seed=base_seed,
        epoch=epoch,
        valid_offsets_per_shard=valid_offsets_per_shard,
        shard_order=order,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _validate_bin_size(path: Path, *, token_count: int) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    byte_size = path.stat().st_size
    if byte_size % 2 != 0:
        raise ValueError(f"invalid uint16 shard byte size (odd): {path}")
    expected = token_count * 2
    if byte_size != expected:
        raise ValueError(
            f"uint16 shard byte size mismatch for {path}: expected {expected}, got {byte_size}"
        )


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
