from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from llm_lab.core.data.pretok_dataset_contract import (
    TinyLlamaP15PretokDatasetConfig,
    TinyLlamaP15RuntimeManifestView,
    build_epoch_sampler_plan,
    load_runtime_manifest_view,
)

RuntimeSplit = Literal["train", "val"]


class TinyLlamaP15PretokIterableDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Finite iterable dataset over tinyllama_p15 pretokenized shards.

    One iterator pass emits each valid offset exactly once for the current epoch.
    """

    def __init__(
        self,
        *,
        root_dir: Path | str,
        split: RuntimeSplit,
        block_size: int,
        base_seed: int = 0,
    ) -> None:
        super().__init__()
        self.config = TinyLlamaP15PretokDatasetConfig(
            root_dir=Path(root_dir),
            split=split,
            block_size=block_size,
            base_seed=base_seed,
        )
        self._manifest_view: TinyLlamaP15RuntimeManifestView = load_runtime_manifest_view(self.config)
        self._epoch = 0

        self._memmaps: list[np.memmap] = []
        for shard in self._manifest_view.eligible_shards:
            mem = np.memmap(shard.bin_path, dtype=np.uint16, mode="r")
            if int(mem.shape[0]) != shard.token_count:
                raise ValueError(
                    f"token count mismatch for {shard.bin_path}: "
                    f"sidecar={shard.token_count} memmap={int(mem.shape[0])}"
                )
            self._memmaps.append(mem)

        self.eligible_shard_count = len(self._manifest_view.eligible_shards)
        self.short_shards_skipped = self._manifest_view.short_shard_stats.short_shards_skipped
        self.short_tokens_skipped = self._manifest_view.short_shard_stats.short_tokens_skipped
        self.total_samples_per_pass = self._manifest_view.total_samples_per_pass

    def __len__(self) -> int:
        # Exact finite pass cardinality under frozen HSU-3A semantics.
        return int(self.total_samples_per_pass)

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative int, got {epoch!r}")
        self._epoch = epoch

    def sample_refs_for_current_epoch(self) -> list[tuple[int, int]]:
        """
        Debug helper: returns (eligible_shard_index, offset) for full current-epoch pass.
        """
        out: list[tuple[int, int]] = []
        for shard_idx, offset in self._iter_shard_offsets():
            out.append((shard_idx, offset))
        return out

    def _iter_shard_offsets(self) -> Iterator[tuple[int, int]]:
        plan = build_epoch_sampler_plan(
            manifest_view=self._manifest_view,
            base_seed=self.config.base_seed,
            epoch=self._epoch,
        )
        for shard_idx, offsets in plan.iter_ordered_pairs():
            for offset in offsets:
                yield int(shard_idx), int(offset)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        if worker_info is not None:
            raise RuntimeError(
                "TinyLlamaP15PretokIterableDataset supports single-worker mode only; "
                "use DataLoader(..., num_workers=0)."
            )

        block_size = self.config.block_size
        window_len = block_size + 1

        for shard_idx, offset in self._iter_shard_offsets():
            mem = self._memmaps[shard_idx]
            window = mem[offset : offset + window_len]
            if int(window.shape[0]) != window_len:
                raise RuntimeError(
                    f"OOB window read attempted: shard_idx={shard_idx} offset={offset} "
                    f"window_len={window_len} actual={int(window.shape[0])}"
                )

            x = torch.from_numpy(np.asarray(window[:-1], dtype=np.int64)).to(dtype=torch.long)
            y = torch.from_numpy(np.asarray(window[1:], dtype=np.int64)).to(dtype=torch.long)
            yield x, y
