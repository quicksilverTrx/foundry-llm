# llm_lab/serving/batching.py
from __future__ import annotations

import torch


def right_pad_and_mask(
    seqs: list[list[int]],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if not seqs:
        raise ValueError("seqs must be non-empty")
    lengths = [len(seq) for seq in seqs]
    max_len = max(lengths)

    bsz = len(seqs)
    input_ids = torch.full((bsz, max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)

    for i, seq in enumerate(seqs):
        l = lengths[i]
        input_ids[i, :l] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, :l] = 1

    return input_ids, attention_mask, lengths
