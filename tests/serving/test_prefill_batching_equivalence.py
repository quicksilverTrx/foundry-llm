# tests/serving/test_prefill_batching_equivalence.py
from __future__ import annotations

import torch

from llm_lab.serving.batching import right_pad_and_mask
from llm_lab.serving.engine import Engine


def test_right_pad_and_mask_shapes_and_values() -> None:
    device = torch.device("cpu")
    seqs = [[5, 6, 7], [9], [1, 2]]
    pad_id = 0

    input_ids, attention_mask, lengths = right_pad_and_mask(seqs, pad_id=pad_id, device=device)

    assert input_ids.shape == (3, 3)
    assert attention_mask.shape == (3, 3)
    assert lengths == [3, 1, 2]

    assert torch.equal(attention_mask[0], torch.tensor([1, 1, 1]))
    assert torch.equal(attention_mask[1], torch.tensor([1, 0, 0]))
    assert torch.equal(attention_mask[2], torch.tensor([1, 1, 0]))

    assert int(input_ids[1, 1].item()) == pad_id
    assert int(input_ids[1, 2].item()) == pad_id
    assert int(input_ids[2, 2].item()) == pad_id


def test_batched_prefill_logits_match_individual_prefill(serving_pkg, serving_device: str) -> None:
    cfg, tok, model = serving_pkg
    engine = Engine(model, tok, block_size=int(getattr(cfg, "block_size", 128)))

    torch.manual_seed(0)
    vocab = int(getattr(cfg, "vocab_size", 32000))
    seqs = [
        torch.randint(0, vocab, (5,), dtype=torch.long).tolist(),
        torch.randint(0, vocab, (9,), dtype=torch.long).tolist(),
        torch.randint(0, vocab, (3,), dtype=torch.long).tolist(),
    ]

    batch_logits, _, _ = engine.prefill_batch(seqs)

    for i, seq in enumerate(seqs):
        x = torch.tensor([seq], dtype=torch.long, device=serving_device)
        m = torch.ones_like(x)
        single_logits, _, _ = engine.prefill(x, m)
        assert torch.allclose(batch_logits[i], single_logits[0], atol=1e-4, rtol=1e-4)


def test_batched_prefill_does_not_nan(serving_pkg) -> None:
    cfg, tok, model = serving_pkg
    engine = Engine(model, tok, block_size=int(getattr(cfg, "block_size", 128)))

    vocab = int(getattr(cfg, "vocab_size", 32000))
    seqs = [
        [1],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [10, 11],
    ]
    seqs = [[tid % vocab for tid in seq] for seq in seqs]

    logits_list, states, _ = engine.prefill_batch(seqs)
    assert len(logits_list) == len(seqs)
    assert len(states) == len(seqs)

    for logits in logits_list:
        assert torch.isfinite(logits).all()
    for i, state in enumerate(states):
        assert state.past_key_values is not None
        assert state.past_len == len(seqs[i])
