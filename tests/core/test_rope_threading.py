# tests/core/test_rope_threading.py
from __future__ import annotations

import torch


def test_mha_threads_rope_scaling_kwargs(monkeypatch):
    """
    Asserts: MultiHeadAttention(attention_type="mha", use_rope=True, rope_scaling_*)
    actually passes rope_scaling_* into apply_rope() (cannot be positional because apply_rope uses *).
    """
    import llm_lab.core.model.attention as attn_mod
    from llm_lab.core.model.attention import MultiHeadAttention, MultiHeadAttentionConfig
    from llm_lab.core.model.pos_encodings import apply_rope as real_apply_rope

    calls = []

    def spy_apply_rope(q, k, position_ids, **kwargs):
        calls.append(dict(kwargs))
        return real_apply_rope(q, k, position_ids, **kwargs)

    monkeypatch.setattr(attn_mod, "apply_rope", spy_apply_rope)

    cfg = MultiHeadAttentionConfig(
        d_model=64,
        n_heads=8,
        dropout=0.0,
        use_rope=True,
        attention_type="mha",
        rope_scaling_type="linear",
        rope_scaling_factor=2.0,
    )
    attn = MultiHeadAttention(cfg).eval()

    B, T = 2, 8
    x = torch.randn(B, T, 64)
    pos = (torch.arange(T) * 50).to(torch.long)  # amplify effect; shape [T]

    _y, _ = attn(x, position_ids=pos)

    # MHA path uses SingleHeadAttention per head -> apply_rope called once per head
    assert len(calls) == cfg.n_heads, f"expected {cfg.n_heads} apply_rope calls, got {len(calls)}"
    for kw in calls:
        print(kw)
        assert kw.get("rope_scaling_type") == "linear"
        assert kw.get("rope_scaling_factor") == 2.0


def test_gqa_threads_rope_scaling_kwargs(monkeypatch):
    """
    Asserts: MultiHeadAttention(attention_type="gqa", use_rope=True, rope_scaling_*)
    passes rope_scaling_* into apply_rope().
    """
    import llm_lab.core.model.attention as attn_mod
    from llm_lab.core.model.attention import MultiHeadAttention, MultiHeadAttentionConfig
    from llm_lab.core.model.pos_encodings import apply_rope as real_apply_rope

    calls = []

    def spy_apply_rope(q, k, position_ids, **kwargs):
        calls.append(dict(kwargs))
        return real_apply_rope(q, k, position_ids, **kwargs)

    monkeypatch.setattr(attn_mod, "apply_rope", spy_apply_rope)

    cfg = MultiHeadAttentionConfig(
        d_model=64,
        n_heads=8,
        dropout=0.0,
        use_rope=True,
        attention_type="gqa",
        num_kv_heads=2,
        rope_scaling_type="linear",
        rope_scaling_factor=3.0,
    )
    attn = MultiHeadAttention(cfg).eval()

    B, T = 2, 8
    x = torch.randn(B, T, 64)
    pos = (torch.arange(T) * 50).to(torch.long)

    _y, _ = attn(x, position_ids=pos)

    # GQA path batches heads -> apply_rope called once
    assert len(calls) == 1, f"expected 1 apply_rope call, got {len(calls)}"
    kw = calls[0]
    assert kw.get("rope_scaling_type") == "linear"
    assert kw.get("rope_scaling_factor") == 3.0


def test_model_threads_rope_scaling_from_config(monkeypatch):
    """
    End-to-end wiring test:
    MiniGPTConfig.rope_scaling_* must flow through TransformerBlockConfig -> (MultiHeadAttentionConfig) -> apply_rope().
    This catches the common bug: scaling fields exist in MiniGPTConfig but aren't passed into attention.
    """
    import llm_lab.core.model.attention as attn_mod
    from llm_lab.core.model.pos_encodings import apply_rope as real_apply_rope
    from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig

    calls = []

    def spy_apply_rope(q, k, position_ids, **kwargs):
        calls.append(dict(kwargs))
        return real_apply_rope(q, k, position_ids, **kwargs)

    monkeypatch.setattr(attn_mod, "apply_rope", spy_apply_rope)

    cfg = MiniGPTConfig(
        arch_family="miniGPT",
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=8,
        d_ff=256,
        block_size=64,
        dropout=0.0,
        pos_encoding_type="rope",
        attention_type="gqa",
        num_kv_heads=2,
        rope_scaling_type="linear",
        rope_scaling_factor=4.0,
        norm_type="layernorm",
        mlp_type="gelu",
    )
    model = MiniGPT(cfg).eval()

    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    _logits, _ = model(input_ids)

    # apply_rope should be called once per layer in GQA path (your current attention implementation)
    assert len(calls) == cfg.n_layers, f"expected {cfg.n_layers} apply_rope calls, got {len(calls)}"
    for kw in calls:
        assert kw.get("rope_scaling_type") == "linear"
        assert kw.get("rope_scaling_factor") == 4.0
