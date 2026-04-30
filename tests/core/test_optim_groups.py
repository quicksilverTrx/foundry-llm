# tests/core/test_optim_groups.py
import torch
import pytest

from llm_lab.core.train.optim import OptimConfig, build_adamw_with_decay_groups
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig


def _tiny_model():
    # Adjust fields to match your MiniGPTConfig signature.
    # Keep it small and fast.
    cfg = MiniGPTConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        block_size=16,
        dropout=0.0,
        norm_type="layernorm",
        mlp_type="gelu",
        pos_encoding_type="learned",
        attention_type="mha",
        # include any other required fields your config needs
    )
    return MiniGPT(cfg)


def test_adamw_groups_partition_trainable_params():
    model = _tiny_model()
    opt_cfg = OptimConfig(lr=1e-3, weight_decay=0.1)

    opt, meta = build_adamw_with_decay_groups(model, opt_cfg)

    assert isinstance(opt, torch.optim.AdamW)

    # Collect trainable params from model
    name_to_param = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert len(name_to_param) > 0

    # Meta should be pure names lists, loggable
    assert "decay" in meta and "no_decay" in meta
    assert isinstance(meta["decay"], list) and isinstance(meta["no_decay"], list)
    assert all(isinstance(x, str) for x in meta["decay"])
    assert all(isinstance(x, str) for x in meta["no_decay"])

    decay_set = set(meta["decay"])
    no_decay_set = set(meta["no_decay"])

    # No overlap
    assert decay_set.isdisjoint(no_decay_set)

    # Partition covers all trainable params
    all_grouped = decay_set | no_decay_set
    assert all_grouped == set(name_to_param.keys())

    # Sanity: both non-empty for GPT-like model
    assert len(decay_set) > 0
    assert len(no_decay_set) > 0


def test_grouping_rule_ndim_1_goes_to_no_decay():
    model = _tiny_model()
    opt_cfg = OptimConfig(lr=1e-3, weight_decay=0.1)
    _, meta = build_adamw_with_decay_groups(model, opt_cfg)

    name_to_param = {n: p for n, p in model.named_parameters() if p.requires_grad}

    for name in meta["no_decay"]:
        assert name_to_param[name].ndim == 1, f"{name} expected ndim==1, got {name_to_param[name].ndim}"

    for name in meta["decay"]:
        assert name_to_param[name].ndim >= 2, f"{name} expected ndim>=2, got {name_to_param[name].ndim}"


def test_optimizer_param_groups_have_correct_weight_decay_values():
    model = _tiny_model()
    wd = 0.2
    opt_cfg = OptimConfig(lr=1e-3, weight_decay=wd)
    opt, meta = build_adamw_with_decay_groups(model, opt_cfg)

    # We can't rely on custom keys in opt.param_groups;
    # we only check weight_decay values in the actual optimizer groups.
    wds = sorted([g.get("weight_decay", None) for g in opt.param_groups])
    assert wds == [0.0, wd]


def test_no_param_object_duplicated_across_groups():
    model = _tiny_model()
    opt_cfg = OptimConfig(lr=1e-3, weight_decay=0.1)
    opt, meta = build_adamw_with_decay_groups(model, opt_cfg)

    # Flatten actual optimizer params from param_groups
    params = []
    for g in opt.param_groups:
        params.extend(list(g["params"]))

    # Same object should not appear twice
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids)), "parameter object duplicated across groups"
