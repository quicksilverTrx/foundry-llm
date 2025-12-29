import torch
from llm_lab.core.model.mlp import FeedForward, FeedForwardConfig

def test_swiglu_gate_can_shut_off_path():
    torch.manual_seed(0)
    cfg = FeedForwardConfig(d_model=8, d_ff=16, dropout=0.0, mlp_type="swiglu")
    mlp = FeedForward(cfg)
    x = torch.randn(2, 3, 8)

    y = mlp(x)

    # Shut the gate deterministically for bias-free w_gate:
    # b = w_gate(x) becomes 0 => SiLU(0)=0 => h=0 => y=0 (since w_down bias=False)
    with torch.no_grad():
        mlp.w_gate.weight.zero_()

    y_shut = mlp(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.isfinite(y_shut).all()

    # Strong invariant for bias-free SwiGLU:
    assert y_shut.abs().mean() < 1e-4
    # And it should be much smaller than the normal output
    assert y_shut.abs().mean() < 0.2 * y.abs().mean()


def test_swiglu_dropout_on_output_projection_changes_train_only():
    torch.manual_seed(0)
    cfg = FeedForwardConfig(d_model=8, d_ff=16, dropout=0.5, mlp_type="swiglu")
    mlp = FeedForward(cfg)
    x = torch.randn(2, 3, 8)

    mlp.train()
    y1 = mlp(x)
    y2 = mlp(x)
    assert not torch.allclose(y1, y2)
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()

    mlp.eval()
    y3 = mlp(x)
    y4 = mlp(x)
    assert torch.allclose(y3, y4)


def test_swiglu_gate_saturates_to_zero_when_forced_negative():
    torch.manual_seed(0)
    cfg = FeedForwardConfig(d_model=8, d_ff=16, dropout=0.0, mlp_type="swiglu")
    mlp = FeedForward(cfg)
    x = torch.randn(2, 3, 8)

    with torch.no_grad():
        mlp.w_gate.weight.zero_()
        if getattr(mlp.w_gate, "bias", None) is not None:
            mlp.w_gate.bias.fill_(-50.0)  # SiLU(-50) ~ 0

    y = mlp(x)
    assert torch.isfinite(y).all()
    assert y.abs().mean().item() < 1e-4