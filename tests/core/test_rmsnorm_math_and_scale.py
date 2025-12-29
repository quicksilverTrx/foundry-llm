import torch
from llm_lab.core.model.norms import RMSNorm

def test_rmsnorm_matches_reference_and_gradients_finite():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, requires_grad=True)
    norm = RMSNorm(8, eps=1e-6)

    y = norm(x)
    # Reference: x * rsqrt(mean(x^2)) * weight
    rms = x.pow(2).mean(dim=-1, keepdim=True)
    y_ref = x * torch.rsqrt(rms + 1e-6) * norm.weight

    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-5)

    loss = y.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.weight.grad).all()

def test_rmsnorm_positive_scale_invariance():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    norm = RMSNorm(8, eps=1e-6)
    norm.weight.data.fill_(1.0)

    y1 = norm(x)
    y2 = norm(3.0 * x)  # positive scaling should not change normalized direction
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5)
