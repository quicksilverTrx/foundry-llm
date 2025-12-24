# scripts/p1_env_sanity.py
import torch
from torch import nn


def run_on_device(device: torch.device, label: str) -> None:
    print(f"\n Running on {label} ({device}) ")

    layer = nn.Linear(16, 4)
    x = torch.randn(2, 16)

    layer = layer.to(device)
    x = x.to(device)

    with torch.no_grad():
        y = layer(x)

    print("Output shape:", tuple(y.shape))
    print("First row:", y[0].cpu().tolist())


def main() -> None:
    # CPU
    cpu = torch.device("cpu")
    run_on_device(cpu, "CPU")

    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps = torch.device("mps")
        run_on_device(mps, "MPS")
    else:
        print("\nMPS not available")

    # CUDA (remote GPU)
    if torch.cuda.is_available():
        print(f"\nCUDA device count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"CUDA device {idx}: {torch.cuda.get_device_name(idx)}")
        cuda0 = torch.device("cuda:0")
        run_on_device(cuda0, "CUDA:0")
    else:
        print("\nCUDA not available")


if __name__ == "__main__":
    main()
