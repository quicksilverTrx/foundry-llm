# scripts/p2_sample_bpe.py
from __future__ import annotations

import torch
from pathlib import Path

from llm_lab.core.package.io import load_model_package
from llm_lab.core.decode.sampling import greedy_decode, sample_with_temperature, sample_top_k, sample_top_p

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    pkg_dir = Path("artifacts/p2_bpe_smoke/3")
    cfg, tokenizer, model = load_model_package(pkg_dir, device=device)

    prompt = "To be"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    out = sample_top_p(
        model=model,
        input_ids=input_ids,
        max_new_tokens=80,
        block_size=cfg.block_size,
        temperature=0.9,
        top_p=0.95,
    )

    text = tokenizer.decode(out[0].tolist())
    print(text)

if __name__ == "__main__":
    main()
