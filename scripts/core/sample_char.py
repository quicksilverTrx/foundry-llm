# scripts/p1_sample_char.py
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from llm_lab.core.tokenization import CharTokenizer
from llm_lab.core.model.gpt import MiniGPT,MiniGPTConfig
from llm_lab.core.decode.sampling import greedy_decode, sample_top_k,sample_with_temperature,sample_top_p

def tokenizer_from_vocab_txt(vocab_path: Path) -> CharTokenizer:
    vocab = vocab_path.read_text(encoding="utf-8")
    stoi = {ch : i for i,ch in enumerate(vocab)}
    itos = {i : ch for i,ch in enumerate(vocab)}
    return CharTokenizer(stoi=stoi,itos=itos,unk_id=None)

def main():
    run_dir = ROOT / "experiments" / "p1_real" / "run_001"
    ckpt = run_dir / "chkpt_epoch_0.pt"
    vocab_path = run_dir / "vocab.txt"

    tok = tokenizer_from_vocab_txt(vocab_path)
    vocab_size = len(tok.stoi)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    block_size = 256
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            serialized = json.load(f)
        cfg = serialized.get("model", {})
        cfg.setdefault("vocab_size", vocab_size)
        cfg.setdefault("block_size", block_size)
        print("loaded model config from file : ", cfg)
        config = MiniGPTConfig(**cfg)
    else:
        config = MiniGPTConfig(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=1024,
            block_size=block_size,
            dropout=0.1,
            pos_encoding_type="learned",
        )
    from dataclasses import asdict
    

    model = MiniGPT(config)
    print("Loaded config and instantiated model",asdict(config))

    state = torch.load(ckpt,map_location=device)

    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    
    prompt = "Today is a good day "
    encoded_prompt = tok.encode(prompt)
    x = torch.tensor([encoded_prompt],device = device)
    with torch.no_grad():
        output_greedy = greedy_decode(model,x,max_new_tokens=250,block_size=block_size)
        output_temperature_sampled = sample_with_temperature(model,x,max_new_tokens=250,block_size=block_size,temperature=0.8)
        output_top_k = sample_top_k(model,x,max_new_tokens=250,block_size=block_size,temperature=0.9,k=50)
        output_top_p = sample_top_p(model,x,max_new_tokens=250,block_size=block_size,temperature=0.8,top_p=0.92)


    decoded_greedy = tok.decode(output_greedy[0].tolist())
    decoded_temperature = tok.decode(output_temperature_sampled[0].tolist())
    decoded_top_k = tok.decode(output_top_k[0].tolist())
    decoded_top_p = tok.decode(output_top_p[0].tolist())

    print("=== Sampling outputs ===")
    print("\n--- Greedy decode ---\n", decoded_greedy)
    print("\n--- Temperature sampling (temp=0.8) ---\n", decoded_temperature)
    print("\n--- Top-k sampling (k=50, temp=0.9) ---\n", decoded_top_k)
    print("\n--- Top-p sampling (top_p=0.92, temp=0.8) ---\n", decoded_top_p)

    print("\nGenerated prompt:", prompt)

if __name__ == "__main__":
    main()
