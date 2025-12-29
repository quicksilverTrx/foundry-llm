from pathlib import Path
import torch

from llm_lab.core.package.io import save_model_package, load_model_package
from llm_lab.core.decode.sampling import greedy_decode
from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig

def test_package_load_and_greedy_decode_smoke(tmp_path: Path):
    text = "hello world hello world hello world"
    tok = SubwordTokenizer.train_from_iterator([text], SubwordTokenizerConfig(vocab_size=50, model_type="bpe"))

    cfg = MiniGPTConfig(vocab_size=len(tok.stoi), d_model=32, n_layers=2, n_heads=2, d_ff=64, block_size=16, dropout=0.0)
    model = MiniGPT(cfg)

    pkg_dir = tmp_path / "pkg"
    save_model_package(pkg_dir, cfg, tok, model, is_best=True)

    cfg2, tok2, model2 = load_model_package(pkg_dir, device="cpu")

    prompt_ids = torch.tensor([tok2.encode("hello")], dtype=torch.long)
    out = greedy_decode(model2, prompt_ids, max_new_tokens=5, block_size=cfg2.block_size)

    assert out.shape[1] == prompt_ids.shape[1] + 5
    assert int(out.max()) < cfg2.vocab_size
    assert int(out.min()) >= 0
