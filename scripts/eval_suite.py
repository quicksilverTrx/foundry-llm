"""
NanoLlama 8L — Evaluation Suite

Loads a MiniGPT checkpoint and runs 11 evaluation tests:
  1.  Sanity check (param count, arch, device)
  2.  Greedy generation (multiple domains)
  3.  Temperature sweep (0.1 → 1.5)
  4.  Top-p / nucleus sampling
  5.  Repetition analysis (n-gram overlap)
  6.  Perplexity on hand-crafted held-out texts
  7.  Next-token entropy (predictability)
  8.  Prompt completion coherence battery
  9.  HellaSwag-style multiple-choice (5 items, pick lowest loss)
  10. Vocabulary coverage (how many tokens ever become top-1)
  11. Baseline comparison (stored numbers; optionally loads a second checkpoint)

Usage:
  python scripts/eval_suite.py --ckpt <path/to/checkpoint.pt>
  python scripts/eval_suite.py --ckpt <path> --test 6        # single test
  python scripts/eval_suite.py --ckpt <path> --baseline_ckpt <path2>  # test 11 live
"""

import sys, math, time, textwrap, argparse
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import tiktoken

from llm_lab.core.model.gpt import MiniGPT, MiniGPTConfig
from llm_lab.core.decode.sampling import (
    greedy_decode,
    sample_with_temperature,
    sample_top_k,
    sample_top_p,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",          required=True,
                    help="Path to primary checkpoint .pt")
parser.add_argument("--baseline_ckpt", default=None,
                    help="Optional baseline checkpoint for test 11 comparison")
parser.add_argument("--test",          type=int, default=0,
                    help="Run only test N (0 = all)")
parser.add_argument("--device",        default=None,
                    help="cuda / mps / cpu  (auto-detected if omitted)")
args = parser.parse_args()

CKPT_PATH    = Path(args.ckpt)
BASELINE_CKPT = Path(args.baseline_ckpt) if args.baseline_ckpt else None

if not CKPT_PATH.exists():
    sys.exit(f"Checkpoint not found: {CKPT_PATH}")

if args.device:
    DEVICE = args.device
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

ENC = tiktoken.get_encoding("gpt2")

# ── helpers ───────────────────────────────────────────────────────────────────
def sep(title="", width=72):
    if title:
        pad = width - len(title) - 8
        print(f"\n{'='*6} {title} {'='*max(pad,1)}")
    else:
        print("─" * width)

def tok(text):
    return ENC.encode(text, allowed_special="all")

def detok(ids):
    return ENC.decode(list(ids))

# ── model loading ─────────────────────────────────────────────────────────────
def load_model(path: Path, device: str = DEVICE):
    ckpt = torch.load(path, map_location="cpu")
    cfg  = MiniGPTConfig(**ckpt["config"])
    mdl  = MiniGPT(cfg)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.to(device).eval()
    return mdl, ckpt.get("step", "?"), ckpt.get("val_loss", float("nan"))

# ── generation wrappers ───────────────────────────────────────────────────────
def gen_greedy(model, prompt_ids, max_new=80):
    inp = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    out = greedy_decode(model, inp, max_new_tokens=max_new,
                        block_size=model.config.block_size)
    return out[0, len(prompt_ids):].tolist()

def gen_temperature(model, prompt_ids, max_new=80, temperature=0.8):
    inp = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    out = sample_with_temperature(model, inp, max_new_tokens=max_new,
                                  block_size=model.config.block_size,
                                  temperature=temperature)
    return out[0, len(prompt_ids):].tolist()

def gen_top_k(model, prompt_ids, max_new=80, temperature=0.8, k=50):
    inp = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    out = sample_top_k(model, inp, max_new_tokens=max_new,
                       block_size=model.config.block_size,
                       temperature=temperature, k=k)
    return out[0, len(prompt_ids):].tolist()

def gen_top_p(model, prompt_ids, max_new=80, temperature=0.8, top_p=0.9):
    inp = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    out = sample_top_p(model, inp, max_new_tokens=max_new,
                       block_size=model.config.block_size,
                       temperature=temperature, top_p=top_p)
    return out[0, len(prompt_ids):].tolist()

# ── analysis helpers ──────────────────────────────────────────────────────────
@torch.no_grad()
def sequence_loss(model, text):
    ids = tok(text)
    if len(ids) < 2:
        return float("nan")
    x = torch.tensor(ids[:-1], dtype=torch.long, device=DEVICE).unsqueeze(0)
    y = torch.tensor(ids[1:],  dtype=torch.long, device=DEVICE)
    logits, _ = model(x)
    return F.cross_entropy(logits[0], y).item()

@torch.no_grad()
def entropy_of_next(model, prompt_ids):
    ids    = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    logits, _ = model(ids)
    probs  = F.softmax(logits[0, -1, :], dim=-1)
    return (-probs * (probs + 1e-9).log()).sum().item() / math.log(2)

def ngram_rep(ids, n=4):
    ngrams = [tuple(ids[i:i+n]) for i in range(len(ids) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST DATA
# ══════════════════════════════════════════════════════════════════════════════
PROMPTS = {
    "science": "The theory of general relativity explains how",
    "code":    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "history": "The French Revolution began in 1789 when",
    "math":    "To solve a quadratic equation ax^2 + bx + c = 0, we use",
    "story":   "Once upon a time in a small village nestled between two mountains,",
    "news":    "Scientists at the university announced today that their research on",
    "list":    "The top five most important factors for learning a new skill are:\n1.",
    "chat":    "Question: What is the capital of France?\nAnswer:",
    "eol":     "<|endoftext|>The weather today in New York is",
}

PERPLEXITY_TEXTS = {
    "wiki_gravity":  ("Gravity is a fundamental force of nature that attracts two "
                      "bodies towards each other. The strength of gravity depends on "
                      "the masses of the objects and the distance between them. "
                      "Isaac Newton described gravity with his law of universal "
                      "gravitation in 1687."),
    "wiki_python":   ("Python is a high-level, general-purpose programming language. "
                      "Its design philosophy emphasizes code readability, through the "
                      "use of significant indentation. Python is dynamically typed and "
                      "garbage-collected. It supports multiple programming paradigms."),
    "random_junk":   "xk9z mf73 qw!! zz55 ab12 lm00 pp77 rr34 ss01 tt98 uu65 vv43",
    "repetitive":    "the " * 20,
    "numbers":       "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
    "code_snippet":  "import numpy as np\nx = np.array([1, 2, 3])\ny = np.dot(x, x)\nprint(y)",
    "dialogue":      '"How are you?" she asked.\n"I am doing well, thank you," he replied.',
}

HELLASWAG_ITEMS = [
    {
        "ctx": "A woman is seen in the kitchen. She begins mixing ingredients in a bowl.",
        "endings": [
            " She adds flour and sugar, then pours in milk and eggs to make batter.",
            " She drives to the supermarket to buy groceries for dinner.",
            " She suddenly falls asleep on the couch while watching television.",
            " She starts painting the walls of her living room with blue paint.",
        ],
        "label": 0,
    },
    {
        "ctx": "A man is shown at a gym. He starts lifting weights.",
        "endings": [
            " He begins typing on his laptop while sitting at his desk.",
            " He picks up the barbell and does several repetitions of bicep curls.",
            " He waters the plants in his garden using a green hose.",
            " He reads a novel about ancient history under a lamp.",
        ],
        "label": 1,
    },
    {
        "ctx": "The teacher writes an equation on the chalkboard and turns to the class.",
        "endings": [
            " She asks the students to copy the homework assignment into their notebooks.",
            " A student raises their hand and asks for clarification on the formula.",
            " Everyone in the room starts cooking pasta on the stove.",
            " The teacher begins swimming laps in the school pool.",
        ],
        "label": 1,
    },
    {
        "ctx": "He opened his laptop and launched the terminal. He typed a command",
        "endings": [
            " to compile the source code and waited for the build to complete.",
            " then went hiking through a dense forest for several hours.",
            " so he could bake a chocolate cake with cream cheese frosting.",
            " and immediately fell asleep on a beach in Hawaii.",
        ],
        "label": 0,
    },
    {
        "ctx": "The dog ran to the door, wagging its tail excitedly.",
        "endings": [
            " The dog jumped into the ocean and swam to a distant island.",
            " It barked twice and waited for its owner to arrive home.",
            " The dog sat down and began solving a calculus problem.",
            " It flew up into the sky like a balloon.",
        ],
        "label": 1,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  TESTS
# ══════════════════════════════════════════════════════════════════════════════
def test_1_sanity(model, step, val_loss):
    sep("TEST 1 — Sanity Check")
    n = sum(p.numel() for p in model.parameters())
    print(f"  Checkpoint step  : {step}")
    print(f"  Val loss (stored): {val_loss:.4f}  →  ppl={math.exp(val_loss):.2f}")
    print(f"  Parameters       : {n/1e6:.2f}M")
    print(f"  Device           : {DEVICE}")
    cfg = model.config
    print(f"  n_layers         : {cfg.n_layers}")
    print(f"  n_heads / kv     : {cfg.n_heads} / {cfg.num_kv_heads}")
    print(f"  d_model / d_ff   : {cfg.d_model} / {cfg.d_ff}")
    print(f"  logit_softcap    : {cfg.logit_softcap}")
    print(f"  qk_norm          : {cfg.qk_norm}")
    print(f"  lm_head bias     : {model.lm_head.bias}")
    print(f"  token_embed std  : {model.token_embed.weight.std().item():.4f}")
    print(f"  Sampling backend : llm_lab.core.decode.sampling")


def test_2_greedy_generation(model):
    sep("TEST 2 — Greedy Generation")
    for name, prompt in PROMPTS.items():
        prompt_ids = tok(prompt)
        gen_ids    = gen_greedy(model, prompt_ids, max_new=80)
        full_text  = detok(prompt_ids + gen_ids)
        print(f"\n  [{name}]")
        print(textwrap.fill(full_text, width=80,
                            initial_indent="    ", subsequent_indent="    "))


def test_3_temperature_sweep(model):
    sep("TEST 3 — Temperature Sweep  (prompt='science')")
    prompt_ids = tok(PROMPTS["science"])
    torch.manual_seed(42)
    for temp in [0.3, 0.7, 1.0, 1.2, 1.5]:
        gen_ids = gen_temperature(model, prompt_ids, max_new=60, temperature=temp)
        rep4    = ngram_rep(gen_ids, n=4)
        text    = detok(gen_ids)
        print(f"\n  temp={temp:.1f}  4gram-rep={rep4:.3f}")
        print(textwrap.fill(text, width=80,
                            initial_indent="    ", subsequent_indent="    "))


def test_4_nucleus_sampling(model):
    sep("TEST 4 — Nucleus (top-p) Sampling  temp=0.8")
    prompt_ids = tok(PROMPTS["story"])
    torch.manual_seed(7)
    for p in [0.5, 0.9, 0.95]:
        gen_ids = gen_top_p(model, prompt_ids, max_new=80, temperature=0.8, top_p=p)
        rep4    = ngram_rep(gen_ids, n=4)
        text    = detok(gen_ids)
        print(f"\n  top_p={p}  4gram-rep={rep4:.3f}")
        print(textwrap.fill(text, width=80,
                            initial_indent="    ", subsequent_indent="    "))


def test_5_repetition_analysis(model):
    sep("TEST 5 — Repetition Analysis")
    print(f"  {'prompt':<12} {'temp':<6} {'4gram-rep':>10} {'unique_tok%':>12} {'len':>6}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*12} {'-'*6}")
    torch.manual_seed(0)
    for name, prompt in list(PROMPTS.items())[:5]:
        prompt_ids = tok(prompt)
        for temp in [0.7, 1.0]:
            gen_ids = gen_top_p(model, prompt_ids, max_new=200,
                                temperature=temp, top_p=0.95)
            rep4 = ngram_rep(gen_ids, n=4)
            uniq = len(set(gen_ids)) / max(len(gen_ids), 1)
            print(f"  {name:<12} {temp:<6.1f} {rep4:>10.3f} {uniq:>12.3f} {len(gen_ids):>6}")


def test_6_perplexity(model):
    sep("TEST 6 — Perplexity on Held-out Texts")
    print(f"  {'text':<20} {'loss':>8} {'ppl':>10}  notes")
    print(f"  {'-'*20} {'-'*8} {'-'*10}")
    for name, text in PERPLEXITY_TEXTS.items():
        loss = sequence_loss(model, text)
        ppl  = math.exp(loss) if not math.isnan(loss) else float("nan")
        note = ""
        if "random" in name: note = "<- expect high"
        if "wiki"   in name: note = "<- expect low"
        if "rep"    in name: note = "<- very low (trivially predictable)"
        print(f"  {name:<20} {loss:>8.4f} {ppl:>10.2f}  {note}", flush=True)


def test_7_entropy(model):
    sep("TEST 7 — Next-Token Entropy (bits)")
    print(f"  {'prompt':<20} {'entropy (bits)':>16}  interpretation")
    print(f"  {'-'*20} {'-'*16}")
    interpretation = {
        "science": "high   -- many plausible physics terms",
        "code":    "low    -- strong code-completion constraint",
        "chat":    "v.low  -- known answer",
        "story":   "high   -- open ended",
        "list":    "medium -- structured but flexible",
    }
    for name, prompt in PROMPTS.items():
        e    = entropy_of_next(model, tok(prompt))
        note = interpretation.get(name, "")
        print(f"  {name:<20} {e:>16.3f}  {note}")
    print(f"\n  (Max possible entropy = log2(50304) = {math.log2(50304):.2f} bits)")


def test_8_coherence_battery(model):
    sep("TEST 8 — Coherence Battery  (top_p=0.92, temp=0.8)")
    coherence_prompts = [
        ("capitals",   "The capital of Japan is"),
        ("arithmetic", "What is 7 multiplied by 8? The answer is"),
        ("opposites",  "Hot is the opposite of"),
        ("category",   "A dog, cat, and bird are all examples of"),
        ("sequence",   "Monday, Tuesday, Wednesday, Thursday,"),
        ("code_fn",    "# Python function to reverse a string\ndef reverse_string(s):"),
        ("edu_fact",   "Photosynthesis is the process by which plants"),
        ("entity",     "Albert Einstein was born in the city of"),
    ]
    torch.manual_seed(123)
    for name, prompt in coherence_prompts:
        ids     = tok(prompt)
        gen_ids = gen_top_p(model, ids, max_new=30, temperature=0.8, top_p=0.92)
        text    = detok(gen_ids)
        print(f"\n  [{name}]  {prompt}")
        print(f"  -> {text.strip()[:120]}")


def test_9_hellaswag(model):
    sep("TEST 9 — HellaSwag-Style Completion Ranking  (5 items)")
    correct = 0
    for i, item in enumerate(HELLASWAG_ITEMS):
        ctx    = item["ctx"]
        label  = item["label"]
        losses = [sequence_loss(model, ctx + e) for e in item["endings"]]
        pred   = int(min(range(4), key=lambda j: losses[j]))
        mark   = "PASS" if pred == label else "FAIL"
        print(f"\n  [{i+1}] {mark}  pred={pred}  gold={label}")
        print(f"       ctx: {ctx[:70]}...")
        for j, (e, l) in enumerate(zip(item["endings"], losses)):
            star = " <-- PRED" if j == pred else (" [gold]" if j == label else "")
            print(f"       {j}: [{l:.3f}]{star}  {e[:60]}")
        if pred == label:
            correct += 1
    print(f"\n  Accuracy: {correct}/{len(HELLASWAG_ITEMS)} = "
          f"{correct/len(HELLASWAG_ITEMS)*100:.0f}%  (random = 25%)")


def test_10_vocab_coverage(model):
    sep("TEST 10 — Vocabulary Coverage  (500 sample steps)")
    print("  Counting distinct tokens that ever become top-1 prediction...\n")
    N_STEPS    = 500
    block_size = model.config.block_size
    torch.manual_seed(999)
    seed_ids   = tok("The quick brown fox jumps over the lazy dog. ")

    buf  = torch.zeros(1, block_size, dtype=torch.long, device=DEVICE)
    L    = len(seed_ids)
    buf[0, :L] = torch.tensor(seed_ids, dtype=torch.long, device=DEVICE)
    pos  = L

    top1_set  = set()
    generated = []
    for _ in range(N_STEPS):
        with torch.no_grad():
            logits, _ = model(buf)
        logits_last = logits[0, pos - 1, :]
        top1_set.add(logits_last.argmax().item())
        probs   = F.softmax(logits_last / 0.9, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        generated.append(next_id)
        buf[0, pos % block_size] = next_id
        pos += 1

    print(f"  Distinct top-1 tokens: {len(top1_set):,} / {model.config.vocab_size:,}  "
          f"({len(top1_set)/model.config.vocab_size*100:.1f}%)")
    counter = Counter(generated)
    print(f"\n  Top 20 most generated tokens:")
    for tid, cnt in counter.most_common(20):
        try:    word = repr(detok([tid]))
        except: word = f"<id={tid}>"
        print(f"    {cnt:5d}x  {word}")


def test_11_comparison(model):
    sep("TEST 11 — Model Comparison")
    # Stored reference numbers (GPT-2 124M at 1.05B tokens on same FineWeb-Edu val shard)
    baseline_val = 3.6273

    if BASELINE_CKPT is not None and BASELINE_CKPT.exists():
        try:
            b_ckpt      = torch.load(BASELINE_CKPT, map_location="cpu")
            baseline_val = b_ckpt.get("val_loss", baseline_val)
            print(f"  Baseline checkpoint loaded from {BASELINE_CKPT.name}")
        except Exception as e:
            print(f"  Baseline load error: {e}  — using stored value {baseline_val}")
    else:
        print(f"  Using stored baseline val loss: {baseline_val}")

    primary_val = 3.3566   # NanoLlama 8L step 4768
    print(f"\n  Baseline   (1.053B tokens) : val={baseline_val:.4f}  "
          f"ppl={math.exp(baseline_val):.2f}")
    print(f"  This model (2.500B tokens) : val={primary_val:.4f}  "
          f"ppl={math.exp(primary_val):.2f}")
    print(f"\n  Gap (absolute) : {baseline_val - primary_val:.4f} nats")
    print(f"  Gap (relative) : {(baseline_val - primary_val)/baseline_val*100:.1f}% lower val loss")
    print(f"\n  Compute-matched @ 1.053B tokens: val≈3.5892")
    print(f"  Compute-matched gap            : {baseline_val - 3.5892:.4f} nats")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  MiniGPT / NanoLlama — Evaluation Suite")
    print(f"  Checkpoint : {CKPT_PATH.name}")
    print(f"  Device     : {DEVICE}")
    print(f"  Sampling   : llm_lab.core.decode.sampling")
    print("=" * 72)

    t0 = time.time()
    print(f"\nLoading model on {DEVICE}...", end=" ", flush=True)
    model, step, val_loss = load_model(CKPT_PATH, DEVICE)
    print(f"done ({time.time()-t0:.1f}s)\n")

    tests = {
        1:  lambda: test_1_sanity(model, step, val_loss),
        2:  lambda: test_2_greedy_generation(model),
        3:  lambda: test_3_temperature_sweep(model),
        4:  lambda: test_4_nucleus_sampling(model),
        5:  lambda: test_5_repetition_analysis(model),
        6:  lambda: test_6_perplexity(model),
        7:  lambda: test_7_entropy(model),
        8:  lambda: test_8_coherence_battery(model),
        9:  lambda: test_9_hellaswag(model),
        10: lambda: test_10_vocab_coverage(model),
        11: lambda: test_11_comparison(model),
    }

    to_run = [args.test] if args.test else list(tests.keys())
    for n in to_run:
        t = time.time()
        tests[n]()
        print(f"\n  [test {n} completed in {time.time()-t:.1f}s]")

    sep()
    print(f"\nAll tests done in {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
