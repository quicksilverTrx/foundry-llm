#!/usr/bin/env python3
"""
Probe runner for NanoLlama-127M v2 ablation probes (N1/N2/N3).

Reads YAML probe configs, translates to train_nanollama_v2.py CLI args,
and runs probes sequentially on the current GPU.

Usage
-----
    # Run all three probes
    python3 scripts/run_nanollama_v2_probes.py \\
        --data-dir /workspace/edu_fineweb10B \\
        --runs-dir /workspace/runs/nanollama_v2/probes \\
        --probes configs/ablation/N1_muon_nanochat.yaml \\
                 configs/ablation/N2_adamw.yaml \\
                 configs/ablation/N3_base_arch_muon.yaml

    # Single probe by name
    python3 scripts/run_nanollama_v2_probes.py \\
        --data-dir /workspace/edu_fineweb10B \\
        --runs-dir /workspace/runs/nanollama_v2/probes \\
        --probe-name N1_muon_nanochat

    # Resume / skip completed
    python3 scripts/run_nanollama_v2_probes.py ... --skip-existing

Appends a results table to <runs-dir>/probe_results.md after each probe.

Stop Gate 2
-----------
After all probes complete, report val@500/1000/1500 to user.  Decision logic:
  N1 wins → Full stack + Muon (nanochat). Proceed to production with N1.
  N2 wins → AdamW + full stack. Muon is not helping at 127M.
  N3 wins → Base arch + Muon. Drop new components.
  All within ±0.05 → Use N1 for novelty.
  Any val@1500 > 5.0 → DEBUG. Do not proceed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


# ── YAML loader (stdlib-only fallback) ────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        with path.open() as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    import re
    result: dict = {}
    current_section: Optional[str] = None
    with path.open() as f:
        for raw_line in f:
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if line.startswith("  ") or line.startswith("\t"):
                if current_section is not None:
                    m = re.match(r'\s+(\w+)\s*:\s*(.*)', line)
                    if m:
                        k, v = m.group(1), m.group(2).strip()
                        result.setdefault(current_section, {})[k] = _parse_val(v)
            else:
                m = re.match(r'(\w+)\s*:\s*(.*)', line)
                if m:
                    k, v = m.group(1), m.group(2).strip()
                    if v == "":
                        current_section = k
                        result[k] = {}
                    else:
                        result[k] = _parse_val(v)
                        current_section = None
    return result


def _parse_val(s: str):
    s = s.strip()
    if s.lower() == "true":  return True
    if s.lower() == "false": return False
    if s.lower() in ("null", "none", "~"): return None
    try: return int(s)
    except ValueError: pass
    try: return float(s)
    except ValueError: pass
    if s.startswith("[") and s.endswith("]"):
        return [_parse_val(x.strip()) for x in s[1:-1].split(",")]
    return s.strip('"\'')


# ── YAML config → argv ────────────────────────────────────────────────────────

def _probe_to_argv(
    cfg: dict,
    data_dir: str,
    run_dir: str,
    probe_name: str,
) -> List[str]:
    """Translate a NanoLlama v2 probe YAML dict to train_nanollama_v2.py argv."""
    argv = [
        "--data_dir",   data_dir,
        "--run_dir",    run_dir,
        "--probe_name", probe_name,
    ]

    # Steps — CRITICAL: both max_steps and run_steps must be passed
    argv += ["--max_steps",  str(cfg.get("max_steps",  9537))]
    if "run_steps" in cfg:
        argv += ["--run_steps", str(cfg["run_steps"])]

    # Batch
    argv += ["--batch_size",  str(cfg.get("batch_size",  16))]
    argv += ["--grad_accum",  str(cfg.get("grad_accum",  32))]
    argv += ["--block_size",  str(cfg.get("block_size",  1024))]

    # LR schedule
    argv += ["--warmup_steps",   str(cfg.get("warmup_steps",   100))]
    argv += ["--warmdown_ratio", str(cfg.get("warmdown_ratio", 0.4))]

    # Eval / logging
    argv += ["--val_every",  str(cfg.get("val_every",  500))]
    argv += ["--val_steps",  str(cfg.get("val_steps",  300))]
    argv += ["--seed",       str(cfg.get("seed",       42))]

    if cfg.get("torch_compile", True):
        argv.append("--compile")

    # Optimizer section
    opt = cfg.get("optimizer", {})
    opt_type = opt.get("type", "muon_adam")
    argv += ["--optimizer",    opt_type]
    argv += ["--muon_lr",      str(opt.get("muon_lr",    0.02))]
    argv += ["--embed_lr",     str(opt.get("embed_lr",   0.3))]
    argv += ["--unembed_lr",   str(opt.get("unembed_lr", 0.004))]
    argv += ["--scalar_lr",    str(opt.get("scalar_lr",  0.5))]
    argv += ["--adam_lr",      str(opt.get("adam_lr",    6e-4))]
    argv += ["--weight_decay", str(opt.get("weight_decay", 0.2))]

    # Architecture overrides (probe N3: v1 arch)
    arch = cfg.get("arch", {})
    for key, flag in [
        ("rope_fraction",   "--rope_fraction"),
        ("n_value_embeds",  "--n_value_embeds"),
        ("n_layers",        "--n_layers"),
        ("d_ff",            "--d_ff"),
        ("attention_type",  "--attention_type"),
        ("num_kv_heads",    "--num_kv_heads"),
        ("mlp_type",        "--mlp_type"),
    ]:
        if key in arch:
            argv += [flag, str(arch[key])]

    if "use_x0_mixin" in arch:
        argv += ["--use_x0_mixin", str(arch["use_x0_mixin"]).lower()]

    return argv


# ── results table helpers ─────────────────────────────────────────────────────

def _append_result(results_md: Path, probe_name: str, summary: dict) -> None:
    val_at   = summary.get("val_at", {})
    val_500  = val_at.get("val@500",  "—")
    val_1000 = val_at.get("val@1000", "—")
    val_1500 = val_at.get("val@1500", "—")
    d1  = summary.get("descent_rate_1", "—")
    d2  = summary.get("descent_rate_2", "—")
    opt = summary.get("optimizer", "—")
    wd  = summary.get("weight_decay", "—")
    rf  = summary.get("rope_fraction", "—")
    nve = summary.get("n_value_embeds", "—")
    x0  = summary.get("use_x0_mixin", "—")

    row = (
        f"| {probe_name:<30} | {str(val_500):<7} | {str(val_1000):<7} | "
        f"{str(val_1500):<7} | {str(d1):<9} | {str(d2):<9} | "
        f"{opt:<10} | {str(wd):<4} | rope={rf} ve={nve} x0={x0} |\n"
    )

    if not results_md.exists():
        results_md.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# NanoLlama-127M v2 Probe Results\n\n"
            "| Probe                          | val@500 | val@1000 | val@1500 | "
            "drate1     | drate2     | optimizer  | wd   | arch                   |\n"
            "|--------------------------------|---------|----------|----------|"
            "-----------|-----------|------------|------|------------------------|\n"
        )
        results_md.write_text(header + row)
    else:
        with results_md.open("a") as f:
            f.write(row)

    print(f"\n{'='*70}")
    print(f"PROBE RESULT: {probe_name}")
    print(f"  val@500={val_500}  val@1000={val_1000}  val@1500={val_1500}")
    print(f"  descent_1={d1}  descent_2={d2}")
    print(f"  optimizer={opt}  wd={wd}  rope={rf}  n_value_embeds={nve}  x0={x0}")
    print(f"{'='*70}\n")


# ── main ──────────────────────────────────────────────────────────────────────

PROBE_DIR = Path(__file__).resolve().parents[1] / "configs" / "ablation"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run NanoLlama v2 ablation probes sequentially",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", required=True,
                    help="Directory with edufineweb_*.npy shards")
    ap.add_argument("--runs-dir", default="/workspace/runs/nanollama_v2/probes",
                    help="Parent directory for per-probe run dirs")
    ap.add_argument("--probes", nargs="*", default=None,
                    help="Paths to probe YAML files")
    ap.add_argument("--probe-name", type=str, default=None,
                    help="Run a single probe by name (e.g. N1_muon_nanochat)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip probes that already have run_summary.json")
    args = ap.parse_args()

    runs_dir   = Path(args.runs_dir)
    results_md = runs_dir / "probe_results.md"

    # Collect probe files.
    if args.probe_name:
        probe_files = [PROBE_DIR / f"{args.probe_name}.yaml"]
    elif args.probes:
        probe_files = [Path(p) for p in args.probes]
    else:
        probe_files = sorted(PROBE_DIR.glob("N*.yaml"))

    if not probe_files:
        print(f"ERROR: no N*.yaml probe configs found in {PROBE_DIR}", file=sys.stderr)
        return 1

    print(f"Running {len(probe_files)} probe(s):")
    for pf in probe_files:
        print(f"  {pf.name}")
    print()

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Import lazily to defer CUDA init.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_nanollama_v2 as trainer_mod

    for probe_file in probe_files:
        if not probe_file.exists():
            print(f"WARNING: {probe_file} not found — skipping", file=sys.stderr)
            continue

        cfg = _load_yaml(probe_file)
        probe_name = cfg.get("probe_name") or probe_file.stem
        run_dir = runs_dir / probe_name

        if args.skip_existing and (run_dir / "run_summary.json").exists():
            print(f"[SKIP] {probe_name} — already completed")
            continue

        print(f"\n{'='*70}")
        print(f"STARTING PROBE: {probe_name}")
        print(f"{'='*70}")

        argv = _probe_to_argv(cfg, args.data_dir, str(run_dir), probe_name)
        print(f"argv: {' '.join(argv)}\n")

        saved_argv = sys.argv
        sys.argv = [str(trainer_mod.__file__)] + argv
        try:
            trainer_mod.main()
        finally:
            sys.argv = saved_argv

        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            with summary_path.open() as f:
                summary = json.load(f)
            _append_result(results_md, probe_name, summary)
        else:
            print(f"WARNING: {summary_path} not found after training", file=sys.stderr)

    print(f"\nAll probes done. Results table: {results_md}")
    print("\n" + "="*70)
    print("STOP GATE 2 — report results to user before starting production run.")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
