#!/usr/bin/env python3
"""
Ablation probe runner for SwiftLlama-350M.

Reads a YAML probe config, translates it to training-script CLI args,
and launches train_swiftllama_350m.py in-process.  Probes run sequentially
by default; use shell parallelism (e.g. GNU parallel) if you want concurrent
runs (not recommended on a single GPU).

Usage:
    python3 scripts/run_ablation_probes.py \\
        --data-dir /workspace/edu_fineweb10B \\
        --runs-dir /workspace/runs/ablation \\
        --probes configs/ablation/A1_current_baseline.yaml \\
                 configs/ablation/A2_muon_with_wd.yaml \\
                 configs/ablation/A3_adamw.yaml

Run a single probe:
    python3 scripts/run_ablation_probes.py \\
        --data-dir /workspace/edu_fineweb10B \\
        --runs-dir /workspace/runs/ablation \\
        --probe-name A1_current_baseline

The runner appends a summary table to /workspace/runs/ablation/probe_results.md
after each probe completes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# ── YAML loader (stdlib only fallback) ────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        with path.open() as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    # Minimal YAML parser for our simple key: value configs (no nesting depth > 2).
    import re
    result: dict = {}
    current_section: Optional[str] = None
    with path.open() as f:
        for raw_line in f:
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            if line.startswith("  ") or line.startswith("\t"):
                # Nested key under current_section.
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
    # List like [0.9, 0.95]
    if s.startswith("[") and s.endswith("]"):
        return [_parse_val(x.strip()) for x in s[1:-1].split(",")]
    return s.strip('"\'')


# ── probe config → argv translation ───────────────────────────────────────────

def _probe_to_argv(
    cfg: dict,
    data_dir: str,
    run_dir: str,
    probe_name: str,
) -> List[str]:
    """Convert a probe YAML config dict to a list of CLI argv strings."""
    argv = [
        "--data_dir",   data_dir,
        "--run_dir",    run_dir,
        "--probe_name", probe_name,
    ]

    # Training constants shared by all probes.
    argv += ["--max_steps",    str(cfg.get("max_steps",    1500))]
    argv += ["--batch_size",   str(cfg.get("batch_size",   8))]
    argv += ["--grad_accum",   str(cfg.get("grad_accum",   16))]
    argv += ["--block_size",   str(cfg.get("block_size",   4096))]
    argv += ["--warmup_steps", str(cfg.get("warmup_steps", 500))]
    argv += ["--val_every",    str(cfg.get("val_every",    500))]
    argv += ["--val_steps",    str(cfg.get("val_steps",    300))]
    argv += ["--seed",         str(cfg.get("seed",         42))]

    if cfg.get("torch_compile", True):
        argv.append("--compile")

    # Optimizer section.
    opt = cfg.get("optimizer", {})
    opt_type = opt.get("type", "muon_adam")
    argv += ["--optimizer", opt_type]
    argv += ["--muon_lr",      str(opt.get("muon_lr",      0.02))]
    argv += ["--adam_lr",      str(opt.get("adam_lr",      6e-4))]
    argv += ["--weight_decay", str(opt.get("weight_decay", 0.1))]

    # Architecture overrides (only emit if explicitly set in config).
    arch = cfg.get("arch", {})
    for key, cli_flag in [
        ("d_ff",           "--d_ff"),
        ("n_layers",       "--n_layers"),
        ("rope_fraction",  "--rope_fraction"),
        ("n_value_embeds", "--n_value_embeds"),
        ("attention_type", "--attention_type"),
        ("num_kv_heads",   "--num_kv_heads"),
        ("mlp_type",       "--mlp_type"),
    ]:
        if key in arch:
            argv += [cli_flag, str(arch[key])]

    if "use_x0_mixin" in arch:
        argv += ["--use_x0_mixin", str(arch["use_x0_mixin"]).lower()]

    return argv


# ── summary helpers ────────────────────────────────────────────────────────────

def _append_result(results_md: Path, probe_name: str, summary: dict) -> None:
    val_at   = summary.get("val_at", {})
    val_500  = val_at.get("val@500",  "—")
    val_1000 = val_at.get("val@1000", "—")
    val_1500 = val_at.get("val@1500", "—")
    d1 = summary.get("descent_rate_1", "—")
    d2 = summary.get("descent_rate_2", "—")
    opt = summary.get("optimizer", "—")
    wd  = summary.get("weight_decay", "—")
    mlr = summary.get("muon_lr", "—")

    row = (
        f"| {probe_name:<30} | {str(val_500):<7} | {str(val_1000):<7} | "
        f"{str(val_1500):<7} | {str(d1):<10} | {str(d2):<10} | "
        f"{opt:<10} | {str(mlr):<8} | {str(wd):<5} |\n"
    )

    if not results_md.exists():
        results_md.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# SwiftLlama-350M Ablation Results\n\n"
            "| Probe                          | val@500 | val@1000 | val@1500 | "
            "drate1      | drate2      | optimizer  | muon_lr  | wd    |\n"
            "|--------------------------------|---------|----------|----------|"
            "------------|------------|------------|----------|-------|\n"
        )
        results_md.write_text(header + row)
    else:
        with results_md.open("a") as f:
            f.write(row)

    print(f"\n{'='*70}")
    print(f"PROBE RESULT: {probe_name}")
    print(f"  val@500={val_500}  val@1000={val_1000}  val@1500={val_1500}")
    print(f"  descent_1={d1}  descent_2={d2}")
    print(f"{'='*70}\n")


# ── main ──────────────────────────────────────────────────────────────────────

PROBE_DIR = Path(__file__).resolve().parents[1] / "configs" / "ablation"

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run SwiftLlama ablation probes sequentially",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", required=True,
                    help="Directory with edufineweb_*.npy shards")
    ap.add_argument("--runs-dir", default="/workspace/runs/ablation",
                    help="Parent directory for per-probe run dirs")
    ap.add_argument("--probes", nargs="*", default=None,
                    help="Paths to probe YAML files (default: all in configs/ablation/)")
    ap.add_argument("--probe-name", type=str, default=None,
                    help="Run a single probe by name (e.g. A1_current_baseline)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip probes that already have a run_summary.json")
    args = ap.parse_args()

    runs_dir     = Path(args.runs_dir)
    results_md   = runs_dir / "probe_results.md"

    # Collect probe files.
    if args.probe_name:
        probe_files = [PROBE_DIR / f"{args.probe_name}.yaml"]
    elif args.probes:
        probe_files = [Path(p) for p in args.probes]
    else:
        probe_files = sorted(PROBE_DIR.glob("*.yaml"))

    if not probe_files:
        print(f"ERROR: no probe configs found in {PROBE_DIR}", file=sys.stderr)
        return 1

    print(f"Running {len(probe_files)} probe(s):")
    for pf in probe_files:
        print(f"  {pf.name}")
    print()

    # Import training main lazily (avoids CUDA init on import).
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import train_swiftllama_350m as trainer_mod

    for probe_file in probe_files:
        if not probe_file.exists():
            print(f"WARNING: {probe_file} not found — skipping", file=sys.stderr)
            continue

        cfg = _load_yaml(probe_file)
        probe_name = cfg.get("probe_name") or probe_file.stem
        run_dir = runs_dir / probe_name

        if args.skip_existing and (run_dir / "run_summary.json").exists():
            print(f"[SKIP] {probe_name} — already has run_summary.json")
            continue

        print(f"\n{'='*70}")
        print(f"STARTING PROBE: {probe_name}")
        print(f"{'='*70}")

        argv = _probe_to_argv(cfg, args.data_dir, str(run_dir), probe_name)
        print(f"argv: {' '.join(argv)}\n")

        # Patch sys.argv and call main().
        saved_argv = sys.argv
        sys.argv = [trainer_mod.__file__] + argv
        try:
            trainer_mod.main()
        finally:
            sys.argv = saved_argv

        # Read summary and append to results table.
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            with summary_path.open() as f:
                summary = json.load(f)
            _append_result(results_md, probe_name, summary)
        else:
            print(f"WARNING: {summary_path} not found after training", file=sys.stderr)

    print(f"\nAll probes complete. Results: {results_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
