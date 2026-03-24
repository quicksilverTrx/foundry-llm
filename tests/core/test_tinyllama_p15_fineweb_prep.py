from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from llm_lab.core.data.pretok_shards import ROOT_MANIFEST_FILENAME
from llm_lab.core.data.tinyllama_p15_tinystories_prep import build_fixed_split_manifest
from llm_lab.core.tokenization.tinyllama_p15_tokenizer_artifact import load_tokenizer_from_artifact_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import tinyllama_p15_prepare_finewebedu as fineweb_prep


def _prepare_script_path() -> Path:
    return REPO_ROOT / "scripts" / "tinyllama_p15_prepare_finewebedu.py"


def _preflight_script_path() -> Path:
    return REPO_ROOT / "scripts" / "p15_preflight_nanollama.py"


def _train_script_path() -> Path:
    return REPO_ROOT / "scripts" / "p15_train_nanollama.py"


def _sample_script_path() -> Path:
    return REPO_ROOT / "scripts" / "p15_sample_package.py"


def _run_subprocess(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _write_synthetic_source(path: Path, *, docs: int = 320) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(docs):
        lines.append(
            f"Document {i} contains token_{i % 37} token_{(i * 7) % 43} and sentence number {i}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_minimal_train_config(path: Path) -> None:
    payload = {
        "model": {"block_size": 16, "vocab_size": 256},
        "tokenizer": {"artifact_dir": "TOKENIZER_PLACEHOLDER", "tokenizer_hash": ""},
        "data": {
            "train_root_dir": "TRAIN_PLACEHOLDER",
            "val_root_dir": "VAL_PLACEHOLDER",
            "train_split": "train",
            "val_split": "val",
        },
        "run": {"output_root": "runs", "run_name": "reuse_smoke"},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_rehearsal_lane_is_deterministic_across_reruns(tmp_path: Path) -> None:
    source_file = tmp_path / "source.txt"
    _write_synthetic_source(source_file, docs=420)

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    report_a = run_a / "report.json"
    report_b = run_b / "report.json"

    common_args = [
        "--lane",
        "rehearsal",
        "--source-file",
        str(source_file),
        "--vocab-size",
        "256",
        "--tokenizer-train-byte-cap",
        "32768",
        "--tokenizer-sample-modulus",
        "1",
        "--tokenizer-sample-acceptance",
        "1",
        "--rehearsal-byte-cap",
        "24576",
        "--max-tokens-per-shard",
        "1024",
    ]

    proc_a = _run_subprocess(
        [
            sys.executable,
            str(_prepare_script_path()),
            *common_args,
            "--data-root",
            str(run_a / "data"),
            "--artifact-root",
            str(run_a / "artifacts"),
            "--report-path",
            str(report_a),
        ],
        cwd=tmp_path,
    )
    assert proc_a.returncode == 0, f"prepare run A failed\nstdout:\n{proc_a.stdout}\nstderr:\n{proc_a.stderr}"

    proc_b = _run_subprocess(
        [
            sys.executable,
            str(_prepare_script_path()),
            *common_args,
            "--data-root",
            str(run_b / "data"),
            "--artifact-root",
            str(run_b / "artifacts"),
            "--report-path",
            str(report_b),
        ],
        cwd=tmp_path,
    )
    assert proc_b.returncode == 0, f"prepare run B failed\nstdout:\n{proc_b.stdout}\nstderr:\n{proc_b.stderr}"

    rep_a = json.loads(report_a.read_text(encoding="utf-8"))
    rep_b = json.loads(report_b.read_text(encoding="utf-8"))

    assert rep_a["tokenizer"]["tokenizer_hash"] == rep_b["tokenizer"]["tokenizer_hash"]
    assert rep_a["lanes"]["rehearsal"]["train_docs"] == rep_b["lanes"]["rehearsal"]["train_docs"]
    assert rep_a["lanes"]["rehearsal"]["val_docs"] == rep_b["lanes"]["rehearsal"]["val_docs"]

    train_a = Path(rep_a["lanes"]["rehearsal"]["artifacts"]["train_txt"]).read_text(encoding="utf-8")
    train_b = Path(rep_b["lanes"]["rehearsal"]["artifacts"]["train_txt"]).read_text(encoding="utf-8")
    val_a = Path(rep_a["lanes"]["rehearsal"]["artifacts"]["valid_txt"]).read_text(encoding="utf-8")
    val_b = Path(rep_b["lanes"]["rehearsal"]["artifacts"]["valid_txt"]).read_text(encoding="utf-8")
    assert train_a == train_b
    assert val_a == val_b

    for rep in (rep_a, rep_b):
        tok_hash = rep["tokenizer"]["tokenizer_hash"]
        rehearsal_artifacts = rep["lanes"]["rehearsal"]["artifacts"]
        shards_train_root = Path(rehearsal_artifacts["shards_train_root"])
        shards_val_root = Path(rehearsal_artifacts["shards_val_root"])
        train_root_manifest = json.loads((shards_train_root / ROOT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        val_root_manifest = json.loads((shards_val_root / ROOT_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert train_root_manifest["tokenizer_provenance"]["tokenizer_hash"] == tok_hash
        assert val_root_manifest["tokenizer_provenance"]["tokenizer_hash"] == tok_hash


def test_canonical_defaults_to_99_1_and_tokenizer_subset_comes_from_train_only(tmp_path: Path) -> None:
    args = fineweb_prep._build_parser().parse_args([])
    assert args.canonical_train_ratio == 0.99
    assert args.tokenizer_train_byte_cap == 4_294_967_296
    assert args.max_tokens_per_shard == 1_000_000

    source_file = tmp_path / "source.txt"
    _write_synthetic_source(source_file, docs=4096)
    report_path = tmp_path / "prepare_report.json"

    proc = _run_subprocess(
        [
            sys.executable,
            str(_prepare_script_path()),
            "--lane",
            "canonical",
            "--source-file",
            str(source_file),
            "--data-root",
            str(tmp_path / "data"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--vocab-size",
            "256",
            "--tokenizer-train-byte-cap",
            "10485760",
            "--tokenizer-sample-modulus",
            "1",
            "--tokenizer-sample-acceptance",
            "1",
            "--max-tokens-per-shard",
            "1024",
            "--report-path",
            str(report_path),
        ],
        cwd=tmp_path,
    )
    assert proc.returncode == 0, f"prepare failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["tokenizer"]["subset_source_split"] == "canonical_train"
    assert report["tokenizer"]["subset_usage_fraction"] == 1.0
    assert report["lanes"]["canonical"]["train_ratio"] == 0.99
    assert report["lanes"]["canonical"]["val_ratio"] == 0.01
    assert report["lanes"]["canonical"]["val_docs"] > 0

    train_txt = Path(report["lanes"]["canonical"]["artifacts"]["train_txt"]).read_text(encoding="utf-8")
    valid_txt = Path(report["lanes"]["canonical"]["artifacts"]["valid_txt"]).read_text(encoding="utf-8")
    subset_txt = Path(report["tokenizer"]["subset_path"]).read_text(encoding="utf-8")

    assert subset_txt == train_txt
    assert subset_txt != valid_txt
    assert report["tokenizer"]["subset_docs"] == report["lanes"]["canonical"]["train_docs"]


def test_fixed_split_manifest_streaming_hash_matches_existing_contract(tmp_path: Path) -> None:
    input_file = tmp_path / "docs.txt"
    docs = ["alpha", "beta gamma", "", "delta", "emoji cafe"]
    input_file.write_text("\n".join(docs) + "\n", encoding="utf-8")

    train_manifest = build_fixed_split_manifest(
        input_file=input_file,
        split="train",
        tokenizer_hash="tok-hash",
        split_seed=7,
    )
    val_manifest = build_fixed_split_manifest(
        input_file=input_file,
        split="val",
        tokenizer_hash="tok-hash",
        split_seed=7,
    )

    non_empty_docs = [doc for doc in docs if doc]
    expected_docs_sha = __import__("hashlib").sha256(
        json.dumps(non_empty_docs, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    expected_train_assignment_sha = __import__("hashlib").sha256(
        json.dumps([1] * len(non_empty_docs), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    expected_val_assignment_sha = __import__("hashlib").sha256(
        json.dumps([0] * len(non_empty_docs), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()

    assert train_manifest["docs_sha256"] == expected_docs_sha
    assert val_manifest["docs_sha256"] == expected_docs_sha
    assert train_manifest["assignment_sha256"] == expected_train_assignment_sha
    assert val_manifest["assignment_sha256"] == expected_val_assignment_sha


def test_canonical_prepare_reuses_complete_artifacts_and_restamps(tmp_path: Path) -> None:
    source_file = tmp_path / "source.txt"
    _write_synthetic_source(source_file, docs=640)
    cfg_path = tmp_path / "runtime_config.json"
    _write_minimal_train_config(cfg_path)
    report_path = tmp_path / "prepare_report.json"

    base_cmd = [
        sys.executable,
        str(_prepare_script_path()),
        "--lane",
        "canonical",
        "--source-file",
        str(source_file),
        "--data-root",
        str(tmp_path / "data"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--vocab-size",
        "256",
        "--tokenizer-train-byte-cap",
        "1048576",
        "--tokenizer-sample-modulus",
        "1",
        "--tokenizer-sample-acceptance",
        "1",
        "--max-tokens-per-shard",
        "1024",
        "--stamp-config-canonical",
        str(cfg_path),
        "--report-path",
        str(report_path),
    ]

    proc_first = _run_subprocess(base_cmd, cwd=tmp_path)
    assert proc_first.returncode == 0, f"prepare failed\nstdout:\n{proc_first.stdout}\nstderr:\n{proc_first.stderr}"
    first_report = json.loads(report_path.read_text(encoding="utf-8"))
    train_manifest = Path(first_report["lanes"]["canonical"]["artifacts"]["train_manifest"])
    val_manifest = Path(first_report["lanes"]["canonical"]["artifacts"]["val_manifest"])
    train_root_manifest = Path(first_report["lanes"]["canonical"]["artifacts"]["shards_train_root"]) / ROOT_MANIFEST_FILENAME
    val_root_manifest = Path(first_report["lanes"]["canonical"]["artifacts"]["shards_val_root"]) / ROOT_MANIFEST_FILENAME
    mtimes_before = {
        path: path.stat().st_mtime_ns
        for path in (train_manifest, val_manifest, train_root_manifest, val_root_manifest)
    }

    time.sleep(0.01)
    proc_second = _run_subprocess(base_cmd, cwd=tmp_path)
    assert proc_second.returncode == 0, f"prepare reuse failed\nstdout:\n{proc_second.stdout}\nstderr:\n{proc_second.stderr}"
    second_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "prepare_reuse_detected" in proc_second.stdout
    assert second_report["reused_existing_artifacts"] is True
    assert second_report["reuse_reason"] == "complete_existing_canonical_prepare"
    for path, mtime_before in mtimes_before.items():
        assert path.stat().st_mtime_ns == mtime_before

    stamped_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert stamped_cfg["tokenizer"]["tokenizer_hash"] == second_report["tokenizer"]["tokenizer_hash"]
    assert "canonical/shards_train" in stamped_cfg["data"]["train_root_dir"]
    assert "canonical/shards_val" in stamped_cfg["data"]["val_root_dir"]


def test_canonical_prepare_partial_artifacts_do_not_falsely_reuse(tmp_path: Path) -> None:
    source_file = tmp_path / "source.txt"
    _write_synthetic_source(source_file, docs=512)
    report_path = tmp_path / "prepare_report.json"

    base_cmd = [
        sys.executable,
        str(_prepare_script_path()),
        "--lane",
        "canonical",
        "--source-file",
        str(source_file),
        "--data-root",
        str(tmp_path / "data"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--vocab-size",
        "256",
        "--tokenizer-train-byte-cap",
        "1048576",
        "--tokenizer-sample-modulus",
        "1",
        "--tokenizer-sample-acceptance",
        "1",
        "--max-tokens-per-shard",
        "1024",
        "--report-path",
        str(report_path),
    ]

    proc_first = _run_subprocess(base_cmd, cwd=tmp_path)
    assert proc_first.returncode == 0, f"prepare failed\nstdout:\n{proc_first.stdout}\nstderr:\n{proc_first.stderr}"
    first_report = json.loads(report_path.read_text(encoding="utf-8"))
    val_manifest = Path(first_report["lanes"]["canonical"]["artifacts"]["val_manifest"])
    val_manifest.unlink()

    proc_second = _run_subprocess(base_cmd, cwd=tmp_path)
    assert proc_second.returncode == 0, f"prepare rebuild failed\nstdout:\n{proc_second.stdout}\nstderr:\n{proc_second.stderr}"
    second_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "prepare_reuse_detected" not in proc_second.stdout
    assert "prepare_canonical_manifests_start" in proc_second.stdout
    assert second_report["reused_existing_artifacts"] is False


def test_rehearsal_lane_local_smoke_produces_receipts(tmp_path: Path) -> None:
    source_file = tmp_path / "source.txt"
    _write_synthetic_source(source_file, docs=380)

    cfg_path = tmp_path / "rehearsal_train_config.json"
    cfg = {
        "model": {
            "arch_family": "nanollama",
            "vocab_size": 256,
            "block_size": 16,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "num_kv_heads": 2,
            "d_ff": 128,
            "dropout": 0.0,
            "norm_type": "rmsnorm",
            "mlp_type": "swiglu",
            "attention_type": "gqa",
            "pos_encoding_type": "rope",
        },
        "tokenizer": {
            "backend_family": "sentencepiece",
            "artifact_dir": "TOKENIZER_PLACEHOLDER",
            "tokenizer_hash": "",
        },
        "data": {
            "train_root_dir": "TRAIN_PLACEHOLDER",
            "val_root_dir": "VAL_PLACEHOLDER",
            "train_split": "train",
            "val_split": "val",
            "base_seed": 7,
        },
        "train": {
            "seed": 7,
            "device": "cpu",
            "dtype": "fp32",
            "batch_size": 2,
            "grad_accum_steps": 1,
            "optimizer": "adamw",
            "lr": 0.001,
            "lr_schedule": {"type": "cosine", "warmup_steps": 1, "min_lr": 0.0001},
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "num_workers": 0,
            "num_epochs": 2,
            "max_steps": 6,
            "eval_every_n_steps": 2,
            "eval_max_batches": 4,
            "log_every_n_steps": 1,
        },
        "run": {
            "contract_tier": "hero_baseline_v1",
            "intent": "test_rehearsal_lane",
            "checkpoint_cadence": "best_only",
            "output_root": str(tmp_path / "runs"),
            "run_name": "fineweb_rehearsal_smoke",
        },
        "samples": {
            "decode": "greedy",
            "max_new_tokens": 8,
            "prompts": ["The best way to solve a hard problem is"],
        },
    }
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    report_path = tmp_path / "prepare_report.json"
    proc_prepare = _run_subprocess(
        [
            sys.executable,
            str(_prepare_script_path()),
            "--lane",
            "rehearsal",
            "--source-file",
            str(source_file),
            "--data-root",
            str(tmp_path / "data"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--vocab-size",
            "256",
            "--tokenizer-train-byte-cap",
            "32768",
            "--tokenizer-sample-modulus",
            "1",
            "--tokenizer-sample-acceptance",
            "1",
            "--rehearsal-byte-cap",
            "28672",
            "--max-tokens-per-shard",
            "512",
            "--stamp-config-rehearsal",
            str(cfg_path),
            "--report-path",
            str(report_path),
        ],
        cwd=tmp_path,
    )
    assert (
        proc_prepare.returncode == 0
    ), f"prepare failed\nstdout:\n{proc_prepare.stdout}\nstderr:\n{proc_prepare.stderr}"

    stamped_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tok_dir = (cfg_path.parent / stamped_cfg["tokenizer"]["artifact_dir"]).resolve()
    tokenizer, _ = load_tokenizer_from_artifact_dir(tok_dir)
    stamped_cfg["model"]["vocab_size"] = len(tokenizer.stoi)
    cfg_path.write_text(json.dumps(stamped_cfg, indent=2, sort_keys=True), encoding="utf-8")

    preflight_path = tmp_path / "preflight.json"
    proc_preflight = _run_subprocess(
        [
            sys.executable,
            str(_preflight_script_path()),
            "--config",
            str(cfg_path),
            "--output",
            str(preflight_path),
        ],
        cwd=tmp_path,
    )
    assert (
        proc_preflight.returncode == 0
    ), f"preflight failed\nstdout:\n{proc_preflight.stdout}\nstderr:\n{proc_preflight.stderr}"

    proc_train = _run_subprocess(
        [
            sys.executable,
            str(_train_script_path()),
            "--config",
            str(cfg_path),
            "--device",
            "cpu",
        ],
        cwd=tmp_path,
    )
    assert proc_train.returncode == 0, f"train failed\nstdout:\n{proc_train.stdout}\nstderr:\n{proc_train.stderr}"

    run_dir = tmp_path / "runs" / "fineweb_rehearsal_smoke"
    assert (run_dir / "checkpoints" / "best_val.pt").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "last.ckpt").exists()
    assert (run_dir / "package" / "checkpoints" / "best_val.pt").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "samples" / "index.json").exists()

    sample_a = tmp_path / "sample_a.json"
    sample_b = tmp_path / "sample_b.json"
    sample_cmd = [
        sys.executable,
        str(_sample_script_path()),
        "--package-dir",
        str((run_dir / "package").resolve()),
        "--prompt",
        "The best way to solve a hard problem is",
        "--max-new-tokens",
        "8",
        "--device",
        "cpu",
    ]
    proc_sample_a = _run_subprocess(sample_cmd + ["--output-file", str(sample_a.resolve())], cwd=tmp_path)
    proc_sample_b = _run_subprocess(sample_cmd + ["--output-file", str(sample_b.resolve())], cwd=tmp_path)
    assert proc_sample_a.returncode == 0
    assert proc_sample_b.returncode == 0
    payload_a = json.loads(sample_a.read_text(encoding="utf-8"))
    payload_b = json.loads(sample_b.read_text(encoding="utf-8"))
    assert payload_a["completion_token_ids"] == payload_b["completion_token_ids"]
    assert payload_a["text"] == payload_b["text"]
