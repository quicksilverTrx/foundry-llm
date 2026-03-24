from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_lab.core.data.pretok_shards import build_tinyllama_p15_pretokenized_shards
from llm_lab.core.data.tinyllama_p15_tinystories_prep import write_fixed_split_manifest
from llm_lab.core.tokenization.tinyllama_p15_tokenizer_artifact import (
    TOKENIZER_ARTIFACT_FILENAMES,
    build_tokenizer_artifact_from_file,
)

_TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
_VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"


def _download_if_missing(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    print(f"Downloading: {url}")
    urlretrieve(url, out_path)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"download failed or empty file: {out_path}")
    return out_path


def _canonicalize_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dst


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare TinyStories tokenizer + split-preserved pretok shards for tinyllama_p15.")
    p.add_argument("--data-dir", type=Path, default=Path("data/tinystories"))
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/tinyllama_p15_tinystories"))
    p.add_argument("--train-url", type=str, default=_TRAIN_URL)
    p.add_argument("--valid-url", type=str, default=_VALID_URL)
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--max-tokens-per-shard", type=int, default=1_000_000)
    p.add_argument("--backend-family", type=str, default="sentencepiece", choices=("sentencepiece", "legacy_bpe", "bpe"))
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--train-ratio", type=float, default=0.98)
    p.add_argument("--report-path", type=Path, default=Path("artifacts/tinyllama_p15_tinystories/prepare_report.json"))
    p.add_argument(
        "--stamp-config",
        type=Path,
        action="append",
        default=[],
        help="Optional config path(s) to stamp with tokenizer hash and artifact roots.",
    )
    return p


def _rel_from(path: Path, *, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _stamp_config(
    *,
    config_path: Path,
    tokenizer_dir: Path,
    shards_train_root: Path,
    shards_val_root: Path,
    tokenizer_hash: str,
) -> None:
    cfg_path = config_path.expanduser().resolve()
    config_dir = cfg_path.parent
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {cfg_path}")

    tok = dict(payload.get("tokenizer", {}))
    data = dict(payload.get("data", {}))

    tok["artifact_dir"] = _rel_from(tokenizer_dir, base_dir=config_dir)
    tok["tokenizer_hash"] = tokenizer_hash
    data["train_root_dir"] = _rel_from(shards_train_root, base_dir=config_dir)
    data["val_root_dir"] = _rel_from(shards_val_root, base_dir=config_dir)
    data.setdefault("train_split", "train")
    data.setdefault("val_split", "val")

    payload["tokenizer"] = tok
    payload["data"] = data
    cfg_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()

    raw_train = _download_if_missing(args.train_url, data_dir / "TinyStoriesV2-GPT4-train.txt")
    raw_valid = _download_if_missing(args.valid_url, data_dir / "TinyStoriesV2-GPT4-valid.txt")

    train_txt = _canonicalize_copy(raw_train, data_dir / "train.txt")
    valid_txt = _canonicalize_copy(raw_valid, data_dir / "valid.txt")

    tokenizer_dir = artifact_root / "tokenizer"
    tokenizer_paths = build_tokenizer_artifact_from_file(
        input_file=train_txt,
        output_dir=tokenizer_dir,
        vocab_size=int(args.vocab_size),
        model_type=args.backend_family,
        train_ratio=float(args.train_ratio),
        split_seed=int(args.split_seed),
    )
    tokenizer_hash = (tokenizer_dir / TOKENIZER_ARTIFACT_FILENAMES["tokenizer_hash"]).read_text(encoding="utf-8").strip()

    manifests_dir = artifact_root / "manifests"
    train_manifest = write_fixed_split_manifest(
        input_file=train_txt,
        split="train",
        tokenizer_hash=tokenizer_hash,
        output_path=manifests_dir / "train_split_manifest.json",
        split_seed=int(args.split_seed),
    )
    val_manifest = write_fixed_split_manifest(
        input_file=valid_txt,
        split="val",
        tokenizer_hash=tokenizer_hash,
        output_path=manifests_dir / "val_split_manifest.json",
        split_seed=int(args.split_seed),
    )

    shards_train_root = artifact_root / "shards_train"
    shards_val_root = artifact_root / "shards_val"

    build_tinyllama_p15_pretokenized_shards(
        input_file=train_txt,
        tokenizer_artifact_dir=tokenizer_dir,
        split_manifest_path=train_manifest,
        output_dir=shards_train_root,
        max_tokens_per_shard=int(args.max_tokens_per_shard),
    )
    build_tinyllama_p15_pretokenized_shards(
        input_file=valid_txt,
        tokenizer_artifact_dir=tokenizer_dir,
        split_manifest_path=val_manifest,
        output_dir=shards_val_root,
        max_tokens_per_shard=int(args.max_tokens_per_shard),
    )

    stamped_configs: list[str] = []
    for cfg in args.stamp_config:
        _stamp_config(
            config_path=cfg,
            tokenizer_dir=tokenizer_dir,
            shards_train_root=shards_train_root,
            shards_val_root=shards_val_root,
            tokenizer_hash=tokenizer_hash,
        )
        stamped_configs.append(str(cfg.expanduser().resolve()))

    report = {
        "data": {
            "raw_train": str(raw_train),
            "raw_valid": str(raw_valid),
            "train_txt": str(train_txt),
            "valid_txt": str(valid_txt),
        },
        "artifacts": {
            "artifact_root": str(artifact_root),
            "tokenizer_dir": str(tokenizer_dir),
            "tokenizer_hash": tokenizer_hash,
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "shards_train_root": str(shards_train_root),
            "shards_val_root": str(shards_val_root),
        },
        "stamped_configs": stamped_configs,
    }
    report_path = args.report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"prepare_report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
