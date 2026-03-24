from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_lab.core.data.pretok_shards import build_tinyllama_p15_pretokenized_shards
from llm_lab.core.data.tinyllama_p15_tinystories_prep import write_fixed_split_manifest
from llm_lab.core.tokenization.tinyllama_p15_tokenizer_artifact import (
    TOKENIZER_ARTIFACT_FILENAMES,
    build_tokenizer_artifact_from_train_file,
)


DEFAULT_DATASET_ID = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_NAME = "sample-10BT"
DEFAULT_TEXT_FIELD = "text"
HEARTBEAT_SECS = 30.0


@dataclass
class LaneCounters:
    train_docs: int = 0
    val_docs: int = 0
    train_bytes: int = 0
    val_bytes: int = 0

    @property
    def total_docs(self) -> int:
        return int(self.train_docs + self.val_docs)

    @property
    def total_bytes(self) -> int:
        return int(self.train_bytes + self.val_bytes)


def _canonicalize_doc(raw: str) -> str:
    return " ".join(str(raw).split()).strip()


def _stable_score(seed: int, index: int, doc: str) -> float:
    payload = f"{seed}\n{index}\n{doc}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)


def _is_train_doc(*, split_seed: int, train_ratio: float, index: int, doc: str) -> bool:
    if not (0.0 < float(train_ratio) < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio!r}")
    return _stable_score(split_seed, index, doc) < float(train_ratio)


def _is_tokenizer_subset_doc(*, selection_seed: int, index: int, doc: str, modulus: int, acceptance: int) -> bool:
    if modulus <= 0:
        raise ValueError(f"tokenizer_sample_modulus must be > 0, got {modulus}")
    if acceptance <= 0 or acceptance > modulus:
        raise ValueError(
            f"tokenizer_sample_acceptance must be in [1, modulus] (acceptance={acceptance}, modulus={modulus})"
        )
    payload = f"{selection_seed}\n{index}\n{doc}\nsubset".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (value % modulus) < acceptance


def _iter_docs_from_source(
    *,
    source_file: Path | None,
    dataset_id: str,
    dataset_name: str,
    text_field: str,
) -> Iterable[str]:
    if source_file is not None:
        with source_file.open("r", encoding="utf-8") as f:
            for line in f:
                yield line.rstrip("\n")
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised via source_file tests
        raise RuntimeError(
            "datasets package is required for Hugging Face streaming mode. "
            "Install with `pip install datasets` in foundry-llm."
        ) from exc

    stream = load_dataset(dataset_id, name=dataset_name, split="train", streaming=True)
    for row in stream:
        if not isinstance(row, dict):
            continue
        text = row.get(text_field, "")
        if text is None:
            continue
        yield str(text)


def _rel_from(path: Path, *, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / float(1024**3):.2f}GiB"


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _path_exists_non_empty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _shard_root_is_complete(root_dir: Path, *, split: str) -> bool:
    root_manifest = root_dir / "tinyllama_p15_shards_manifest.json"
    if not _path_exists_non_empty(root_manifest):
        return False
    try:
        payload = _read_json_object(root_manifest)
    except Exception:
        return False
    inventory = payload.get("shard_inventory")
    if not isinstance(inventory, list) or not inventory:
        return False
    entries = [entry for entry in inventory if isinstance(entry, dict) and entry.get("split") == split]
    if not entries:
        return False
    for entry in entries:
        bin_rel = entry.get("bin")
        sidecar_rel = entry.get("sidecar")
        if not isinstance(bin_rel, str) or not isinstance(sidecar_rel, str):
            return False
        if not _path_exists_non_empty(root_dir / Path(bin_rel)):
            return False
        if not _path_exists_non_empty(root_dir / Path(sidecar_rel)):
            return False
    return True


def _canonical_reuse_payload(
    *,
    data_root: Path,
    artifact_root: Path,
    tokenizer_artifact_dir: Path,
    tokenizer_subset_path: Path,
    report_path: Path,
) -> dict[str, Any] | None:
    canonical_data_dir = data_root / "canonical"
    canonical_art_root = artifact_root / "canonical"
    canonical_manifests_dir = canonical_art_root / "manifests"
    canonical_train_txt = canonical_data_dir / "train.txt"
    canonical_valid_txt = canonical_data_dir / "valid.txt"
    train_manifest_path = canonical_manifests_dir / "train_split_manifest.json"
    val_manifest_path = canonical_manifests_dir / "val_split_manifest.json"
    shards_train_root = canonical_art_root / "shards_train"
    shards_val_root = canonical_art_root / "shards_val"
    tokenizer_hash_path = tokenizer_artifact_dir / TOKENIZER_ARTIFACT_FILENAMES["tokenizer_hash"]

    required_files = (
        tokenizer_hash_path,
        canonical_train_txt,
        canonical_valid_txt,
        train_manifest_path,
        val_manifest_path,
    )
    if not all(_path_exists_non_empty(path) for path in required_files):
        return None
    if not _shard_root_is_complete(shards_train_root, split="train"):
        return None
    if not _shard_root_is_complete(shards_val_root, split="val"):
        return None

    tokenizer_hash = tokenizer_hash_path.read_text(encoding="utf-8").strip()
    if not tokenizer_hash:
        return None

    train_manifest = _read_json_object(train_manifest_path)
    val_manifest = _read_json_object(val_manifest_path)
    previous_report = _read_json_object(report_path) if _path_exists_non_empty(report_path) else {}
    previous_source = previous_report.get("source", {}) if isinstance(previous_report.get("source"), dict) else {}
    previous_tokenizer = (
        previous_report.get("tokenizer", {}) if isinstance(previous_report.get("tokenizer"), dict) else {}
    )

    train_docs = int(train_manifest.get("total_docs", 0))
    val_docs = int(val_manifest.get("total_docs", 0))
    train_bytes = canonical_train_txt.stat().st_size
    val_bytes = canonical_valid_txt.stat().st_size
    subset_bytes = tokenizer_subset_path.stat().st_size if tokenizer_subset_path.exists() else 0

    return {
        "source": {
            "source_file": previous_source.get("source_file"),
            "dataset_id": previous_source.get("dataset_id"),
            "dataset_name": previous_source.get("dataset_name"),
            "text_field": previous_source.get("text_field"),
            "seen_docs": previous_source.get("seen_docs"),
            "non_empty_docs": previous_source.get("non_empty_docs", train_docs + val_docs),
        },
        "tokenizer": {
            "artifact_dir": str(tokenizer_artifact_dir),
            "tokenizer_hash": tokenizer_hash,
            "trained_this_run": False,
            "subset_path": str(tokenizer_subset_path),
            "subset_source_split": previous_tokenizer.get("subset_source_split", "canonical_train"),
            "subset_docs": int(previous_tokenizer.get("subset_docs", 0)),
            "subset_bytes": int(previous_tokenizer.get("subset_bytes", subset_bytes)),
            "subset_byte_cap": int(previous_tokenizer.get("subset_byte_cap", 0)),
            "subset_usage_fraction": float(previous_tokenizer.get("subset_usage_fraction", 1.0)),
            "selection_seed": int(previous_tokenizer.get("selection_seed", 1337)),
            "sample_modulus": int(previous_tokenizer.get("sample_modulus", 1000)),
            "sample_acceptance": int(previous_tokenizer.get("sample_acceptance", 100)),
        },
        "lanes": {
            "canonical": {
                "enabled": True,
                "train_ratio": 0.99,
                "val_ratio": 0.01,
                "split_seed": int(train_manifest.get("split_seed", 1337)),
                "train_docs": train_docs,
                "val_docs": val_docs,
                "total_docs": train_docs + val_docs,
                "train_bytes": train_bytes,
                "val_bytes": val_bytes,
                "total_bytes": train_bytes + val_bytes,
                "artifacts": {
                    "train_txt": str(canonical_train_txt),
                    "valid_txt": str(canonical_valid_txt),
                    "train_manifest": str(train_manifest_path),
                    "val_manifest": str(val_manifest_path),
                    "shards_train_root": str(shards_train_root),
                    "shards_val_root": str(shards_val_root),
                },
                "stamped_configs": [],
            },
            "rehearsal": {
                "enabled": False,
                "train_ratio": 0.0,
                "val_ratio": 0.0,
                "split_seed": 0,
                "train_docs": 0,
                "val_docs": 0,
                "total_docs": 0,
                "train_bytes": 0,
                "val_bytes": 0,
                "total_bytes": 0,
                "byte_cap": 0,
                "artifacts": None,
                "stamped_configs": [],
            },
        },
        "reused_existing_artifacts": True,
        "reuse_reason": "complete_existing_canonical_prepare",
    }


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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Prepare FineWeb-Edu sample-10BT artifacts for tinyllama_p15 with two lanes: "
            "canonical hero data and local rehearsal subset."
        )
    )
    p.add_argument("--lane", choices=("canonical", "rehearsal", "both"), default="rehearsal")
    p.add_argument("--source-file", type=Path, default=None, help="Optional local one-doc-per-line source file.")

    p.add_argument("--dataset-id", type=str, default=DEFAULT_DATASET_ID)
    p.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    p.add_argument("--text-field", type=str, default=DEFAULT_TEXT_FIELD)

    p.add_argument("--data-root", type=Path, default=Path("data/finewebedu_sample10bt"))
    p.add_argument("--artifact-root", type=Path, default=Path("artifacts/p15/finewebedu_sample10bt"))
    p.add_argument("--tokenizer-artifact-dir", type=Path, default=None)
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--backend-family", type=str, default="sentencepiece", choices=("sentencepiece",))
    p.add_argument("--force-retrain-tokenizer", action="store_true")
    p.add_argument("--tokenizer-train-byte-cap", type=int, default=4_294_967_296)
    p.add_argument("--tokenizer-selection-seed", type=int, default=1337)
    p.add_argument("--tokenizer-sample-modulus", type=int, default=1000)
    p.add_argument("--tokenizer-sample-acceptance", type=int, default=100)

    p.add_argument("--canonical-train-ratio", type=float, default=0.99)
    p.add_argument("--canonical-split-seed", type=int, default=1337)

    p.add_argument("--rehearsal-byte-cap", type=int, default=128 * 1024 * 1024)
    p.add_argument("--rehearsal-train-ratio", type=float, default=0.98)
    p.add_argument("--rehearsal-split-seed", type=int, default=2026)

    p.add_argument("--max-docs", type=int, default=None, help="Optional cap for development/testing only.")
    p.add_argument("--max-tokens-per-shard", type=int, default=1_000_000)
    p.add_argument(
        "--stamp-config-canonical",
        type=Path,
        action="append",
        default=[],
        help="Config path(s) to stamp with canonical lane shard roots.",
    )
    p.add_argument(
        "--stamp-config-rehearsal",
        type=Path,
        action="append",
        default=[],
        help="Config path(s) to stamp with rehearsal lane shard roots.",
    )
    p.add_argument("--report-path", type=Path, default=Path("artifacts/p15/finewebedu_sample10bt/prepare_report.json"))
    return p


def main() -> int:
    args = _build_parser().parse_args()
    start_time = time.monotonic()

    lane_canonical = args.lane in ("canonical", "both")
    lane_rehearsal = args.lane in ("rehearsal", "both")
    if not lane_canonical and not lane_rehearsal:
        raise ValueError(f"invalid lane selection: {args.lane!r}")

    data_root = args.data_root.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    source_file = args.source_file.expanduser().resolve() if args.source_file is not None else None
    if source_file is not None and not source_file.exists():
        raise FileNotFoundError(source_file)

    tokenizer_artifact_dir = (
        args.tokenizer_artifact_dir.expanduser().resolve()
        if args.tokenizer_artifact_dir is not None
        else (artifact_root / "tokenizer_sp16k").resolve()
    )
    tokenizer_subset_path = artifact_root / "tokenizer_training_subset.txt"

    canonical_data_dir = data_root / "canonical"
    rehearsal_data_dir = data_root / "rehearsal"

    canonical_train_txt = canonical_data_dir / "train.txt"
    canonical_valid_txt = canonical_data_dir / "valid.txt"
    rehearsal_train_txt = rehearsal_data_dir / "train.txt"
    rehearsal_valid_txt = rehearsal_data_dir / "valid.txt"

    canonical_counters = LaneCounters()
    rehearsal_counters = LaneCounters()

    tokenizer_hash_path = tokenizer_artifact_dir / TOKENIZER_ARTIFACT_FILENAMES["tokenizer_hash"]
    need_tokenizer_training = args.force_retrain_tokenizer or (not tokenizer_hash_path.exists())
    tokenizer_subset_docs = 0
    tokenizer_subset_bytes = 0

    artifact_root.mkdir(parents=True, exist_ok=True)
    if lane_canonical:
        canonical_data_dir.mkdir(parents=True, exist_ok=True)
    if lane_rehearsal:
        rehearsal_data_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_path.expanduser().resolve()

    if lane_canonical and not lane_rehearsal and not args.force_retrain_tokenizer:
        reuse_report = _canonical_reuse_payload(
            data_root=data_root,
            artifact_root=artifact_root,
            tokenizer_artifact_dir=tokenizer_artifact_dir,
            tokenizer_subset_path=tokenizer_subset_path,
            report_path=report_path,
        )
        if reuse_report is not None:
            canonical_artifacts = reuse_report["lanes"]["canonical"]["artifacts"]
            assert isinstance(canonical_artifacts, dict)
            shards_train_root = Path(str(canonical_artifacts["shards_train_root"]))
            shards_val_root = Path(str(canonical_artifacts["shards_val_root"]))
            tokenizer_hash = str(reuse_report["tokenizer"]["tokenizer_hash"])
            print(
                "prepare_reuse_detected "
                f"tokenizer_hash={tokenizer_hash} "
                f"train_txt={canonical_artifacts['train_txt']} "
                f"valid_txt={canonical_artifacts['valid_txt']}",
                flush=True,
            )
            stamped_canonical: list[str] = []
            for cfg in args.stamp_config_canonical:
                _stamp_config(
                    config_path=cfg,
                    tokenizer_dir=tokenizer_artifact_dir,
                    shards_train_root=shards_train_root,
                    shards_val_root=shards_val_root,
                    tokenizer_hash=tokenizer_hash,
                )
                stamped_canonical.append(str(cfg.expanduser().resolve()))
            reuse_report["lanes"]["canonical"]["stamped_configs"] = stamped_canonical
            reuse_report["report_path"] = str(report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(reuse_report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(
                "prepare_reuse_configs_stamped "
                f"count={len(stamped_canonical)} "
                f"report_path={report_path}",
                flush=True,
            )
            print("prepare_reuse_done", flush=True)
            return 0

    source_non_empty_docs = 0
    source_seen_docs = 0
    rehearsal_done = not lane_rehearsal
    last_heartbeat = time.monotonic()

    print(
        "prepare_start "
        f"lane={args.lane} "
        f"dataset={args.dataset_id}/{args.dataset_name} "
        f"source_file={source_file if source_file is not None else 'hf_stream'} "
        f"canonical_train_ratio={args.canonical_train_ratio} "
        f"tokenizer_byte_cap={args.tokenizer_train_byte_cap} "
        f"max_tokens_per_shard={args.max_tokens_per_shard}",
        flush=True,
    )

    with (
        canonical_train_txt.open("w", encoding="utf-8") if lane_canonical else open("/dev/null", "w", encoding="utf-8") as c_train_f,
        canonical_valid_txt.open("w", encoding="utf-8") if lane_canonical else open("/dev/null", "w", encoding="utf-8") as c_val_f,
        rehearsal_train_txt.open("w", encoding="utf-8") if lane_rehearsal else open("/dev/null", "w", encoding="utf-8") as r_train_f,
        rehearsal_valid_txt.open("w", encoding="utf-8") if lane_rehearsal else open("/dev/null", "w", encoding="utf-8") as r_val_f,
        tokenizer_subset_path.open("w", encoding="utf-8") if need_tokenizer_training else open("/dev/null", "w", encoding="utf-8") as tok_subset_f,
    ):
        for raw_doc in _iter_docs_from_source(
            source_file=source_file,
            dataset_id=str(args.dataset_id),
            dataset_name=str(args.dataset_name),
            text_field=str(args.text_field),
        ):
            doc = _canonicalize_doc(raw_doc)
            source_seen_docs += 1
            if not doc:
                continue
            if args.max_docs is not None and source_non_empty_docs >= int(args.max_docs):
                break
            source_non_empty_docs += 1
            doc_index = source_non_empty_docs - 1
            doc_byte_size = len(doc.encode("utf-8")) + 1

            canonical_is_train = False
            rehearsal_is_train = False

            if lane_canonical:
                canonical_is_train = _is_train_doc(
                    split_seed=int(args.canonical_split_seed),
                    train_ratio=float(args.canonical_train_ratio),
                    index=doc_index,
                    doc=doc,
                )
                if canonical_is_train:
                    c_train_f.write(doc + "\n")
                    canonical_counters.train_docs += 1
                    canonical_counters.train_bytes += doc_byte_size
                else:
                    c_val_f.write(doc + "\n")
                    canonical_counters.val_docs += 1
                    canonical_counters.val_bytes += doc_byte_size

            if lane_rehearsal and not rehearsal_done:
                if (rehearsal_counters.total_bytes + doc_byte_size) <= int(args.rehearsal_byte_cap):
                    rehearsal_is_train = _is_train_doc(
                        split_seed=int(args.rehearsal_split_seed),
                        train_ratio=float(args.rehearsal_train_ratio),
                        index=doc_index,
                        doc=doc,
                    )
                    if rehearsal_is_train:
                        r_train_f.write(doc + "\n")
                        rehearsal_counters.train_docs += 1
                        rehearsal_counters.train_bytes += doc_byte_size
                    else:
                        r_val_f.write(doc + "\n")
                        rehearsal_counters.val_docs += 1
                        rehearsal_counters.val_bytes += doc_byte_size
                if rehearsal_counters.total_bytes >= int(args.rehearsal_byte_cap):
                    rehearsal_done = True

            tokenizer_candidate_is_train = canonical_is_train if lane_canonical else rehearsal_is_train
            if (
                need_tokenizer_training
                and tokenizer_candidate_is_train
                and tokenizer_subset_bytes < int(args.tokenizer_train_byte_cap)
            ):
                if _is_tokenizer_subset_doc(
                    selection_seed=int(args.tokenizer_selection_seed),
                    index=doc_index,
                    doc=doc,
                    modulus=int(args.tokenizer_sample_modulus),
                    acceptance=int(args.tokenizer_sample_acceptance),
                ):
                    if (tokenizer_subset_bytes + doc_byte_size) <= int(args.tokenizer_train_byte_cap):
                        tok_subset_f.write(doc + "\n")
                        tokenizer_subset_docs += 1
                        tokenizer_subset_bytes += doc_byte_size

            if not lane_canonical:
                tokenizer_done = (not need_tokenizer_training) or (
                    tokenizer_subset_bytes >= int(args.tokenizer_train_byte_cap)
                )
                if rehearsal_done and tokenizer_done:
                    break

            now = time.monotonic()
            if (now - last_heartbeat) >= HEARTBEAT_SECS:
                print(
                    "prepare_heartbeat "
                    f"elapsed_secs={int(now - start_time)} "
                    f"seen_docs={source_seen_docs} "
                    f"non_empty_docs={source_non_empty_docs} "
                    f"canonical_train_docs={canonical_counters.train_docs} "
                    f"canonical_val_docs={canonical_counters.val_docs} "
                    f"canonical_total_bytes={canonical_counters.total_bytes} "
                    f"canonical_total_gib={_format_gib(canonical_counters.total_bytes)} "
                    f"tokenizer_subset_docs={tokenizer_subset_docs} "
                    f"tokenizer_subset_bytes={tokenizer_subset_bytes} "
                    f"tokenizer_subset_gib={_format_gib(tokenizer_subset_bytes)}",
                    flush=True,
                )
                last_heartbeat = now

    if lane_canonical and (canonical_counters.train_docs <= 0 or canonical_counters.val_docs <= 0):
        raise RuntimeError("canonical lane produced empty train or val split; adjust ratio/seed/source.")
    if lane_rehearsal and (rehearsal_counters.train_docs <= 0 or rehearsal_counters.val_docs <= 0):
        raise RuntimeError("rehearsal lane produced empty train or val split; increase rehearsal_byte_cap.")

    if need_tokenizer_training:
        if tokenizer_subset_docs <= 0:
            raise RuntimeError(
                "tokenizer subset is empty; adjust tokenizer sampling controls or increase tokenizer_train_byte_cap."
            )
        print(
            "prepare_tokenizer_start "
            f"subset_docs={tokenizer_subset_docs} "
            f"subset_bytes={tokenizer_subset_bytes} "
            f"subset_gib={_format_gib(tokenizer_subset_bytes)} "
            f"artifact_dir={tokenizer_artifact_dir}",
            flush=True,
        )
        build_tokenizer_artifact_from_train_file(
            input_file=tokenizer_subset_path,
            output_dir=tokenizer_artifact_dir,
            vocab_size=int(args.vocab_size),
            model_type=str(args.backend_family),  # type: ignore[arg-type]
        )
        print("prepare_tokenizer_done", flush=True)
    tokenizer_hash = tokenizer_hash_path.read_text(encoding="utf-8").strip()
    if not tokenizer_hash:
        raise RuntimeError(f"tokenizer hash missing/empty after preparation: {tokenizer_hash_path}")

    stamped_canonical: list[str] = []
    stamped_rehearsal: list[str] = []
    canonical_artifacts: dict[str, str] | None = None
    rehearsal_artifacts: dict[str, str] | None = None

    if lane_canonical:
        print("prepare_canonical_manifests_start", flush=True)
        canonical_art_root = artifact_root / "canonical"
        canonical_manifests_dir = canonical_art_root / "manifests"
        canonical_train_manifest = write_fixed_split_manifest(
            input_file=canonical_train_txt,
            split="train",
            tokenizer_hash=tokenizer_hash,
            output_path=canonical_manifests_dir / "train_split_manifest.json",
            split_seed=int(args.canonical_split_seed),
        )
        canonical_val_manifest = write_fixed_split_manifest(
            input_file=canonical_valid_txt,
            split="val",
            tokenizer_hash=tokenizer_hash,
            output_path=canonical_manifests_dir / "val_split_manifest.json",
            split_seed=int(args.canonical_split_seed),
        )
        canonical_shards_train_root = canonical_art_root / "shards_train"
        canonical_shards_val_root = canonical_art_root / "shards_val"
        print("prepare_canonical_shards_train_start", flush=True)
        build_tinyllama_p15_pretokenized_shards(
            input_file=canonical_train_txt,
            tokenizer_artifact_dir=tokenizer_artifact_dir,
            split_manifest_path=canonical_train_manifest,
            output_dir=canonical_shards_train_root,
            max_tokens_per_shard=int(args.max_tokens_per_shard),
        )
        print("prepare_canonical_shards_train_done", flush=True)
        print("prepare_canonical_shards_val_start", flush=True)
        build_tinyllama_p15_pretokenized_shards(
            input_file=canonical_valid_txt,
            tokenizer_artifact_dir=tokenizer_artifact_dir,
            split_manifest_path=canonical_val_manifest,
            output_dir=canonical_shards_val_root,
            max_tokens_per_shard=int(args.max_tokens_per_shard),
        )
        print("prepare_canonical_shards_val_done", flush=True)
        for cfg in args.stamp_config_canonical:
            _stamp_config(
                config_path=cfg,
                tokenizer_dir=tokenizer_artifact_dir,
                shards_train_root=canonical_shards_train_root,
                shards_val_root=canonical_shards_val_root,
                tokenizer_hash=tokenizer_hash,
            )
            stamped_canonical.append(str(cfg.expanduser().resolve()))
        canonical_artifacts = {
            "train_txt": str(canonical_train_txt),
            "valid_txt": str(canonical_valid_txt),
            "train_manifest": str(canonical_train_manifest),
            "val_manifest": str(canonical_val_manifest),
            "shards_train_root": str(canonical_shards_train_root),
            "shards_val_root": str(canonical_shards_val_root),
        }

    if lane_rehearsal:
        print("prepare_rehearsal_manifests_start", flush=True)
        rehearsal_art_root = artifact_root / "rehearsal"
        rehearsal_manifests_dir = rehearsal_art_root / "manifests"
        rehearsal_train_manifest = write_fixed_split_manifest(
            input_file=rehearsal_train_txt,
            split="train",
            tokenizer_hash=tokenizer_hash,
            output_path=rehearsal_manifests_dir / "train_split_manifest.json",
            split_seed=int(args.rehearsal_split_seed),
        )
        rehearsal_val_manifest = write_fixed_split_manifest(
            input_file=rehearsal_valid_txt,
            split="val",
            tokenizer_hash=tokenizer_hash,
            output_path=rehearsal_manifests_dir / "val_split_manifest.json",
            split_seed=int(args.rehearsal_split_seed),
        )
        rehearsal_shards_train_root = rehearsal_art_root / "shards_train"
        rehearsal_shards_val_root = rehearsal_art_root / "shards_val"
        print("prepare_rehearsal_shards_train_start", flush=True)
        build_tinyllama_p15_pretokenized_shards(
            input_file=rehearsal_train_txt,
            tokenizer_artifact_dir=tokenizer_artifact_dir,
            split_manifest_path=rehearsal_train_manifest,
            output_dir=rehearsal_shards_train_root,
            max_tokens_per_shard=int(args.max_tokens_per_shard),
        )
        print("prepare_rehearsal_shards_train_done", flush=True)
        print("prepare_rehearsal_shards_val_start", flush=True)
        build_tinyllama_p15_pretokenized_shards(
            input_file=rehearsal_valid_txt,
            tokenizer_artifact_dir=tokenizer_artifact_dir,
            split_manifest_path=rehearsal_val_manifest,
            output_dir=rehearsal_shards_val_root,
            max_tokens_per_shard=int(args.max_tokens_per_shard),
        )
        print("prepare_rehearsal_shards_val_done", flush=True)
        for cfg in args.stamp_config_rehearsal:
            _stamp_config(
                config_path=cfg,
                tokenizer_dir=tokenizer_artifact_dir,
                shards_train_root=rehearsal_shards_train_root,
                shards_val_root=rehearsal_shards_val_root,
                tokenizer_hash=tokenizer_hash,
            )
            stamped_rehearsal.append(str(cfg.expanduser().resolve()))
        rehearsal_artifacts = {
            "train_txt": str(rehearsal_train_txt),
            "valid_txt": str(rehearsal_valid_txt),
            "train_manifest": str(rehearsal_train_manifest),
            "val_manifest": str(rehearsal_val_manifest),
            "shards_train_root": str(rehearsal_shards_train_root),
            "shards_val_root": str(rehearsal_shards_val_root),
        }

    report = {
        "source": {
            "source_file": str(source_file) if source_file is not None else None,
            "dataset_id": str(args.dataset_id),
            "dataset_name": str(args.dataset_name),
            "text_field": str(args.text_field),
            "seen_docs": int(source_seen_docs),
            "non_empty_docs": int(source_non_empty_docs),
        },
        "tokenizer": {
            "artifact_dir": str(tokenizer_artifact_dir),
            "tokenizer_hash": tokenizer_hash,
            "trained_this_run": bool(need_tokenizer_training),
            "subset_path": str(tokenizer_subset_path),
            "subset_source_split": "canonical_train" if lane_canonical else "rehearsal_train",
            "subset_docs": int(tokenizer_subset_docs),
            "subset_bytes": int(tokenizer_subset_bytes),
            "subset_byte_cap": int(args.tokenizer_train_byte_cap),
            "subset_usage_fraction": 1.0,
            "selection_seed": int(args.tokenizer_selection_seed),
            "sample_modulus": int(args.tokenizer_sample_modulus),
            "sample_acceptance": int(args.tokenizer_sample_acceptance),
        },
        "lanes": {
            "canonical": {
                "enabled": lane_canonical,
                "train_ratio": float(args.canonical_train_ratio),
                "val_ratio": round(1.0 - float(args.canonical_train_ratio), 8),
                "split_seed": int(args.canonical_split_seed),
                "train_docs": canonical_counters.train_docs,
                "val_docs": canonical_counters.val_docs,
                "total_docs": canonical_counters.total_docs,
                "train_bytes": canonical_counters.train_bytes,
                "val_bytes": canonical_counters.val_bytes,
                "total_bytes": canonical_counters.total_bytes,
                "artifacts": canonical_artifacts,
                "stamped_configs": stamped_canonical,
            },
            "rehearsal": {
                "enabled": lane_rehearsal,
                "train_ratio": float(args.rehearsal_train_ratio),
                "val_ratio": round(1.0 - float(args.rehearsal_train_ratio), 8),
                "split_seed": int(args.rehearsal_split_seed),
                "train_docs": rehearsal_counters.train_docs,
                "val_docs": rehearsal_counters.val_docs,
                "total_docs": rehearsal_counters.total_docs,
                "train_bytes": rehearsal_counters.train_bytes,
                "val_bytes": rehearsal_counters.val_bytes,
                "total_bytes": rehearsal_counters.total_bytes,
                "byte_cap": int(args.rehearsal_byte_cap),
                "artifacts": rehearsal_artifacts,
                "stamped_configs": stamped_rehearsal,
            },
        },
        "reused_existing_artifacts": False,
        "reuse_reason": None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "prepare_done "
        f"elapsed_secs={int(time.monotonic() - start_time)} "
        f"report_path={report_path}",
        flush=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"prepare_report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
