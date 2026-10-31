#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed adapter for WeMath Pro/Standard sampler outputs.

The model-facing JSONL produced here contains only the public problem fields,
the row's own ground truth for the deterministic verifier, and a stable row ID.
Family/group metadata is written to a separate analysis-only JSONL and is never
passed to Solver, Judge, Critic, or Author prompts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence


DatasetKind = Literal["pro", "standard"]
DIFFICULTY_ORDER = ("base", "x", "y", "z", "xy", "xz", "yz", "xyz")
MAIN_INPUT_KEYS = frozenset({"question", "image_path", "ground_truth"})
FORBIDDEN_MANIFEST_KEYS = frozenset(
    {
        "answer",
        "ground_truth",
        "groundtruth",
        "groudtruth",
        "teacher",
        "rejudge",
        "verifier",
        "verifier_result",
        "correctness",
        "reviewed_answer",
        "scarac",
        "sidecar",
    }
)


@dataclass(frozen=True)
class PreparationResult:
    dataset_kind: DatasetKind
    record_count: int
    model_input_path: Path
    metadata_path: Path
    report_path: Path
    metadata_by_question_id: Dict[str, Dict[str, Any]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {line_number} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}: line {line_number} must be a JSON object"
                )
            rows.append(payload)
    return rows


def _safe_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if text:
        return text[:80]
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def _canonical_image_ref(value: Any, base: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return os.path.normpath(str(path))


def _forbidden_keys(value: Any) -> List[str]:
    found: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_MANIFEST_KEYS:
                found.append(normalized)
            found.extend(_forbidden_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_forbidden_keys(child))
    return found


def _detect_kind(
    manifest_rows: Sequence[Mapping[str, Any]],
    requested: str,
) -> DatasetKind:
    if requested not in {"auto", "pro", "standard"}:
        raise ValueError(f"Unsupported dataset kind: {requested!r}")
    has_pro = all(
        "family_id" in row and "difficulty" in row
        for row in manifest_rows
    )
    has_standard = all("standard_idx" in row for row in manifest_rows)
    if has_pro == has_standard:
        raise ValueError(
            "Cannot uniquely identify manifest as Pro or Standard"
        )
    detected: DatasetKind = "pro" if has_pro else "standard"
    if requested != "auto" and requested != detected:
        raise ValueError(
            f"--dataset-kind={requested} conflicts with detected {detected}"
        )
    return detected


def _validate_main_row(row: Mapping[str, Any], line_number: int) -> None:
    actual = frozenset(row)
    if actual != MAIN_INPUT_KEYS:
        raise ValueError(
            f"main JSONL line {line_number} must contain exactly "
            f"{sorted(MAIN_INPUT_KEYS)}; got {sorted(actual)}"
        )
    if not isinstance(row["question"], str) or not row["question"].strip():
        raise ValueError(f"main JSONL line {line_number} has empty question")
    if not isinstance(row["image_path"], str) or not row["image_path"].strip():
        raise ValueError(f"main JSONL line {line_number} has empty image_path")
    if row["ground_truth"] in (None, ""):
        raise ValueError(f"main JSONL line {line_number} has empty ground_truth")


def _validate_manifest_common(
    manifest: Mapping[str, Any],
    *,
    index: int,
    main_row: Mapping[str, Any],
    input_base: Path,
) -> None:
    if manifest.get("sample_index") != index:
        raise ValueError(
            f"manifest line {index + 1}: expected sample_index={index}, "
            f"got {manifest.get('sample_index')!r}"
        )
    forbidden = sorted(set(_forbidden_keys(manifest)))
    if forbidden:
        raise ValueError(
            f"manifest line {index + 1} contains forbidden oracle/audit "
            f"keys: {forbidden}"
        )
    expected_image = _canonical_image_ref(
        main_row["image_path"],
        input_base,
    )
    manifest_image = _canonical_image_ref(
        manifest.get("exported_image_path"),
        input_base,
    )
    if not manifest_image or manifest_image != expected_image:
        raise ValueError(
            f"manifest line {index + 1} image does not match main JSONL: "
            f"{manifest.get('exported_image_path')!r} vs "
            f"{main_row['image_path']!r}"
        )


def _pro_metadata(
    manifest: Mapping[str, Any],
    *,
    index: int,
) -> tuple[str, Dict[str, Any]]:
    family_id = str(manifest.get("family_id", "") or "").strip()
    difficulty = str(manifest.get("difficulty", "") or "").strip()
    source_row_index = manifest.get("source_row_index")
    if not family_id:
        raise ValueError(f"manifest line {index + 1}: empty family_id")
    if difficulty not in DIFFICULTY_ORDER:
        raise ValueError(
            f"manifest line {index + 1}: invalid difficulty={difficulty!r}"
        )
    if not isinstance(source_row_index, int) or source_row_index < 0:
        raise ValueError(
            f"manifest line {index + 1}: invalid source_row_index"
        )
    expected_flags = {
        "step_x": "x" in difficulty,
        "visual_y": "y" in difficulty,
        "context_z": "z" in difficulty,
    }
    if manifest.get("axis_flags") != expected_flags:
        raise ValueError(
            f"manifest line {index + 1}: axis_flags do not match "
            f"difficulty={difficulty!r}"
        )
    question_id = (
        f"wemath_pro__family_{_safe_component(family_id)}"
        f"__{difficulty}__row_{source_row_index}"
    )
    metadata = {
        "dataset_kind": "pro",
        "sample_index": index,
        "family_order": manifest.get("family_order"),
        "family_id": family_id,
        "difficulty": difficulty,
        "axis_flags": expected_flags,
        "source_row_index": source_row_index,
        "source_idx": manifest.get("source_idx"),
        "image_sha256": manifest.get("image_sha256"),
        "knowledge_points": manifest.get("knowledge_points") or [],
    }
    return question_id, metadata


def _standard_metadata(
    manifest: Mapping[str, Any],
    *,
    index: int,
) -> tuple[str, Dict[str, Any]]:
    standard_idx = str(manifest.get("standard_idx", "") or "").strip()
    source_row_index = manifest.get("source_row_index")
    source_id = str(manifest.get("source_id", "") or "").strip()
    if not standard_idx:
        raise ValueError(f"manifest line {index + 1}: empty standard_idx")
    if not isinstance(source_row_index, int) or source_row_index < 0:
        raise ValueError(
            f"manifest line {index + 1}: invalid source_row_index"
        )
    question_id = (
        f"wemath_standard__idx_{_safe_component(standard_idx)}"
        f"__id_{_safe_component(source_id or source_row_index)}"
        f"__row_{source_row_index}"
    )
    metadata = {
        "dataset_kind": "standard",
        "sample_index": index,
        "standard_idx": standard_idx,
        "source_id": source_id or None,
        "source_row_index": source_row_index,
        "image_sha256": manifest.get("image_sha256"),
        "question_sha256": manifest.get("question_sha256"),
        "knowledge_level1": manifest.get("knowledge_level1"),
        "knowledge_level2": manifest.get("knowledge_level2"),
        "knowledge_level3": manifest.get("knowledge_level3"),
        "knowledge_level4": manifest.get("knowledge_level4"),
        "knowledge": manifest.get("knowledge"),
        "principle": manifest.get("principle"),
    }
    return question_id, metadata


def prepare_wemath_input(
    *,
    input_jsonl: Path,
    sample_manifest: Path,
    run_dir: Path,
    dataset_kind: str = "auto",
) -> PreparationResult:
    """Validate aligned sampler outputs and create the model-facing input."""

    if not input_jsonl.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_jsonl}")
    if not sample_manifest.is_file():
        raise FileNotFoundError(
            f"SAMPLE_MANIFEST.jsonl does not exist: {sample_manifest}"
        )
    main_rows = _read_jsonl(input_jsonl)
    manifest_rows = _read_jsonl(sample_manifest)
    if not main_rows:
        raise ValueError("Input JSONL is empty")
    if len(main_rows) != len(manifest_rows):
        raise ValueError(
            f"Main JSONL has {len(main_rows)} rows but manifest has "
            f"{len(manifest_rows)} rows"
        )
    detected = _detect_kind(manifest_rows, dataset_kind)

    prepared_rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    metadata_by_id: Dict[str, Dict[str, Any]] = {}
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    group_counts: Dict[str, int] = {}
    pro_difficulties: Dict[str, set[str]] = {}
    input_base = input_jsonl.parent

    for index, (main_row, manifest) in enumerate(
        zip(main_rows, manifest_rows)
    ):
        _validate_main_row(main_row, index + 1)
        _validate_manifest_common(
            manifest,
            index=index,
            main_row=main_row,
            input_base=input_base,
        )
        if detected == "pro":
            question_id, metadata = _pro_metadata(manifest, index=index)
            group_key = str(metadata["family_id"])
        else:
            question_id, metadata = _standard_metadata(
                manifest,
                index=index,
            )
            group_key = str(metadata["standard_idx"])
        if question_id in seen_ids:
            raise ValueError(f"Duplicate derived question_id: {question_id}")
        seen_ids.add(question_id)
        seen_groups.add(group_key)
        group_counts[group_key] = group_counts.get(group_key, 0) + 1
        if detected == "pro":
            pro_difficulties.setdefault(group_key, set()).add(
                str(metadata["difficulty"])
            )
        prepared_rows.append(
            {
                "question_id": question_id,
                "question": main_row["question"],
                "image_path": main_row["image_path"],
                "ground_truth": main_row["ground_truth"],
            }
        )
        metadata_record = {"question_id": question_id, **metadata}
        metadata_rows.append(metadata_record)
        metadata_by_id[question_id] = metadata

    if detected == "pro":
        expected = set(DIFFICULTY_ORDER)
        invalid_families = {
            family: {
                "row_count": group_counts[family],
                "difficulties": sorted(
                    pro_difficulties[family],
                    key=DIFFICULTY_ORDER.index,
                ),
            }
            for family in sorted(group_counts)
            if group_counts[family] != len(DIFFICULTY_ORDER)
            or pro_difficulties[family] != expected
        }
        if invalid_families:
            raise ValueError(
                "Pro experiment requires complete eight-vertex families; "
                f"invalid families: {invalid_families}"
            )
    elif any(count != 1 for count in group_counts.values()):
        duplicates = {
            group: count
            for group, count in group_counts.items()
            if count != 1
        }
        raise ValueError(
            "Standard experiment requires at most one row per idx; "
            f"duplicates: {duplicates}"
        )

    adapter_dir = run_dir / "wemath_input"
    model_input_path = adapter_dir / "model_input.jsonl"
    metadata_path = adapter_dir / "dataset_metadata.jsonl"
    report_path = adapter_dir / "preparation_report.json"
    _atomic_write_text(
        model_input_path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in prepared_rows
        ),
    )
    _atomic_write_text(
        metadata_path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in metadata_rows
        ),
    )
    report = {
        "status": "complete",
        "dataset_kind": detected,
        "input_jsonl": str(input_jsonl.resolve()),
        "input_sha256": sha256_file(input_jsonl),
        "sample_manifest": str(sample_manifest.resolve()),
        "sample_manifest_sha256": sha256_file(sample_manifest),
        "record_count": len(prepared_rows),
        "unique_question_ids": len(seen_ids),
        "unique_analysis_groups": len(seen_groups),
        "model_input_path": str(model_input_path.resolve()),
        "model_input_sha256": sha256_file(model_input_path),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "model_input_fields": [
            "question_id",
            "question",
            "image_path",
            "ground_truth",
        ],
        "metadata_passed_to_model_prompts": False,
        "audit_or_verifier_sidecar_used": False,
        "invariants": {
            "main_and_manifest_row_counts_match": True,
            "stable_question_ids_unique": True,
            "manifest_contains_no_oracle_or_audit_keys": True,
            "manifest_images_align_with_main_jsonl": True,
            "each_row_keeps_its_own_ground_truth": True,
        },
    }
    if detected == "pro":
        report["pro_family_counts"] = group_counts
        report["all_pro_families_are_complete_eight_vertices"] = all(
            count == 8 for count in group_counts.values()
        )
    else:
        report["standard_idx_unique"] = len(seen_groups) == len(prepared_rows)
    _atomic_write_json(report_path, report)
    return PreparationResult(
        dataset_kind=detected,
        record_count=len(prepared_rows),
        model_input_path=model_input_path,
        metadata_path=metadata_path,
        report_path=report_path,
        metadata_by_question_id=metadata_by_id,
    )


def attach_dataset_metadata(
    output_jsonl: Path,
    metadata_by_question_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Attach analysis metadata to canonical results without changing prompts."""

    if not output_jsonl.exists():
        return
    rows = _read_jsonl(output_jsonl)
    updated: List[Dict[str, Any]] = []
    for row in rows:
        question_id = str(row.get("question_id", "") or "")
        metadata = metadata_by_question_id.get(question_id)
        if metadata is not None:
            row["dataset_metadata"] = dict(metadata)
        updated.append(row)
    _atomic_write_text(
        output_jsonl,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in updated
        ),
    )
