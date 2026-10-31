#!/usr/bin/env python3
"""v7 WeMath data-production experiment: forced C-F, blind crossover.

The verifier organizes parent pairs outside the model prompt.  Main DPO data
uses blind crossover.  A deterministic ~10% additional oracle-hinted arm is
saved as a control and excluded from trainable DPO exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import sys
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import v7_io as io
from v7_protocol import (
    OPERATORS,
    SCRIPT_VERSION,
    CrossoverTask,
    ProbeCandidate,
    build_crossover_tasks,
    content_hash,
    crossover_messages,
    evaluate_dpo_pair,
    guided_process_audit_messages,
    length_ratio,
    normalized_text,
    parse_process_audit,
    parse_trajectory_judge,
    plan_probe_pool,
    process_audit_messages,
    solver_messages,
    stable_sampling_seed,
    text_similarity,
    trajectory_judge_messages,
)
from verifier_v4 import (
    VERIFIER_VERSION,
    VerifierPolicy,
    default_policy,
    infer_answer_type,
    load_sidecar,
    sidecar_entry_to_dict,
    verify_answer,
)
from wemath_adapter import prepare_wemath_input


DEFAULT_MODEL_PATH = "/data/Qwen/Qwen3-VL-8B-Instruct"
STAGE_ORDER = {
    "initialized": 0,
    "round0_generated": 1,
    "round0_completed": 2,
    "offspring_generated": 3,
    "offspring_completed": 4,
    "audits_completed": 5,
    "completed": 6,
}


def setup_logger(path: Path, extra: Optional[Path]) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"v7_{path}")
    for existing in logger.handlers:
        existing.close()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(path, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if extra is not None and extra.resolve() != path.resolve():
        extra.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(extra, mode="a", encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate lineage-aligned DPO candidates with forced C-F blind crossover"
    )
    parser.add_argument("--dataset-kind", choices=("auto", "pro", "standard"), default="auto")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--verifier-sidecar",
        type=Path,
        default=HERE / "configs" / "verifier_review_overrides_v5_256.jsonl",
    )
    parser.add_argument("--backend", choices=("vllm", "mock"), default="vllm")
    parser.add_argument("--mock-script", type=Path, default=None)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--round0-count", type=int, default=16)
    parser.add_argument("--children-per-operator", type=int, default=2)
    parser.add_argument("--hinted-control-fraction", type=float, default=0.10)
    parser.add_argument("--solver-max-tokens", type=int, default=4096)
    parser.add_argument("--crossover-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--process-judge-max-tokens", type=int, default=2048)
    parser.add_argument("--generation-micro-batch-size", type=int, default=4)
    parser.add_argument("--judge-micro-batch-size", type=int, default=8)
    parser.add_argument("--duplicate-retries", type=int, default=4)
    parser.add_argument("--judge-parse-retries", type=int, default=2)
    parser.add_argument("--solver-temperature", type=float, default=0.8)
    parser.add_argument("--crossover-temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-length-ratio", type=float, default=0.70)
    parser.add_argument("--max-length-ratio", type=float, default=1.30)
    parser.add_argument("--max-correct-parent-similarity", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.round0_count != 16:
        parser.error("v7 main protocol fixes --round0-count at 16")
    if args.children_per_operator < 1:
        parser.error("--children-per-operator must be positive")
    if not 0.0 <= args.hinted_control_fraction <= 1.0:
        parser.error("--hinted-control-fraction must be in [0,1]")
    if args.generation_micro_batch_size < 1 or args.judge_micro_batch_size < 1:
        parser.error("micro-batch sizes must be positive")
    if not 0.0 < args.min_length_ratio <= args.max_length_ratio:
        parser.error("invalid length-ratio interval")
    if args.backend == "mock" and args.mock_script is None and not args.dry_run:
        parser.error("--backend mock requires --mock-script")
    return args


def state_path(run_dir: Path, question_id: str) -> Path:
    return run_dir / "question_states" / f"{io.safe_filename(question_id)}.json"


def save_state(run_dir: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = io.utc_now()
    io.atomic_json(state_path(run_dir, str(state["question_id"])), state)


def load_states(run_dir: Path) -> List[Dict[str, Any]]:
    paths = sorted((run_dir / "question_states").glob("*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def rebuild_exports(run_dir: Path, output_jsonl: Path) -> Dict[str, Any]:
    states = sorted(load_states(run_dir), key=lambda item: str(item.get("question_id", "")))
    io.atomic_jsonl(output_jsonl, states)
    exports = run_dir / "exports"
    trajectory_rows: List[Dict[str, Any]] = []
    judge_rationale_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    metrics: Counter[str] = Counter()

    for state in states:
        question_id = str(state.get("question_id", ""))
        if state.get("status") not in {"completed", "skipped"}:
            failed_rows.append(
                {
                    "question_id": question_id,
                    "status": state.get("status"),
                    "stage": state.get("stage"),
                    "error": state.get("error", ""),
                }
            )
        plan = dict(state.get("pool_plan") or {})
        mode = str(plan.get("mode", "not_planned"))
        mode_counts[mode] += 1
        trajectories = [
            *list(state.get("round0_trajectories") or []),
            *list(state.get("offspring") or []),
        ]
        for item in trajectories:
            trajectory_rows.append({"question_id": question_id, **item})
            for field in ("blind_judge", "verifier_guided_judge"):
                judge = dict(item.get(field) or {})
                if not judge:
                    continue
                judge_rationale_rows.append(
                    {
                        "question_id": question_id,
                        "question": state.get("question"),
                        "image_path": state.get("image_path"),
                        "trajectory_id": item.get("trajectory_id"),
                        "generation": item.get("generation"),
                        "candidate_response": item.get("raw_response", ""),
                        "candidate_final_answer": item.get("final_answer", ""),
                        "offline_verifier_correct": (
                            item.get("verifier") or {}
                        ).get("compatible_is_correct"),
                        "judge_mode": judge.get("mode"),
                        "judge_rationale": judge.get("judge_rationale", ""),
                        "result_verdict": judge.get("result_verdict"),
                        "process_verdict": judge.get("process_verdict"),
                        "score": judge.get("score"),
                        "confidence": judge.get("confidence"),
                        "parse_ok": judge.get("parse_ok", False),
                        "raw_response": judge.get("raw_response", ""),
                        "analysis_only_not_used_by_evolution": True,
                        "future_judge_dpo_status": "candidate_only_no_pair_constructed",
                    }
                )
                metrics[f"{field}_total"] += 1
                metrics[f"{field}_parse_ok"] += int(judge.get("parse_ok") is True)
        for item in state.get("pair_evaluations") or []:
            row = {
                "question_id": question_id,
                "question": state.get("question"),
                "image_path": state.get("image_path"),
                **item,
            }
            if item.get("eligible"):
                pair_rows.append(row)
                if item.get("pair_type") == "error_repair":
                    metrics["cf_to_c_dpo_candidate_pairs"] += 1
                elif item.get("pair_type") == "failure_failure_rescue":
                    metrics["ff_to_c_dpo_candidate_pairs"] += 1
            if item.get("arm") == "oracle_hinted":
                control_rows.append(row)
            guided_audit = item.get("verifier_guided_process_audit") or {}
            metrics["verifier_guided_process_audit_total"] += int(bool(guided_audit))
            metrics["verifier_guided_process_audit_parse_ok"] += int(
                guided_audit.get("parse_ok") is True
            )
        for item in state.get("offspring") or []:
            task = item.get("task", {})
            key = f"offspring_{str(task.get('pair_type', '')).lower()}_{task.get('arm', '')}"
            metrics[key] += 1
            if (item.get("verifier") or {}).get("compatible_is_correct") is True:
                metrics[key + "_correct"] += 1
                if task.get("pair_type") == "CF" and task.get("arm") == "blind":
                    metrics["cf_to_c_correct_children"] += 1
                elif task.get("pair_type") == "CF" and task.get("arm") == "oracle_hinted":
                    metrics["oracle_hinted_cf_to_c_correct_children"] += 1
                elif task.get("pair_type") == "FF" and task.get("arm") == "blind":
                    metrics["ff_to_c_correct_children"] += 1
        metrics["round0_trajectories"] += len(state.get("round0_trajectories") or [])
        metrics["dpo_pairs"] += sum(1 for item in state.get("pair_evaluations") or [] if item.get("eligible"))
        metrics["completed_questions"] += int(state.get("status") == "completed")
        metrics["failed_or_interrupted_questions"] += int(state.get("status") not in {"completed", "skipped"})
        summary_rows.append(
            {
                "question_id": question_id,
                "status": state.get("status"),
                "stage": state.get("stage"),
                "pool_mode": mode,
                "pool_counts": plan.get("counts", {}),
                "crossover_task_count": len(state.get("crossover_tasks") or []),
                "offspring_count": len(state.get("offspring") or []),
                "dpo_pair_count": sum(
                    1 for item in state.get("pair_evaluations") or [] if item.get("eligible")
                ),
            }
        )

    io.atomic_jsonl(exports / "all_trajectories.jsonl", trajectory_rows)
    io.atomic_jsonl(exports / "judge_rationales.jsonl", judge_rationale_rows)
    io.atomic_jsonl(exports / "dpo_pairs_full.jsonl", pair_rows)
    io.atomic_jsonl(exports / "oracle_hinted_control.jsonl", control_rows)
    io.atomic_jsonl(exports / "failed_records.jsonl", failed_rows)
    io.atomic_jsonl(exports / "question_summary.jsonl", summary_rows)
    report = {
        "schema_version": 1,
        "generated_at": io.utc_now(),
        "question_count": len(states),
        "pool_mode_counts": dict(sorted(mode_counts.items())),
        "metrics": dict(sorted(metrics.items())),
        "notes": {
            "dpo_pairs_full": "blind arm only; candidate data, no training performed",
            "oracle_hinted_control": "control only; excluded from dpo_pairs_full",
        },
    }
    io.atomic_json(exports / "metrics.json", report)
    return report


def code_hashes() -> Dict[str, str]:
    names = (
        "run_self_judge_fusion_v7_wemath_dpo.py",
        "v7_protocol.py",
        "v7_io.py",
        "verifier_v4.py",
        "wemath_adapter.py",
    )
    return {name: io.sha256_file(HERE / name) for name in names}


def make_manifest(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "status": "dry_run" if args.dry_run else "running",
        "started_at": io.utc_now(),
        "input_jsonl": str(args.input_jsonl.resolve()),
        "input_sha256": io.sha256_file(args.input_jsonl),
        "sample_manifest": str(args.sample_manifest.resolve()),
        "sample_manifest_sha256": io.sha256_file(args.sample_manifest),
        "verifier_sidecar": str(args.verifier_sidecar.resolve()) if args.verifier_sidecar else None,
        "verifier_sidecar_sha256": io.sha256_file(args.verifier_sidecar) if args.verifier_sidecar else None,
        "output_jsonl": str(args.output_jsonl.resolve()),
        "run_dir": str(run_dir.resolve()),
        "code_sha256": code_hashes(),
        "runtime_config": {
            "dataset_kind": args.dataset_kind,
            "backend": args.backend,
            "mock_script_sha256": (
                io.sha256_file(args.mock_script) if args.mock_script is not None else None
            ),
            "model_path": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "solver_max_tokens": args.solver_max_tokens,
            "crossover_max_tokens": args.crossover_max_tokens,
            "process_judge_max_tokens": args.process_judge_max_tokens,
            "judge_max_tokens": args.judge_max_tokens,
            "generation_micro_batch_size": args.generation_micro_batch_size,
            "judge_micro_batch_size": args.judge_micro_batch_size,
            "duplicate_retries": args.duplicate_retries,
            "judge_parse_retries": args.judge_parse_retries,
            "solver_temperature": args.solver_temperature,
            "crossover_temperature": args.crossover_temperature,
            "top_p": args.top_p,
            "min_length_ratio": args.min_length_ratio,
            "max_length_ratio": args.max_length_ratio,
            "max_correct_parent_similarity": args.max_correct_parent_similarity,
            "seed": args.seed,
            "start_line": args.start_line,
            "limit": args.limit,
        },
        "protocol": {
            "round0_count": 16,
            "selection": "2C+2F when available; 4 forced C-F pairs; one F-F rescue pair",
            "all_f": "select 4F and enumerate 6 F-F pairs",
            "crossover": "blind main arm; C/F identities external only",
            "oracle_hinted_control_fraction": args.hinted_control_fraction,
            "oracle_hinted_is_additional": True,
            "oracle_hinted_excluded_from_dpo": True,
            "operators": list(OPERATORS),
            "children_per_operator": args.children_per_operator,
            "process_judge_passes": 2,
            "trajectory_judges": ["blind", "verifier_guided"],
            "judge_output_protocol": "<think>auditable rationale</think><answer>strict JSON</answer>",
            "guided_judge_used_for_dpo_filter": False,
            "trajectory_judge_outputs_analysis_only": True,
            "judge_outputs_used_for_parent_selection": False,
            "judge_outputs_passed_to_crossover": False,
            "judge_outputs_passed_to_future_generation": False,
        },
        "server_validated": False,
        "dpo_training_performed": False,
        "scientific_improvement_claimed": False,
    }


def prepare_run_dir(run_dir: Path, *, resume: bool) -> None:
    existing = {path.name for path in run_dir.iterdir()} if run_dir.is_dir() else set()
    harmless = {"stdout.log", "stderr.log"}
    if existing - harmless and not resume:
        raise ValueError(f"run directory is not empty; use a new versioned path: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "question_states").mkdir(exist_ok=True)
    (run_dir / "exports").mkdir(exist_ok=True)
    (run_dir / "raw_calls").mkdir(exist_ok=True)


def batching(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def policy_for_question(
    *,
    question_id: str,
    ground_truth: Any,
    image_path: str,
    sidecar: Mapping[str, Any],
) -> Tuple[str, VerifierPolicy, Dict[str, Any]]:
    entry = sidecar.get(question_id)
    if entry is not None:
        if entry.image_basename and Path(image_path).name != entry.image_basename:
            raise ValueError(
                f"sidecar image mismatch for {question_id}: {entry.image_basename} vs {Path(image_path).name}"
            )
        answer_type = entry.policy.answer_type
        return answer_type, entry.policy, sidecar_entry_to_dict(entry)
    answer_type = infer_answer_type(ground_truth)
    policy = default_policy(answer_type=answer_type)
    return answer_type, policy, {
        "question_id": question_id,
        "policy": asdict(policy),
        "source": "default_policy",
    }


def attach_verifier(
    *,
    final_answer: str,
    ground_truth: Any,
    answer_type: str,
    policy: VerifierPolicy,
    question: str,
) -> Dict[str, Any]:
    return asdict(
        verify_answer(
            final_answer,
            ground_truth,
            answer_type,
            policy=policy,
            question=question,
        )
    )


def generate_round0(
    *,
    backend: Any,
    image: Any,
    question_id: str,
    question: str,
    image_path: str,
    ground_truth: Any,
    answer_type: str,
    policy: VerifierPolicy,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    accepted: Dict[int, Dict[str, Any]] = {}
    hashes: set[str] = set()
    pending = list(range(args.round0_count))
    for retry in range(args.duplicate_retries + 1):
        if not pending:
            break
        retry_pending: List[int] = []
        for slots in batching(pending, args.generation_micro_batch_size):
            slot_seeds = [
                stable_sampling_seed(
                    args.seed,
                    question_id,
                    "solver_round0",
                    slot,
                    retry,
                )
                for slot in slots
            ]
            messages = [
                solver_messages(
                    question=question,
                    answer_type=answer_type,
                    slot=slot,
                    retry=retry,
                    vision_placeholder=io.VISION_PLACEHOLDER,
                )
                for slot in slots
            ]
            prompts = [backend.render(system, user) for system, user in messages]
            completions = backend.generate(
                prompts=prompts,
                image=image,
                image_path=image_path,
                role="solver_round0",
                question_id=question_id,
                sampling={
                    "n": 1,
                    "temperature": args.solver_temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.solver_max_tokens,
                },
                contexts=[
                    {"slot": slot, "retry": retry, "sampling_seed": sampling_seed}
                    for slot, sampling_seed in zip(slots, slot_seeds)
                ],
                request_seeds=slot_seeds,
            )
            for slot, sampling_seed, completion in zip(slots, slot_seeds, completions):
                reasoning, final_answer = io.parse_solver(completion.text)
                digest = content_hash(completion.text)
                if not completion.text or digest in hashes:
                    retry_pending.append(slot)
                    continue
                hashes.add(digest)
                verifier = attach_verifier(
                    final_answer=final_answer,
                    ground_truth=ground_truth,
                    answer_type=answer_type,
                    policy=policy,
                    question=question,
                )
                accepted[slot] = {
                    "trajectory_id": f"g0_s{slot:02d}",
                    "generation": 0,
                    "slot": slot,
                    "parent_ids": [],
                    "role": "solver",
                    "raw_response": completion.text,
                    "reasoning": reasoning,
                    "final_answer": final_answer,
                    "finish_reason": completion.finish_reason,
                    "content_hash": digest,
                    "sampling_seed": sampling_seed,
                    "verifier": verifier,
                    "generation_retry": retry,
                }
        pending = retry_pending
    if len(accepted) != args.round0_count:
        raise RuntimeError(
            f"Round 0 produced {len(accepted)}/{args.round0_count} unique trajectories; "
            f"unfilled slots={pending}"
        )
    return [accepted[index] for index in sorted(accepted)]


def candidates_from_round0(rows: Sequence[Mapping[str, Any]]) -> List[ProbeCandidate]:
    result: List[ProbeCandidate] = []
    for row in rows:
        verifier = dict(row.get("verifier") or {})
        result.append(
            ProbeCandidate(
                trajectory_id=str(row["trajectory_id"]),
                raw_response=str(row.get("raw_response", "")),
                reasoning=str(row.get("reasoning", "")),
                final_answer=str(row.get("final_answer", "")),
                verifier_correct=verifier.get("compatible_is_correct"),
                verifier_status=str(verifier.get("status", "")),
                finish_reason=str(row.get("finish_reason", "")),
            )
        )
    return result


def public_verifier_evidence(verifier: Mapping[str, Any]) -> Dict[str, Any]:
    """Return result-layer evidence without the reference or normalized reference."""

    compatible = verifier.get("compatible_is_correct")
    outcome = "correct" if compatible is True else "incorrect" if compatible is False else "unresolved"
    return {
        "outcome": outcome,
        "status": verifier.get("status"),
        "answer_type": verifier.get("answer_type"),
        "decision_source": verifier.get("decision_source", "deterministic_rule"),
        "reference_answer_hidden": True,
    }


def run_trajectory_judges(
    *,
    backend: Any,
    image: Any,
    question_id: str,
    question: str,
    image_path: str,
    trajectories: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    for mode, output_field, role in (
        ("blind", "blind_judge", "judge_blind"),
        ("verifier_guided", "verifier_guided_judge", "judge_verifier_guided"),
    ):
        requests: List[Dict[str, Any]] = []
        for trajectory in trajectories:
            evidence = (
                public_verifier_evidence(trajectory.get("verifier") or {})
                if mode == "verifier_guided"
                else None
            )
            system, user = trajectory_judge_messages(
                question=question,
                trajectory=str(trajectory.get("raw_response", "")),
                mode=mode,
                verifier_evidence=evidence,
                vision_placeholder=io.VISION_PLACEHOLDER,
            )
            requests.append(
                {
                    "trajectory": trajectory,
                    "trajectory_id": trajectory["trajectory_id"],
                    "prompt": backend.render(system, user),
                    "verifier_evidence": evidence,
                }
            )
        pending = requests
        for retry in range(args.judge_parse_retries + 1):
            if not pending:
                break
            retry_requests: List[Dict[str, Any]] = []
            for batch in batching(pending, args.judge_micro_batch_size):
                completions = backend.generate(
                    prompts=[str(item["prompt"]) for item in batch],
                    image=image,
                    image_path=image_path,
                    role=role,
                    question_id=question_id,
                    sampling={
                        "n": 1,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": args.judge_max_tokens,
                        "seed": args.seed,
                    },
                    contexts=[
                        {
                            "trajectory_id": item["trajectory_id"],
                            "judge_mode": mode,
                            "parse_retry": retry,
                            "verifier_evidence": item["verifier_evidence"],
                        }
                        for item in batch
                    ],
                )
                for request, completion in zip(batch, completions):
                    parsed = parse_trajectory_judge(completion.text, mode=mode)
                    parsed["parse_retry"] = retry
                    request["last_parse"] = parsed
                    if parsed.get("parse_ok"):
                        request["trajectory"][output_field] = parsed
                    else:
                        retry_requests.append(request)
            pending = retry_requests
        for request in pending:
            request["trajectory"][output_field] = {
                **dict(request.get("last_parse") or {}),
                "mode": mode,
                "parse_ok": False,
                "parse_retries_exhausted": True,
                "verifier_evidence": request["verifier_evidence"],
            }


def generate_offspring(
    *,
    backend: Any,
    image: Any,
    question_id: str,
    question: str,
    image_path: str,
    ground_truth: Any,
    answer_type: str,
    policy: VerifierPolicy,
    round0: Sequence[Mapping[str, Any]],
    tasks: Sequence[CrossoverTask],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    parent_by_id = {str(item["trajectory_id"]): dict(item) for item in round0}
    parent_text = {key: str(value.get("raw_response", "")) for key, value in parent_by_id.items()}
    offspring: List[Dict[str, Any]] = []
    for replicate in sorted({task.replicate for task in tasks}):
        replicate_tasks = [task for task in tasks if task.replicate == replicate]
        for task_batch in batching(replicate_tasks, args.generation_micro_batch_size):
            task_seeds = [
                stable_sampling_seed(
                    args.seed,
                    question_id,
                    "crossover",
                    task.task_id,
                )
                for task in task_batch
            ]
            messages = [
                crossover_messages(
                    question=question,
                    answer_type=answer_type,
                    task=task,
                    parent_text_by_id=parent_text,
                    vision_placeholder=io.VISION_PLACEHOLDER,
                )
                for task in task_batch
            ]
            prompts = [backend.render(system, user) for system, user in messages]
            completions = backend.generate(
                prompts=prompts,
                image=image,
                image_path=image_path,
                role="crossover_oracle_hinted" if all(task.arm == "oracle_hinted" for task in task_batch) else "crossover",
                question_id=question_id,
                sampling={
                    "n": 1,
                    "temperature": args.crossover_temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.crossover_max_tokens,
                },
                contexts=[
                    {**asdict(task), "sampling_seed": sampling_seed}
                    for task, sampling_seed in zip(task_batch, task_seeds)
                ],
                request_seeds=task_seeds,
            )
            for task, sampling_seed, completion in zip(task_batch, task_seeds, completions):
                reasoning, final_answer = io.parse_solver(completion.text)
                verifier = attach_verifier(
                    final_answer=final_answer,
                    ground_truth=ground_truth,
                    answer_type=answer_type,
                    policy=policy,
                    question=question,
                )
                offspring.append(
                    {
                        "trajectory_id": f"g1_o{len(offspring):03d}",
                        "generation": 1,
                        "parent_ids": list(task.parent_ids),
                        "role": "crossover",
                        "task": asdict(task),
                        "raw_response": completion.text,
                        "reasoning": reasoning,
                        "final_answer": final_answer,
                        "finish_reason": completion.finish_reason,
                        "content_hash": content_hash(completion.text),
                        "sampling_seed": sampling_seed,
                        "verifier": verifier,
                    }
                )
    return offspring


def cheap_comparison_pass(
    *,
    child: Mapping[str, Any],
    wrong_parent: Mapping[str, Any],
    correct_parent: Optional[Mapping[str, Any]],
    policy: VerifierPolicy,
    args: argparse.Namespace,
) -> bool:
    if policy.gt_status.lower() != "trusted":
        return False
    if (child.get("verifier") or {}).get("compatible_is_correct") is not True:
        return False
    if (wrong_parent.get("verifier") or {}).get("compatible_is_correct") is not False:
        return False
    ratio = length_ratio(child.get("raw_response", ""), wrong_parent.get("raw_response", ""))
    if not args.min_length_ratio <= ratio <= args.max_length_ratio:
        return False
    if normalized_text(child.get("raw_response")) == normalized_text(wrong_parent.get("raw_response")):
        return False
    if correct_parent is not None and text_similarity(
        child.get("raw_response", ""), correct_parent.get("raw_response", "")
    ) > args.max_correct_parent_similarity:
        return False
    return True


def run_process_audits(
    *,
    backend: Any,
    image: Any,
    question_id: str,
    question: str,
    image_path: str,
    comparisons: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    blind_requests: List[Dict[str, Any]] = []
    for comparison in comparisons:
        for pass_index in (0, 1):
            system, user = process_audit_messages(
                question=question,
                wrong_parent=str(comparison["wrong_parent_text"]),
                child=str(comparison["child_text"]),
                pass_index=pass_index,
                vision_placeholder=io.VISION_PLACEHOLDER,
            )
            blind_requests.append(
                {
                    "comparison_id": comparison["comparison_id"],
                    "pass_index": pass_index,
                    "prompt": backend.render(system, user),
                }
            )
    blind_results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    pending = blind_requests
    for retry in range(args.judge_parse_retries + 1):
        if not pending:
            break
        retry_requests: List[Dict[str, Any]] = []
        for batch in batching(pending, args.judge_micro_batch_size):
            completions = backend.generate(
                prompts=[str(item["prompt"]) for item in batch],
                image=image,
                image_path=image_path,
                role="process_judge",
                question_id=question_id,
                sampling={
                    "n": 1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": args.process_judge_max_tokens,
                    "seed": args.seed,
                },
                contexts=[
                    {
                        "comparison_id": item["comparison_id"],
                        "pass_index": item["pass_index"],
                        "parse_retry": retry,
                    }
                    for item in batch
                ],
            )
            for request, completion in zip(batch, completions):
                parsed = parse_process_audit(completion.text)
                parsed["mode"] = "blind"
                parsed["pass_index"] = request["pass_index"]
                parsed["parse_retry"] = retry
                if parsed.get("parse_ok"):
                    blind_results.setdefault(request["comparison_id"], {})[request["pass_index"]] = parsed
                else:
                    request["last_parse"] = parsed
                    retry_requests.append(request)
        pending = retry_requests
    for request in pending:
        blind_results.setdefault(request["comparison_id"], {})[request["pass_index"]] = {
            **dict(request.get("last_parse") or {}),
            "parse_ok": False,
            "pass_index": request["pass_index"],
            "parse_retries_exhausted": True,
        }
    guided_requests: List[Dict[str, Any]] = []
    for comparison in comparisons:
        evidence = {
            "source_parent": public_verifier_evidence(
                comparison["wrong_parent"].get("verifier") or {}
            ),
            "offspring": public_verifier_evidence(
                comparison["child"].get("verifier") or {}
            ),
        }
        system, user = guided_process_audit_messages(
            question=question,
            wrong_parent=str(comparison["wrong_parent_text"]),
            child=str(comparison["child_text"]),
            verifier_evidence=evidence,
            vision_placeholder=io.VISION_PLACEHOLDER,
        )
        guided_requests.append(
            {
                "comparison_id": comparison["comparison_id"],
                "prompt": backend.render(system, user),
                "verifier_evidence": evidence,
            }
        )
    guided_results: Dict[str, Dict[str, Any]] = {}
    pending_guided = guided_requests
    for retry in range(args.judge_parse_retries + 1):
        if not pending_guided:
            break
        retry_guided: List[Dict[str, Any]] = []
        for batch in batching(pending_guided, args.judge_micro_batch_size):
            completions = backend.generate(
                prompts=[str(item["prompt"]) for item in batch],
                image=image,
                image_path=image_path,
                role="process_judge_guided",
                question_id=question_id,
                sampling={
                    "n": 1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": args.process_judge_max_tokens,
                    "seed": args.seed,
                },
                contexts=[
                    {
                        "comparison_id": item["comparison_id"],
                        "judge_mode": "verifier_guided",
                        "parse_retry": retry,
                        "verifier_evidence": item["verifier_evidence"],
                    }
                    for item in batch
                ],
            )
            for request, completion in zip(batch, completions):
                parsed = parse_process_audit(completion.text)
                parsed["mode"] = "verifier_guided"
                parsed["parse_retry"] = retry
                if parsed.get("parse_ok"):
                    guided_results[request["comparison_id"]] = parsed
                else:
                    request["last_parse"] = parsed
                    retry_guided.append(request)
        pending_guided = retry_guided
    for request in pending_guided:
        guided_results[request["comparison_id"]] = {
            **dict(request.get("last_parse") or {}),
            "mode": "verifier_guided",
            "parse_ok": False,
            "parse_retries_exhausted": True,
            "verifier_evidence": request["verifier_evidence"],
        }
    return {
        comparison["comparison_id"]: {
            "blind": [
                blind_results.get(comparison["comparison_id"], {}).get(
                    pass_index,
                    {"parse_ok": False, "pass_index": pass_index, "missing": True},
                )
                for pass_index in (0, 1)
            ],
            "verifier_guided": guided_results.get(
                comparison["comparison_id"],
                {"parse_ok": False, "missing": True},
            ),
        }
        for comparison in comparisons
    }


def build_pair_evaluations(
    *,
    backend: Any,
    image: Any,
    question_id: str,
    question: str,
    image_path: str,
    policy: VerifierPolicy,
    gt_audit_status: str,
    round0: Sequence[Mapping[str, Any]],
    offspring: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    parent_by_id = {str(item["trajectory_id"]): dict(item) for item in round0}
    comparisons: List[Dict[str, Any]] = []
    for child in offspring:
        task = dict(child.get("task") or {})
        correct_parent = (
            parent_by_id.get(str(task.get("correct_parent_id")))
            if task.get("correct_parent_id")
            else None
        )
        for wrong_id in task.get("wrong_parent_ids") or []:
            wrong_parent = parent_by_id[str(wrong_id)]
            comparison_id = f"{child['trajectory_id']}__vs__{wrong_id}"
            comparisons.append(
                {
                    "comparison_id": comparison_id,
                    "child": dict(child),
                    "wrong_parent": wrong_parent,
                    "correct_parent": correct_parent,
                    "child_text": child.get("raw_response", ""),
                    "wrong_parent_text": wrong_parent.get("raw_response", ""),
                    "cheap_pass": cheap_comparison_pass(
                        child=child,
                        wrong_parent=wrong_parent,
                        correct_parent=correct_parent,
                        policy=policy,
                        args=args,
                    ),
                }
            )
    audit_targets = [item for item in comparisons if item["cheap_pass"]]
    audits = run_process_audits(
        backend=backend,
        image=image,
        question_id=question_id,
        question=question,
        image_path=image_path,
        comparisons=audit_targets,
        args=args,
    ) if audit_targets else {}

    evaluations: List[Dict[str, Any]] = []
    for comparison in comparisons:
        child = comparison["child"]
        wrong = comparison["wrong_parent"]
        correct = comparison["correct_parent"]
        task = dict(child.get("task") or {})
        audit_bundle = audits.get(
            comparison["comparison_id"],
            {"blind": [], "verifier_guided": {"parse_ok": False, "missing": True}},
        )
        pair_audits = audit_bundle["blind"]
        evaluation = evaluate_dpo_pair(
            arm=str(task.get("arm", "")),
            pair_type=str(task.get("pair_type", "")),
            child_text=str(child.get("raw_response", "")),
            wrong_parent_text=str(wrong.get("raw_response", "")),
            correct_parent_text=(str(correct.get("raw_response", "")) if correct else None),
            child_verifier_correct=(child.get("verifier") or {}).get("compatible_is_correct"),
            wrong_parent_verifier_correct=(wrong.get("verifier") or {}).get("compatible_is_correct"),
            gt_status=policy.gt_status,
            audits=pair_audits,
            min_length_ratio=args.min_length_ratio,
            max_length_ratio=args.max_length_ratio,
            max_correct_parent_similarity=args.max_correct_parent_similarity,
        )
        evaluations.append(
            {
                "comparison_id": comparison["comparison_id"],
                "arm": task.get("arm"),
                "crossover_task_id": task.get("task_id"),
                "lineage_pair_type": task.get("pair_type"),
                "chosen_id": child["trajectory_id"],
                "rejected_id": wrong["trajectory_id"],
                "correct_parent_id": task.get("correct_parent_id"),
                "parent_ids": list(task.get("parent_ids") or []),
                "chosen": child.get("raw_response", ""),
                "rejected": wrong.get("raw_response", ""),
                "chosen_verifier": True if (child.get("verifier") or {}).get("compatible_is_correct") is True else False,
                "rejected_verifier": False if (wrong.get("verifier") or {}).get("compatible_is_correct") is False else None,
                "gt_audit_status": policy.gt_status,
                "gt_reference_audit_status": gt_audit_status,
                "training_ready_without_additional_gt_audit": bool(
                    evaluation.get("eligible")
                    and gt_audit_status == "sidecar_reviewed_trusted"
                ),
                "gt_exposed_during_generation": False,
                "oracle_hint_exposed": task.get("arm") == "oracle_hinted",
                "verifier_guided_process_audit": audit_bundle["verifier_guided"],
                "guided_audit_used_for_dpo_filter": False,
                **evaluation,
            }
        )
    return evaluations


def process_question(
    *,
    row: Mapping[str, Any],
    backend: Any,
    sidecar: Mapping[str, Any],
    run_dir: Path,
    input_base: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, Any]:
    question_id = str(row["question_id"])
    question = str(row["question"])
    raw_image_path = Path(str(row["image_path"]))
    image_path = str(raw_image_path if raw_image_path.is_absolute() else input_base / raw_image_path)
    ground_truth = row["ground_truth"]
    previous_path = state_path(run_dir, question_id)
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else None
    if previous and previous.get("status") in {"completed", "skipped"}:
        logger.info("Resume skip completed question id=%s", question_id)
        return previous

    answer_type, policy, policy_record = policy_for_question(
        question_id=question_id,
        ground_truth=ground_truth,
        image_path=image_path,
        sidecar=sidecar,
    )
    if policy.gt_status.lower() != "trusted":
        gt_audit_status = "sidecar_marked_suspect"
    elif policy_record.get("source") == "default_policy":
        gt_audit_status = "dataset_reference_unreviewed"
    else:
        gt_audit_status = "sidecar_reviewed_trusted"
    state: Dict[str, Any] = previous or {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "question_id": question_id,
        "question": question,
        "image_path": image_path,
        "dataset_metadata": row.get("dataset_metadata", {}),
        "oracle_evaluation": {
            "reference": ground_truth,
            "answer_type": answer_type,
            "policy": policy_record,
            "gt_reference_audit_status": gt_audit_status,
            "reference_passed_to_solver_or_crossover": False,
        },
        "stage": "initialized",
        "status": "running",
        "created_at": io.utc_now(),
    }
    state["status"] = "running"
    state.pop("error", None)
    save_state(run_dir, state)

    image = io.open_image_rgb(image_path)
    try:
        if STAGE_ORDER.get(str(state.get("stage")), 0) < STAGE_ORDER["round0_generated"]:
            logger.info("Round0 start id=%s count=16", question_id)
            state["round0_trajectories"] = generate_round0(
                backend=backend,
                image=image,
                question_id=question_id,
                question=question,
                image_path=image_path,
                ground_truth=ground_truth,
                answer_type=answer_type,
                policy=policy,
                args=args,
            )
            state["stage"] = "round0_generated"
            save_state(run_dir, state)

        if STAGE_ORDER.get(str(state.get("stage")), 0) < STAGE_ORDER["round0_completed"]:
            run_trajectory_judges(
                backend=backend,
                image=image,
                question_id=question_id,
                question=question,
                image_path=image_path,
                trajectories=state["round0_trajectories"],
                args=args,
            )
            state["pool_plan"] = plan_probe_pool(
                candidates_from_round0(state["round0_trajectories"])
            )
            tasks = build_crossover_tasks(
                question_id=question_id,
                pool_plan=state["pool_plan"],
                children_per_operator=args.children_per_operator,
                hinted_fraction=args.hinted_control_fraction,
                seed=args.seed,
            )
            state["crossover_tasks"] = [asdict(task) for task in tasks]
            state["stage"] = "round0_completed"
            save_state(run_dir, state)
            logger.info(
                "Round0 complete id=%s mode=%s C=%s F=%s tasks=%d",
                question_id,
                state["pool_plan"]["mode"],
                state["pool_plan"]["counts"]["usable_correct"],
                state["pool_plan"]["counts"]["usable_wrong"],
                len(tasks),
            )

        if STAGE_ORDER.get(str(state.get("stage")), 0) < STAGE_ORDER["offspring_generated"]:
            tasks = [CrossoverTask(**item) for item in state.get("crossover_tasks", [])]
            state["offspring"] = generate_offspring(
                backend=backend,
                image=image,
                question_id=question_id,
                question=question,
                image_path=image_path,
                ground_truth=ground_truth,
                answer_type=answer_type,
                policy=policy,
                round0=state["round0_trajectories"],
                tasks=tasks,
                args=args,
            ) if tasks else []
            state["stage"] = "offspring_generated"
            save_state(run_dir, state)

        if STAGE_ORDER.get(str(state.get("stage")), 0) < STAGE_ORDER["offspring_completed"]:
            run_trajectory_judges(
                backend=backend,
                image=image,
                question_id=question_id,
                question=question,
                image_path=image_path,
                trajectories=state.get("offspring", []),
                args=args,
            )
            state["stage"] = "offspring_completed"
            save_state(run_dir, state)

        if STAGE_ORDER.get(str(state.get("stage")), 0) < STAGE_ORDER["audits_completed"]:
            state["pair_evaluations"] = build_pair_evaluations(
                backend=backend,
                image=image,
                question_id=question_id,
                question=question,
                image_path=image_path,
                policy=policy,
                gt_audit_status=gt_audit_status,
                round0=state["round0_trajectories"],
                offspring=state.get("offspring", []),
                args=args,
            )
            state["stage"] = "audits_completed"
            save_state(run_dir, state)

        state["stage"] = "completed"
        state["status"] = "completed"
        state["finished_at"] = io.utc_now()
        state["invariants"] = {
            "round0_exactly_16": len(state.get("round0_trajectories", [])) == 16,
            "round0_unique_hashes": len(
                {item.get("content_hash") for item in state.get("round0_trajectories", [])}
            ) == len(state.get("round0_trajectories", [])),
            "blind_dpo_has_no_oracle_hint": all(
                not item.get("oracle_hint_exposed")
                for item in state.get("pair_evaluations", [])
                if item.get("eligible")
            ),
            "eligible_pairs_are_c_to_f_or_ff_rescue": all(
                item.get("pair_type") in {"error_repair", "failure_failure_rescue"}
                for item in state.get("pair_evaluations", [])
                if item.get("eligible")
            ),
        }
        save_state(run_dir, state)
        return state
    except Exception as exc:
        state["status"] = "engine_dead" if io.is_engine_dead(exc) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        state["failed_at"] = io.utc_now()
        save_state(run_dir, state)
        raise
    finally:
        image.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    for path in (args.input_jsonl, args.sample_manifest, args.verifier_sidecar):
        if path is not None and not path.is_file():
            print(f"Required file does not exist: {path}", file=sys.stderr)
            return 2
    run_dir = args.run_dir or (
        args.output_jsonl.parent / f"{args.output_jsonl.stem}_artifacts_experiment"
    )
    try:
        prepare_run_dir(run_dir, resume=args.resume)
        if args.output_jsonl.exists() and not args.resume:
            raise ValueError(
                f"output already exists; use a new versioned filename or --resume: {args.output_jsonl}"
            )
        logger = setup_logger(run_dir / "run.log", args.log_file)
        manifest_path = run_dir / "run_manifest.json"
        manifest = make_manifest(args, run_dir)
        if args.resume and manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in (
                "input_sha256",
                "sample_manifest_sha256",
                "verifier_sidecar_sha256",
                "code_sha256",
                "runtime_config",
                "protocol",
                "output_jsonl",
            ):
                if previous.get(key) != manifest.get(key):
                    raise ValueError(f"resume refused: {key} changed")
            manifest["original_started_at"] = previous.get("started_at")
            manifest["resumed_at"] = io.utc_now()
        io.atomic_json(manifest_path, manifest)
        executed = (
            (f"CUDA_VISIBLE_DEVICES={shlex.quote(os.environ['CUDA_VISIBLE_DEVICES'])} " if os.environ.get("CUDA_VISIBLE_DEVICES") else "")
            + " ".join(shlex.quote(item) for item in sys.argv)
        )
        io.atomic_text(run_dir / "executed_command.txt", executed + "\n")
        io.append_jsonl(
            run_dir / "execution_history.jsonl",
            {
                "started_at": io.utc_now(),
                "command": executed,
                "cwd": str(Path.cwd()),
                "resume": args.resume,
                "dry_run": args.dry_run,
            },
        )
        io.atomic_json(
            run_dir / "resolved_config.json",
            {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()},
        )
        io.atomic_json(run_dir / "environment_summary.json", io.collect_environment_summary())

        preparation = prepare_wemath_input(
            input_jsonl=args.input_jsonl,
            sample_manifest=args.sample_manifest,
            run_dir=run_dir,
            dataset_kind=args.dataset_kind,
        )
        rows = io.load_jsonl(preparation.model_input_path)
        for row in rows:
            row["dataset_metadata"] = preparation.metadata_by_question_id.get(str(row["question_id"]), {})
        start = max(0, args.start_line - 1)
        rows = rows[start : (start + args.limit if args.limit else None)]
        missing_images = []
        for row in rows:
            raw = Path(str(row["image_path"]))
            resolved = raw if raw.is_absolute() else args.input_jsonl.parent / raw
            if not resolved.is_file():
                missing_images.append({"question_id": row["question_id"], "image_path": str(resolved)})
        preflight = {
            "status": "passed" if not missing_images else "failed",
            "selected_question_count": len(rows),
            "round0_calls_planned": len(rows) * 16,
            "maximum_blind_cf_offspring_if_all_mixed": len(rows) * 4 * len(OPERATORS) * args.children_per_operator,
            "maximum_blind_ff_offspring_if_all_mixed": len(rows) * len(OPERATORS) * args.children_per_operator,
            "hinted_control_is_additional": True,
            "missing_images": missing_images,
            "verifier_version": VERIFIER_VERSION,
        }
        io.atomic_json(run_dir / "preflight_report.json", preflight)
        if missing_images:
            raise ValueError(f"preflight found {len(missing_images)} missing images")
        if args.dry_run:
            manifest["status"] = "dry_run_completed"
            manifest["finished_at"] = io.utc_now()
            io.atomic_json(manifest_path, manifest)
            logger.info("Dry run complete for %d questions", len(rows))
            return 0

        recorder = io.CallRecorder(run_dir)
        if args.backend == "mock":
            script = json.loads(args.mock_script.read_text(encoding="utf-8"))
            backend: Any = io.ScriptedMockBackend(script, recorder)
        else:
            backend = io.VllmBackend(
                model_path=args.model_path,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                recorder=recorder,
            )
        sidecar = load_sidecar(args.verifier_sidecar)
        completed_now = 0
        failed_now = 0
        for index, row in enumerate(rows, start=1):
            logger.info("Question %d/%d id=%s", index, len(rows), row["question_id"])
            try:
                result = process_question(
                    row=row,
                    backend=backend,
                    sidecar=sidecar,
                    run_dir=run_dir,
                    input_base=args.input_jsonl.parent,
                    args=args,
                    logger=logger,
                )
                completed_now += int(result.get("status") == "completed")
            except Exception as exc:
                failed_now += 1
                logger.error("Question failed id=%s error=%s", row["question_id"], exc)
                rebuild_exports(run_dir, args.output_jsonl)
                if io.is_engine_dead(exc):
                    logger.critical("EngineDeadError detected; stopping immediately to avoid chained failures")
                    manifest["status"] = "engine_dead_stopped"
                    manifest["engine_dead_question_id"] = row["question_id"]
                    manifest["finished_at"] = io.utc_now()
                    io.atomic_json(manifest_path, manifest)
                    return 3
                continue
            rebuild_exports(run_dir, args.output_jsonl)

        metrics = rebuild_exports(run_dir, args.output_jsonl)
        all_states = load_states(run_dir)
        invariants = {
            "output_question_ids_unique": len({item["question_id"] for item in all_states}) == len(all_states),
            "completed_round0_exactly_16": all(
                len(item.get("round0_trajectories", [])) == 16
                for item in all_states
                if item.get("status") == "completed"
            ),
            "all_completed_question_invariants_true": all(
                all((item.get("invariants") or {}).values())
                for item in all_states
                if item.get("status") == "completed"
            ),
            "latest_verifier_loaded": VERIFIER_VERSION == "4.1.0-v5-256-regression-20260806",
        }
        io.atomic_json(run_dir / "invariant_report.json", invariants)
        manifest["status"] = "completed" if failed_now == 0 else "completed_with_failures"
        manifest["completed_now"] = completed_now
        manifest["failed_now"] = failed_now
        manifest["metrics"] = metrics
        manifest["finished_at"] = io.utc_now()
        io.atomic_json(manifest_path, manifest)
        return 0 if failed_now == 0 and all(invariants.values()) else 1
    except Exception as exc:
        print(f"Fatal v7 error: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
