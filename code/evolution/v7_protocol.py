#!/usr/bin/env python3
"""Pure planning, prompt, and DPO-filter logic for the v7 experiment.

This module deliberately has no vLLM dependency.  C/F labels are used only to
construct an external task plan; crossover prompts receive anonymous parents.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from verifier_v4 import generator_answer_constraint


SCRIPT_VERSION = "7.2.0-per-request-seeds-20260806"
ORACLE_HINT_SENTENCE = (
    "Exactly one of the two parent trajectories has a correct final result and "
    "the other has an incorrect final result. You are not told which is which."
)
OPERATORS = ("preserve_repair", "independent_recombination")

TAGGED_RATIONALE_INSTRUCTION = r"""Output exactly two blocks and no other text:
<think>
Write a concise, auditable judgment rationale. Check the claimed result, the
key reasoning steps, image grounding, the earliest suspected error, and genuine
uncertainty. Use no more than six numbered checks. This is an explicit report
for the experiment, not a claim that hidden internal model state is observable.
</think>
<answer>
ONE STRICT JSON OBJECT REQUIRED BY THE SCHEMA
</answer>
Do not place JSON inside <think>. Do not omit either closing tag.
"""


@dataclass(frozen=True)
class ProbeCandidate:
    trajectory_id: str
    raw_response: str
    reasoning: str
    final_answer: str
    verifier_correct: Optional[bool]
    verifier_status: str
    finish_reason: str = ""

    @property
    def usable(self) -> bool:
        return bool(
            self.raw_response.strip()
            and self.final_answer.strip()
            and len(self.reasoning.strip()) >= 40
            and self.finish_reason != "length"
            and self.verifier_correct is not None
        )


@dataclass(frozen=True)
class CrossoverTask:
    task_id: str
    pair_type: str
    arm: str
    operator: str
    replicate: int
    parent_ids: Tuple[str, str]
    correct_parent_id: Optional[str]
    wrong_parent_ids: Tuple[str, ...]
    prompt_parent_ids: Tuple[str, str]


def stable_fraction(key: str, *, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def stable_sampling_seed(base_seed: int, *parts: Any) -> int:
    """Derive a reproducible per-request vLLM seed without Python hash()."""

    payload = json.dumps(
        [int(base_seed), *[str(part) for part in parts]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def normalized_text(text: Any) -> str:
    value = str(text or "").lower()
    value = re.sub(r"<think>|</think>|<answer>|</answer>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def content_hash(text: Any) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def text_similarity(left: Any, right: Any) -> float:
    a = normalized_text(left)
    b = normalized_text(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def tokenish_length(text: Any) -> int:
    return len(re.findall(r"\w+|[^\w\s]", str(text or ""), flags=re.UNICODE))


def length_ratio(chosen: Any, rejected: Any) -> float:
    denominator = max(1, tokenish_length(rejected))
    return tokenish_length(chosen) / denominator


def answer_cluster(candidate: ProbeCandidate) -> str:
    return normalized_text(candidate.final_answer)


def _base_quality(candidate: ProbeCandidate) -> Tuple[int, int, str]:
    # Deterministic structural quality only.  No Self-Judge score is used here.
    reasonable_size = min(len(candidate.reasoning), 4000)
    return (1 if candidate.finish_reason != "length" else 0, reasonable_size, candidate.trajectory_id)


def select_diverse(
    candidates: Sequence[ProbeCandidate],
    count: int,
    *,
    prefer_answer_diversity: bool,
) -> List[ProbeCandidate]:
    remaining = sorted((item for item in candidates if item.usable), key=_base_quality, reverse=True)
    if not remaining or count <= 0:
        return []
    selected = [remaining.pop(0)]
    while remaining and len(selected) < count:
        def key(item: ProbeCandidate) -> Tuple[int, float, int, str]:
            new_cluster = int(
                prefer_answer_diversity
                and answer_cluster(item) not in {answer_cluster(old) for old in selected}
            )
            minimum_distance = min(
                1.0 - text_similarity(item.reasoning, old.reasoning)
                for old in selected
            )
            return (new_cluster, minimum_distance, min(len(item.reasoning), 4000), item.trajectory_id)

        chosen = max(remaining, key=key)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def plan_probe_pool(candidates: Sequence[ProbeCandidate]) -> Dict[str, Any]:
    correct = [item for item in candidates if item.usable and item.verifier_correct is True]
    wrong = [item for item in candidates if item.usable and item.verifier_correct is False]
    unresolved = [item for item in candidates if not item.usable or item.verifier_correct is None]

    selected_correct: List[ProbeCandidate] = []
    selected_wrong: List[ProbeCandidate] = []
    mode = "insufficient"
    if len(correct) >= 2 and len(wrong) >= 2:
        mode = "mixed_2c2f"
        selected_correct = select_diverse(correct, 2, prefer_answer_diversity=False)
        selected_wrong = select_diverse(wrong, 2, prefer_answer_diversity=True)
    elif not correct and len(wrong) >= 4:
        mode = "all_f_rescue"
        selected_wrong = select_diverse(wrong, 4, prefer_answer_diversity=True)
    elif correct and not wrong:
        mode = "all_c_no_crossover"
    elif len(wrong) >= 2:
        # No forced C-F data, but F-F exploration remains useful.
        mode = "insufficient_cf_ff_only"
        selected_wrong = select_diverse(wrong, min(4, len(wrong)), prefer_answer_diversity=True)

    cf_pairs = [
        (correct_item.trajectory_id, wrong_item.trajectory_id)
        for correct_item in selected_correct
        for wrong_item in selected_wrong
    ] if mode == "mixed_2c2f" else []
    ff_pairs = [
        (left.trajectory_id, right.trajectory_id)
        for left, right in itertools.combinations(selected_wrong, 2)
    ]
    if mode == "mixed_2c2f":
        # The primary screenshot design asks for 4 C-F combinations and one
        # additional F-F rescue attempt, not every possible F-F permutation.
        ff_pairs = ff_pairs[:1]

    return {
        "mode": mode,
        "counts": {
            "round0": len(candidates),
            "usable_correct": len(correct),
            "usable_wrong": len(wrong),
            "unresolved_or_low_quality": len(unresolved),
        },
        "selected_correct_ids": [item.trajectory_id for item in selected_correct],
        "selected_wrong_ids": [item.trajectory_id for item in selected_wrong],
        "cf_pairs": [list(pair) for pair in cf_pairs],
        "ff_pairs": [list(pair) for pair in ff_pairs],
    }


def _ordered_parents(
    pair: Tuple[str, str],
    *,
    question_id: str,
    operator: str,
    replicate: int,
    seed: int,
) -> Tuple[str, str]:
    key = f"{question_id}:{pair[0]}:{pair[1]}:{operator}:{replicate}:order"
    return pair if stable_fraction(key, seed=seed) < 0.5 else (pair[1], pair[0])


def build_crossover_tasks(
    *,
    question_id: str,
    pool_plan: Mapping[str, Any],
    operators: Sequence[str] = OPERATORS,
    children_per_operator: int = 2,
    hinted_fraction: float = 0.10,
    seed: int = 42,
) -> List[CrossoverTask]:
    if children_per_operator < 1:
        raise ValueError("children_per_operator must be positive")
    if not 0.0 <= hinted_fraction <= 1.0:
        raise ValueError("hinted_fraction must be between 0 and 1")
    unknown = sorted(set(operators) - set(OPERATORS))
    if unknown:
        raise ValueError(f"unsupported crossover operators: {unknown}")

    tasks: List[CrossoverTask] = []
    for pair_type, pairs in (("CF", pool_plan.get("cf_pairs", [])), ("FF", pool_plan.get("ff_pairs", []))):
        for pair_index, raw_pair in enumerate(pairs):
            pair = (str(raw_pair[0]), str(raw_pair[1]))
            for operator in operators:
                for replicate in range(children_per_operator):
                    base_key = f"{question_id}:{pair_type}:{pair_index}:{operator}:{replicate}"
                    ordered = _ordered_parents(
                        pair,
                        question_id=question_id,
                        operator=operator,
                        replicate=replicate,
                        seed=seed,
                    )
                    correct_parent = pair[0] if pair_type == "CF" else None
                    wrong_parents = (pair[1],) if pair_type == "CF" else pair
                    blind = CrossoverTask(
                        task_id=f"{base_key}:blind",
                        pair_type=pair_type,
                        arm="blind",
                        operator=operator,
                        replicate=replicate,
                        parent_ids=pair,
                        correct_parent_id=correct_parent,
                        wrong_parent_ids=tuple(wrong_parents),
                        prompt_parent_ids=ordered,
                    )
                    tasks.append(blind)
                    # The hinted arm is additional control data; it never replaces
                    # a blind main-data attempt and applies only to C-F tasks.
                    if pair_type == "CF" and stable_fraction(base_key + ":hint", seed=seed) < hinted_fraction:
                        tasks.append(
                            CrossoverTask(
                                **{
                                    **asdict(blind),
                                    "task_id": f"{base_key}:hinted",
                                    "arm": "oracle_hinted",
                                }
                            )
                        )
    return tasks


def solver_messages(
    *,
    question: str,
    answer_type: str,
    slot: int,
    retry: int,
    vision_placeholder: str,
) -> Tuple[str, str]:
    system = (
        "You are the Solver role. Solve the visual mathematics problem independently. "
        "Use the image and question only. Return reasoning inside <think>...</think> "
        "and the final answer inside <answer>...</answer>."
    )
    user = (
        f"{vision_placeholder}\nQuestion:\n{question}\n\n"
        f"Independent probe slot: {slot}; diversity retry: {retry}. "
        "Use a genuinely independent reasoning route when possible.\n"
        f"{generator_answer_constraint(answer_type)}"
    )
    return system, user


def crossover_messages(
    *,
    question: str,
    answer_type: str,
    task: CrossoverTask,
    parent_text_by_id: Mapping[str, str],
    vision_placeholder: str,
) -> Tuple[str, str]:
    if task.arm not in {"blind", "oracle_hinted"}:
        raise ValueError(f"unknown arm: {task.arm}")
    left_id, right_id = task.prompt_parent_ids
    operator_instruction = {
        "preserve_repair": (
            "Preserve independently valid steps, locate contradictions, repair them, "
            "and derive a complete solution."
        ),
        "independent_recombination": (
            "Rebuild the solution using a meaningfully different organization; use "
            "both parents only as fallible evidence and recheck every key step."
        ),
    }[task.operator]
    system = (
        "You are the blind Crossover role. You receive two anonymous candidate "
        "trajectories. Never assume either candidate is correct. Re-evaluate the "
        "problem from the image and question, combine only information you can "
        "independently justify, and output one complete solution."
    )
    hint = f"\nControlled hint: {ORACLE_HINT_SENTENCE}\n" if task.arm == "oracle_hinted" else "\n"
    user = (
        f"{vision_placeholder}\nQuestion:\n{question}\n\n"
        f"Parent A:\n{parent_text_by_id[left_id]}\n\n"
        f"Parent B:\n{parent_text_by_id[right_id]}\n"
        f"{hint}Crossover instruction: {operator_instruction}\n"
        f"Independent offspring sample index: {task.replicate}. Recheck the problem "
        "and produce an independent sample rather than copying another offspring.\n"
        "Return reasoning inside <think>...</think> and the final answer inside "
        "<answer>...</answer>.\n"
        f"{generator_answer_constraint(answer_type)}"
    )
    return system, user


PROCESS_AUDIT_SCHEMA = r"""Return exactly one JSON object:
{
  "child_repaired_parent_error": true|false,
  "new_critical_error_introduced": true|false,
  "stable_prefix": "shared valid prefix or empty string",
  "first_parent_error": "earliest concrete parent error",
  "repair_summary": "what the offspring repaired",
  "shared_reasoning_ratio": 0.0-1.0,
  "dpo_pair_recommended": true|false,
  "confidence": 0.0-1.0
}
"""


def process_audit_messages(
    *,
    question: str,
    wrong_parent: str,
    child: str,
    pass_index: int,
    vision_placeholder: str,
) -> Tuple[str, str]:
    if pass_index not in {0, 1}:
        raise ValueError("pass_index must be 0 or 1")
    system = (
        "You are a blind Process Judge for a parent-offspring lineage. You have no "
        "ground truth, verifier result, correctness label, teacher signal, or score. "
        "Judge whether the offspring genuinely repairs a concrete reasoning error "
        "in the source parent without introducing a new critical error."
    )
    if pass_index == 0:
        first_label, first_text = "Source parent", wrong_parent
        second_label, second_text = "Offspring", child
    else:
        first_label, first_text = "Offspring", child
        second_label, second_text = "Source parent", wrong_parent
    user = (
        f"{vision_placeholder}\nQuestion:\n{question}\n\n"
        f"{first_label}:\n{first_text}\n\n{second_label}:\n{second_text}\n\n"
        "The order is intentionally varied. Evaluate the stated lineage roles, not "
        "position or writing style. Do not infer any hidden reference answer.\n"
        f"{PROCESS_AUDIT_SCHEMA}\n{TAGGED_RATIONALE_INSTRUCTION}"
    )
    return system, user


def guided_process_audit_messages(
    *,
    question: str,
    wrong_parent: str,
    child: str,
    verifier_evidence: Mapping[str, Any],
    vision_placeholder: str,
) -> Tuple[str, str]:
    system = (
        "You are a verifier-guided Process Judge for a parent-offspring lineage. "
        "The deterministic verifier evaluated final answers only; the reference "
        "answer itself is hidden. Use the result-layer evidence to focus your audit, "
        "but independently inspect the full process and explicitly note any reason "
        "to distrust the verifier."
    )
    user = (
        f"{vision_placeholder}\nQuestion:\n{question}\n\n"
        f"Source parent:\n{wrong_parent}\n\nOffspring:\n{child}\n\n"
        "Final-result evidence with the reference hidden:\n"
        f"{json.dumps(dict(verifier_evidence), ensure_ascii=False)}\n\n"
        "Determine whether the offspring genuinely repairs the parent process. "
        "Do not reconstruct or request the hidden reference answer.\n"
        f"{PROCESS_AUDIT_SCHEMA}\n{TAGGED_RATIONALE_INSTRUCTION}"
    )
    return system, user


TRAJECTORY_JUDGE_SCHEMA = r"""The <answer> JSON must have exactly these semantics:
{
  "result_verdict": "LikelyCorrect"|"LikelyIncorrect"|"Uncertain",
  "process_verdict": "Sound"|"MinorIssue"|"MajorError"|"Uncertain",
  "score": 0-100,
  "confidence": 0.0-1.0,
  "first_error": "earliest concrete error or empty string",
  "error_type": "short category or empty string",
  "valid_prefix": "reasoning worth preserving or empty string",
  "useful_information": "reusable information or empty string",
  "fusion_advice": "specific repair advice or empty string"
}
"""


def trajectory_judge_messages(
    *,
    question: str,
    trajectory: str,
    mode: str,
    verifier_evidence: Optional[Mapping[str, Any]],
    vision_placeholder: str,
) -> Tuple[str, str]:
    if mode == "blind":
        system = (
            "You are the blind Self-Judge role. Evaluate the candidate using only "
            "the image, question, and candidate trajectory. You have no reference "
            "answer, verifier result, teacher label, or prior correctness label."
        )
        evidence = (
            "No verifier or reference evidence is available. Judge result and "
            "process independently, and preserve uncertainty when warranted."
        )
    elif mode == "verifier_guided":
        if verifier_evidence is None:
            raise ValueError("verifier_guided mode requires verifier_evidence")
        system = (
            "You are the verifier-guided Self-Judge role. A deterministic verifier "
            "evaluated only the final answer; the reference answer itself is hidden. "
            "Use that result-layer evidence, then independently audit the reasoning "
            "process. Do not turn the verifier label into a second process score."
        )
        evidence = (
            "Final-result evidence with the reference hidden:\n"
            + json.dumps(dict(verifier_evidence), ensure_ascii=False)
        )
    else:
        raise ValueError(f"unsupported trajectory judge mode: {mode}")
    user = (
        f"{vision_placeholder}\nQuestion:\n{question}\n\n"
        f"Candidate trajectory:\n{trajectory}\n\n{evidence}\n\n"
        f"{TRAJECTORY_JUDGE_SCHEMA}\n{TAGGED_RATIONALE_INSTRUCTION}"
    )
    return system, user


def extract_tagged_rationale_json(raw: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    text = str(raw or "").strip()
    think_match = re.fullmatch(
        r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*",
        text,
        flags=re.I | re.S,
    )
    if not think_match:
        return None, None
    rationale = think_match.group(1).strip()
    answer_text = think_match.group(2).strip()
    if not rationale:
        return None, None
    try:
        parsed = json.loads(answer_text)
    except (json.JSONDecodeError, TypeError):
        return rationale, None
    return rationale, parsed if isinstance(parsed, dict) else None


def parse_trajectory_judge(raw: str, *, mode: str) -> Dict[str, Any]:
    rationale, parsed = extract_tagged_rationale_json(raw)
    if rationale is None or parsed is None:
        return {"mode": mode, "parse_ok": False, "raw_response": raw}
    result_verdict = parsed.get("result_verdict")
    process_verdict = parsed.get("process_verdict")
    if result_verdict not in {"LikelyCorrect", "LikelyIncorrect", "Uncertain"}:
        return {"mode": mode, "parse_ok": False, "raw_response": raw, "judge_rationale": rationale}
    if process_verdict not in {"Sound", "MinorIssue", "MajorError", "Uncertain"}:
        return {"mode": mode, "parse_ok": False, "raw_response": raw, "judge_rationale": rationale}
    try:
        score = float(parsed["score"])
        confidence = float(parsed["confidence"])
    except (KeyError, TypeError, ValueError):
        return {"mode": mode, "parse_ok": False, "raw_response": raw, "judge_rationale": rationale}
    if not (0.0 <= score <= 100.0 and 0.0 <= confidence <= 1.0):
        return {"mode": mode, "parse_ok": False, "raw_response": raw, "judge_rationale": rationale}
    result: Dict[str, Any] = {
        "mode": mode,
        "parse_ok": True,
        "judge_rationale": rationale,
        "result_verdict": result_verdict,
        "process_verdict": process_verdict,
        "score": score,
        "confidence": confidence,
        "raw_response": raw,
    }
    for key in ("first_error", "error_type", "valid_prefix", "useful_information", "fusion_advice"):
        result[key] = str(parsed.get(key, ""))
    return result


def parse_process_audit(raw: str) -> Dict[str, Any]:
    rationale, parsed = extract_tagged_rationale_json(raw)
    if rationale is None or parsed is None:
        return {"parse_ok": False, "raw_response": raw}
    required_bools = (
        "child_repaired_parent_error",
        "new_critical_error_introduced",
        "dpo_pair_recommended",
    )
    if any(type(parsed.get(name)) is not bool for name in required_bools):
        return {"parse_ok": False, "raw_response": raw, "parsed": parsed}
    try:
        ratio = float(parsed["shared_reasoning_ratio"])
        confidence = float(parsed["confidence"])
    except (KeyError, TypeError, ValueError):
        return {"parse_ok": False, "raw_response": raw, "parsed": parsed}
    if not (0.0 <= ratio <= 1.0 and 0.0 <= confidence <= 1.0):
        return {"parse_ok": False, "raw_response": raw, "parsed": parsed}
    return {
        "parse_ok": True,
        "judge_rationale": rationale,
        "child_repaired_parent_error": parsed["child_repaired_parent_error"],
        "new_critical_error_introduced": parsed["new_critical_error_introduced"],
        "stable_prefix": str(parsed.get("stable_prefix", "")),
        "first_parent_error": str(parsed.get("first_parent_error", "")),
        "repair_summary": str(parsed.get("repair_summary", "")),
        "shared_reasoning_ratio": ratio,
        "dpo_pair_recommended": parsed["dpo_pair_recommended"],
        "confidence": confidence,
        "raw_response": raw,
    }


def evaluate_dpo_pair(
    *,
    arm: str,
    pair_type: str,
    child_text: str,
    wrong_parent_text: str,
    correct_parent_text: Optional[str],
    child_verifier_correct: Optional[bool],
    wrong_parent_verifier_correct: Optional[bool],
    gt_status: str,
    audits: Sequence[Mapping[str, Any]],
    min_length_ratio: float = 0.70,
    max_length_ratio: float = 1.30,
    max_correct_parent_similarity: float = 0.92,
) -> Dict[str, Any]:
    ratio = length_ratio(child_text, wrong_parent_text)
    child_wrong_similarity = text_similarity(child_text, wrong_parent_text)
    child_correct_similarity = (
        text_similarity(child_text, correct_parent_text)
        if correct_parent_text is not None
        else None
    )
    reasons: List[str] = []
    if arm != "blind":
        reasons.append("oracle_hinted_control_excluded")
    if gt_status.lower() != "trusted":
        reasons.append("gt_not_trusted")
    if child_verifier_correct is not True:
        reasons.append("child_not_verifier_correct")
    if wrong_parent_verifier_correct is not False:
        reasons.append("rejected_parent_not_verifier_wrong")
    if not (min_length_ratio <= ratio <= max_length_ratio):
        reasons.append("length_ratio_out_of_range")
    if normalized_text(child_text) == normalized_text(wrong_parent_text):
        reasons.append("child_duplicates_wrong_parent")
    if child_correct_similarity is not None and child_correct_similarity > max_correct_parent_similarity:
        reasons.append("child_too_similar_to_correct_parent")
    if len(audits) != 2 or not all(bool(item.get("parse_ok")) for item in audits):
        reasons.append("two_parseable_process_audits_required")
    elif not all(
        item.get("child_repaired_parent_error") is True
        and item.get("new_critical_error_introduced") is False
        and item.get("dpo_pair_recommended") is True
        for item in audits
    ):
        reasons.append("process_audit_not_unanimous")
    return {
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "pair_type": (
            "failure_failure_rescue" if pair_type == "FF" else "error_repair"
        ),
        "length_ratio": ratio,
        "child_wrong_parent_similarity": child_wrong_similarity,
        "child_correct_parent_similarity": child_correct_similarity,
        "process_audits": [dict(item) for item in audits],
    }
