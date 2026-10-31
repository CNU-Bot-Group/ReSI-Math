#!/usr/bin/env python3
import re

BOXED_RE = re.compile(
    r"\\boxed\s*\{\s*([01])\s*\}",
    flags=re.I,
)

def extract_final_verdict(text):
    matches = BOXED_RE.findall(str(text or ""))
    return matches[-1] if matches else None

def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    **kwargs,
):
    if solution_str is None:
        solution_str = (
            kwargs.get("response")
            or kwargs.get("solution")
            or kwargs.get("completion")
            or ""
        )

    gt = str(ground_truth or "").strip()

    if gt not in {"0", "1"}:
        return 0.0

    pred = extract_final_verdict(solution_str)

    if pred is None:
        return 0.0

    return 1.0 if pred == gt else 0.0
