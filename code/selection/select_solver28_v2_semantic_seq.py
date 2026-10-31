#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pandas as pd


# ============================================================
# Format / parse-failure sentinels
# These are NOT semantic mathematical answers.
# ============================================================

PARSE_FAIL_EXACT = {
    "",
    "<NO_BOX>",
    "<NO_ANSWER>",
    "<PARSE_FAIL>",
    "<PARSE_FAILED>",
    "<NONE>",
    "NONE",
    "NULL",
    "N/A",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_histogram(x):
    """
    Accept:
      dict
      JSON string
      Python-dict string
    """
    if isinstance(x, dict):
        return {str(k): int(v) for k, v in x.items()}

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}

        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return {
                    str(k): int(v)
                    for k, v in obj.items()
                }
        except Exception:
            pass

        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return {
                    str(k): int(v)
                    for k, v in obj.items()
                }
        except Exception:
            pass

    raise ValueError(
        f"Cannot parse answer_histogram: {x!r}"
    )


def unwrap_full_wrapper(s: str):
    """
    Conservative wrapper normalization only.
    No algebraic equivalence is attempted here.

    Examples:
      \\boxed{4} -> 4
      \\text{A}  -> A
      $4$        -> 4
    """

    s = str(s).strip()

    # Markdown / LaTeX math delimiters.
    while len(s) >= 2 and (
        (s.startswith("$") and s.endswith("$"))
        or
        (s.startswith("`") and s.endswith("`"))
    ):
        s = s[1:-1].strip()

    wrappers = [
        "boxed",
        "text",
        "mathrm",
        "mathbf",
        "mathit",
        "operatorname",
    ]

    changed = True

    while changed:
        changed = False

        for w in wrappers:
            prefix = "\\" + w + "{"

            if not (
                s.startswith(prefix)
                and s.endswith("}")
            ):
                continue

            start = len(prefix)
            depth = 1
            valid = True

            for i in range(start, len(s)):
                ch = s[i]

                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1

                    if depth == 0:
                        if i != len(s) - 1:
                            valid = False
                        break

            if valid and depth == 0:
                s = s[start:-1].strip()
                changed = True
                break

    # Formatting whitespace only.
    s = re.sub(r"\s+", "", s)

    return s


def is_parse_failure(answer: str):
    s = str(answer).strip()

    if s.upper() in {
        x.upper()
        for x in PARSE_FAIL_EXACT
    }:
        return True

    # Profiler-style generic format sentinels.
    su = s.upper()

    if (
        su.startswith("<NO_")
        and su.endswith(">")
    ):
        return True

    if (
        "PARSE_FAIL" in su
        and su.startswith("<")
        and su.endswith(">")
    ):
        return True

    return False


def semantic_histogram(raw_hist):
    """
    Returns:
      semantic Counter
      format-failure count
    """

    sem = Counter()
    fail_count = 0

    for raw_answer, count in raw_hist.items():

        count = int(count)

        if is_parse_failure(raw_answer):
            fail_count += count
            continue

        a = unwrap_full_wrapper(raw_answer)

        if is_parse_failure(a):
            fail_count += count
            continue

        if not a:
            fail_count += count
            continue

        sem[a] += count

    return sem, fail_count


def disagreement(hist: Counter):
    """
    Gini disagreement:
        1 - sum p_i^2
    """
    n = sum(hist.values())

    if n <= 0:
        return 0.0

    return 1.0 - sum(
        (c / n) ** 2
        for c in hist.values()
    )


def entropy(hist: Counter):
    n = sum(hist.values())

    if n <= 0:
        return 0.0

    out = 0.0

    for c in hist.values():
        p = c / n

        if p > 0:
            out -= p * math.log(p)

    return out


def get_first(obj, keys, default=None):
    for k in keys:
        if k in obj:
            return obj[k]
    return default


def load_profile_files(profile_dir: Path):
    per_q = profile_dir / "per_question"

    if not per_q.exists():
        raise FileNotFoundError(
            f"Missing per_question dir: {per_q}"
        )

    files = sorted(per_q.glob("*.json"))

    if not files:
        raise RuntimeError(
            f"No JSON profiles under {per_q}"
        )

    return files


def build_repeat_dataframe(
    source_df,
    row_index,
    repeat_time,
    selected_meta,
):
    base = source_df.iloc[[row_index]].copy()

    frames = []

    for repeat_id in range(repeat_time):
        x = base.copy(deep=True)

        old_extra = x.iloc[0].get(
            "extra_info",
            {}
        )

        if not isinstance(old_extra, dict):
            old_extra = {}

        extra = deepcopy(old_extra)

        extra.update({
            "selector_version":
                "v2_semantic_no_box_excluded",

            "selector_repeat_id":
                repeat_id,

            "selector_repeat_time":
                repeat_time,

            "selector_score_v2":
                selected_meta[
                    "selector_score_v2"
                ],
        })

        x.at[
            x.index[0],
            "extra_info"
        ] = extra

        frames.append(x)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--profile-dir",
        required=True,
    )

    ap.add_argument(
        "--source-parquet",
        required=True,
    )

    ap.add_argument(
        "--outdir",
        required=True,
    )

    ap.add_argument(
        "--expected-n",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--repeat-time",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--exclude-row-index",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Original source row indices that have already been "
            "used for training and must not be selected again."
        ),
    )

    args = ap.parse_args()

    excluded_row_indices = set(
        int(x) for x in (args.exclude_row_index or [])
    )

    if excluded_row_indices:
        print(
            "exclude row indices:",
            sorted(excluded_row_indices),
        )

    profile_dir = Path(args.profile_dir)
    source_parquet = Path(args.source_parquet)
    outdir = Path(args.outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = load_profile_files(
        profile_dir
    )

    print(
        f"Profile JSON files = {len(files)}"
    )

    rows = []

    for f in files:

        obj = load_json(f)

        row_index = int(
            get_first(
                obj,
                [
                    "row_index",
                    "source_row_index",
                    "index",
                ],
            )
        )

        if row_index in excluded_row_indices:
            print(
                f"[EXCLUDE USED QUESTION] row_index={row_index} profile={f.name}"
            )
            continue


        question_id = str(
            get_first(
                obj,
                [
                    "question_id",
                    "qid",
                    "id",
                ],
                "",
            )
        )

        correct = int(
            get_first(
                obj,
                [
                    "correct_count",
                    "num_correct",
                ],
                0,
            )
        )

        wrong_value = get_first(
            obj,
            [
                "wrong_count",
                "incorrect_count",
                "num_wrong",
            ],
            None,
        )

        hist_raw = parse_histogram(
            get_first(
                obj,
                [
                    "answer_histogram",
                    "normalized_answer_histogram",
                ],
                {},
            )
        )

        hist_total = sum(
            int(v)
            for v in hist_raw.values()
        )

        # expected-n is the source of truth when given.
        n = args.expected_n

        if hist_total != n:
            raise RuntimeError(
                f"{f.name}: histogram total "
                f"{hist_total} != expected-n {n}"
            )

        wrong = (
            int(wrong_value)
            if wrong_value is not None
            else n - correct
        )

        if correct + wrong != n:
            raise RuntimeError(
                f"{f.name}: correct+wrong="
                f"{correct+wrong}, expected {n}"
            )

        p = correct / n

        reward_variance = (
            p * (1.0 - p)
        )

        raw_counter = Counter(
            {
                str(k): int(v)
                for k, v
                in hist_raw.items()
            }
        )

        raw_D = disagreement(
            raw_counter
        )

        sem_hist, format_fail_count = \
            semantic_histogram(
                hist_raw
            )

        parsed_count = sum(
            sem_hist.values()
        )

        if (
            parsed_count
            + format_fail_count
            != n
        ):
            raise RuntimeError(
                f"{f.name}: parsed "
                f"{parsed_count} + fail "
                f"{format_fail_count} != {n}"
            )

        parse_rate = (
            parsed_count / n
        )

        format_fail_rate = (
            format_fail_count / n
        )

        D_sem = disagreement(
            sem_hist
        )

        H_sem = entropy(
            sem_hist
        )

        # v1 equivalent audit score:
        score_v1_recomputed = (
            reward_variance
            * raw_D
        )

        # v2:
        #
        # reward uncertainty
        # × semantic output disagreement
        # × probability that output is parseable
        score_v2 = (
            reward_variance
            * D_sem
            * parse_rate
        )

        mixed_reward = (
            correct > 0
            and wrong > 0
        )

        rows.append({
            "row_index":
                row_index,

            "question_id":
                question_id,

            "n":
                n,

            "correct_count":
                correct,

            "wrong_count":
                wrong,

            "pass_rate":
                p,

            "reward_variance":
                reward_variance,

            # ----------------------------
            # RAW / selector-v1 audit
            # ----------------------------
            "raw_unique_answer_count":
                len(raw_counter),

            "raw_answer_disagreement":
                raw_D,

            "selector_score_v1_recomputed":
                score_v1_recomputed,

            "raw_answer_histogram":
                json.dumps(
                    dict(raw_counter),
                    ensure_ascii=False,
                    sort_keys=True,
                ),

            # ----------------------------
            # FORMAT
            # ----------------------------
            "parsed_count":
                parsed_count,

            "format_fail_count":
                format_fail_count,

            "parse_rate":
                parse_rate,

            "format_fail_rate":
                format_fail_rate,

            # ----------------------------
            # SEMANTIC
            # ----------------------------
            "semantic_unique_answer_count":
                len(sem_hist),

            "semantic_answer_disagreement":               D_sem,

            "semantic_answer_entropy":
                H_sem,

            "semantic_answer_histogram":
                json.dumps(
                    dict(sem_hist),
                    ensure_ascii=False,
                    sort_keys=True,
                ),

            # ----------------------------
            # V2
            # ----------------------------
            "selector_score_v2":
                score_v2,

            "distance_to_half":
                abs(p - 0.5),

            "mixed_reward":
                mixed_reward,

            "profile_json":
                str(f),
        })

    # Only mixed-reward questions are eligible.
    mixed = [
        x for x in rows
        if x["mixed_reward"]
    ]

    if not mixed:
        raise RuntimeError(
            "No mixed-reward questions found."
        )

    # v2 ranking
    mixed.sort(
        key=lambda x: (
            -x["selector_score_v2"],
            -x["reward_variance"],
            -x[
                "semantic_answer_disagreement"
            ],
            -x["parse_rate"],
            x["distance_to_half"],
            x["row_index"],
        )
    )

    all_rows = sorted(
        rows,
        key=lambda x: x["row_index"],
    )

    def write_csv(path, data):
        if not data:
            return

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            w = csv.DictWriter(
                f,
                fieldnames=list(
                    data[0].keys()
                ),
            )

            w.writeheader()
            w.writerows(data)

    write_csv(
        outdir / "selector_v2_all.csv",
        all_rows,
    )

    write_csv(
        outdir / "selector_v2_mixed.csv",
        mixed,
    )

    selected = deepcopy(
        mixed[0]
    )

    selected["selector_version"] = (
        "v2_semantic_no_box_excluded"
    )

    selected["selector_formula"] = (
        "p*(1-p)"
        " * D_semantic"
        " * parse_rate"
    )

    selected["parse_fail_policy"] = (
        "parse failures remain incorrect "
        "for reward/pass_rate, but are "
        "excluded from semantic diversity"
    )

    (
        outdir / "selected_v2.json"
    ).write_text(
        json.dumps(
            selected,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Build training parquet for future round.
    # Does NOT modify current R1.
    # ========================================================

    source_df = pd.read_parquet(
        source_parquet
    )

    idx = selected["row_index"]

    if int(idx) in excluded_row_indices:
        raise RuntimeError(
            f"Selector returned excluded row_index={idx}"
        )


    if not (
        0 <= idx < len(source_df)
    ):
        raise IndexError(
            f"selected row_index={idx}, "
            f"source rows={len(source_df)}"
        )

    single = source_df.iloc[
        [idx]
    ].copy()

    single.to_parquet(
        outdir / "selected_single.parquet",
        index=False,
    )

    repeat_df = build_repeat_dataframe(
        source_df=source_df,
        row_index=idx,
        repeat_time=args.repeat_time,
        selected_meta=selected,
    )

    repeat_df.to_parquet(
        outdir
        / f"selected_repeat{args.repeat_time}.parquet",
        index=False,
    )

    # ========================================================
    # Console audit
    # ========================================================

    print()
    print("=" * 118)
    print(
        "SELECTOR V2 — "
        "semantic disagreement only"
    )
    print("=" * 118)

    print(
        f"{'rank':>4} "
        f"{'row':>4} "
        f"{'correct':>8} "
        f"{'p':>7} "
        f"{'parse':>7} "
        f"{'D_raw':>8} "
        f"{'D_sem':>8} "
        f"{'score_v1':>10} "
        f"{'score_v2':>10} "
        f"question_id"
    )

    print("-" * 118)

    for rank, x in enumerate(
        mixed[:15],
        start=1,
    ):
        print(
            f"{rank:>4} "
            f"{x['row_index']:>4} "
            f"{x['correct_count']:>3}/"
            f"{x['n']:<4} "
            f"{x['pass_rate']:>7.4f} "
            f"{x['parse_rate']:>7.4f} "
            f"{x['raw_answer_disagreement']:>8.4f} "
            f"{x['semantic_answer_disagreement']:>8.4f} "
            f"{x['selector_score_v1_recomputed']:>10.6f} "
            f"{x['selector_score_v2']:>10.6f} "
            f"{x['question_id']}"
        )

    print()
    print("=" * 118)
    print("SELECTED V2")
    print("=" * 118)

    print(
        json.dumps(
            selected,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "OUTPUT:",
        outdir,
    )


if __name__ == "__main__":
    main()
