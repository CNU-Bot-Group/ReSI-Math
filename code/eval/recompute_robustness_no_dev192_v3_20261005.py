#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path

ROOT = Path("/root/autodl-tmp/self-judge")
RES  = ROOT / "benchmark-results"

OUT = ROOT / "reports/robustness_no_dev192_20261005/v3"
OUT.mkdir(parents=True, exist_ok=True)

MATH_DEV = (
    ROOT
    / "benchmark/solver-benchmarks/math500_splits/"
      "math500_dev96_seed20260901.jsonl"
)

WEMATH_FULL = (
    ROOT
    / "benchmark/We-Math/data/testmini.json"
)

WEMATH_DEV = (
    ROOT
    / "benchmark/We-Math/data/testmini_mini96.json"
)

MODELS = {
    "Base": {
        "math": (
            RES
            / "solver-oneshot-r1-dev192/full/base/"
              "math500/generations.jsonl"
        ),
        "wemath": (
            RES
            / "wemath-vllm-final6/base.jsonl"
        ),
        "math500": 85.80,
        "wemath1740": 71.61,
    },

    "Q1": {
        "math": (
            RES
            / "solver-oneshot-r1-dev192/full/step500/"
              "math500/generations.jsonl"
        ),
        "wemath": (
            RES
            / "solver-oneshot-r1-dev192/full/step500/"
              "wemath1740/parts/part_0.jsonl"
        ),
        "math500": 86.80,
        "wemath1740": 71.95,
    },

    "Q2": {
        "math": (
            RES
            / "sequential-one-shot-auto/q2/full/step500/"
              "math500/generations.jsonl"
        ),
        "wemath": (
            RES
            / "sequential-one-shot-auto/q2/full/step500/"
              "wemath1740/parts/part_0.jsonl"
        ),
        "math500": 86.00,
        "wemath1740": 73.05,
    },

    "Q3": {
        "math": (
            RES
            / "q3-selectorv2-row12-700-20260907/full/step650/"
              "math500/generations.jsonl"
        ),
        "wemath": (
            RES
            / "q3-selectorv2-row12-700-20260907/full/step650/"
              "wemath1740/parts/part_0.jsonl"
        ),
        "math500": 86.80,
        "wemath1740": 73.51,
    },

    "Fixed-Seq": {
        "math": (
            RES
            / "solver28-full700-20260924/full/step250/"
              "math500/generations.jsonl"
        ),
        "wemath": (
            RES
            / "solver28-full700-20260924/full/step250/"
              "wemath1740/parts/part_0.jsonl"
        ),
        "math500": 86.80,
        "wemath1740": 72.13,
    },

    "Full-Batch": {
        "math": (
            RES
            / "solver28-fullbatch28-n16-budget33600-20260927/"
              "full/step20/math500/generations.jsonl"
        ),
        "wemath": (
            RES
            / "solver28-fullbatch28-n16-budget33600-20260927/"
              "full/step20/wemath1740/parts/part_0.jsonl"
        ),
        "math500": 85.40,
        "wemath1740": 71.44,
    },
}


def read_jsonl(path):
    path = Path(path)
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


# ============================================================
# Exact official WE-MATH response parser
# copied from merge_score_wemath.py
# ============================================================

def official_extract_letter(response):
    if response is None:
        return None

    tail = str(response).split("Answer")[-1].strip()

    tail = re.sub(
        r"[>><<:.]",
        "",
        tail,
    ).strip()

    if tail and tail[0] in "ABCDEFGH":
        return tail[0]

    return None


def wemath_correct(row):
    pred = official_extract_letter(
        str(row.get("response", "") or "")
    )

    gold = str(
        row.get("answer", "") or ""
    ).strip()

    return pred == gold


# ============================================================
# Stable WE-MATH identity
# ============================================================

IDENTITY_FIELDS = (
    "ID",
    "key",
    "question",
    "option",
    "answer",
    "image_path",
    "question number",
)


def wm_identity(row):
    return tuple(
        str(row.get(k, "")).strip()
        for k in IDENTITY_FIELDS
    )


# ============================================================
# Load frozen DEV96 definitions
# ============================================================

math_dev_rows = read_jsonl(MATH_DEV)

if len(math_dev_rows) != 96:
    raise RuntimeError(
        f"MATH DEV expected 96, "
        f"got {len(math_dev_rows)}"
    )

math_dev_ids = {
    str(r["id"])
    for r in math_dev_rows
}

if len(math_dev_ids) != 96:
    raise RuntimeError(
        f"MATH DEV unique IDs = "
        f"{len(math_dev_ids)}"
    )


wm_full_dataset = read_json(WEMATH_FULL)
wm_dev_dataset = read_json(WEMATH_DEV)

if len(wm_full_dataset) != 1740:
    raise RuntimeError(
        f"WE-MATH full expected 1740, "
        f"got {len(wm_full_dataset)}"
    )

if len(wm_dev_dataset) != 96:
    raise RuntimeError(
        f"WE-MATH DEV expected 96, "
        f"got {len(wm_dev_dataset)}"
    )


# ============================================================
# Map frozen Mini96 back to original full-testmini indices
# ============================================================

full_identity_to_indices = {}

for idx, row in enumerate(wm_full_dataset):
    key = wm_identity(row)

    full_identity_to_indices.setdefault(
        key,
        [],
    ).append(idx)


duplicate_full_keys = {
    k: v
    for k, v in full_identity_to_indices.items()
    if len(v) != 1
}

if duplicate_full_keys:
    raise RuntimeError(
        "WE-MATH identity is not unique in "
        f"full dataset: {len(duplicate_full_keys)} "
        "duplicate keys."
    )


wm_dev_indices = set()

for row in wm_dev_dataset:
    key = wm_identity(row)

    matches = full_identity_to_indices.get(
        key,
        [],
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Cannot uniquely map Mini96 row:\n"
            + json.dumps(
                row,
                ensure_ascii=False,
                indent=2,
            )[:2000]
        )

    wm_dev_indices.add(
        matches[0]
    )


if len(wm_dev_indices) != 96:
    raise RuntimeError(
        f"Mapped WE-MATH DEV indices="
        f"{len(wm_dev_indices)}, expected 96"
    )


print(
    "MATH DEV96 IDs       =",
    len(math_dev_ids),
)

print(
    "WE-MATH DEV96 indices =",
    len(wm_dev_indices),
)


# ============================================================
# Evaluate each model
# ============================================================

results = []

for name, cfg in MODELS.items():

    print()
    print("=" * 90)
    print(name)
    print("=" * 90)

    mp = cfg["math"]
    wp = cfg["wemath"]

    if not mp.exists():
        raise FileNotFoundError(mp)

    if not wp.exists():
        raise FileNotFoundError(wp)


    # --------------------------------------------------------
    # MATH500
    # --------------------------------------------------------

    mrows = read_jsonl(mp)

    if len(mrows) != 500:
        raise RuntimeError(
            f"{name}: MATH rows="
            f"{len(mrows)}, expected 500"
        )

    math_full_correct = sum(
        bool(r.get("correct"))
        for r in mrows
    )

    math_full_acc = (
        100.0 *
        math_full_correct /
        len(mrows)
    )

    if round(math_full_acc, 2) != cfg["math500"]:
        raise RuntimeError(
            f"{name}: MATH500 mismatch. "
            f"raw={math_full_acc:.2f}, "
            f"expected={cfg['math500']:.2f}"
        )

    math_dev_in_full = [
        r
        for r in mrows
        if str(r.get("id")) in math_dev_ids
    ]

    if len(math_dev_in_full) != 96:
        raise RuntimeError(
            f"{name}: MATH DEV matches="
            f"{len(math_dev_in_full)}, "
            "expected 96"
        )

    math404_rows = [
        r
        for r in mrows
        if str(r.get("id"))
        not in math_dev_ids
    ]

    if len(math404_rows) != 404:
        raise RuntimeError(
            f"{name}: MATH404 rows="
            f"{len(math404_rows)}"
        )

    math404_correct = sum(
        bool(r.get("correct"))
        for r in math404_rows
    )

    math404_acc = (
        100.0 *
        math404_correct /
        404
    )


    # --------------------------------------------------------
    # WE-MATH1740
    # --------------------------------------------------------

    wrows = read_jsonl(wp)

    if len(wrows) != 1740:
        raise RuntimeError(
            f"{name}: WE-MATH rows="
            f"{len(wrows)}, expected 1740"
        )

    # Every raw prediction should point back to the
    # original official dataset index.
    pred_indices = {
        int(r["_dataset_index"])
        for r in wrows
    }

    if len(pred_indices) != 1740:
        raise RuntimeError(
            f"{name}: unique _dataset_index="
            f"{len(pred_indices)}, expected 1740"
        )

    if pred_indices != set(range(1740)):
        missing = sorted(
            set(range(1740)) -
            pred_indices
        )

        extra = sorted(
            pred_indices -
            set(range(1740))
        )

        raise RuntimeError(
            f"{name}: WE-MATH index mismatch. "
            f"missing={missing[:10]} "
            f"extra={extra[:10]}"
        )

    # Re-sort exactly into canonical original order.
    by_idx = {
        int(r["_dataset_index"]): r
        for r in wrows
    }

    wrows = [
        by_idx[i]
        for i in range(1740)
    ]

    wemath_full_correct = sum(
        wemath_correct(r)
        for r in wrows
    )

    wemath_full_acc = (
        100.0 *
        wemath_full_correct /
        1740
    )

    # Critical integrity test:
    # reproduce published main-table result from raw responses.
    if round(wemath_full_acc, 2) != cfg["wemath1740"]:
        raise RuntimeError(
            f"{name}: WE-MATH full mismatch. "
            f"official-parser raw="
            f"{wemath_full_acc:.4f}%, "
            f"expected={cfg['wemath1740']:.2f}%"
        )


    # --------------------------------------------------------
    # Remove exact frozen WE-MATH DEV96
    # --------------------------------------------------------

    wemath_dev_rows = [
        r
        for i, r in enumerate(wrows)
        if i in wm_dev_indices
    ]

    wemath1644_rows = [
        r
        for i, r in enumerate(wrows)
        if i not in wm_dev_indices
    ]

    if len(wemath_dev_rows) != 96:
        raise RuntimeError(
            f"{name}: WE DEV rows="
            f"{len(wemath_dev_rows)}"
        )

    if len(wemath1644_rows) != 1644:
        raise RuntimeError(
            f"{name}: WE1644 rows="
            f"{len(wemath1644_rows)}"
        )

    wemath1644_correct = sum(
        wemath_correct(r)
        for r in wemath1644_rows
    )

    wemath1644_acc = (
        100.0 *
        wemath1644_correct /
        1644
    )


    # --------------------------------------------------------
    # Final averages
    # --------------------------------------------------------

    full_avg = (
        cfg["math500"] +
        cfg["wemath1740"]
    ) / 2.0

    avg_no_dev = (
        math404_acc +
        wemath1644_acc
    ) / 2.0


    math_dev_correct = (
        math_full_correct -
        math404_correct
    )

    wemath_dev_correct = (
        wemath_full_correct -
        wemath1644_correct
    )


    result = {
        "Model": name,

        "MATH500": cfg["math500"],

        "MATH404_correct":
            math404_correct,

        "MATH404_w/o_DEV":
            math404_acc,

        "MATH_DEV96_correct":
            math_dev_correct,

        "WE-MATH_testmini":
            cfg["wemath1740"],

        "WE-MATH1644_correct":
            wemath1644_correct,

        "WE-MATH1644_w/o_DEV":
            wemath1644_acc,

        "WE-MATH_DEV96_correct":
            wemath_dev_correct,

        "Full_Avg":
            full_avg,

        "Avg_w/o_DEV":
            avg_no_dev,
    }

    results.append(result)

    print(
        f"MATH500   = "
        f"{math_full_correct}/500 "
        f"= {math_full_acc:.2f}%"
    )

    print(
        f"MATH DEV96 = "
        f"{math_dev_correct}/96"
    )

    print(
        f"MATH404    = "
        f"{math404_correct}/404 "
        f"= {math404_acc:.4f}%"
    )

    print(
        f"WE1740     = "
        f"{wemath_full_correct}/1740 "
        f"= {wemath_full_acc:.4f}%"
    )

    print(
        f"WE DEV96   = "
        f"{wemath_dev_correct}/96"
    )

    print(
        f"WE1644     = "
        f"{wemath1644_correct}/1644 "
        f"= {wemath1644_acc:.4f}%"
    )


# ============================================================
# Final Table 7
# ============================================================

print()
print("=" * 110)
print(
    "TABLE 7 — ROBUSTNESS AFTER REMOVING DEV192"
)
print("=" * 110)

header = (
    f"{'Model':12s}"
    f"{'MATH500':>10s}"
    f"{'MATH404':>11s}"
    f"{'WE1740':>10s}"
    f"{'WE1644':>11s}"
    f"{'FullAvg':>11s}"
    f"{'NoDEVAvg':>12s}"
)

print(header)
print("-" * len(header))

for r in results:
    print(
        f"{r['Model']:12s}"
        f"{r['MATH500']:10.2f}"
        f"{r['MATH404_w/o_DEV']:11.2f}"
        f"{r['WE-MATH_testmini']:10.2f}"
        f"{r['WE-MATH1644_w/o_DEV']:11.2f}"
        f"{r['Full_Avg']:11.3f}"
        f"{r['Avg_w/o_DEV']:12.3f}"
    )


# ============================================================
# Save CSV
# ============================================================

csv_path = OUT / "table7_robustness_no_dev192.csv"

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys()),
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# Save LaTeX rows
# ============================================================

tex_path = OUT / "table7_rows.tex"

with tex_path.open(
    "w",
    encoding="utf-8",
) as f:

    for r in results:

        f.write(
            f"{r['Model']} & "
            f"{r['MATH500']:.2f} & "
            f"{r['MATH404_w/o_DEV']:.2f} & "
            f"{r['WE-MATH_testmini']:.2f} & "
            f"{r['WE-MATH1644_w/o_DEV']:.2f} & "
            f"{r['Full_Avg']:.3f} & "
            f"{r['Avg_w/o_DEV']:.3f} "
            "\\\\\n"
        )


print()
print(
    "CSV   =",
    csv_path,
)

print(
    "LATEX =",
    tex_path,
)
