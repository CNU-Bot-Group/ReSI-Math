#!/usr/bin/env python3

import json
import os
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/root/autodl-tmp/self-judge")

B2 = ROOT / "imports/unpacked_2b/qwen3vl2b_solver_portable_20261007"
B4 = ROOT / "imports/unpacked_4b/qwen3vl4b_ga_solver_complete_20261007"

OUT = ROOT / "datasets/exact_budget33600"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_PROMPTS = 2100
ROLLOUT_N = 16
TARGET_ROLLOUTS = TARGET_PROMPTS * ROLLOUT_N

assert TARGET_ROLLOUTS == 33600


def get_qid(row):
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        return extra.get("question_id")
    return None


def fix_2b_images(row):
    """
    2B portable package stores all image files under:
      B2/images/<filename>
    """
    row = deepcopy(row)

    images = row.get("images")
    if not images:
        return row

    fixed = []

    for item in images:
        item = dict(item)
        old = item.get("image")

        if old:
            name = os.path.basename(old)
            new = B2 / "images" / name

            if not new.exists():
                raise FileNotFoundError(
                    f"2B image missing:\n"
                    f"old={old}\n"
                    f"expected={new}"
                )

            item["image"] = str(new)

        fixed.append(item)

    row["images"] = fixed
    return row


def fix_4b_images(row):
    """
    4B self-contained package JSONL stores paths like:
      images/train/xxx.png
    """
    row = deepcopy(row)

    images = row.get("images")
    if not images:
        return row

    fixed = []

    for item in images:
        item = dict(item)
        old = item.get("image")

        if old:
            p = Path(old)

            if p.is_absolute():
                # Prefer packaged copy by basename.
                candidate = B4 / "images/train" / p.name
            else:
                candidate = B4 / p

            if not candidate.exists():
                raise FileNotFoundError(
                    f"4B image missing:\n"
                    f"old={old}\n"
                    f"expected={candidate}"
                )

            item["image"] = str(candidate)

        fixed.append(item)

    row["images"] = fixed
    return row


def load_2b():
    path = B2 / "solver/train.parquet"

    rows = pq.read_table(path).to_pylist()

    if len(rows) != 69:
        raise RuntimeError(
            f"Expected 69 rows for 2B, got {len(rows)}"
        )

    rows = [fix_2b_images(x) for x in rows]

    return rows


def load_4b():
    path = B4 / "data/train.jsonl"

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if len(rows) != 32:
        raise RuntimeError(
            f"Expected 32 rows for 4B, got {len(rows)}"
        )

    rows = [fix_4b_images(x) for x in rows]

    return rows


def build_stream(rows, model_name):
    n = len(rows)

    stream = [
        deepcopy(rows[i % n])
        for i in range(TARGET_PROMPTS)
    ]

    assert len(stream) == TARGET_PROMPTS

    out_path = OUT / f"{model_name}_train_exact2100_n16_budget33600.parquet"

    table = pa.Table.from_pylist(stream)
    pq.write_table(
        table,
        out_path,
        compression="zstd"
    )

    full_cycles = TARGET_PROMPTS // n
    remainder = TARGET_PROMPTS % n
    global_steps_fullbatch = full_cycles + (1 if remainder else 0)

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    print("pool_size             =", n)
    print("target_prompt_slots   =", TARGET_PROMPTS)
    print("rollout_n             =", ROLLOUT_N)
    print("target_rollouts       =", TARGET_ROLLOUTS)

    print("full_cycles           =", full_cycles)
    print("final_partial_batch   =", remainder)
    print("fullbatch_steps       =", global_steps_fullbatch)

    print("fixed_seq_steps       =", TARGET_PROMPTS)

    print("output                =", out_path)

    print()
    print("first 5 question ids:")
    for x in stream[:5]:
        print(" ", get_qid(x))

    print()
    print("last 5 question ids:")
    for x in stream[-5:]:
        print(" ", get_qid(x))

    print()

    check = pq.read_table(out_path)

    assert check.num_rows == TARGET_PROMPTS

    print("PARQUET_VERIFY_ROWS   =", check.num_rows)
    print("STATUS                = OK")


def main():
    rows2 = load_2b()
    rows4 = load_4b()

    build_stream(rows2, "qwen3vl2b")
    build_stream(rows4, "qwen3vl4b")


if __name__ == "__main__":
    main()
