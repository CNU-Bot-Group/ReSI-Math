#!/usr/bin/env python3
import json
import random
import re
from pathlib import Path
from collections import defaultdict, Counter

SEED = 20260831
CAP = 4
MAX_PIXELS = 2007040

ROOT = Path("/root/autodl-tmp/self-judge")
RUN = ROOT / "datasets/evolution_v7_243source_20260811"

TRAJ = RUN / "experiment/exports/all_trajectories.jsonl"
STATES = RUN / "experiment/question_states"
WEMATH = ROOT / "datasets/We-Math2.0-Pro"

OUT = ROOT / "datasets/judge_rlvr_243source_v1/cap4"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)


def clean(x):
    return str(x or "").strip()


def walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)


def deep_first(obj, keys):
    keys = set(keys)
    for d in walk(obj):
        for k in keys:
            if k in d:
                v = d[k]
                if isinstance(v, str) and v.strip():
                    return v.strip()
                if v is not None and not isinstance(v, str):
                    return v
    return None


def family_id(qid):
    m = re.search(r"__(family_\d+)__", qid)
    return m.group(1) if m else qid


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


# ============================================================
# 1. Build image basename index
# ============================================================

print("=" * 80)
print("BUILD IMAGE INDEX")

image_index = defaultdict(list)

for root in [WEMATH, RUN]:
    print("SCAN:", root, "exists =", root.exists())

    if not root.exists():
        continue

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".webp", ".bmp"
        }:
            image_index[p.name].append(p.resolve())

print("unique image basenames =", len(image_index))


def raw_image_ref(v):
    if isinstance(v, str):
        return v.strip()

    if isinstance(v, dict):
        for k in ["path", "image", "image_path", "file"]:
            if isinstance(v.get(k), str) and v[k].strip():
                return v[k].strip()

    if isinstance(v, list) and v:
        return raw_image_ref(v[0])

    return ""


def resolve_image(v):
    ref = raw_image_ref(v)

    if not ref:
        return ""

    p = Path(ref)

    if p.is_absolute() and p.exists():
        return str(p.resolve())

    candidates = [
        WEMATH / ref,
        WEMATH / "converted_jsonl_v1" / ref,
        WEMATH / "converted_jsonl_v1" / "图片" / p.name,
    ]

    for c in candidates:
        if c.exists():
            return str(c.resolve())

    hits = image_index.get(p.name, [])

    if len(hits) == 1:
        return str(hits[0])

    wemath_hits = [x for x in hits if str(WEMATH) in str(x)]

    if len(wemath_hits) == 1:
        return str(wemath_hits[0])

    return ""


# ============================================================
# 2. Recover question + image metadata from question_states
# ============================================================

print()
print("=" * 80)
print("BUILD QUESTION METADATA")

qmeta = {}
bad_states = []

state_files = sorted(STATES.glob("*.json"))
print("state files =", len(state_files))

for fp in state_files:
    try:
        state = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        bad_states.append({
            "file": str(fp),
            "reason": "bad_json",
            "detail": str(e),
        })
        continue

    qid = clean(
        deep_first(
            state,
            ["question_id", "qid", "problem_id"]
        )
    )

    question = clean(
        deep_first(
            state,
            ["question", "problem"]
        )
    )

    image_raw = deep_first(
        state,
        [
            "image_path",
            "image_file",
            "source_image",
            "image",
        ]
    )

    image = resolve_image(image_raw)

    if not qid:
        bad_states.append({
            "file": str(fp),
            "reason": "missing_qid",
        })
        continue

    if not question:
        bad_states.append({
            "file": str(fp),
            "question_id": qid,
            "reason": "missing_question",
        })
        continue

    if not image:
        bad_states.append({
            "file": str(fp),
            "question_id": qid,
            "reason": "missing_image",
            "raw_image": raw_image_ref(image_raw),
        })
        continue

    qmeta[qid] = {
        "question": question,
        "image_path": image,
        "family_id": family_id(qid),
    }

print("usable question metadata =", len(qmeta))
print("bad states =", len(bad_states))


# ============================================================
# 3. Load trusted C/F trajectories
# ============================================================

print()
print("=" * 80)
print("LOAD TRAJECTORIES")

raw = read_jsonl(TRAJ)

groups = defaultdict(lambda: {"C": [], "F": []})
reject = Counter()

for x in raw:
    verifier = x.get("verifier") or {}
    truth = verifier.get("is_correct")
    policy = verifier.get("policy") or {}

    if truth not in (True, False):
        reject["unknown_truth"] += 1
        continue

    if policy.get("gt_status") != "trusted":
        reject["untrusted_gt"] += 1
        continue

    candidate = clean(
        x.get("raw_response") or x.get("reasoning")
    )

    if not candidate:
        reject["empty_candidate"] += 1
        continue

    y = dict(x)
    y["_candidate"] = candidate
    y["_truth"] = "C" if truth else "F"
    y["_label"] = "1" if truth else "0"

    groups[y["question_id"]][y["_truth"]].append(y)

print("raw rows =", len(raw))
print(
    "known rows =",
    sum(len(v["C"]) + len(v["F"]) for v in groups.values())
)
print("reject =", dict(reject))


# ============================================================
# 4. Exact dedupe + per-question cap4
# ============================================================

def dedup(items):
    seen = set()
    out = []

    for x in items:
        key = clean(x.get("content_hash"))

        if not key:
            key = re.sub(
                r"\s+",
                " ",
                x["_candidate"]
            ).strip()

        if key in seen:
            continue

        seen.add(key)
        out.append(x)

    return out


def sample_diverse(items, k):
    if len(items) <= k:
        return list(items)

    by_gen = defaultdict(list)

    for x in items:
        by_gen[x.get("generation")].append(x)

    for arr in by_gen.values():
        random.shuffle(arr)

    keep = []

    # Make g0/g1 both represented when possible.
    for g in sorted(by_gen, key=str):
        if len(keep) < k and by_gen[g]:
            keep.append(by_gen[g].pop())

    rest = []
    for arr in by_gen.values():
        rest.extend(arr)

    random.shuffle(rest)
    keep.extend(rest[:k - len(keep)])

    return keep


selected = []
qstats = []

for qid in sorted(groups):
    C = dedup(groups[qid]["C"])
    F = dedup(groups[qid]["F"])

    k = min(len(C), len(F), CAP)

    if k == 0:
        continue

    C = sample_diverse(C, k)
    F = sample_diverse(F, k)

    selected.extend(C)
    selected.extend(F)

    qstats.append({
        "question_id": qid,
        "family_id": family_id(qid),
        "k": k,
        "rows": 2 * k,
    })

random.shuffle(selected)

print()
print("=" * 80)
print("CAP4 RESULT")
print("questions =", len(qstats))
print("rows =", len(selected))
print("truth =", Counter(x["_truth"] for x in selected))
print("generation =", Counter(x.get("generation") for x in selected))


# ============================================================
# 5. Exact JudgeRLVR judging prompt
# ============================================================

def make_prompt(question, candidate):
    return (
        "<image>\n"
        "Given the question:\n"
        f"{question}\n"
        "And the proposed answer:\n"
        f"{candidate}\n"
        "Analyze concisely whether the answer is correct. "
        "After your analysis, put only your final judgment "
        "in \\boxed{} as 1 for correct and 0 for incorrect."
    )


# ============================================================
# 6. Join metadata and create VERL records
# ============================================================

canonical = []
verl = []
missing = []

for x in selected:
    qid = x["question_id"]

    if qid not in qmeta:
        missing.append({
            "question_id": qid,
            "trajectory_id": x.get("trajectory_id"),
            "reason": "missing_question_metadata",
        })
        continue

    m = qmeta[qid]

    image = m["image_path"]

    if not Path(image).exists():
        missing.append({
            "question_id": qid,
            "trajectory_id": x.get("trajectory_id"),
            "reason": "image_not_found",
            "image": image,
        })
        continue

    candidate = x["_candidate"]
    question = m["question"]
    label = x["_label"]

    sample_id = f"{qid}::{x['trajectory_id']}"

    prompt = make_prompt(
        question,
        candidate,
    )

    canonical.append({
        "sample_id": sample_id,
        "question_id": qid,
        "family_id": m["family_id"],
        "trajectory_id": x["trajectory_id"],
        "generation": x.get("generation"),
        "question": question,
        "image_path": image,
        "candidate_solution": candidate,
        "outcome_truth": x["_truth"],
        "expected_verdict": label,
        "content_hash": x.get("content_hash"),
        "prompt_text": prompt,
    })

    verl.append({
        "data_source": "judge_rlvr_v8",

        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],

        "images": [
            {
                "image": image,
                "max_pixels": MAX_PIXELS,
            }
        ],

        "ability": "judge",

        "reward_model": {
            "style": "rule",
            "ground_truth": label,
        },

        "extra_info": {
            "task": "judge_rlvr",
            "sample_id": sample_id,
            "question_id": qid,
            "family_id": m["family_id"],
            "trajectory_id": x["trajectory_id"],
            "generation": x.get("generation"),
            "outcome_truth": x["_truth"],
            "expected_verdict": label,
            "reward_verifier": "judge_binary_boxed",
            "label_source": "v8_deterministic_verifier",
            "source_pool": "v8_shard01_cap4",
        },
    })


# ============================================================
# 7. Save JSONL
# ============================================================

write_jsonl(
    OUT / "judge_rlvr_cap4.jsonl",
    canonical
)

write_jsonl(
    OUT / "question_balance_stats.jsonl",
    qstats
)

write_jsonl(
    OUT / "missing_metadata.jsonl",
    missing
)

write_jsonl(
    OUT / "bad_question_states.jsonl",
    bad_states
)


# ============================================================
# 8. Strict validation
# ============================================================

truth_counts = Counter(
    x["outcome_truth"]
    for x in canonical
)

label_counts = Counter(
    x["expected_verdict"]
    for x in canonical
)

questions = {
    x["question_id"]
    for x in canonical
}

families = { x["family_id"]
    for x in canonical
}

generation_counts = Counter(
    x["generation"]
    for x in canonical
)

print()
print("=" * 80)
print("STRICT VALIDATION")
print("canonical rows =", len(canonical))
print("verl rows =", len(verl))
print("questions =", len(questions))
print("families =", len(families))
print("truth =", truth_counts)
print("labels =", label_counts)
print("generation =", generation_counts)
print("missing metadata =", len(missing))

if missing:
    raise RuntimeError(
        "Missing question/image metadata. DO NOT TRAIN."
    )

assert len(canonical) == len(verl)
assert truth_counts["C"] == truth_counts["F"]
assert label_counts["1"] == label_counts["0"]

# Generic structural validation for arbitrary source pools.
assert len(canonical) > 0
assert len(canonical) == len(verl), (
    len(canonical), len(verl)
)
assert sum(truth_counts.values()) == len(canonical), truth_counts
assert sum(label_counts.values()) == len(canonical), label_counts
assert truth_counts["C"] == truth_counts["F"], truth_counts
assert label_counts["1"] == label_counts["0"], label_counts

print("[PASS] strict validation passed")


# ============================================================
# 9. Save VERL parquet
# ============================================================

from datasets import Dataset

PARQUET = OUT / "judge_rlvr_cap4.parquet"

Dataset.from_list(
    verl
).to_parquet(
    str(PARQUET)
)


summary = {
    "rows": len(canonical),
    "questions": len(questions),
    "families": len(families),
    "truth_counts": dict(truth_counts),
    "label_counts": dict(label_counts),
    "generation_counts": dict(generation_counts),
    "cap_per_class": CAP,
    "seed": SEED,
    "missing_metadata": len(missing),
    "prompt_style": "JudgeRLVR_Appendix_A",
    "reward_verifier": "judge_binary_boxed",
    "jsonl": str(OUT / "judge_rlvr_cap4.jsonl"),
    "parquet": str(PARQUET),
}

(OUT / "summary.json").write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 80)
print("FIRST SAMPLE")
print(json.dumps(
    canonical[0],
    ensure_ascii=False,
    indent=2,
)[:3500])

print()
print("=" * 80)
print("DONE")
print("JSONL  =", OUT / "judge_rlvr_cap4.jsonl")
print("PARQUET=", PARQUET)
print("SUMMARY=", OUT / "summary.json")
