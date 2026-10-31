#!/usr/bin/env python3

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# ============================================================
# Utils
# ============================================================

def atomic_json_dump(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def to_py(x):
    """Convert numpy/object containers recursively to plain Python."""
    if hasattr(x, "tolist"):
        x = x.tolist()

    if isinstance(x, dict):
        return {str(k): to_py(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [to_py(v) for v in x]

    return x


# ============================================================
# Boxed answer extraction
# ============================================================

def extract_last_boxed(text: str):
    """
    Brace-aware extraction of the last \\boxed{...}.
    """
    if not text:
        return None

    marker = r"\boxed{"
    starts = []

    pos = 0
    while True:
        i = text.find(marker, pos)
        if i < 0:
            break
        starts.append(i)
        pos = i + len(marker)

    if not starts:
        return None

    # Try from the last boxed backwards.
    for start in reversed(starts):
        i = start + len(marker)
        depth = 1
        j = i

        while j < len(text):
            c = text[j]

            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1

                if depth == 0:
                    return text[i:j].strip()

            j += 1

    return None


def normalize_answer(ans):
    """
    Used ONLY for output-disagreement clustering.
    Correctness itself is determined by the training reward function.
    """
    if ans is None:
        return "<NO_BOX>"

    s = str(ans).strip()

    # common LaTeX wrappers
    s = s.replace("$", "")
    s = s.replace(r"\(", "")
    s = s.replace(r"\)", "")
    s = s.replace(r"\[", "")
    s = s.replace(r"\]", "")

    # Normalize semantically identical choice wrappers:
    # \text{A}, \mathrm{A}, \mathbf{A} -> A
    wrapper_patterns = [
        r"\\text\{([^{}]+)\}",
        r"\\mathrm\{([^{}]+)\}",
        r"\\mathbf\{([^{}]+)\}",
        r"\\operatorname\{([^{}]+)\}",
    ]

    for pat in wrapper_patterns:
        m = re.fullmatch(pat, s)
        if m:
            s = m.group(1).strip()
            break

    # remove spaces
    s = re.sub(r"\s+", "", s)

    # normalize some common forms
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\tfrac", r"\frac")

    # choice answers
    m = re.fullmatch(r"[\(\[]?([A-Ha-h])[\)\].]?", s)
    if m:
        return m.group(1).upper()

    # normalize final punctuation
    s = s.rstrip(".,;:")

    return s if s else "<EMPTY_BOX>"


# ============================================================
# Load the EXACT Solver-RLVR reward function
# ============================================================

def load_reward_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "solver_reward_module",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reward module: {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "compute_score"):
        raise RuntimeError(
            f"{path} has no compute_score() function"
        )

    return mod.compute_score


def call_reward(
    reward_fn,
    response,
    ground_truth,
    extra_info,
    reward_model,
):
    """
    Adapt to the installed reward function signature instead of
    hard-coding one assumed signature.
    """

    sig = inspect.signature(reward_fn)

    values = {
        "solution_str": response,
        "response": response,
        "output": response,
        "prediction": response,

        "ground_truth": ground_truth,
        "gt": ground_truth,
        "reference": ground_truth,
        "answer": ground_truth,

        "extra_info": extra_info,
        "reward_model": reward_model,
        "data_source": "math",
    }

    kwargs = {}

    missing = []

    for name, param in sig.parameters.items():

        # *args / **kwargs are never required named arguments.
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name in values:
            kwargs[name] = values[name]

        elif param.default is inspect.Parameter.empty:
            missing.append(name)

    if missing:
        raise RuntimeError(
            "Cannot satisfy compute_score signature. "
            f"Signature={sig}, missing required args={missing}"
        )

    result = reward_fn(**kwargs)

    if isinstance(result, dict):
        for key in [
            "score",
            "reward",
            "value",
            "accuracy",
        ]:
            if key in result:
                result = result[key]
                break

    try:
        return float(result)
    except Exception as e:
        raise RuntimeError(
            f"Unexpected compute_score result: {result!r}"
        ) from e


# ============================================================
# Qwen3-VL prompt construction
# ============================================================

def build_vllm_input(processor, row):
    prompt_obj = to_py(row["prompt"])
    images_obj = to_py(row["images"])

    if not isinstance(prompt_obj, list) or not prompt_obj:
        raise ValueError(f"Bad prompt: {prompt_obj!r}")

    if not isinstance(images_obj, list) or not images_obj:
        raise ValueError(f"Bad images: {images_obj!r}")

    # Current dataset is one-image-per-question.
    image_path = images_obj[0]["image"]

    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    # Extract original user text.
    user_msg = prompt_obj[0]

    text = str(user_msg["content"])

    # Dataset currently stores literal <image>.
    # Let Qwen chat template insert its own vision placeholder.
    text = re.sub(
        r"^\s*<image>\s*",
        "",
        text,
        count=1,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": text,
                },
            ],
        }
    ]

    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image = Image.open(image_path).convert("RGB")

    return {
        "prompt": rendered,
        "multi_modal_data": {
            "image": image,
        },
    }, image_path, text


# ============================================================
# Statistics
# ============================================================

def gini_disagreement(values):
    """
    1 - sum p_i^2
    """
    n = len(values)

    if n == 0:
        return 0.0

    counts = Counter(values)

    return 1.0 - sum(
        (c / n) ** 2
        for c in counts.values()
    )


def entropy(values):
    n = len(values)

    if n == 0:
        return 0.0

    counts = Counter(values)

    h = 0.0

    for c in counts.values():
        p = c / n
        h -= p * math.log(p + 1e-12)

    return h


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        default=(
            "/root/autodl-tmp/self-judge/"
            "datasets/rlvr_pilot_v1/solver/"
            "train_wemath_official_gt.parquet"
        ),
    )

    ap.add_argument(
        "--model",
        default=(
            "/root/autodl-tmp/self-judge/"
            "models/Qwen3-VL-8B-Instruct"
        ),
    )

    ap.add_argument(
        "--reward-file",
        default=(
            "/root/autodl-tmp/self-judge/"
            "rlvr/reward_solver_judge_v1.py"
        ),
    )

    ap.add_argument(
        "--outdir",
        default=(
            "/root/autodl-tmp/self-judge/"
            "outputs/solver28_profile/"
            "base_n16_seed20260901"
        ),
    )

    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=20260901)

    ap.add_argument(
        "--overwrite",
        action="store_true",
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only profile first N questions; 0 means all.",
    )

    ap.add_argument(
        "--adapter",
        default="",
        help="Optional PEFT LoRA adapter directory.",
    )

    ap.add_argument(
        "--max-lora-rank",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--row-start",
        type=int,
        default=None,
        help="Original dataframe row start, inclusive.",
    )

    ap.add_argument(
        "--row-end",
        type=int,
        default=None,
        help="Original dataframe row end, exclusive.",
    )

    args = ap.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)

    adapter_path = None
    if args.adapter:
        adapter_path = Path(args.adapter)

        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"Missing adapter_config.json: {adapter_path}"
            )

        if not (adapter_path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(
                f"Missing adapter_model.safetensors: {adapter_path}"
            )

        print("adapter   :", adapter_path)
    reward_path = Path(args.reward_file)
    outdir = Path(args.outdir)

    perq_dir = outdir / "per_question"

    outdir.mkdir(parents=True, exist_ok=True)
    perq_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("SOLVER28 BASE N16 PROFILING")
    print("=" * 90)

    print("data      :", data_path)
    print("model     :", model_path)
    print("reward    :", reward_path)
    print("outdir    :", outdir)
    print("n         :", args.n)
    print("temperature:", args.temperature)
    print("top_p     :", args.top_p)
    print("seed      :", args.seed)

    df = pd.read_parquet(data_path)

    if len(df) != 28:
        raise RuntimeError(
            f"Expected 28 rows, got {len(df)}"
        )

    print("\nrows =", len(df))

    if args.limit > 0:
        df = df.iloc[:args.limit]
        print("profiling limit =", len(df))


    if args.row_start is not None or args.row_end is not None:
        start = 0 if args.row_start is None else args.row_start
        end = len(df) if args.row_end is None else args.row_end

        if not (0 <= start <= end <= len(df)):
            raise ValueError(
                f"Invalid row range [{start}, {end}) for {len(df)} rows"
            )

        # iloc preserves the original dataframe indices.
        # This is important because selector row_index must remain 0..27.
        df = df.iloc[start:end]

        print(
            f"profiling original rows [{start}, {end}) -> {len(df)} questions"
        )
    # --------------------------------------------------------
    # Validate reward function
    # --------------------------------------------------------

    reward_fn = load_reward_module(reward_path)

    print(
        "reward signature =",
        inspect.signature(reward_fn),
    )

    # --------------------------------------------------------
    # Preflight image paths
    # --------------------------------------------------------

    missing_images = []

    for _, row in df.iterrows():
        imgs = to_py(row["images"])

        for item in imgs:
            p = item.get("image")

            if not p or not os.path.isfile(p):
                missing_images.append(p)

    if missing_images:
        print("\nMISSING IMAGES:")
        for x in missing_images:
            print(x)

        raise RuntimeError(
            f"{len(missing_images)} image paths missing"
        )

    print("[PASS] all image paths exist")

    # --------------------------------------------------------
    # Processor / vLLM
    # --------------------------------------------------------

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    print("Loading vLLM...")

    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        limit_mm_per_prompt={
            "image": 1,
        },
        enable_lora=bool(adapter_path),
        max_lora_rank=args.max_lora_rank,
    )

    lora_request = None

    if adapter_path is not None:
        lora_request = LoRARequest(
            "solver_policy_adapter",
            1,
            str(adapter_path),
        )


    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # One question at a time:
    # n=16 means 16 simultaneous generations for one question.
    # Safer for 1×A800 and gives natural resumability.
    # --------------------------------------------------------

    for row_idx, row in df.iterrows():

        extra = to_py(row["extra_info"])
        reward_model = to_py(row["reward_model"])

        qid = str(
            extra.get(
                "question_id",
                f"row_{row_idx}",
            )
        )

        safe_qid = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            qid,
        )

        save_path = (
            perq_dir
            / f"{row_idx:02d}__{safe_qid}.json"
        )

        if save_path.exists() and not args.overwrite:
            try:
                old = json.loads(
                    save_path.read_text(
                        encoding="utf-8"
                    )
                )

                if (
                    old.get("n_requested") == args.n
                    and len(old.get("rollouts", [])) == args.n
                ):
                    print(
                        f"[SKIP] {row_idx:02d} {qid} "
                        f"already has {args.n} rollouts"
                    )
                    continue

            except Exception:
                pass

        gt = str(
            reward_model.get("ground_truth", "")
        )

        print()
        print("=" * 90)
        print(
            f"[{row_idx + 1:02d}/28] {qid}"
        )
        print("GT:", gt)
        print("=" * 90)

        vllm_input, image_path, question_text = (
            build_vllm_input(
                processor,
                row,
            )
        )

        outputs = llm.generate(
            [vllm_input],
            sampling_params=sampling,
            use_tqdm=True,
            lora_request=lora_request,
        )

        if len(outputs) != 1:
            raise RuntimeError(
                f"Expected 1 request output, got {len(outputs)}"
            )

        gens = outputs[0].outputs

        if len(gens) != args.n:
            raise RuntimeError(
                f"{qid}: expected {args.n} generations, "
                f"got {len(gens)}"
            )

        rollouts = []

        for j, gen in enumerate(gens):

            text = gen.text

            boxed = extract_last_boxed(text)
            norm = normalize_answer(boxed)

            reward = call_reward(
                reward_fn=reward_fn,
                response=text,
                ground_truth=gt,
                extra_info=extra,
                reward_model=reward_model,
            )

            rollouts.append(
                {
                    "rollout_id": j,
                    "reward": reward,
                    "is_correct": bool(reward > 0.5),
                    "boxed_answer": boxed,
                    "normalized_answer": norm,
                    "response": text,
                    "finish_reason": str(
                        getattr(
                            gen,
                            "finish_reason",
                            None,
                        )
                    ),
                }
            )

        correct = sum(
            1
            for x in rollouts
            if x["is_correct"]
        )

        p = correct / args.n
        reward_var = p * (1.0 - p)

        answers = [
            x["normalized_answer"]
            for x in rollouts
        ]

        disagreement = gini_disagreement(
            answers
        )

        answer_entropy = entropy(
            answers
        )

        selector_score = (
            reward_var
            * disagreement
        )

        hist = dict(
            Counter(answers).most_common()
        )

        result = {
            "row_index": int(row_idx),
            "question_id": qid,
            "question": extra.get(
                "question",
                question_text,
            ),
            "image": image_path,

            "ground_truth": gt,
            "reward_verifier": extra.get(
                "reward_verifier"
            ),

            "n_requested": args.n,

            "correct_count": correct,
            "wrong_count": args.n - correct,
            "pass_rate": p,
            "reward_variance": reward_var,

            "unique_answer_count": len(
                set(answers)
            ),
            "answer_disagreement": disagreement,
            "answer_entropy": answer_entropy,

            "selector_score": selector_score,

            "answer_histogram": hist,

            "rollouts": rollouts,
        }

        atomic_json_dump(
            result,
            save_path,
        )

        print(
            f"C/F={correct}/{args.n-correct} "
            f"p={p:.4f} "
            f"Var={reward_var:.4f} "
            f"Dout={disagreement:.4f} "
            f"Score={selector_score:.4f}"
        )

        print(
            "answers:",
            hist,
        )

    # ========================================================
    # Rebuild ranking from all completed per-question files.
    # ========================================================

    rows = []

    for pth in sorted(perq_dir.glob("*.json")):

        x = json.loads(
            pth.read_text(
                encoding="utf-8"
            )
        )

        if len(x.get("rollouts", [])) != args.n:
            continue

        rows.append(x)

    if len(rows) != 28:
        print(
            f"\nWARNING: only {len(rows)}/28 "
            "questions are complete."
        )

    # High composite first.
    # Then high reward variance.
    # Then high disagreement.
    rows.sort(
        key=lambda x: (
            -x["selector_score"],
            -x["reward_variance"],
            -x["answer_disagreement"],
            abs(x["pass_rate"] - 0.5),
            x["row_index"],
        )
    )

    ranking = []

    for rank, x in enumerate(rows, 1):

        ranking.append(
            {
                "rank": rank,

                "row_index": x["row_index"],
                "question_id": x["question_id"],

                "correct_count": x["correct_count"],
                "wrong_count": x["wrong_count"],

                "pass_rate": x["pass_rate"],
                "reward_variance": x["reward_variance"],

                "unique_answer_count": x[
                    "unique_answer_count"
                ],

                "answer_disagreement": x[
                    "answer_disagreement"
                ],

                "answer_entropy": x[
                    "answer_entropy"
                ],

                "selector_score": x[
                    "selector_score"
                ],

                "ground_truth": x["ground_truth"],
                "reward_verifier": x[
                    "reward_verifier"
                ],

                "answer_histogram": json.dumps(
                    x["answer_histogram"],
                    ensure_ascii=False,
                ),

                "question": x["question"],
                "image": x["image"],
            }
        )

    rank_df = pd.DataFrame(ranking)

    csv_path = outdir / "ranking.csv"
    json_path = outdir / "ranking.json"
    md_path = outdir / "ranking.md"

    rank_df.to_csv(
        csv_path,
        index=False,
    )

    atomic_json_dump(
        ranking,
        json_path,
    )

    # --------------------------------------------------------
    # Markdown summary
    # --------------------------------------------------------

    lines = []

    lines.append(
        "# Solver28 Base n=16 Profiling Ranking"
    )
    lines.append("")

    lines.append(
        "| Rank | Row | QID | C/16 | p | p(1-p) | "
        "UniqueAns | D_out | Selector |"
    )

    lines.append(
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|"
    )

    for x in ranking:

        lines.append(
            f"| {x['rank']} "
            f"| {x['row_index']} "
            f"| {x['question_id']} "
            f"| {x['correct_count']}/{args.n} "
            f"| {x['pass_rate']:.4f} "
            f"| {x['reward_variance']:.4f} "
            f"| {x['unique_answer_count']} "
            f"| {x['answer_disagreement']:.4f} "
            f"| **{x['selector_score']:.4f}** |"
        )

    md_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("FINAL RANKING")
    print("=" * 90)

    display_cols = [
        "rank",
        "row_index",
        "question_id",
        "correct_count",
        "pass_rate",
        "reward_variance",
        "unique_answer_count",
        "answer_disagreement",
        "selector_score",
    ]

    if len(rank_df):
        print(
            rank_df[
                display_cols
            ].to_string(
                index=False,
                max_colwidth=60,
            )
        )

    print()
    print("SAVED:")
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", md_path)
    print(" ", perq_dir)

    print()
    print("=" * 90)
    print("TOP MIXED-REWARD CANDIDATES")
    print("=" * 90)

    top = [
        x
        for x in ranking
        if 0.0 < x["pass_rate"] < 1.0
    ][:8]

    for x in top:
        print(
            f"#{x['rank']:02d} "
            f"row={x['row_index']:02d} "
            f"C={x['correct_count']:02d}/{args.n} "
            f"p={x['pass_rate']:.3f} "
            f"Var={x['reward_variance']:.4f} "
            f"D={x['answer_disagreement']:.4f} "
            f"S={x['selector_score']:.4f} "
            f"{x['question_id']}"
        )


if __name__ == "__main__":
    main()
