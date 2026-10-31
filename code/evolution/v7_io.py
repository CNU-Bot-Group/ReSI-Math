#!/usr/bin/env python3
"""Transactional storage and model backends for v7."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
    )


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def apply_chat(tokenizer: Any, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def open_image_rgb(path: str) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for model execution") from exc
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def extract_last_boxed(text: str) -> Optional[str]:
    matches: List[str] = []
    for marker in (r"\boxed", r"\fbox"):
        cursor = 0
        while True:
            index = text.find(marker, cursor)
            if index < 0:
                break
            brace = index + len(marker)
            while brace < len(text) and text[brace].isspace():
                brace += 1
            if brace >= len(text) or text[brace] != "{":
                cursor = index + len(marker)
                continue
            depth = 0
            for end in range(brace, len(text)):
                if text[end] == "{":
                    depth += 1
                elif text[end] == "}":
                    depth -= 1
                    if depth == 0:
                        matches.append(text[brace + 1 : end].strip())
                        cursor = end + 1
                        break
            else:
                cursor = brace + 1
    return matches[-1] if matches else None


def parse_solver(raw: str) -> tuple[str, str]:
    text = str(raw or "").strip()
    think = re.search(r"<think>(.*?)</think>", text, flags=re.I | re.S)
    answer = re.search(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    reasoning = think.group(1).strip() if think else text
    final_answer = answer.group(1).strip() if answer else ""
    if not final_answer:
        boxed = extract_last_boxed(text)
        if boxed is not None:
            final_answer = rf"\boxed{{{boxed}}}"
    return reasoning, final_answer


def collect_environment_summary() -> Dict[str, Any]:
    packages: Dict[str, Optional[str]] = {}
    for name in ("vllm", "torch", "transformers", "Pillow", "sympy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    nvidia: Dict[str, Any] = {"returncode": None, "stdout": "", "stderr": ""}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        nvidia = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        nvidia["stderr"] = f"{type(exc).__name__}: {exc}"
    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": str(Path.cwd()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "package_versions": packages,
        "nvidia_smi": nvidia,
    }


def is_engine_dead(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "enginedeaderror" in text or "engine core encountered an issue" in text


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str = ""


def build_sampling_by_prompt(
    sampling: Mapping[str, Any],
    *,
    prompt_count: int,
    request_seeds: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Expand common sampling settings into auditable per-prompt settings."""

    if prompt_count < 0:
        raise ValueError("prompt_count must be non-negative")
    if request_seeds is not None and len(request_seeds) != prompt_count:
        raise ValueError(
            f"request_seeds length {len(request_seeds)} does not match "
            f"prompt count {prompt_count}"
        )
    common = dict(sampling)
    result: List[Dict[str, Any]] = []
    for index in range(prompt_count):
        item = dict(common)
        if request_seeds is not None:
            item["seed"] = int(request_seeds[index])
        result.append(item)
    return result


class CallRecorder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        raw_dir = run_dir / "raw_calls"
        raw_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            int(path.stem.split("_")[-1])
            for path in raw_dir.glob("call_*.json")
            if path.stem.split("_")[-1].isdigit()
        ]
        self.counter = max(existing, default=0)

    def record(self, payload: Dict[str, Any]) -> str:
        self.counter += 1
        call_id = f"call_{self.counter:06d}"
        full = {"schema_version": 1, "call_id": call_id, **payload}
        atomic_json(self.run_dir / "raw_calls" / f"{call_id}.json", full)
        append_jsonl(
            self.run_dir / "model_call_index.jsonl",
            {
                "call_id": call_id,
                "role": full.get("role"),
                "question_id": full.get("question_id"),
                "status": full.get("status"),
                "started_at": full.get("started_at"),
                "finished_at": full.get("finished_at"),
            },
        )
        return call_id


class VllmBackend:
    def __init__(
        self,
        *,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        recorder: CallRecorder,
    ) -> None:
        try:
            from transformers import AutoProcessor
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("transformers and vLLM are required on the server") from exc
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=max_model_len,
            disable_log_stats=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            compilation_config={"level": 0},
        )
        self.recorder = recorder

    def render(self, system: str, user: str) -> str:
        return apply_chat(self.tokenizer, system, user)

    def generate(
        self,
        *,
        prompts: Sequence[str],
        image: Any,
        image_path: str,
        role: str,
        question_id: str,
        sampling: Mapping[str, Any],
        contexts: Sequence[Mapping[str, Any]],
        request_seeds: Optional[Sequence[int]] = None,
    ) -> List[Completion]:
        from vllm import SamplingParams

        requests = []
        for prompt in prompts:
            request: Dict[str, Any] = {"prompt": prompt}
            if image is not None:
                request["multi_modal_data"] = {"image": image}
            requests.append(request)
        sampling_by_prompt = build_sampling_by_prompt(
            sampling,
            prompt_count=len(prompts),
            request_seeds=request_seeds,
        )
        sampling_params = [SamplingParams(**item) for item in sampling_by_prompt]
        started_at = utc_now()
        started_perf = time.perf_counter()
        completions: List[Completion] = []
        status = "completed"
        error = ""
        try:
            outputs = self.llm.generate(requests, sampling_params)
            for output in outputs:
                if output.outputs:
                    first = output.outputs[0]
                    completions.append(
                        Completion(
                            text=str(first.text or "").strip(),
                            finish_reason=str(getattr(first, "finish_reason", "") or ""),
                        )
                    )
                else:
                    completions.append(Completion(text="", finish_reason="empty"))
            while len(completions) < len(prompts):
                completions.append(Completion(text="", finish_reason="missing"))
            return completions
        except Exception as exc:
            status = "engine_dead" if is_engine_dead(exc) else "failed"
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            raise
        finally:
            self.recorder.record(
                {
                    "role": role,
                    "question_id": question_id,
                    "status": status,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "duration_seconds": time.perf_counter() - started_perf,
                    "prompts": list(prompts),
                    "responses": [
                        {"text": item.text, "finish_reason": item.finish_reason}
                        for item in completions
                    ],
                    "sampling": dict(sampling),
                    "sampling_by_prompt": sampling_by_prompt,
                    "contexts": [dict(item) for item in contexts],
                    "image_path": image_path,
                    "error": error,
                }
            )


class ScriptedMockBackend:
    """Deterministic backend for local CPU tests and protocol demonstrations."""

    def __init__(self, script: Mapping[str, Sequence[str]], recorder: CallRecorder):
        self.script = {key: list(value) for key, value in script.items()}
        self.offsets: Dict[str, int] = {}
        self.recorder = recorder

    def render(self, system: str, user: str) -> str:
        return f"SYSTEM:\n{system}\nUSER:\n{user}"

    def generate(
        self,
        *,
        prompts: Sequence[str],
        image: Any,
        image_path: str,
        role: str,
        question_id: str,
        sampling: Mapping[str, Any],
        contexts: Sequence[Mapping[str, Any]],
        request_seeds: Optional[Sequence[int]] = None,
    ) -> List[Completion]:
        sampling_by_prompt = build_sampling_by_prompt(
            sampling,
            prompt_count=len(prompts),
            request_seeds=request_seeds,
        )
        values = self.script.get(role, [])
        start = self.offsets.get(role, 0)
        selected = values[start : start + len(prompts)]
        self.offsets[role] = start + len(prompts)
        if len(selected) != len(prompts):
            raise RuntimeError(
                f"mock role {role!r} needs {len(prompts)} responses at offset {start}; "
                f"only {len(values) - start} remain"
            )
        completions = [Completion(text=value, finish_reason="stop") for value in selected]
        self.recorder.record(
            {
                "role": role,
                "question_id": question_id,
                "status": "completed",
                "started_at": utc_now(),
                "finished_at": utc_now(),
                "duration_seconds": 0.0,
                "prompts": list(prompts),
                "responses": [
                    {"text": item.text, "finish_reason": item.finish_reason}
                    for item in completions
                ],
                "sampling": dict(sampling),
                "sampling_by_prompt": sampling_by_prompt,
                "contexts": [dict(item) for item in contexts],
                "image_path": image_path,
                "mock": True,
                "error": "",
            }
        )
        return completions
