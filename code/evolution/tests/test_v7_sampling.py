#!/usr/bin/env python3

from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import v7_io  # noqa: E402
from v7_protocol import stable_sampling_seed  # noqa: E402


class SamplingSeedTests(unittest.TestCase):
    def test_stable_sampling_seeds_are_reproducible_and_per_slot(self) -> None:
        first = [
            stable_sampling_seed(42, "question-1", "solver_round0", slot, 0)
            for slot in range(16)
        ]
        second = [
            stable_sampling_seed(42, "question-1", "solver_round0", slot, 0)
            for slot in range(16)
        ]
        retry = [
            stable_sampling_seed(42, "question-1", "solver_round0", slot, 1)
            for slot in range(16)
        ]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 16)
        self.assertTrue(all(left != right for left, right in zip(first, retry)))

    def test_sampling_expansion_pairs_one_seed_with_each_prompt(self) -> None:
        common = {"temperature": 0.8, "top_p": 0.95, "max_tokens": 128}
        expanded = v7_io.build_sampling_by_prompt(
            common,
            prompt_count=3,
            request_seeds=[101, 202, 303],
        )
        self.assertEqual([item["seed"] for item in expanded], [101, 202, 303])
        self.assertTrue(all(item["temperature"] == 0.8 for item in expanded))
        self.assertNotIn("seed", common)

    def test_sampling_expansion_rejects_seed_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            v7_io.build_sampling_by_prompt(
                {"temperature": 0.8},
                prompt_count=2,
                request_seeds=[101],
            )

    def test_vllm_backend_passes_per_prompt_sampling_params_as_a_list(self) -> None:
        class FakeSamplingParams:
            def __init__(self, **values: object) -> None:
                self.values = values

        class FakeLlm:
            def __init__(self) -> None:
                self.params = None

            def generate(self, requests: object, params: object) -> list[object]:
                self.params = params
                return [
                    SimpleNamespace(
                        outputs=[SimpleNamespace(text=f"answer-{index}", finish_reason="stop")]
                    )
                    for index in range(len(requests))
                ]

        class FakeRecorder:
            def __init__(self) -> None:
                self.payload = None

            def record(self, payload: object) -> str:
                self.payload = payload
                return "call_000001"

        fake_module = types.ModuleType("vllm")
        fake_module.SamplingParams = FakeSamplingParams
        backend = object.__new__(v7_io.VllmBackend)
        backend.llm = FakeLlm()
        backend.recorder = FakeRecorder()

        with mock.patch.dict(sys.modules, {"vllm": fake_module}):
            completions = backend.generate(
                prompts=["prompt-a", "prompt-b"],
                image=None,
                image_path="image.png",
                role="solver_round0",
                question_id="question-1",
                sampling={"temperature": 0.8, "max_tokens": 64},
                contexts=[{"slot": 0}, {"slot": 1}],
                request_seeds=[111, 222],
            )

        self.assertEqual([item.text for item in completions], ["answer-0", "answer-1"])
        self.assertIsInstance(backend.llm.params, list)
        self.assertEqual(
            [item.values["seed"] for item in backend.llm.params],
            [111, 222],
        )
        self.assertEqual(
            [item["seed"] for item in backend.recorder.payload["sampling_by_prompt"]],
            [111, 222],
        )


if __name__ == "__main__":
    unittest.main()
