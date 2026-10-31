#!/usr/bin/env python3

from __future__ import annotations

import json
import base64
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import run_self_judge_fusion_v7_wemath_dpo as runner  # noqa: E402


class DryRunTests(unittest.TestCase):
    def test_standard_one_row_preflight_without_model(self) -> None:
        self.assertEqual(
            runner.DEFAULT_MODEL_PATH,
            "/data/Qwen/Qwen3-VL-8B-Instruct",
        )
        all_in_one = (HERE / "run_v7_all_in_one.shell").read_text(encoding="utf-8")
        self.assertIn("RESI_MATH_EVOLUTION_INPUT_JSONL", all_in_one)
        self.assertIn("RESI_MATH_EVOLUTION_SAMPLE_MANIFEST", all_in_one)
        self.assertNotIn("/home/", all_in_one)
        self.assertIn("--duplicate-retries 4", all_in_one)
        self.assertIn("sampling_by_prompt", (HERE / "v7_io.py").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            image.write_bytes(b"not-opened-during-dry-run")
            input_path = root / "input.jsonl"
            manifest_path = root / "SAMPLE_MANIFEST.jsonl"
            output_path = root / "result.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "question": "What is 2+3?",
                        "image_path": str(image),
                        "ground_truth": "5",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_index": 0,
                        "standard_idx": "0",
                        "source_row_index": 0,
                        "source_id": "demo",
                        "exported_image_path": str(image),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            code = runner.main(
                [
                    "--dataset-kind",
                    "standard",
                    "--sample-manifest",
                    str(manifest_path),
                    "--input-jsonl",
                    str(input_path),
                    "--output-jsonl",
                    str(output_path),
                    "--run-dir",
                    str(root / "artifacts"),
                    "--dry-run",
                ]
            )
            self.assertEqual(code, 0)
            report = json.loads(
                (root / "artifacts" / "preflight_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["selected_question_count"], 1)
            self.assertEqual(report["round0_calls_planned"], 16)
            self.assertEqual(report["status"], "passed")

    def test_mock_end_to_end_produces_lineage_dpo_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            image.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                )
            )
            input_path = root / "input.jsonl"
            manifest_path = root / "SAMPLE_MANIFEST.jsonl"
            output_path = root / "result.jsonl"
            script_path = root / "mock.json"
            input_path.write_text(
                json.dumps(
                    {
                        "question": "What is 2+3?",
                        "image_path": str(image),
                        "ground_truth": "5",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_index": 0,
                        "standard_idx": "0",
                        "source_row_index": 0,
                        "source_id": "demo",
                        "exported_image_path": str(image),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            solver = []
            for index in range(16):
                answer = "5" if index < 8 else str(20 + index)
                solver.append(
                    "<think>We inspect the problem carefully and develop a complete "
                    f"multi-step derivation for independent probe number {index}.</think>"
                    f"<answer>{answer}</answer>"
                )
            crossover = [
                "<think>Rechecking both anonymous candidates from first principles, "
                "we repair the arithmetic conflict and derive the requested quantity.</think>"
                "<answer>5</answer>"
                for _ in range(20)
            ]
            audit = json.dumps(
                {
                    "child_repaired_parent_error": True,
                    "new_critical_error_introduced": False,
                    "stable_prefix": "rechecked setup",
                    "first_parent_error": "arithmetic",
                    "repair_summary": "recomputed the sum",
                    "shared_reasoning_ratio": 0.5,
                    "dpo_pair_recommended": True,
                    "confidence": 0.9,
                }
            )
            tagged_audit = (
                "<think>1. The parent has a concrete arithmetic error. "
                "2. The offspring recomputes that step without a new error.</think>"
                f"<answer>{audit}</answer>"
            )
            trajectory_judge_json = json.dumps(
                {
                    "result_verdict": "LikelyCorrect",
                    "process_verdict": "Sound",
                    "score": 90,
                    "confidence": 0.85,
                    "first_error": "",
                    "error_type": "",
                    "valid_prefix": "complete derivation",
                    "useful_information": "arithmetic setup",
                    "fusion_advice": "recheck each operation",
                }
            )
            tagged_trajectory_judge = (
                "<think>1. I checked the claimed result. 2. The visible steps are "
                "consistent with the candidate's conclusion. "
                "JUDGE_RATIONALE_CANARY_NOT_FOR_CROSSOVER</think>"
                f"<answer>{trajectory_judge_json}</answer>"
            )
            script_path.write_text(
                json.dumps(
                    {
                        "solver_round0": solver,
                        "crossover": crossover,
                        "judge_blind": [tagged_trajectory_judge for _ in range(36)],
                        "judge_verifier_guided": [tagged_trajectory_judge for _ in range(36)],
                        "process_judge": [tagged_audit for _ in range(48)],
                        "process_judge_guided": [tagged_audit for _ in range(24)],
                    }
                ),
                encoding="utf-8",
            )
            code = runner.main(
                [
                    "--dataset-kind",
                    "standard",
                    "--sample-manifest",
                    str(manifest_path),
                    "--input-jsonl",
                    str(input_path),
                    "--output-jsonl",
                    str(output_path),
                    "--run-dir",
                    str(root / "artifacts"),
                    "--backend",
                    "mock",
                    "--mock-script",
                    str(script_path),
                    "--hinted-control-fraction",
                    "0",
                ]
            )
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["pool_plan"]["mode"], "mixed_2c2f")
            self.assertEqual(len(row["round0_trajectories"]), 16)
            self.assertEqual(len(row["crossover_tasks"]), 20)
            self.assertTrue(
                all(
                    item["blind_judge"]["judge_rationale"]
                    and item["verifier_guided_judge"]["judge_rationale"]
                    for item in [*row["round0_trajectories"], *row["offspring"]]
                )
            )
            rationale_rows = [
                json.loads(line)
                for line in (root / "artifacts" / "exports" / "judge_rationales.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rationale_rows), 72)
            self.assertTrue(
                all(
                    item["analysis_only_not_used_by_evolution"] is True
                    and item["future_judge_dpo_status"]
                    == "candidate_only_no_pair_constructed"
                    for item in rationale_rows
                )
            )
            pairs = [
                json.loads(line)
                for line in (root / "artifacts" / "exports" / "dpo_pairs_full.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(pairs), 24)
            self.assertTrue(all(not item["oracle_hint_exposed"] for item in pairs))
            raw_calls = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in (root / "artifacts" / "raw_calls").glob("call_*.json")
            ]
            solver_calls = [item for item in raw_calls if item["role"] == "solver_round0"]
            solver_seeds = [
                settings["seed"]
                for call in solver_calls
                for settings in call["sampling_by_prompt"]
            ]
            self.assertEqual(len(solver_seeds), 16)
            self.assertEqual(len(set(solver_seeds)), 16)
            for call in solver_calls:
                self.assertEqual(
                    [item["seed"] for item in call["sampling_by_prompt"]],
                    [item["sampling_seed"] for item in call["contexts"]],
                )

            crossover_calls = [item for item in raw_calls if item["role"] == "crossover"]
            crossover_seeds = [
                settings["seed"]
                for call in crossover_calls
                for settings in call["sampling_by_prompt"]
            ]
            self.assertEqual(len(crossover_seeds), 20)
            self.assertEqual(len(set(crossover_seeds)), 20)
            for call in crossover_calls:
                self.assertEqual(
                    [item["seed"] for item in call["sampling_by_prompt"]],
                    [item["sampling_seed"] for item in call["contexts"]],
                )
            crossover_prompts = [
                prompt
                for path in (root / "artifacts" / "raw_calls").glob("call_*.json")
                for payload in [json.loads(path.read_text(encoding="utf-8"))]
                if payload["role"] == "crossover"
                for prompt in payload["prompts"]
            ]
            self.assertTrue(crossover_prompts)
            self.assertTrue(
                all(
                    "JUDGE_RATIONALE_CANARY_NOT_FOR_CROSSOVER" not in prompt
                    for prompt in crossover_prompts
                )
            )
            self.assertTrue(
                all(
                    item["gt_reference_audit_status"] == "dataset_reference_unreviewed"
                    and item["training_ready_without_additional_gt_audit"] is False
                    for item in pairs
                )
            )

            # Resume must replace/rebuild canonical output by question_id and
            # must not request any scripted model responses for a completed row.
            resume_code = runner.main(
                [
                    "--dataset-kind",
                    "standard",
                    "--sample-manifest",
                    str(manifest_path),
                    "--input-jsonl",
                    str(input_path),
                    "--output-jsonl",
                    str(output_path),
                    "--run-dir",
                    str(root / "artifacts"),
                    "--backend",
                    "mock",
                    "--mock-script",
                    str(script_path),
                    "--hinted-control-fraction",
                    "0",
                    "--resume",
                ]
            )
            self.assertEqual(resume_code, 0)
            resumed_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(resumed_rows), 1)


if __name__ == "__main__":
    unittest.main()
