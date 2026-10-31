#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from v7_protocol import (  # noqa: E402
    ORACLE_HINT_SENTENCE,
    ProbeCandidate,
    build_crossover_tasks,
    crossover_messages,
    evaluate_dpo_pair,
    parse_process_audit,
    parse_trajectory_judge,
    plan_probe_pool,
    trajectory_judge_messages,
)
import v7_io  # noqa: E402


def candidate(index: int, correct: bool) -> ProbeCandidate:
    answer = "5" if correct else str(100 + index)
    return ProbeCandidate(
        trajectory_id=f"g0_s{index:02d}",
        raw_response=(
            f"<think>Candidate {index} develops a substantive and independently "
            f"checkable mathematical derivation with several explicit steps.</think>"
            f"<answer>{answer}</answer>"
        ),
        reasoning=(
            f"Candidate {index} develops a substantive and independently checkable "
            "mathematical derivation with several explicit steps."
        ),
        final_answer=answer,
        verifier_correct=correct,
        verifier_status="verified_exact" if correct else "verified_incorrect",
        finish_reason="stop",
    )


class PoolPlanningTests(unittest.TestCase):
    def test_mixed_pool_is_2c2f_with_four_cf_and_one_ff_pair(self) -> None:
        plan = plan_probe_pool(
            [candidate(index, index < 8) for index in range(16)]
        )
        self.assertEqual(plan["mode"], "mixed_2c2f")
        self.assertEqual(len(plan["selected_correct_ids"]), 2)
        self.assertEqual(len(plan["selected_wrong_ids"]), 2)
        self.assertEqual(len(plan["cf_pairs"]), 4)
        self.assertEqual(len(plan["ff_pairs"]), 1)

    def test_all_f_selects_four_and_enumerates_six_pairs(self) -> None:
        plan = plan_probe_pool([candidate(index, False) for index in range(16)])
        self.assertEqual(plan["mode"], "all_f_rescue")
        self.assertEqual(len(plan["selected_wrong_ids"]), 4)
        self.assertEqual(len(plan["cf_pairs"]), 0)
        self.assertEqual(len(plan["ff_pairs"]), 6)

    def test_all_c_is_recorded_without_crossover(self) -> None:
        plan = plan_probe_pool([candidate(index, True) for index in range(16)])
        self.assertEqual(plan["mode"], "all_c_no_crossover")
        self.assertFalse(plan["cf_pairs"])
        self.assertFalse(plan["ff_pairs"])


class PromptIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = plan_probe_pool(
            [candidate(index, index < 8) for index in range(16)]
        )
        self.by_id = {
            item.trajectory_id: item.raw_response
            for item in [candidate(index, index < 8) for index in range(16)]
        }

    def test_blind_tasks_are_main_and_hints_are_additional(self) -> None:
        tasks = build_crossover_tasks(
            question_id="q",
            pool_plan=self.plan,
            children_per_operator=2,
            hinted_fraction=1.0,
            seed=42,
        )
        blind_cf = [item for item in tasks if item.pair_type == "CF" and item.arm == "blind"]
        hinted = [item for item in tasks if item.arm == "oracle_hinted"]
        blind_ff = [item for item in tasks if item.pair_type == "FF" and item.arm == "blind"]
        self.assertEqual(len(blind_cf), 16)
        self.assertEqual(len(hinted), 16)
        self.assertEqual(len(blind_ff), 4)

    def test_blind_prompt_has_anonymous_parents_and_no_oracle_labels(self) -> None:
        task = next(
            item
            for item in build_crossover_tasks(
                question_id="q",
                pool_plan=self.plan,
                children_per_operator=1,
                hinted_fraction=0.0,
            )
            if item.pair_type == "CF"
        )
        system, user = crossover_messages(
            question="Find the requested value.",
            answer_type="numeric",
            task=task,
            parent_text_by_id=self.by_id,
            vision_placeholder="<IMAGE>",
        )
        rendered = system + "\n" + user
        self.assertIn("Parent A", rendered)
        self.assertIn("Parent B", rendered)
        self.assertNotIn(ORACLE_HINT_SENTENCE, rendered)
        self.assertNotIn("correct_parent_id", rendered)
        self.assertNotIn("wrong_parent_ids", rendered)
        self.assertNotIn("C-F", rendered)
        self.assertNotIn("verifier", rendered.lower())

    def test_hinted_prompt_reveals_only_aggregate_hint(self) -> None:
        task = next(
            item
            for item in build_crossover_tasks(
                question_id="q",
                pool_plan=self.plan,
                children_per_operator=1,
                hinted_fraction=1.0,
            )
            if item.arm == "oracle_hinted"
        )
        system, user = crossover_messages(
            question="Find the requested value.",
            answer_type="numeric",
            task=task,
            parent_text_by_id=self.by_id,
            vision_placeholder="<IMAGE>",
        )
        rendered = system + "\n" + user
        self.assertEqual(rendered.count(ORACLE_HINT_SENTENCE), 1)
        self.assertNotIn(task.correct_parent_id or "IMPOSSIBLE", rendered)
        for wrong_id in task.wrong_parent_ids:
            self.assertNotIn(wrong_id, rendered)

    def test_ten_percent_hints_are_deterministic_and_approximately_ten_percent(self) -> None:
        hinted = 0
        blind_cf = 0
        for index in range(200):
            tasks = build_crossover_tasks(
                question_id=f"q{index}",
                pool_plan=self.plan,
                children_per_operator=2,
                hinted_fraction=0.10,
                seed=42,
            )
            hinted += sum(item.arm == "oracle_hinted" for item in tasks)
            blind_cf += sum(item.arm == "blind" and item.pair_type == "CF" for item in tasks)
        fraction = hinted / blind_cf
        self.assertGreater(fraction, 0.08)
        self.assertLess(fraction, 0.12)


class DpoFilterTests(unittest.TestCase):
    AUDIT = {
        "parse_ok": True,
        "child_repaired_parent_error": True,
        "new_critical_error_introduced": False,
        "dpo_pair_recommended": True,
        "shared_reasoning_ratio": 0.6,
        "confidence": 0.9,
    }

    def test_blind_repair_requires_two_unanimous_audits(self) -> None:
        result = evaluate_dpo_pair(
            arm="blind",
            pair_type="CF",
            child_text="Shared setup. Corrected step gives the answer five.",
            wrong_parent_text="Shared setup. Incorrect step gives the answer nine.",
            correct_parent_text="A completely different correct derivation arrives at five.",
            child_verifier_correct=True,
            wrong_parent_verifier_correct=False,
            gt_status="trusted",
            audits=[self.AUDIT, self.AUDIT],
            min_length_ratio=0.5,
            max_length_ratio=2.0,
        )
        self.assertTrue(result["eligible"], result)

    def test_hinted_control_is_never_dpo_eligible(self) -> None:
        result = evaluate_dpo_pair(
            arm="oracle_hinted",
            pair_type="CF",
            child_text="Shared setup. Corrected step gives the answer five.",
            wrong_parent_text="Shared setup. Incorrect step gives the answer nine.",
            correct_parent_text="A completely different correct derivation arrives at five.",
            child_verifier_correct=True,
            wrong_parent_verifier_correct=False,
            gt_status="trusted",
            audits=[self.AUDIT, self.AUDIT],
            min_length_ratio=0.5,
            max_length_ratio=2.0,
        )
        self.assertFalse(result["eligible"])
        self.assertIn("oracle_hinted_control_excluded", result["rejection_reasons"])

    def test_one_disagreeing_audit_rejects_pair(self) -> None:
        disagree = {**self.AUDIT, "child_repaired_parent_error": False}
        result = evaluate_dpo_pair(
            arm="blind",
            pair_type="FF",
            child_text="Shared setup. Corrected step gives the answer five.",
            wrong_parent_text="Shared setup. Incorrect step gives the answer nine.",
            correct_parent_text=None,
            child_verifier_correct=True,
            wrong_parent_verifier_correct=False,
            gt_status="trusted",
            audits=[self.AUDIT, disagree],
            min_length_ratio=0.5,
            max_length_ratio=2.0,
        )
        self.assertFalse(result["eligible"])
        self.assertIn("process_audit_not_unanimous", result["rejection_reasons"])

    def test_process_json_parser_is_fail_closed(self) -> None:
        malformed = parse_process_audit('{"child_repaired_parent_error": "yes"}')
        self.assertFalse(malformed["parse_ok"])

    def test_tagged_trajectory_judge_keeps_visible_rationale_and_json_separate(self) -> None:
        raw = (
            "<think>1. Check the result. 2. Inspect the first doubtful step.</think>"
            '<answer>{"result_verdict":"Uncertain","process_verdict":"MinorIssue",'
            '"score":55,"confidence":0.4,"first_error":"step 2",'
            '"error_type":"arithmetic","valid_prefix":"step 1",'
            '"useful_information":"setup","fusion_advice":"recompute"}</answer>'
        )
        parsed = parse_trajectory_judge(raw, mode="blind")
        self.assertTrue(parsed["parse_ok"])
        self.assertIn("Inspect", parsed["judge_rationale"])
        self.assertEqual(parsed["result_verdict"], "Uncertain")

    def test_guided_judge_prompt_hides_reference_answer(self) -> None:
        system, user = trajectory_judge_messages(
            question="Question without a secret.",
            trajectory="Candidate answer.",
            mode="verifier_guided",
            verifier_evidence={
                "outcome": "incorrect",
                "status": "verified_incorrect",
                "reference_answer_hidden": True,
            },
            vision_placeholder="<IMAGE>",
        )
        rendered = system + user
        self.assertIn("<think>", rendered)
        self.assertIn("reference answer itself is hidden", rendered)
        self.assertNotIn("GT_CANARY_8675309", rendered)

    def test_engine_dead_error_is_distinguished_from_normal_failure(self) -> None:
        class EngineDeadError(RuntimeError):
            pass

        self.assertTrue(v7_io.is_engine_dead(EngineDeadError("worker stopped")))
        self.assertFalse(v7_io.is_engine_dead(RuntimeError("ordinary parse failure")))


if __name__ == "__main__":
    unittest.main()
