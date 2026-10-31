import unittest

from verifier_v4 import (
    MODEL_EQUIVALENCE_INSTRUCTION,
    VerifierPolicy,
    apply_model_equivalence_fallback,
    build_model_equivalence_prompt,
    extract_option_label,
    generator_answer_constraint,
    parse_model_equivalence_output,
    verify_answer,
)


class VerifierV4Tests(unittest.TestCase):
    def test_text_wrapper_is_a_not_e(self):
        label, status = extract_option_label(r"\boxed{\text{A}}")
        self.assertEqual((label, status), ("A", "parsed_label"))

    def test_all_a_to_h_wrapper_forms(self):
        for label in "ABCDEFGH":
            with self.subTest(label=label):
                parsed, status = extract_option_label(
                    rf"\boxed{{\mathrm{{{label}}}}}"
                )
                self.assertEqual((parsed, status), (label, "parsed_label"))

    def test_option_content_maps_to_question_label(self):
        question = "Choose.\nA. (0, 6) and (4, 2)\nB. (2, 4) and (4, 2)"
        label, status = extract_option_label(
            "(0, 6) and (4, 2)",
            question=question,
        )
        self.assertEqual((label, status), ("A", "parsed_option_content"))

    def test_ambiguous_word_is_not_guessed(self):
        label, status = extract_option_label("BECAUSE")
        self.assertIsNone(label)
        self.assertEqual(status, "parse_failed")

    def test_option_verification_handles_text_wrapper(self):
        result = verify_answer(
            r"\boxed{\text{A}}",
            "A",
            "option",
            policy=VerifierPolicy(answer_type="option"),
        )
        self.assertTrue(result.compatible_is_correct)
        self.assertEqual(result.normalized_prediction, "A")

    def test_unparsed_option_is_unknown_and_fallback_eligible(self):
        result = verify_answer(
            "not a label",
            "A",
            "option",
            policy=VerifierPolicy(answer_type="option"),
        )
        self.assertIsNone(result.compatible_is_correct)
        self.assertEqual(result.status, "parse_failed")
        self.assertTrue(result.model_fallback_eligible)

    def test_binary_model_fallback_protocol(self):
        result = verify_answer(
            "not a label",
            "A",
            "option",
            policy=VerifierPolicy(answer_type="option"),
        )
        resolved = apply_model_equivalence_fallback(
            result,
            raw_model_output="1",
        )
        self.assertTrue(resolved.compatible_is_correct)
        self.assertTrue(resolved.oracle_conditioned)
        self.assertEqual(resolved.decision_source, "model_equivalence_fallback")
        self.assertIsNone(parse_model_equivalence_output("1 because equal"))

    def test_minimal_model_prompt(self):
        prompt = build_model_equivalence_prompt("A", r"\text{A}")
        self.assertEqual(prompt.splitlines()[0], MODEL_EQUIVALENCE_INSTRUCTION)
        self.assertIn("标准GT：A", prompt)
        self.assertIn(r"模型回答GT：\text{A}", prompt)
        self.assertNotIn("Question", prompt)
        self.assertNotIn("verifier", prompt.lower())

    def test_generator_option_constraint(self):
        constraint = generator_answer_constraint("option")
        self.assertIn("<answer>A</answer>", constraint)
        self.assertIn("A-H", constraint)

    def test_constant_expression_without_sympy(self):
        correct = verify_answer(
            r"\boxed{\frac{5}{8}}",
            "5/8",
            "expression",
            policy=VerifierPolicy(answer_type="expression"),
        )
        wrong = verify_answer(
            r"\boxed{\frac{3\sqrt{57}}{8}}",
            "5/8",
            "expression",
            policy=VerifierPolicy(answer_type="expression"),
        )
        self.assertTrue(correct.compatible_is_correct)
        self.assertFalse(wrong.compatible_is_correct)

    def test_inline_option_content_maps_to_label(self):
        cases = [
            "Choose the value, A. $\\frac{1}{2}$; B. $1$; C. $2$; D. $4$",
            "Which value is correct? A.$\\frac{1}{2}$; B.$1$; C.$2$; D.$4$",
            "Coordinates? A. (0,0); B. (-4,-8); C. (4,8); D. (8,4)",
        ]
        predictions = [r"\boxed{\frac12}", r"\frac{1}{2}", "(-4,-8)"]
        expected = ["A", "A", "B"]
        for question, prediction, label in zip(cases, predictions, expected):
            with self.subTest(question=question):
                parsed, status = extract_option_label(
                    prediction,
                    question=question,
                )
                self.assertEqual((parsed, status), (label, "parsed_option_content"))

    def test_single_geometry_point_is_not_parsed_as_option_block(self):
        label, status = extract_option_label(
            "(0,0)",
            question="Point A. (0,0) is rotated about point B.",
        )
        self.assertEqual((label, status), (None, "parse_failed"))

    def test_tuple_latex_normalization_is_inside_verifier(self):
        result = verify_answer(
            r"\boxed{\left(\frac{26}{5},0\right)}",
            "('5.2', '0')",
            "tuple",
            policy=VerifierPolicy(answer_type="tuple"),
        )
        self.assertTrue(result.compatible_is_correct)

        scalar = verify_answer(
            r"\boxed{\dfrac{26}{5}}",
            "('5.2', '0')",
            "tuple",
            policy=VerifierPolicy(answer_type="tuple"),
        )
        self.assertFalse(scalar.compatible_is_correct)

    def test_expression_equation_format_normalization(self):
        equivalent = verify_answer(
            r"(x - 2)^2 + (y + 3)^2 = 5",
            r"(x-2)^{2}+(y+3)^{2}=5",
            "expression",
            policy=VerifierPolicy(answer_type="expression"),
        )
        wrong = verify_answer(
            "y=-3x",
            "y=2x",
            "expression",
            policy=VerifierPolicy(answer_type="expression"),
        )
        self.assertTrue(equivalent.compatible_is_correct)
        self.assertFalse(wrong.compatible_is_correct)

    def test_exact_math_equivalences_from_v5_256(self):
        cases = [
            (r"\dfrac{2\sqrt{13}}{13}", r"\frac{2\sqrt{13}}{13}"),
            (r"2/\sqrt{13}", r"\frac{2\sqrt{13}}{13}"),
            (r"45^\circ", "45"),
            (r"\tan^{-1}(e)", r"\arctan e"),
            (r"\arctan(e)", r"\arctan e"),
            (r"4\sqrt2", r"4\\sqrt2"),
        ]
        for prediction, reference in cases:
            with self.subTest(prediction=prediction, reference=reference):
                result = verify_answer(
                    prediction,
                    reference,
                    "exact",
                    policy=VerifierPolicy(answer_type="exact"),
                )
                self.assertTrue(result.compatible_is_correct)

    def test_conservative_relation_aliases(self):
        for prediction in (
            "intersect",
            "intersecting",
            "intersect at two points",
            r"\text{intersect}",
        ):
            with self.subTest(prediction=prediction):
                result = verify_answer(
                    prediction,
                    "they intersect.",
                    "exact",
                    policy=VerifierPolicy(answer_type="exact"),
                )
                self.assertTrue(result.compatible_is_correct)

        for prediction in ("externally separate", "externally tangent"):
            with self.subTest(prediction=prediction):
                result = verify_answer(
                    prediction,
                    "they intersect.",
                    "exact",
                    policy=VerifierPolicy(answer_type="exact"),
                )
                self.assertFalse(result.compatible_is_correct)

    def test_constant_expression_equivalence_and_inequivalence(self):
        equivalent = verify_answer(
            r"\frac{2-\sqrt2}{4}",
            r"\frac12-\frac{\sqrt2}{4}",
            "expression",
            policy=VerifierPolicy(answer_type="expression"),
        )
        wrong = verify_answer(
            r"\frac{3\sqrt{13}}{13}",
            r"\frac{2\sqrt{13}}{13}",
            "exact",
            policy=VerifierPolicy(answer_type="exact"),
        )
        self.assertTrue(equivalent.compatible_is_correct)
        self.assertFalse(wrong.compatible_is_correct)

    def test_scalar_reference_does_not_silently_accept_tuple_answer(self):
        result = verify_answer(
            r"(\arctan(e),0)",
            r"\arctan e",
            "exact",
            policy=VerifierPolicy(answer_type="exact"),
        )
        self.assertFalse(result.compatible_is_correct)

    def test_suspect_gt_produces_no_label_and_no_model_fallback(self):
        result = verify_answer(
            r"(\arctan(e),0)",
            r"\arctan e",
            "exact",
            policy=VerifierPolicy(answer_type="exact", gt_status="suspect"),
        )
        self.assertIsNone(result.compatible_is_correct)
        self.assertEqual(result.status, "gt_suspect")
        self.assertFalse(result.model_fallback_eligible)


if __name__ == "__main__":
    unittest.main()
