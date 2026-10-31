#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-layer answer verification for the two golden-question experiments.

Layer 1 constrains newly generated answers to a canonical answer envelope.
Layer 2 is this module's deterministic, answer-type-aware rule verifier.
Layer 3 is an explicitly oracle-conditioned Qwen equivalence fallback which is
only eligible when layer 2 cannot parse or support the candidate.  The model
fallback is *not* described as deterministic verification.

The deterministic functions in this file never call a model.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
VERIFIER_VERSION = "4.1.0-v5-256-regression-20260806"


@dataclass(frozen=True)
class VerifierPolicy:
    """Per-question verification policy loaded from the sidecar."""

    answer_type: str = "auto"
    gt_status: str = "trusted"
    reference_override: Optional[Any] = None
    reference_source: str = "dataset"
    strict_abs_tol: float = 1e-8
    strict_rel_tol: float = 1e-6
    compatible_abs_tol: float = 0.0
    compatible_rel_tol: float = 1e-3
    rounding_digits: Optional[int] = None
    angle_unit: Optional[str] = None
    accepted_references: Tuple[Any, ...] = ()
    legacy_pi_value: Optional[float] = None
    notes: str = ""


@dataclass(frozen=True)
class VerifierSidecarEntry:
    question_id: str
    policy: VerifierPolicy
    image_basename: str = ""
    source_line: Optional[int] = None


@dataclass
class VerifierResult:
    status: str = "unsupported_valid"
    is_correct: Optional[bool] = None
    strict_is_correct: Optional[bool] = None
    compatible_is_correct: Optional[bool] = None
    answer_type: str = "unknown"
    normalized_prediction: Optional[str] = None
    normalized_reference: Optional[str] = None
    reference_source: str = "dataset"
    accepted_by: str = ""
    absolute_error: Optional[float] = None
    relative_error: Optional[float] = None
    policy: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    decision_source: str = "deterministic_rule"
    deterministic_status: str = ""
    model_fallback_eligible: bool = False
    model_fallback_used: bool = False
    oracle_conditioned: bool = False


def normalize_answer_type(value: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", str(value).strip().lower())
    aliases = {
        "number": "numeric",
        "float": "numeric",
        "decimal": "numeric",
        "integer": "integer",
        "fraction": "rational",
        "ratio": "rational",
        "percentage": "probability",
        "percent": "probability",
        "symbolic": "expression",
        "choice": "option",
        "multiple_choice": "option",
        "mcq": "option",
        "list": "tuple",
        "string": "exact",
        "text": "exact",
    }
    return aliases.get(normalized, normalized)


def extract_last_boxed(text: str) -> Optional[str]:
    matches: List[str] = []
    for marker in (r"\boxed", r"\fbox"):
        start = 0
        while True:
            index = text.find(marker, start)
            if index < 0:
                break
            brace = index + len(marker)
            while brace < len(text) and text[brace].isspace():
                brace += 1
            if brace >= len(text) or text[brace] != "{":
                start = index + len(marker)
                continue
            depth = 0
            for end in range(brace, len(text)):
                char = text[end]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        matches.append(text[brace + 1 : end].strip())
                        start = end + 1
                        break
            else:
                start = brace + 1
    return matches[-1] if matches else None


def clean_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return repr(value)
    text = str(value).strip()
    answer_matches = ANSWER_RE.findall(text)
    if answer_matches:
        text = answer_matches[-1].strip()
    boxed = extract_last_boxed(text)
    if boxed is not None:
        text = boxed
    text = text.strip().strip("$ ").strip()
    text = re.sub(
        r"\\text\{(?:degrees?|radians?|rad|mm|cm|m|kg|gb/month|students?)\}\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*(?:degrees?|radians?|rad|°|mm|cm|kg|GB/month|students?)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _collapse_latex_command_escapes(value: Any) -> str:
    """Collapse JSON/export escape duplication before a LaTeX command.

    Some WeMath references contain two literal backslashes (for example
    ``4\\\\sqrt2``) while model answers contain one.  Only runs immediately
    followed by an ASCII command name are collapsed; arbitrary backslashes in
    prose are left untouched.
    """

    return re.sub(
        r"\\+(?=[A-Za-z])",
        lambda _match: "\\",
        str(value),
    )


def _unwrap_full_text_wrapper(value: Any) -> str:
    text = clean_answer(value).strip()
    previous = None
    wrapper = re.compile(
        r"^\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}$",
        flags=re.IGNORECASE,
    )
    while text and text != previous:
        previous = text
        match = wrapper.fullmatch(text)
        if match:
            text = match.group(1).strip()
    return text


def _canonical_math_text(value: Any) -> str:
    """Conservatively normalize notation without doing algebra.

    Equality after this function is safe evidence of equivalence.  Inequality
    is *not* evidence of mathematical inequivalence and callers must not use it
    to fabricate a negative label.
    """

    text = _collapse_latex_command_escapes(_unwrap_full_text_wrapper(value))
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    text = text.replace("−", "-").replace("×", r"\cdot").replace("÷", "/")
    text = text.replace("π", r"\pi")
    text = text.replace("**", "^")
    text = text.replace("$", "")

    # Common compact TeX emitted by models: \frac12 and \sqrt2.
    text = re.sub(
        r"\\frac\s*([A-Za-z0-9])\s*([A-Za-z0-9])",
        r"\\frac{\1}{\2}",
        text,
    )
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*([A-Za-z0-9])",
        r"\\frac{\1}{\2}",
        text,
    )
    text = re.sub(
        r"\\frac\s*([A-Za-z0-9])\s*\{([^{}]+)\}",
        r"\\frac{\1}{\2}",
        text,
    )
    text = re.sub(r"\\sqrt\s*([A-Za-z0-9]+)", r"\\sqrt{\1}", text)

    # Canonicalize the inverse tangent spellings seen in the v5-256 export.
    text = re.sub(
        r"\\tan\s*\^\s*(?:\{\s*-1\s*\}|-1)",
        r"\\arctan",
        text,
    )
    text = re.sub(
        r"\\arctan\s*\(\s*([^()]*)\s*\)",
        r"\\arctan{\1}",
        text,
    )
    text = re.sub(
        r"\\arctan\s+([A-Za-z0-9]+)",
        r"\\arctan{\1}",
        text,
    )

    # Braces around a plain exponent carry no mathematical distinction.
    text = re.sub(r"\^\s*\{\s*([^{}]+)\s*\}", r"^\1", text)
    # Exact-answer records sometimes omit the degree suffix in the reference.
    text = re.sub(
        r"(?:\^\s*\\circ|\\text\{degrees?\}|degrees?|\u00b0)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", "", text).strip()


def _canonical_relation_text(value: Any) -> Optional[str]:
    """Map only a small allowlist of complete relation phrases."""

    text = _unwrap_full_text_wrapper(value).strip().lower()
    text = re.sub(r"[.\s]+$", "", text)
    text = re.sub(r"^(?:they|the\s+(?:two\s+)?circles?)\s+", "", text)
    aliases = {
        "intersect",
        "intersecting",
        "intersect at two points",
        "intersect in two points",
    }
    return "relation:intersect" if text in aliases else None


def infer_answer_type(reference: Any) -> str:
    if isinstance(reference, (list, tuple)):
        return "tuple"
    if isinstance(reference, set):
        return "set"
    text = clean_answer(reference)
    if re.fullmatch(r"(?:[A-H]|\(?[A-H]\)?)", text.upper()):
        return "option"
    if text.endswith("%") or text.endswith(r"\%"):
        return "probability"
    if _parse_numeric_value(text) is not None:
        return "numeric"
    if len(text) >= 2 and text[0] in "([{" and text[-1] in ")]}":
        return "tuple"
    if re.search(r"(?:\\pi|\\sqrt|pi|sqrt|[A-Za-z])", text):
        return "expression"
    return "exact"


def _latex_to_sympy_text(value: Any) -> str:
    text = _canonical_math_text(value)
    text = text.replace(r"\,", "").replace(",", "")
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    text = text.replace("π", "pi").replace(r"\pi", "pi")
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace(r"\%", "%")
    sqrt_pattern = re.compile(r"\\sqrt\{([^{}]+)\}")
    while sqrt_pattern.search(text):
        text = sqrt_pattern.sub(r"sqrt(\1)", text)
    # Convert inner radicals first so a numerator such as
    # ``3\sqrt{57}`` no longer contains braces when the outer fraction is
    # processed.
    fraction_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while fraction_pattern.search(text):
        text = fraction_pattern.sub(r"((\1)/(\2))", text)
    text = text.replace("^", "**")
    function_pattern = re.compile(
        r"\\(arctan|arcsin|arccos|sin|cos|tan|exp)\{([^{}]+)\}"
    )
    function_names = {
        "arctan": "atan",
        "arcsin": "asin",
        "arccos": "acos",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "exp": "exp",
    }
    while function_pattern.search(text):
        text = function_pattern.sub(
            lambda match: f"{function_names[match.group(1)]}({match.group(2)})",
            text,
        )
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"(?<=\d)\s*(?=(?:pi|e|sqrt|\())", "*", text)
    text = re.sub(r"(?<=\))\s*(?=(?:\d|pi|e|\())", "*", text)
    return text.strip()


def _parse_numeric_value(
    value: Any,
    *,
    legacy_pi_value: Optional[float] = None,
) -> Optional[Decimal]:
    text = _latex_to_sympy_text(value)
    if not text:
        return None
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    try:
        if re.fullmatch(r"[+-]?\d+\s*/\s*[+-]?\d+", text):
            fraction = Fraction(text.replace(" ", ""))
            result = Decimal(fraction.numerator) / Decimal(fraction.denominator)
        elif re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
            text,
        ):
            result = Decimal(text)
        else:
            numeric = _safe_numeric_expression(
                text,
                legacy_pi_value=legacy_pi_value,
            )
            if numeric is None:
                try:
                    import sympy
                except ImportError:
                    return None
                expression = sympy.sympify(
                    text,
                    locals={
                        "pi": sympy.Float(legacy_pi_value)
                        if legacy_pi_value is not None
                        else sympy.pi,
                        "e": sympy.E,
                        "sqrt": sympy.sqrt,
                        "sin": sympy.sin,
                        "cos": sympy.cos,
                        "tan": sympy.tan,
                        "acos": sympy.acos,
                        "asin": sympy.asin,
                        "atan": sympy.atan,
                        "exp": sympy.exp,
                    },
                )
                if expression.free_symbols:
                    return None
                result = Decimal(str(sympy.N(expression, 30)))
            else:
                result = Decimal(str(numeric))
    except (InvalidOperation, ZeroDivisionError, ValueError, TypeError):
        return None
    except Exception:
        return None
    return result / Decimal(100) if percent else result


def _safe_numeric_expression(
    text: str,
    *,
    legacy_pi_value: Optional[float] = None,
) -> Optional[float]:
    """Evaluate a small arithmetic grammar without using Python ``eval``."""

    try:
        root = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    names = {
        "pi": float(legacy_pi_value) if legacy_pi_value is not None else math.pi,
        "e": math.e,
    }
    functions = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "acos": math.acos,
        "asin": math.asin,
        "atan": math.atan,
        "exp": math.exp,
    }

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(
                node.value,
                bool,
            ):
                return float(node.value)
            raise ValueError("non-numeric constant")
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op,
            (ast.UAdd, ast.USub),
        ):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow),
        ):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if abs(right) > 100:
                raise ValueError("exponent too large")
            return left**right
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in functions
            and len(node.args) == 1
            and not node.keywords
        ):
            return functions[node.func.id](visit(node.args[0]))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    try:
        value = visit(root)
    except (ArithmeticError, OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


_LATEX_OPTION_WRAPPER_RE = re.compile(
    r"^\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}$",
    re.IGNORECASE,
)
_OPTION_PREFIX_RE = re.compile(
    r"^(?:THE\s+)?(?:FINAL\s+)?(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:|=)?\s*",
    re.IGNORECASE,
)


def _unwrap_option_text(value: Any) -> str:
    """Remove only recognized outer wrappers before looking for A--H.

    The old verifier searched the whole LaTeX string and therefore read the
    ``E`` in ``\\text{A}`` as option E.  This routine never searches inside an
    arbitrary word.
    """

    text = clean_answer(value).strip()
    previous = None
    while text and text != previous:
        previous = text
        text = text.strip().strip("$").strip()
        wrapper = _LATEX_OPTION_WRAPPER_RE.fullmatch(text)
        if wrapper:
            text = wrapper.group(1).strip()
            continue
        text = _OPTION_PREFIX_RE.sub("", text).strip()
        if (
            len(text) >= 2
            and (text[0], text[-1]) in {("(", ")"), ("[", "]")}
        ):
            text = text[1:-1].strip()
    return text


def _normalize_option(value: Any) -> Optional[str]:
    text = _unwrap_option_text(value).upper()
    match = re.fullmatch(
        r"[\(\[]?\s*([A-H])\s*[\)\]]?\s*[\.\。]?",
        text,
    )
    # Deliberately require the entire isolated answer region to be one label.
    # Searching arbitrary prose would turn "not a label" into option A.
    return match.group(1) if match else None


def parse_question_options(question: str) -> Dict[str, str]:
    """Parse an ordered A--H option block, including inline ``; B.`` forms.

    A single apparent marker is intentionally rejected.  Requiring at least
    two ascending labels prevents ordinary geometry prose such as ``point A.``
    from being treated as a multiple-choice block.
    """

    raw = str(question)
    candidates = list(
        re.finditer(
            r"(?:^|[\n;；?？,，])\s*([A-H])\s*[\.\):：]\s*",
            raw,
            flags=re.IGNORECASE,
        )
    )
    runs: List[List[re.Match[str]]] = []
    current: List[re.Match[str]] = []
    for match in candidates:
        label = match.group(1).upper()
        if not current:
            current = [match]
        elif ord(label) == ord(current[-1].group(1).upper()) + 1:
            current.append(match)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = [match]
    if len(current) >= 2:
        runs.append(current)
    if not runs:
        return {}
    matches = max(runs, key=len)

    options: Dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        content = raw[start:end].strip()
        if content:
            options[match.group(1).upper()] = content
    return options


def _normalized_option_content(value: Any) -> str:
    text = _canonical_math_text(value)
    return text.strip(" .。;；")


def extract_option_label(
    prediction: Any,
    *,
    question: str = "",
) -> Tuple[Optional[str], str]:
    """Return a deterministic option label plus a precise parse status."""

    direct = _normalize_option(prediction)
    if direct is not None:
        return direct, "parsed_label"
    candidate = _normalized_option_content(prediction)
    if not candidate:
        return None, "parse_failed"
    matched = [
        label
        for label, content in parse_question_options(question).items()
        if _normalized_option_content(content) == candidate
    ]
    if len(matched) == 1:
        return matched[0], "parsed_option_content"
    if len(matched) > 1:
        return None, "ambiguous"
    return None, "parse_failed"


def generator_answer_constraint(answer_type: str) -> str:
    """Layer-1 output instruction appended to Solver/Author prompts."""

    if normalize_answer_type(answer_type) == "option":
        return (
            "Final-answer format constraint: output exactly one option label A-H "
            "inside <answer>...</answer>, for example <answer>A</answer>. Do not "
            "place words, LaTeX wrappers, or option content inside that tag."
        )
    return (
        "Final-answer format constraint: put only the final mathematical answer "
        "inside <answer>...</answer>."
    )


MODEL_EQUIVALENCE_INSTRUCTION = (
    "判定输入的两者是否在数学意义上相等，相等输出1，不等输出0"
)


def build_model_equivalence_prompt(reference: Any, prediction: Any) -> str:
    """Build the deliberately minimal, GT-conditioned layer-3 prompt."""

    return (
        f"{MODEL_EQUIVALENCE_INSTRUCTION}\n"
        f"标准GT：{clean_answer(reference)}\n"
        f"模型回答GT：{clean_answer(prediction)}"
    )


def parse_model_equivalence_output(raw: str) -> Optional[bool]:
    """Accept only an exact binary response; prose is a protocol failure."""

    normalized = str(raw or "").strip()
    if normalized == "1":
        return True
    if normalized == "0":
        return False
    return None


def apply_model_equivalence_fallback(
    deterministic: VerifierResult,
    *,
    raw_model_output: str,
) -> VerifierResult:
    """Attach a strict layer-3 decision without hiding its oracle use."""

    if not deterministic.model_fallback_eligible:
        raise ValueError(
            "model equivalence fallback is only allowed after unresolved rule verification"
        )
    decision = parse_model_equivalence_output(raw_model_output)
    if decision is None:
        deterministic.status = "verifier_error"
        deterministic.reason += "; model equivalence output was not exactly 0 or 1"
        deterministic.model_fallback_used = True
        deterministic.oracle_conditioned = True
        deterministic.decision_source = "model_equivalence_fallback_error"
        return deterministic
    deterministic.status = (
        "verified_equivalent_model" if decision else "verified_incorrect_model"
    )
    deterministic.is_correct = decision
    deterministic.strict_is_correct = None
    deterministic.compatible_is_correct = decision
    deterministic.accepted_by = "qwen_gt_candidate_equivalence"
    deterministic.reason += "; unresolved rule result resolved by strict binary model fallback"
    deterministic.model_fallback_used = True
    deterministic.oracle_conditioned = True
    deterministic.decision_source = "model_equivalence_fallback"
    return deterministic


def _split_container(value: Any) -> Optional[List[str]]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    text = _collapse_latex_command_escapes(clean_answer(value))
    text = text.replace(r"\left", "").replace(r"\right", "").strip()
    try:
        literal = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        literal = None
    if isinstance(literal, (list, tuple, set)):
        return [str(item) for item in literal]
    if len(text) < 2 or text[0] not in "([{" or text[-1] not in ")]}":
        return None
    inner = text[1:-1].strip()
    if not inner:
        return []
    items: List[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(inner):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(inner[start:index].strip())
            start = index + 1
    items.append(inner[start:].strip())
    return items if all(items) else None


def _normalize_scalar(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    number = _parse_numeric_value(text)
    if number is not None:
        return str(number.normalize())
    return _canonical_math_text(text).lower()


def _numeric_error(
    prediction: Decimal,
    reference: Decimal,
) -> Tuple[Decimal, Decimal]:
    absolute = abs(prediction - reference)
    scale = max(abs(reference), Decimal(1))
    return absolute, absolute / scale


def _numeric_pass(
    absolute: Decimal,
    relative: Decimal,
    *,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    return (
        absolute <= Decimal(str(abs_tol))
        or relative <= Decimal(str(rel_tol))
    )


def _parse_affine_equation(
    value: Any,
) -> Optional[Tuple[str, str, Decimal, Decimal]]:
    """Parse only ``dependent = slope * independent + intercept``.

    This narrow parser keeps common school-level line equations deterministic
    even when SymPy is unavailable.  It intentionally does not claim to be a
    general equation parser.
    """

    text = _latex_to_sympy_text(value).lower().replace(" ", "")
    if text.count("=") != 1:
        return None
    left, right = text.split("=", 1)
    if not re.fullmatch(r"[a-z]", left):
        return None
    independent_matches = re.findall(r"[a-z]", right)
    if len(set(independent_matches)) != 1:
        return None
    independent = independent_matches[0]
    if independent == left or right.count(independent) != 1:
        return None
    position = right.index(independent)
    slope_text = right[:position].rstrip("*")
    intercept_text = right[position + 1 :]
    if slope_text in {"", "+"}:
        slope = Decimal(1)
    elif slope_text == "-":
        slope = Decimal(-1)
    else:
        slope = _parse_numeric_value(slope_text)
        if slope is None:
            return None
    if not intercept_text:
        intercept = Decimal(0)
    else:
        intercept = _parse_numeric_value(intercept_text)
        if intercept is None:
            return None
    return left, independent, slope, intercept


def _policy_dict(policy: VerifierPolicy) -> Dict[str, Any]:
    value = asdict(policy)
    value["accepted_references"] = list(policy.accepted_references)
    return value


def _result(
    *,
    status: str,
    strict: Optional[bool],
    compatible: Optional[bool],
    kind: str,
    prediction: Optional[str],
    reference: Optional[str],
    policy: VerifierPolicy,
    accepted_by: str,
    reason: str,
    absolute_error: Optional[Decimal] = None,
    relative_error: Optional[Decimal] = None,
) -> VerifierResult:
    unresolved = status in {
        "parse_failed",
        "ambiguous",
        "unsupported",
        "unsupported_valid",
        "invalid_prediction",
        "verifier_error",
    }
    return VerifierResult(
        status=status,
        is_correct=compatible,
        strict_is_correct=strict,
        compatible_is_correct=compatible,
        answer_type=kind,
        normalized_prediction=prediction,
        normalized_reference=reference,
        reference_source=policy.reference_source,
        accepted_by=accepted_by,
        absolute_error=float(absolute_error) if absolute_error is not None else None,
        relative_error=float(relative_error) if relative_error is not None else None,
        policy=_policy_dict(policy),
        reason=reason,
        deterministic_status=status,
        model_fallback_eligible=unresolved,
    )


def _verify_numeric(
    prediction: Any,
    references: Sequence[Any],
    kind: str,
    policy: VerifierPolicy,
) -> VerifierResult:
    predicted_number = _parse_numeric_value(prediction)
    if predicted_number is None:
        return _result(
            status="invalid_prediction",
            strict=False,
            compatible=False,
            kind=kind,
            prediction=None,
            reference=clean_answer(references[0]),
            policy=policy,
            accepted_by="",
            reason="cannot parse prediction as a finite numeric value",
        )

    if kind == "angle" and policy.angle_unit:
        raw_prediction = str(prediction).lower()
        explicit_radian = bool(re.search(r"\b(?:rad|radian)s?\b", raw_prediction))
        explicit_degree = "°" in raw_prediction or bool(
            re.search(r"\bdegrees?\b", raw_prediction)
        )
        target_unit = policy.angle_unit.strip().lower()
        if target_unit.startswith("deg") and explicit_radian:
            predicted_number = predicted_number * Decimal(
                str(180.0 / math.pi)
            )
        elif target_unit.startswith("rad") and explicit_degree:
            predicted_number = predicted_number * Decimal(
                str(math.pi / 180.0)
            )

    parsed_references: List[Tuple[Any, Decimal]] = []
    for reference in references:
        parsed = _parse_numeric_value(reference)
        if parsed is not None:
            parsed_references.append((reference, parsed))
    if not parsed_references:
        return _result(
            status="unsupported_valid",
            strict=None,
            compatible=None,
            kind=kind,
            prediction=str(predicted_number.normalize()),
            reference=None,
            policy=policy,
            accepted_by="",
            reason="prediction is valid, but the reference is not numerically parseable",
        )

    best: Optional[Tuple[Any, Decimal, Decimal, Decimal]] = None
    for raw_reference, expected in parsed_references:
        absolute, relative = _numeric_error(predicted_number, expected)
        candidate = (raw_reference, expected, absolute, relative)
        if best is None or (absolute, relative) < (best[2], best[3]):
            best = candidate
    assert best is not None
    raw_reference, expected, absolute, relative = best

    strict_abs_tol = policy.strict_abs_tol
    if policy.rounding_digits is not None:
        strict_abs_tol = max(
            strict_abs_tol,
            0.5 * (10.0 ** (-policy.rounding_digits)),
        )
    strict = _numeric_pass(
        absolute,
        relative,
        abs_tol=strict_abs_tol,
        rel_tol=policy.strict_rel_tol,
    )
    compatible = strict or _numeric_pass(
        absolute,
        relative,
        abs_tol=max(policy.compatible_abs_tol, strict_abs_tol),
        rel_tol=max(policy.compatible_rel_tol, policy.strict_rel_tol),
    )

    accepted_by = ""
    if strict:
        accepted_by = (
            f"rounded_to_{policy.rounding_digits}_decimals"
            if policy.rounding_digits is not None
            and absolute > Decimal(str(policy.strict_abs_tol))
            else "strict_numeric"
        )
    elif compatible:
        accepted_by = "compatible_numeric"

    if not compatible and policy.legacy_pi_value is not None:
        prediction_legacy = _parse_numeric_value(
            prediction,
            legacy_pi_value=policy.legacy_pi_value,
        )
        if prediction_legacy is not None:
            legacy_absolute, legacy_relative = _numeric_error(
                prediction_legacy,
                expected,
            )
            if _numeric_pass(
                legacy_absolute,
                legacy_relative,
                abs_tol=max(policy.compatible_abs_tol, strict_abs_tol),
                rel_tol=max(policy.compatible_rel_tol, policy.strict_rel_tol),
            ):
                compatible = True
                accepted_by = f"legacy_pi={policy.legacy_pi_value}"
                absolute, relative = legacy_absolute, legacy_relative

    if strict:
        status = "verified_exact" if absolute == 0 else "verified_equivalent"
    elif compatible:
        status = "verified_approx"
    else:
        status = "verified_incorrect"
    return _result(
        status=status,
        strict=strict,
        compatible=compatible,
        kind=kind,
        prediction=str(predicted_number.normalize()),
        reference=str(expected.normalize()),
        policy=policy,
        accepted_by=accepted_by,
        absolute_error=absolute,
        relative_error=relative,
        reason=(
            f"numeric comparison; strict(abs={strict_abs_tol}, "
            f"rel={policy.strict_rel_tol}); "
            f"compatible(abs={max(policy.compatible_abs_tol, strict_abs_tol)}, "
            f"rel={max(policy.compatible_rel_tol, policy.strict_rel_tol)})"
        ),
    )


def _verify_expression(
    prediction: Any,
    references: Sequence[Any],
    policy: VerifierPolicy,
) -> VerifierResult:
    predicted_canonical = _canonical_math_text(prediction)
    reference_canonical = [
        _canonical_math_text(reference) for reference in references
    ]
    if predicted_canonical and predicted_canonical in reference_canonical:
        matched = reference_canonical.index(predicted_canonical)
        return _result(
            status="verified_equivalent",
            strict=True,
            compatible=True,
            kind="expression",
            prediction=predicted_canonical,
            reference=reference_canonical[matched],
            policy=policy,
            accepted_by="conservative_notation_normalization",
            reason="expressions are identical after notation-only normalization",
        )

    # Constant expressions should remain verifiable on a CPU environment
    # without SymPy.  The safe numeric grammar supports fractions, radicals,
    # pi, and elementary one-argument functions.
    numeric_first = _verify_numeric(
        prediction,
        references,
        "expression",
        policy,
    )
    if numeric_first.status != "invalid_prediction":
        return numeric_first

    predicted_affine = _parse_affine_equation(prediction)
    affine_references = [
        parsed
        for parsed in (_parse_affine_equation(reference) for reference in references)
        if parsed is not None
    ]
    if predicted_affine is not None and affine_references:
        correct = predicted_affine in affine_references
        return _result(
            status="verified_equivalent" if correct else "verified_incorrect",
            strict=correct,
            compatible=correct,
            kind="expression",
            prediction=repr(predicted_affine),
            reference=repr(affine_references[0]),
            policy=policy,
            accepted_by="affine_equation_normalization" if correct else "",
            reason="both sides match the deterministic affine-equation grammar",
        )

    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except ImportError:
        return _result(
            status="unsupported_valid" if clean_answer(prediction) else "invalid_prediction",
            strict=None,
            compatible=None,
            kind="expression",
            prediction=clean_answer(prediction) or None,
            reference=clean_answer(references[0]),
            policy=policy,
            accepted_by="",
            reason="SymPy unavailable; symbolic correctness was not fabricated",
        )

    predicted_text = _latex_to_sympy_text(prediction)
    if not predicted_text:
        return _result(
            status="invalid_prediction",
            strict=False,
            compatible=False,
            kind="expression",
            prediction=None,
            reference=clean_answer(references[0]),
            policy=policy,
            accepted_by="",
            reason="empty expression prediction",
        )
    transformations = standard_transformations + (
        implicit_multiplication_application,
    )
    local_dict = {
        "pi": sympy.pi,
        "e": sympy.E,
        "sqrt": sympy.sqrt,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "acos": sympy.acos,
        "asin": sympy.asin,
        "atan": sympy.atan,
        "exp": sympy.exp,
    }

    def parse_relation(text: str) -> Tuple[Any, bool]:
        if text.count("=") > 1:
            raise ValueError("more than one equality sign")
        if "=" in text:
            left, right = text.split("=", 1)
            if not left or not right:
                raise ValueError("empty side of equality")
            left_expr = parse_expr(
                left,
                local_dict=local_dict,
                transformations=transformations,
                evaluate=True,
            )
            right_expr = parse_expr(
                right,
                local_dict=local_dict,
                transformations=transformations,
                evaluate=True,
            )
            return sympy.simplify(left_expr - right_expr), True
        return (
            parse_expr(
                text,
                local_dict=local_dict,
                transformations=transformations,
                evaluate=True,
            ),
            False,
        )

    try:
        predicted_expression, predicted_is_relation = parse_relation(
            predicted_text
        )
    except Exception as exc:
        return _result(
            status="invalid_prediction",
            strict=False,
            compatible=False,
            kind="expression",
            prediction=predicted_text,
            reference=clean_answer(references[0]),
            policy=policy,
            accepted_by="",
            reason=f"cannot parse prediction expression: {type(exc).__name__}: {exc}",
        )

    for reference in references:
        reference_text = _latex_to_sympy_text(reference)
        try:
            reference_expression, reference_is_relation = parse_relation(
                reference_text
            )
            if predicted_is_relation != reference_is_relation:
                continue
            if predicted_is_relation:
                difference = sympy.simplify(
                    predicted_expression - reference_expression
                )
                sum_difference = sympy.simplify(
                    predicted_expression + reference_expression
                )
                equivalent = difference == 0 or sum_difference == 0
                if not equivalent and reference_expression != 0:
                    ratio = sympy.simplify(
                        predicted_expression / reference_expression
                    )
                    equivalent = bool(
                        ratio != 0 and not getattr(ratio, "free_symbols", set())
                    )
            else:
                equivalent = bool(
                    sympy.simplify(
                        predicted_expression - reference_expression
                    )
                    == 0
                )
        except Exception:
            continue
        if equivalent:
            exact_text = str(predicted_expression) == str(reference_expression)
            return _result(
                status="verified_exact" if exact_text else "verified_equivalent",
                strict=True,
                compatible=True,
                kind="expression",
                prediction=str(predicted_expression),
                reference=str(reference_expression),
                policy=policy,
                accepted_by="sympy_equivalence",
                reason=(
                    "symbolic equations have proportional residuals"
                    if predicted_is_relation
                    else "symbolic expressions simplify to the same value"
                ),
            )

    return _result(
        status="verified_incorrect",
        strict=False,
        compatible=False,
        kind="expression",
        prediction=str(predicted_expression),
        reference=clean_answer(references[0]),
        policy=policy,
        accepted_by="",
        reason="both symbolic forms parsed but are not mathematically equivalent",
    )


def verify_answer(
    prediction: Any,
    reference: Any,
    answer_type: Optional[str] = None,
    *,
    policy: Optional[VerifierPolicy] = None,
    question: str = "",
) -> VerifierResult:
    """Verify one answer without calling a model."""

    active = policy or VerifierPolicy(answer_type=answer_type or "auto")
    kind = normalize_answer_type(
        answer_type
        or (active.answer_type if active.answer_type != "auto" else "")
        or infer_answer_type(reference)
    )
    chosen_reference = (
        active.reference_override
        if active.reference_override not in (None, "")
        else reference
    )
    references = [chosen_reference, *active.accepted_references]

    if active.gt_status.lower() != "trusted":
        prediction_text = clean_answer(prediction)
        return _result(
            status="gt_suspect",
            strict=None,
            compatible=None,
            kind=kind,
            prediction=prediction_text or None,
            reference=clean_answer(chosen_reference) or None,
            policy=active,
            accepted_by="",
            reason=(
                "reference is marked suspect/ambiguous in the verifier sidecar; "
                "no correctness label was produced"
            ),
        )

    if kind == "option":
        predicted, option_parse_status = extract_option_label(
            prediction,
            question=question,
        )
        expected_options = [
            option for option in (_normalize_option(item) for item in references)
            if option is not None
        ]
        if predicted is None:
            return _result(
                status=option_parse_status,
                strict=None,
                compatible=None,
                kind=kind,
                prediction=None,
                reference=expected_options[0] if expected_options else None,
                policy=active,
                accepted_by="",
                reason=(
                    "deterministic option extraction did not produce one "
                    "unambiguous A-H label or exact option-content match"
                ),
            )
        if not expected_options:
            return _result(
                status="unsupported_valid",
                strict=None,
                compatible=None,
                kind=kind,
                prediction=predicted,
                reference=None,
                policy=active,
                accepted_by="",
                reason="prediction is a valid A-H option but reference is not parseable",
            )
        correct = predicted in expected_options
        return _result(
            status="verified_exact" if correct else "verified_incorrect",
            strict=correct,
            compatible=correct,
            kind=kind,
            prediction=predicted,
            reference=expected_options[0],
            policy=active,
            accepted_by="exact_option" if correct else "",
            reason="exact option comparison with labels A-H",
        )

    if kind in {"numeric", "integer", "rational", "probability", "angle"}:
        result = _verify_numeric(prediction, references, kind, active)
        if kind == "integer" and result.compatible_is_correct:
            value = _parse_numeric_value(prediction)
            if value is None or value != value.to_integral_value():
                result.status = "verified_incorrect"
                result.is_correct = False
                result.strict_is_correct = False
                result.compatible_is_correct = False
                result.accepted_by = ""
                result.reason += "; integer answer required"
        return result

    if kind == "expression":
        return _verify_expression(prediction, references, active)

    if kind in {"tuple", "set"}:
        predicted_items = _split_container(prediction)
        reference_items = _split_container(chosen_reference)
        if predicted_items is None:
            return _result(
                status="invalid_prediction",
                strict=False,
                compatible=False,
                kind=kind,
                prediction=None,
                reference=clean_answer(chosen_reference),
                policy=active,
                accepted_by="",
                reason=f"cannot parse prediction as {kind}",
            )
        if reference_items is None:
            return _result(
                status="unsupported_valid",
                strict=None,
                compatible=None,
                kind=kind,
                prediction=repr(predicted_items),
                reference=None,
                policy=active,
                accepted_by="",
                reason=f"prediction is a valid {kind}, but reference is not parseable",
            )
        predicted_normalized = tuple(_normalize_scalar(item) for item in predicted_items)
        reference_normalized = tuple(_normalize_scalar(item) for item in reference_items)
        if kind == "set":
            predicted_normalized = tuple(sorted(predicted_normalized))
            reference_normalized = tuple(sorted(reference_normalized))
        correct = predicted_normalized == reference_normalized
        return _result(
            status="verified_equivalent" if correct else "verified_incorrect",
            strict=correct,
            compatible=correct,
            kind=kind,
            prediction=repr(predicted_normalized),
            reference=repr(reference_normalized),
            policy=active,
            accepted_by=f"{kind}_normalization" if correct else "",
            reason="unordered set comparison" if kind == "set" else "ordered tuple comparison",
        )

    predicted_text = re.sub(
        r"\s+",
        " ",
        _unwrap_full_text_wrapper(prediction),
    ).strip().lower()
    reference_texts = [
        re.sub(r"\s+", " ", _unwrap_full_text_wrapper(item)).strip().lower()
        for item in references
    ]
    if not predicted_text:
        return _result(
            status="invalid_prediction",
            strict=False,
            compatible=False,
            kind="exact",
            prediction=None,
            reference=reference_texts[0],
            policy=active,
            accepted_by="",
            reason="empty exact-string prediction",
        )
    if predicted_text in reference_texts:
        return _result(
            status="verified_exact",
            strict=True,
            compatible=True,
            kind="exact",
            prediction=predicted_text,
            reference=reference_texts[0],
            policy=active,
            accepted_by="normalized_exact_string",
            reason="normalized exact-string comparison",
        )

    predicted_relation = _canonical_relation_text(prediction)
    reference_relations = [
        _canonical_relation_text(item) for item in references
    ]
    if predicted_relation and predicted_relation in reference_relations:
        return _result(
            status="verified_equivalent",
            strict=True,
            compatible=True,
            kind="exact",
            prediction=predicted_relation,
            reference=predicted_relation,
            policy=active,
            accepted_by="relation_phrase_allowlist",
            reason="complete relation phrase matches a conservative semantic alias",
        )

    predicted_math = _canonical_math_text(prediction).lower()
    reference_math = [
        _canonical_math_text(item).lower() for item in references
    ]
    if predicted_math and predicted_math in reference_math:
        return _result(
            status="verified_equivalent",
            strict=True,
            compatible=True,
            kind="exact",
            prediction=predicted_math,
            reference=predicted_math,
            policy=active,
            accepted_by="conservative_notation_normalization",
            reason="answers are identical after notation-only normalization",
        )

    predicted_number = _parse_numeric_value(prediction)
    numeric_references = [
        parsed
        for parsed in (_parse_numeric_value(item) for item in references)
        if parsed is not None
    ]
    if predicted_number is not None and numeric_references:
        errors = [
            (*_numeric_error(predicted_number, expected), expected)
            for expected in numeric_references
        ]
        absolute, relative, expected = min(
            errors,
            key=lambda item: (item[0], item[1]),
        )
        correct = _numeric_pass(
            absolute,
            relative,
            abs_tol=active.strict_abs_tol,
            rel_tol=active.strict_rel_tol,
        )
        return _result(
            status="verified_equivalent" if correct else "verified_incorrect",
            strict=correct,
            compatible=correct,
            kind="exact",
            prediction=str(predicted_number.normalize()),
            reference=str(expected.normalize()),
            policy=active,
            accepted_by="numeric_constant_equivalence" if correct else "",
            absolute_error=absolute,
            relative_error=relative,
            reason="both exact fields are finite constant mathematical expressions",
        )

    correct = False
    return _result(
        status="verified_incorrect",
        strict=correct,
        compatible=correct,
        kind="exact",
        prediction=predicted_text,
        reference=reference_texts[0],
        policy=active,
        accepted_by="",
        reason="no approved exact, semantic, notation, or numeric equivalence matched",
    )


def load_sidecar(path: Optional[Path]) -> Dict[str, VerifierSidecarEntry]:
    """Load a JSONL sidecar keyed by unique ``question_id``."""

    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Verifier sidecar does not exist: {path}")
    entries: Dict[str, VerifierSidecarEntry] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            question_id = str(raw.get("question_id", "") or "").strip()
            if not question_id:
                raise ValueError(f"{path}:{line_number}: missing question_id")
            if question_id in entries:
                raise ValueError(
                    f"{path}:{line_number}: duplicate question_id {question_id}"
                )
            policy_raw = raw.get("policy", raw)
            if not isinstance(policy_raw, dict):
                raise ValueError(f"{path}:{line_number}: policy must be an object")
            accepted = policy_raw.get("accepted_references", ())
            if not isinstance(accepted, (list, tuple)):
                raise ValueError(
                    f"{path}:{line_number}: accepted_references must be a list"
                )
            policy = VerifierPolicy(
                answer_type=str(policy_raw.get("answer_type", "auto") or "auto"),
                gt_status=str(policy_raw.get("gt_status", "trusted") or "trusted"),
                reference_override=policy_raw.get("reference_override"),
                reference_source=str(
                    policy_raw.get("reference_source", "dataset") or "dataset"
                ),
                strict_abs_tol=float(policy_raw.get("strict_abs_tol", 1e-8)),
                strict_rel_tol=float(policy_raw.get("strict_rel_tol", 1e-6)),
                compatible_abs_tol=float(policy_raw.get("compatible_abs_tol", 0.0)),
                compatible_rel_tol=float(policy_raw.get("compatible_rel_tol", 1e-3)),
                rounding_digits=(
                    int(policy_raw["rounding_digits"])
                    if policy_raw.get("rounding_digits") is not None
                    else None
                ),
                angle_unit=(
                    str(policy_raw["angle_unit"])
                    if policy_raw.get("angle_unit") is not None
                    else None
                ),
                accepted_references=tuple(accepted),
                legacy_pi_value=(
                    float(policy_raw["legacy_pi_value"])
                    if policy_raw.get("legacy_pi_value") is not None
                    else None
                ),
                notes=str(policy_raw.get("notes", "") or ""),
            )
            entries[question_id] = VerifierSidecarEntry(
                question_id=question_id,
                policy=policy,
                image_basename=str(raw.get("image_basename", "") or ""),
                source_line=(
                    int(raw["source_line"])
                    if raw.get("source_line") is not None
                    else None
                ),
            )
    return entries


def default_policy(
    *,
    answer_type: str,
    numeric_tolerance: float = 1e-6,
) -> VerifierPolicy:
    """Safe fallback for records without a sidecar entry."""

    normalized = normalize_answer_type(answer_type or "auto")
    return VerifierPolicy(
        answer_type=normalized,
        strict_abs_tol=1e-8,
        strict_rel_tol=min(float(numeric_tolerance), 1e-6),
        compatible_rel_tol=1e-3,
        notes="automatic fallback; no question-specific convention enabled",
    )


def sidecar_entry_to_dict(entry: VerifierSidecarEntry) -> Dict[str, Any]:
    value = asdict(entry)
    value["policy"]["accepted_references"] = list(
        entry.policy.accepted_references
    )
    return value
