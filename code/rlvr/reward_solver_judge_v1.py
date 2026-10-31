import re


def _extract_answer(text):
    text = str(text or "")

    xs = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.I | re.S,
    )
    if xs:
        return xs[-1].strip()

    xs = re.findall(
        r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
        text,
        flags=re.S,
    )
    if xs:
        return xs[-1].strip()

    xs = re.findall(
        r"(?:final\s+answer|answer)\s*[:\uFF1A]\s*([^\n]+)",
        text,
        flags=re.I,
    )
    if xs:
        return xs[-1].strip()

    return None


def _judge_verdict(text):
    text = str(text or "")

    xs = re.findall(
        r'["\']?result_verdict["\']?\s*:\s*["\']?'
        r'(LikelyCorrect|LikelyIncorrect)["\']?',
        text,
        flags=re.I,
    )

    if not xs:
        xs = re.findall(
            r"\b(LikelyCorrect|LikelyIncorrect)\b",
            text,
            flags=re.I,
        )

    if not xs:
        return None

    return (
        "LikelyCorrect"
        if xs[-1].lower() == "likelycorrect"
        else "LikelyIncorrect"
    )


# Mathematical equivalence fallback

def _replace_frac(text):

    text = str(text)

    while r"\frac" in text:

        pos = text.find(r"\frac")
        i = pos + len(r"\frac")

        while i < len(text) and text[i].isspace():
            i += 1

        if i >= len(text) or text[i] != "{":
            break

        def read_group(s, start):
            depth = 0

            for j in range(start, len(s)):

                if s[j] == "{":
                    depth += 1

                elif s[j] == "}":
                    depth -= 1

                    if depth == 0:
                        return s[start + 1:j], j + 1

            raise ValueError("unbalanced braces")

        num, j = read_group(text, i)

        while j < len(text) and text[j].isspace():
            j += 1

        if j >= len(text) or text[j] != "{":
            break

        den, k = read_group(text, j)

        repl = (
            "(("
            + _replace_frac(num)
            + ")/("
            + _replace_frac(den)
            + "))"
        )

        text = (
            text[:pos]
            + repl
            + text[k:]
        )

    return text


def _to_sympy_text(x):

    x = str(x).strip()

    x = x.replace("$", "")
    x = x.replace(r"\left", "")
    x = x.replace(r"\right", "")
    x = x.replace(r"\,", "")
    x = x.replace(" ", "")

    x = _replace_frac(x)

    x = re.sub(
        r"\\sqrt\{([^{}]+)\}",
        r"sqrt(\1)",
        x,
    )

    x = x.replace("^", "**")

    return x


def _math_equivalent(a, b):

    if a is None or b is None:
        return False

    a = str(a).strip()
    b = str(b).strip()

    # exact normalized
    na = re.sub(r"\s+", "", a)
    nb = re.sub(r"\s+", "", b)

    if na == nb:
        return True

    # multiple choice
    ma = re.fullmatch(
        r"(?:option\s*)?([A-Da-d])[\.\)]?",
        a,
        flags=re.I,
    )

    mb = re.fullmatch(
        r"(?:option\s*)?([A-Da-d])[\.\)]?",
        b,
        flags=re.I,
    )

    if ma and mb:
        return (
            ma.group(1).upper()
            ==
            mb.group(1).upper()
        )

    # numeric
    try:

        fa = float(
            a.replace(",", "")
        )

        fb = float(
            b.replace(",", "")
        )

        if abs(fa - fb) <= max(
            1e-6,
            1e-6 * abs(fb),
        ):
            return True

    except Exception:
        pass

    # symbolic
    try:
        import sympy as sp

        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor,
        )

        transformations = (
            standard_transformations
            + (
                implicit_multiplication_application,
                convert_xor,
            )
        )

        ea = parse_expr(
            _to_sympy_text(a),
            local_dict={
                "sqrt": sp.sqrt,
            },
            transformations=transformations,
            evaluate=True,
        )

        eb = parse_expr(
            _to_sympy_text(b),
            local_dict={
                "sqrt": sp.sqrt,
            },
            transformations=transformations,
            evaluate=True,
        )

        return bool(
            sp.simplify(ea - eb) == 0
        )

    except Exception:
        return False





def _normalize_degree_notation(value):
    """
    Normalize equivalent degree notations without changing
    the underlying mathematical value.

    Examples:
        45^\\circ      -> 45°
        45^{\\circ}    -> 45°
        45 °           -> 45°
    """
    if value is None:
        return value

    import re

    s = str(value)

    # LaTeX: 45^{\circ}
    s = re.sub(
        r"\^\s*\{\s*\\circ\s*\}",
        "°",
        s,
    )

    # LaTeX: 45^\circ
    s = re.sub(
        r"\^\s*\\circ\b",
        "°",
        s,
    )

    # Remaining standalone \circ
    s = re.sub(
        r"\\circ\b",
        "°",
        s,
    )

    # Normalize spacing before degree symbol
    s = re.sub(
        r"\s+°",
        "°",
        s,
    )

    return s

def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    # Normalize equivalent degree-symbol formats.
    solution_str = _normalize_degree_notation(solution_str)
    ground_truth = _normalize_degree_notation(ground_truth)

    extra_info = extra_info if isinstance(extra_info, dict) else {}

    task = str(
        extra_info.get("task")
        or data_source
        or ""
    ).lower()

    # Judge RLVR
    if task == "judge_rlvr" or data_source == "judge_rlvr":

        pred = _judge_verdict(solution_str)

        return 1.0 if pred == str(ground_truth).strip() else 0.0

    # Solver RLVR: use VERL's math_dapo verifier first.
    try:
        from verl.utils.reward_score import math_dapo

        result = math_dapo.compute_score(
            solution_str,
            ground_truth,
        )

        if isinstance(result, dict):
            score = result.get(
                "score",
                result.get("reward", 0.0),
            )
        elif isinstance(result, (tuple, list)):
            score = result[0]
        else:
            score = result

        # math_dapo positive => definitely correct.
        # If it returns 0, do NOT immediately mark wrong:
        # continue to our task-specific fallback parser below.
        if float(score) > 0:
            return 1.0

    except Exception:
        pass

    # Lightweight fallback when math_dapo does not verify the answer.
    pred = _extract_answer(solution_str)

    if pred is None:
        return 0.0

    a = re.sub(r"\s+", "", str(pred))
    b = re.sub(r"\s+", "", str(ground_truth))

    if a.upper() == b.upper():
        return 1.0

    # symbolic / numeric equivalence fallback
    if _math_equivalent(
        pred,
        ground_truth,
    ):
        return 1.0

    return 0.0
