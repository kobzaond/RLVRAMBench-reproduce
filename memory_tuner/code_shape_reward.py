"""Non-executing syntax/shape reward for code-generation memory workloads."""

from __future__ import annotations

import ast
import re


def extract_python(solution: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", solution, re.DOTALL)
    return (fenced.group(1) if fenced else solution).strip()


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Score code structure without running untrusted generated programs."""
    del data_source, ground_truth, extra_info, kwargs
    code = extract_python(solution_str)
    if not code:
        return 0.0
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0
    has_input = False
    has_output = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            has_input |= node.func.id == "input"
            has_output |= node.func.id == "print"
        elif isinstance(node.func, ast.Attribute):
            has_input |= node.func.attr in {"read", "readline"}
            has_output |= node.func.attr in {"write", "writelines"}
    if has_input and has_output:
        return 1.0
    if has_input or has_output:
        return 0.5
    return 0.25
