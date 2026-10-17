"""Generated Python execution is disabled until a trusted isolated service is integrated.

An AST filter or subprocess is NOT an isolation boundary. Do not replace this
fail-closed gate with eval/exec or a candidate-printable completion marker.
Protocol v2 requires host-enforced isolation, credential-free workers, resource
limits, parent-enforced termination and harness completion attested outside the
candidate process. No such evaluator ships with this repository.
"""

import ast
import math
import operator

PROTOCOL_VERSION = "isolated-evaluation-v2-unavailable"


class IsolatedEvaluatorUnavailable(RuntimeError):
    pass


def require_isolated_evaluator():
    raise IsolatedEvaluatorUnavailable(
        "Generated Python evaluation requires a trusted isolated evaluator; "
        "no evaluator is configured. Refusing to execute or assign a score."
    )


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "pow": pow,
}


def arithmetic_value(expression):
    """Bounded numeric AST interpreter, not a Python execution sandbox."""
    if not isinstance(expression, str) or len(expression) > 512:
        return None

    def bounded(value):
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError("non-numeric value")
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("numeric limit")
        return value

    def visit(node, depth=0):
        if depth > 20:
            raise ValueError("expression depth")
        if isinstance(node, ast.Constant):
            return bounded(node.value)
        if isinstance(node, ast.Name) and node.id in {"pi", "e"}:
            return getattr(math, node.id)
        if isinstance(node, ast.UnaryOp) and type(node.op) in {ast.UAdd, ast.USub}:
            value = visit(node.operand, depth + 1)
            return bounded(value if isinstance(node.op, ast.UAdd) else -value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent limit")
            return bounded(_BINARY[type(node.op)](left, right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _FUNCTIONS or node.keywords or not 1 <= len(node.args) <= 8:
                raise ValueError("unsupported call")
            args = [visit(arg, depth + 1) for arg in node.args]
            if node.func.id == "pow" and (len(args) != 2 or abs(args[1]) > 100):
                raise ValueError("exponent limit")
            if node.func.id == "round" and len(args) > 1 and abs(args[1]) > 100:
                raise ValueError("precision limit")
            return bounded(_FUNCTIONS[node.func.id](*args))
        raise ValueError("unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            return None
        return float(visit(tree.body))
    except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError):
        return None


def evaluate_python(*, code, tests=None, entry_point=None, expected_stdout=None, timeout=10):
    require_isolated_evaluator()
