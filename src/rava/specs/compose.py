from __future__ import annotations

from typing import Iterable

from rava.specs.schema import Verdict


def negate(v: Verdict) -> Verdict:
    if v == Verdict.PASS:
        return Verdict.FAIL
    if v == Verdict.FAIL:
        return Verdict.PASS
    return Verdict.UNCERTAIN


def and_semantics(values: Iterable[Verdict]) -> Verdict:
    vals = list(values)
    if any(v == Verdict.FAIL for v in vals):
        return Verdict.FAIL
    if vals and all(v == Verdict.PASS for v in vals):
        return Verdict.PASS
    return Verdict.UNCERTAIN


def or_semantics(values: Iterable[Verdict]) -> Verdict:
    vals = list(values)
    if any(v == Verdict.PASS for v in vals):
        return Verdict.PASS
    if vals and all(v == Verdict.FAIL for v in vals):
        return Verdict.FAIL
    return Verdict.UNCERTAIN


def implies_semantics(a: Verdict, b: Verdict) -> Verdict:
    # a -> b  ==  (not a) OR b in three-valued logic.
    return or_semantics([negate(a), b])


def evaluate_expression(expr: dict, atoms: dict[str, Verdict]) -> Verdict:
    op = str(expr.get("op", "ATOM")).upper()

    if op == "ATOM":
        atom_id = expr["id"]
        return atoms.get(atom_id, Verdict.UNCERTAIN)

    if op == "AND":
        args = [evaluate_expression(arg, atoms) for arg in expr.get("args", [])]
        return and_semantics(args)

    if op == "OR":
        args = [evaluate_expression(arg, atoms) for arg in expr.get("args", [])]
        return or_semantics(args)

    if op == "IMPLIES":
        args = expr.get("args", [])
        if len(args) != 2:
            return Verdict.UNCERTAIN
        return implies_semantics(
            evaluate_expression(args[0], atoms),
            evaluate_expression(args[1], atoms),
        )

    return Verdict.UNCERTAIN
