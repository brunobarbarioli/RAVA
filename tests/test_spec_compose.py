from rava.specs.compose import and_semantics, evaluate_expression, implies_semantics, or_semantics
from rava.specs.schema import Verdict


def test_and_semantics_with_uncertain():
    assert and_semantics([Verdict.PASS, Verdict.UNCERTAIN]) == Verdict.UNCERTAIN
    assert and_semantics([Verdict.PASS, Verdict.PASS]) == Verdict.PASS
    assert and_semantics([Verdict.PASS, Verdict.FAIL]) == Verdict.FAIL


def test_or_semantics_with_uncertain():
    assert or_semantics([Verdict.FAIL, Verdict.UNCERTAIN]) == Verdict.UNCERTAIN
    assert or_semantics([Verdict.FAIL, Verdict.FAIL]) == Verdict.FAIL
    assert or_semantics([Verdict.PASS, Verdict.FAIL]) == Verdict.PASS


def test_implies_semantics_three_valued():
    assert implies_semantics(Verdict.PASS, Verdict.FAIL) == Verdict.FAIL
    assert implies_semantics(Verdict.FAIL, Verdict.FAIL) == Verdict.PASS
    assert implies_semantics(Verdict.UNCERTAIN, Verdict.FAIL) == Verdict.UNCERTAIN


def test_expression_eval():
    expr = {
        "op": "IMPLIES",
        "args": [
            {"op": "AND", "args": [{"op": "ATOM", "id": "A"}, {"op": "ATOM", "id": "B"}]},
            {"op": "OR", "args": [{"op": "ATOM", "id": "C"}, {"op": "ATOM", "id": "D"}]},
        ],
    }
    atoms = {"A": Verdict.PASS, "B": Verdict.PASS, "C": Verdict.FAIL, "D": Verdict.PASS}
    assert evaluate_expression(expr, atoms) == Verdict.PASS
