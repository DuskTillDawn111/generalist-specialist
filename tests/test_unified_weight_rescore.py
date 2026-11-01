from fractions import Fraction
import unittest

from tools.unified_weight_rescore_20261023 import (
    TERMS, boxed_linear_maximum, evaluable, group_rankings, solve,
)


def test_opposing_pairs_have_zero_best_worst_gain():
    positive = dict.fromkeys(TERMS, 0)
    positive.update(Q=1, E=-1)
    negative = {t: -v for t, v in positive.items()}
    result = solve([positive, negative])
    assert abs(result["minimum_gain"]) < 1e-14
    assert Fraction(result["upper_bound_certificate"]["upper_bound_exact"]) == 0


def test_dominated_candidate_has_certified_negative_bound_when_all_metrics_matter():
    result = solve([dict.fromkeys(TERMS, -1)], floor=.01, cap=.5)
    assert result["minimum_gain"] == -1
    assert result["upper_bound_certificate"]["strict_positive_impossible_certified"]


def test_box_cap_prevents_all_weight_being_assigned_to_best_term():
    coefficients = dict.fromkeys(TERMS, 0)
    coefficients["E"] = 2
    value, weights = boxed_linear_maximum(coefficients, ".1", ".4")
    assert value == Fraction(4, 5)
    assert sum(weights.values()) == 1


def test_missing_energy_is_unknown_but_complete_calibration_is_rescorable():
    row = dict(eligible=True, utility_complete=True, rescorable_numeric=True,
               excluded_reason=["NOT_FORMAL_EFFICACY_PHASE:calibration"], terms=dict.fromkeys(TERMS, 1))
    assert evaluable(row)
    row["terms"]["E"] = None
    assert not evaluable(row)


def test_ties_keep_shared_rank_and_different_contexts_stay_separate():
    base = dict(eligible=True, utility_complete=True, rescorable_numeric=True,
                formal_efficacy_eligible=True, terms=dict.fromkeys(TERMS, 1),
                comparison_group_signature="same", study="test", workload={"text": 1},
                metric_definition_signature="m", pool_signature="p", cohort_signature="c",
                deployment={"text": 2}, cohort_identity_complete=True)
    rows = [dict(base, label="a"), dict(base, label="b"),
            dict(base, label="different_load", comparison_group_signature="other", workload={"text": 2})]
    rankings, contrasts = group_rankings(rows, {"equal": dict.fromkeys(TERMS, Fraction(1, 6))})
    assert len(rankings) == 2 and len(contrasts) == 1
    tied = next(r for r in rankings if len(r["candidates"]) == 2)
    assert [r["rank"] for r in tied["candidates"]] == [1, 1]
    assert contrasts[0]["U_a_minus_U_b"] == 0


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for name, test in sorted(globals().items())
                              if name.startswith("test_") and callable(test))


if __name__ == "__main__":
    unittest.main()
