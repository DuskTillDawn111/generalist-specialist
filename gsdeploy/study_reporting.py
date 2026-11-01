"""Paired static-study summaries; requests are never independent repetitions.

Integrity checking belongs to the collection importer. These functions consume
verified run summaries, preserve missing blocks, and make no global GPU claim.
"""
import itertools
import math
import statistics

from scipy.stats import t as student_t


TERMS = ("Q", "T", "L", "P", "M", "E")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def paired_comparisons(rows, winner_key, planned_seeds, *, planned_plan_keys=None, alpha=.05):
    """Compare a frozen selected plan against each frozen rival on paired seeds.

    The Student interval assumes independent, approximately normal seed-level
    differences. The exact two-sided sign-flip test additionally assumes a
    symmetric null distribution. With three blocks its smallest p is .25.
    Neither test extrapolates beyond the fixed inputs and measured candidates.
    """
    seeds = list(planned_seeds)
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("Need at least two distinct predeclared seed blocks")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one")
    keys = list(planned_plan_keys) if planned_plan_keys is not None else None
    if keys is not None and (len(keys) != len(set(keys)) or winner_key not in keys):
        raise ValueError("Declared candidates must be unique and include the selected plan")
    groups = {key: {} for key in keys or []}
    for row in rows:
        key, seed = row["plan_key"], row["seed"]
        if keys is not None and key not in groups:
            raise ValueError("Run outside predeclared candidate roster")
        if seed not in seeds:
            raise ValueError("Run outside predeclared seed blocks")
        if seed in groups.setdefault(key, {}):
            raise ValueError("Repeated plan/seed cannot count as independent evidence")
        utility = row.get("utility", {})
        if utility.get("eligible") is not True or not all(_finite(utility.get(term + "_norm")) for term in TERMS) or not _finite(utility.get("U")):
            raise ValueError("Only verified complete six-term measured utilities may be compared")
        groups[key][seed] = row
    comparisons = []
    selected = groups.get(winner_key, {})
    for rival_key, rival in sorted(groups.items()):
        if rival_key == winner_key:
            continue
        complete = [seed for seed in seeds if seed in selected and seed in rival]
        missing = [seed for seed in seeds if seed not in complete]
        entry = {"winner_key": winner_key, "rival_key": rival_key,
                 "paired_seeds": complete, "missing_seeds": missing,
                 "status": "INCOMPLETE_BLOCKS" if missing else "COMPLETE_PAIRED_BLOCKS"}
        # Partial results stay descriptive; no early stopping on favorable seeds.
        delta = [selected[seed]["utility"]["U"] - rival[seed]["utility"]["U"] for seed in complete]
        entry["U_differences"] = delta
        entry["mean_U_difference"] = statistics.fmean(delta) if delta else None
        if not missing:
            n = len(delta)
            estimate = statistics.fmean(delta)
            se = statistics.stdev(delta) / math.sqrt(n)
            half = float(student_t.ppf(1 - alpha / 2, n - 1)) * se
            observed = abs(math.fsum(delta))
            total = 2 ** n
            exceed = sum(abs(math.fsum(s * d for s, d in zip(signs, delta))) >= observed - 1e-12
                         for signs in itertools.product((-1, 1), repeat=n))
            entry.update({"standard_error": se,
                          "student_interval": [estimate - half, estimate + half],
                          "student_confidence_level": 1 - alpha,
                          "exact_two_sided_sign_flip_p": exceed / total,
                          "all_observed_differences_positive": all(d > 0 for d in delta),
                          "mean_term_differences": {term: statistics.fmean(
                              selected[seed]["utility"][term + "_norm"] - rival[seed]["utility"][term + "_norm"]
                              for seed in seeds) for term in TERMS}})
        comparisons.append(entry)
    # Holm correction is valid under arbitrary dependence among rival tests.
    tested = sorted((c for c in comparisons if "exact_two_sided_sign_flip_p" in c),
                    key=lambda c: c["exact_two_sided_sign_flip_p"])
    adjusted = 0.0
    for index, comparison in enumerate(tested):
        adjusted = max(adjusted, min(1.0, (len(comparisons) - index) * comparison["exact_two_sided_sign_flip_p"]))
        comparison["holm_adjusted_sign_flip_p"] = adjusted
        comparison["sign_flip_significant_after_holm"] = adjusted < alpha
    return {"schema": "gsdeploy.paired_measured_comparisons.v1", "winner_key": winner_key,
            "planned_seeds": seeds, "comparisons": comparisons,
            "all_planned_blocks_complete": all(set(group) == set(seeds) for group in groups.values()) and bool(groups),
            "independent_units": "predeclared trace seed blocks, conditional on fixed input pool; repeated requests are not independent quality samples",
            "assumptions": {"student_interval": "independent approximately normal seed-level differences",
                            "sign_flip": "independent seed blocks and symmetric paired null differences"},
            "smallest_possible_two_sided_sign_flip_p": 2 / 2 ** len(seeds),
            "physical_global_optimality_certified": False,
            "scope": "Measured predeclared candidates and seeds only; predictor certificates reported separately"}


def measured_summary(rows, weights):
    """Retain each run and report six weighted costs without renormalizing."""
    if set(weights) != set(TERMS) or not all(_finite(v) and v >= 0 for v in weights.values()):
        raise ValueError("Six finite nonnegative utility weights required")
    results = []
    for row in rows:
        utility = row["utility"]
        if utility.get("eligible") is not True or not _finite(utility.get("U")):
            raise ValueError("Measured utility is incomplete")
        terms = {term: utility.get(term + "_norm") for term in TERMS}
        if not all(_finite(v) for v in terms.values()):
            raise ValueError("Missing normalized utility component")
        costs = {term: weights[term] * terms[term] for term in TERMS}
        if not math.isclose(utility["U"], -math.fsum(costs.values()), abs_tol=1e-8, rel_tol=1e-8):
            raise ValueError("Six-term utility does not match declared scoring weights")
        results.append({**row, "weighted_costs": costs})
    return results
