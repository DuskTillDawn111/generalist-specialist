"""Retrospective utility with P95 weight exactly zero; old evidence stays intact."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from pathlib import Path

from tools.unified_weight_rescore_20261023 import (
    TERMS, evaluable, exact_dot, group_rankings, projected_simplex,
    rankings, rational, read, receipt, write,
)

ACTIVE = "QTLME"


def solve_five(gains, floor=0., cap=1.):
    """Maximize the smallest contrast with P fixed to zero, using exact witnesses."""
    from scipy.optimize import linprog
    lo, hi = rational(floor), rational(cap)
    if 5*lo > 1 or 5*hi < 1:
        return dict(status="EMPTY_BOX", floor=floor, cap=cap)
    result = linprog([0.]*5+[-1.],
                     A_ub=[[-float(rational(g[t])) for t in ACTIVE]+[1.] for g in gains],
                     b_ub=[0.]*len(gains), A_eq=[[1.]*5+[0.]], b_eq=[1.],
                     bounds=[(floor, cap)]*5+[(None, None)], method="highs",
                     options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    if not result.success:
        return dict(status="SOLVER_OTHER", message=result.message, floor=floor, cap=cap)
    weights = dict(zip(ACTIVE, projected_simplex(result.x[:5], floor, cap)))
    weights["P"] = Fraction()
    values = [exact_dot(g, weights) for g in gains]
    alpha = projected_simplex([-float(v) for v in result.ineqlin.marginals], 0., 1.)
    coefficients = {t: sum(a*rational(g[t]) for a,g in zip(alpha, gains)) for t in ACTIVE}
    upper_weights = dict.fromkeys(ACTIVE, lo)
    remaining = 1-5*lo
    for t in sorted(ACTIVE, key=coefficients.get, reverse=True):
        delta = min(remaining, hi-lo)
        upper_weights[t] += delta
        remaining -= delta
    assert remaining == 0
    upper = sum(coefficients[t]*upper_weights[t] for t in ACTIVE)
    assert upper >= min(values)
    assert weights["P"] == 0 and sum(weights.values()) == 1
    return dict(status="SOLVER_OPTIMAL", floor_active_terms=floor, cap_active_terms=cap,
                weights_exact={t: str(weights[t]) for t in TERMS},
                weights={t: float(weights[t]) for t in TERMS},
                gains_exact=[str(v) for v in values], gains=[float(v) for v in values],
                minimum_gain=float(min(values)), all_strictly_positive=all(v>0 for v in values),
                upper_bound_certificate=dict(alpha_exact=[str(v) for v in alpha],
                                             upper_bound_exact=str(upper), upper_bound=float(upper),
                                             primal_dual_gap=float(upper-min(values)),
                                             strict_positive_impossible_certified=upper<=0,
                                             proof="min_i g_i·w <= sum_i alpha_i*g_i·w <= five-term bounded-simplex maximum; P=0"))


def analyze(root, output):
    root, output = Path(root), Path(output)
    base = root/"artifacts/unified-weight-study-20261023"
    inventory_path = base/"inventory-v3/inventory.json"
    previous_path = base/"rescore-v3/analysis.json"
    pairs_path = base/"pair-scores-v2/analysis.json"
    rows = read(inventory_path)["rows"]
    previous, pairs = read(previous_path), read(pairs_path)
    schemes, aliases = {}, {}
    for parent, scheme in previous["schemes"].items():
        w = {t: rational(scheme["weights_exact"][t]) for t in TERMS}
        total = sum(w[t] for t in ACTIVE)
        if not total:
            continue
        weights = {t: w[t]/total if t in ACTIVE else Fraction() for t in TERMS}
        key = tuple(weights[t] for t in TERMS)
        if key not in aliases:
            label = "five_equal" if parent=="original_equal" else parent+"_without_P"
            aliases[key] = label
            schemes[label] = dict(weights=weights, parents=[])
        schemes[aliases[key]]["parents"].append(parent)
    latest = previous["latest"]
    domains = [(floor, cap) for floor in (0., .01, .02, .05, .10, .20) for cap in (1., .5)]
    family_fits = {}
    for name, family in pairs["repeated_comparison_families"].items():
        vectors = [pairs["pairs"][i]["gain_exact"] for i in family["pair_indices"]]
        family_fits[name] = dict(pair_indices=family["pair_indices"], seed_order=family["seed_order"],
                                 pair_count=len(vectors),
                                 fits=[solve_five(vectors, floor, cap) for floor,cap in domains])
    best_all = []
    from tools.unified_weight_rescore_20261023 import gain, ROLES
    alternatives = [gain(latest["rows"][f"coupled_spill-seed{s}"]["terms"], latest["rows"][f"{r}-seed{s}"]["terms"])
                    for s in (303,304) for r in ROLES if r!="coupled_spill"]
    for floor, cap in domains:
        best_all.append(solve_five(alternatives, floor, cap))
    # Include the fresh five-term maximin witnesses in the same uniform scoring.
    for fit in family_fits["coupled303_304_matched"]["fits"]:
        weights = {t: rational(fit["weights_exact"][t]) for t in TERMS}
        key = tuple(weights[t] for t in TERMS)
        if fit["all_strictly_positive"] and key not in aliases:
            label = f"five_maximin_floor{fit['floor_active_terms']}_cap{fit['cap_active_terms']}"
            aliases[key] = label
            schemes[label] = dict(weights=weights, parents=[])
    weights_only = {label:s["weights"] for label,s in schemes.items()}
    group_results, contrasts = group_rankings(rows, weights_only)
    records, declared = [], []
    for label, weights in weights_only.items():
        assert weights["P"]==0 and sum(weights.values())==1 and all(v>=0 for v in weights.values())
        for row in rows:
            known = evaluable(row)
            value = -exact_dot(row["terms"],weights) if known else None
            records.append(dict(scheme=label,label=row["label"],seed=row["seed"],study=row["study"],
                                role=row.get("role"),formal_efficacy_eligible=row["formal_efficacy_eligible"],
                                U=float(value) if known else None,U_exact=str(value) if known else None,
                                excluded_reason=row["excluded_reason"]))
        for pair in pairs["pairs"]:
            value = exact_dot(pair["gain_exact"],weights)
            declared.append(dict(scheme=label,study=pair["study"],seed=pair["seed"],
                                 text_rps=pair["workload"].get("text"),family=pair["family"],
                                 spill=pair["pair"]["spill"],control=pair["pair"]["control"],
                                 comparison_kind=pair["pair"]["comparison_kind"],
                                 delta_U=float(value),delta_exact=str(value)))
    result = dict(schema="gsdeploy.no_p95_posthoc_rescore.v1",source=receipt(__file__),
                  inventory=receipt(inventory_path),previous=receipt(previous_path),pairs=receipt(pairs_path),
                  record_count=len(rows),numeric_records=sum(evaluable(r) for r in rows),
                  formal_phase_numeric_records=sum(evaluable(r) and r["formal_efficacy_eligible"] for r in rows),
                  scheme_count=len(schemes),P_weight_exactly_zero=True,
                  schemes={label:dict(weights={t:float(w) for t,w in s["weights"].items()},
                                     weights_exact={t:str(w) for t,w in s["weights"].items()},
                                     parent_schemes=s["parents"],
                                     latest_pair_gains=[float(exact_dot(g,s["weights"])) for g in latest["paired_gains"]],
                                     latest_four_candidate_ranking=rankings(latest,s["weights"]))
                           for label,s in schemes.items()},
                  family_five_term_maximin=family_fits,latest_best_of_all_candidates=best_all,
                  comprehensive_group_rankings=len(group_results),comprehensive_contrasts=len(contrasts),
                  declared_spill_contrasts=len(declared),GPU_runs_added=0,
                  scope="Retrospective utility only; P95 measurements remain diagnostic; original admission and stability unchanged; no independent future-positive or full-pool optimality claim")
    output.mkdir(parents=True,exist_ok=False)
    for name,obj in [("analysis",result),("rescored-runs",records),("group-rankings",group_results),
                     ("pairwise-contrasts",contrasts),("declared-contrasts",declared)]:
        write(output/(name+".json"),obj)
    for name,entries in [("rescored-runs",records),("declared-contrasts",declared)]:
        with (output/(name+".csv")).open("x",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(entries[0]))
            writer.writeheader(); writer.writerows(entries)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); result=analyze(a.root,a.output)
    print({k:result[k] for k in ("record_count","numeric_records","scheme_count")})
