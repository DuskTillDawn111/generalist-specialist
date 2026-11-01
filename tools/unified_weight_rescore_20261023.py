"""Reweight every inventoried native result; never change a measured run."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict
from itertools import combinations

TERMS = "QTLPME"
ROLES = ("coupled_spill", "matched_no_spill", "legacy_latency", "other_mixed")


def rational(value):
    return Fraction(str(value))


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def exact_dot(gain, weights):
    return sum((rational(gain[t])*rational(weights[t]) for t in TERMS), Fraction())


def evaluable(row):
    terms = row.get("terms", {})
    return (row.get("eligible") is True and row.get("utility_complete") is True
            and all(type(terms.get(t)) in (int, float) and math.isfinite(terms[t])
                    and terms[t] >= 0 for t in TERMS)
            and row.get("rescorable_numeric") is True)


def group_rankings(rows, schemes):
    """Retain strict inventory contexts and keep calibration out of formal ranking."""
    grouped = defaultdict(list)
    for row in rows:
        if evaluable(row):
            tier = "formal_stage" if row.get("formal_efficacy_eligible") else "diagnostic_only"
            grouped[(row["comparison_group_signature"], tier)].append(row)
    ranks, contrasts = [], []
    for (signature, tier), members in sorted(grouped.items()):
        first = members[0]
        context = dict(comparison_group_signature=signature, evidence_tier=tier,
                       study=first["study"], seed=first.get("seed"), workload=first["workload"],
                       metric_definition_signature=first["metric_definition_signature"],
                       pool_signature=first["pool_signature"], cohort_signature=first["cohort_signature"],
                       cohort_identity_complete=all(r.get("cohort_identity_complete") for r in members))
        for name, weights in schemes.items():
            scored = sorted([(-exact_dot(r["terms"], weights), r) for r in members],
                            key=lambda item: (-item[0], item[1]["label"]))
            entries = []
            previous, rank = None, 0
            for index, (score, row) in enumerate(scored, 1):
                if previous is None or score != previous:
                    rank = index
                previous = score
                entries.append(dict(rank=rank, label=row["label"], role=row.get("role"),
                                    stage=row.get("stage"), deployment=row["deployment"],
                                    declared_text_spill_allowed=row.get("declared_text_spill_allowed"),
                                    text_generalist_spills=row.get("text_generalist_spills"),
                                    U=float(score), U_exact=str(score)))
            ranks.append(dict(**context, scheme=name, candidates=entries,
                              scope="Observed compatible candidates only; no global optimality claim"))
            for (score_a, a), (score_b, b) in combinations(scored, 2):
                contrasts.append(dict(**context, scheme=name, candidate_a=a["label"], candidate_b=b["label"],
                                      U_a_minus_U_b=float(score_a-score_b),
                                      delta_exact=str(score_a-score_b), same_deployment=a["deployment"]==b["deployment"],
                                      pure_spill_causal_claim=False,
                                      interpretation="Comprehensive contrast; strict routing ablation requires separate policy identity check"))
    return ranks, contrasts


def gain(candidate, control):
    return {t: str(rational(control[t])-rational(candidate[t])) for t in TERMS}


def boxed_linear_maximum(coefficients, floor, cap):
    lo, hi = rational(floor), rational(cap)
    weights = {t:lo for t in TERMS}
    remaining = 1-sum(weights.values())
    if remaining < 0 or 6*hi < 1:
        raise ValueError("Empty bounded simplex")
    for t in sorted(TERMS, key=lambda t:rational(coefficients[t]), reverse=True):
        change = min(remaining, hi-weights[t])
        weights[t] += change
        remaining -= change
    assert remaining == 0
    return exact_dot(coefficients, weights), weights


def projected_simplex(values, lo, hi):
    lo, hi = rational(lo), rational(hi)
    weights = [min(hi, max(lo, rational(float(v)))) for v in values]
    remaining = 1-sum(weights)
    for index in range(len(weights)):
        shift = min(remaining, hi-weights[index]) if remaining>0 else max(remaining, lo-weights[index])
        weights[index] += shift
        remaining -= shift
        if remaining == 0:
            break
    if remaining:
        raise ValueError("Could not project solver witness into exact simplex")
    return weights


def solve(gains, floor=0., cap=1., objective="maximin", target=1e-6):
    """LP with exact primal witnesses and an exact maximin upper bound."""
    from scipy.optimize import linprog
    if not gains or any(g is None for g in gains):
        return dict(status="UNKNOWN", objective=objective)
    if 6*rational(floor)>1 or 6*rational(cap)<1:
        return dict(status="EMPTY_BOX", objective=objective)
    numeric = [[float(rational(g[t])) for t in TERMS] for g in gains]
    if objective == "maximin":
        c = [0.]*6+[-1.]
        aub = [[-v for v in row]+[1.] for row in numeric]
        bub = [0.]*len(gains)
        aeq, bounds = [[1.]*6+[0.]], [(floor,cap)]*6+[(None,None)]
    elif objective == "nearest_L1":
        c = [0.]*6+[1.]*6
        # An explicit tiny guard avoids presenting a rounded boundary witness
        # as meeting a stricter target than it actually meets.
        aub = [[-v for v in row]+[0.]*6 for row in numeric]
        bub = [-(target+1e-9)]*len(gains)
        for i in range(6):
            row = [0.]*12; row[i]=1.; row[6+i]=-1.
            aub.append(row); bub.append(1/6)
            row = [0.]*12; row[i]=-1.; row[6+i]=-1.
            aub.append(row); bub.append(-1/6)
        aeq, bounds = [[1.]*6+[0.]*6], [(floor,cap)]*6+[(0,None)]*6
    else:
        raise ValueError("Unknown objective")
    result = linprog(c, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=[1.], bounds=bounds,
                     method="highs", options=dict(primal_feasibility_tolerance=1e-9,
                                                  dual_feasibility_tolerance=1e-9))
    report = dict(objective=objective, floor=floor, cap=cap, solver_status=int(result.status),
                  status="SOLVER_OPTIMAL" if result.success else "SOLVER_INFEASIBLE" if result.status==2 else "SOLVER_OTHER",
                  solver_message=result.message, requested_positive_margin=target if objective=="nearest_L1" else None)
    if not result.success:
        return report
    weights = dict(zip(TERMS, projected_simplex(result.x[:6], floor, cap)))
    values = [exact_dot(g, weights) for g in gains]
    report.update(weights={t:float(w) for t,w in weights.items()},
                  weights_exact={t:str(w) for t,w in weights.items()},
                  exact_simplex_verified=sum(weights.values())==1,
                  exact_box_verified=all(rational(floor)<=w<=rational(cap) for w in weights.values()),
                  gains=[float(v) for v in values], gains_exact=[str(v) for v in values],
                  minimum_gain=float(min(values)), mean_gain=float(sum(values)/len(values)),
                  all_strictly_positive=all(v>0 for v in values),
                  L1_distance_from_equal=float(sum(abs(w-Fraction(1,6)) for w in weights.values())))
    if objective == "nearest_L1":
        report["target_exactly_verified"] = all(v>=rational(target) for v in values)
        report["nearest_scope"] = "Numerical L1 optimum; exact witness verifies feasibility, not uniqueness"
    else:
        alpha = projected_simplex([-float(v) for v in result.ineqlin.marginals], 0., 1.)
        combined = {t:sum(a*rational(g[t]) for a,g in zip(alpha,gains)) for t in TERMS}
        upper, upper_weights = boxed_linear_maximum(combined, floor, cap)
        assert upper >= min(values)
        report["upper_bound_certificate"] = dict(
            mixing_coefficients_exact=[str(v) for v in alpha],
            combined_gain_exact={t:str(v) for t,v in combined.items()},
            upper_bound_exact=str(upper), upper_bound=float(upper),
            primal_dual_gap=float(upper-min(values)),
            strict_positive_impossible_certified=upper<=0,
            proof="min_i(g_i·w) <= sum_i(alpha_i*g_i·w) <= max_box_simplex(sum_i alpha_i*g_i)·w; alpha>=0,sum alpha=1",
            maximizing_weight_exact={t:str(v) for t,v in upper_weights.items()})
    return report


def latest_study(root):
    path = root / "artifacts/queue-coupled-study-20261023/native-endpoint-diagnostics-v1/diagnostics.json"
    data = read(path)
    rows = {label:dict(role=r["role"], seed=r["seed"], terms={t:r["utility"][t+"_norm"] for t in TERMS},
                      raw=r["raw"], deployment=r["deployment"])
            for label,r in data["runs"].items()}
    paired = [gain(rows[f"coupled_spill-seed{s}"]["terms"],rows[f"matched_no_spill-seed{s}"]["terms"])
              for s in (303,304)]
    alternatives = [gain(rows[f"coupled_spill-seed{s}"]["terms"],rows[f"{role}-seed{s}"]["terms"])
                    for s in (303,304) for role in ROLES if role!="coupled_spill"]
    studies = []
    for floor in (0.,.01,.02,.05,.10):
        for cap in (1.,.5):
            studies.append(dict(floor=floor,cap=cap,maximin=solve(paired,floor,cap),
                                nearest_L1=solve(paired,floor,cap,"nearest_L1"),
                                best_of_all_four=solve(alternatives,floor,cap)))
    control = rows["legacy_latency-seed304"]["terms"]
    candidate = rows["coupled_spill-seed304"]["terms"]
    domination = gain(candidate,control)
    assert all(rational(v)<=0 for v in domination.values()) and any(rational(v)<0 for v in domination.values())
    return dict(source=receipt(path),rows=rows,seed_order=[303,304],paired_gains=paired,
                studies=studies,legacy_304_dominates_new_coupled=dict(
                    gain_exact=domination,all_nonpositive=True,
                    scope="Fixed seed304 candidate table only; not a full-pool or future-policy impossibility"))


def lower_latency_family(gains):
    """Only lower original L/P, leaving Q/T/M/E mutually equal."""
    from scipy.optimize import linprog
    # Maximize m with gain >= m*(4+l+p), a fractional objective. Feasibility
    # for a fixed positive margin remains a linear two-variable problem.
    target=1e-6
    constraints=[];bounds=[]
    for g in gains:
        a=sum(rational(g[t]) for t in "QTME")
        constraints.append([float(-rational(g["L"])+rational(target)),
                            float(-rational(g["P"])+rational(target))])
        bounds.append(float(a-4*rational(target)))
    # Maximize l+p: retain as much of the original latency preference as possible.
    result=linprog([-1.,-1.],A_ub=constraints,b_ub=bounds,bounds=[(0,1),(0,1)],method="highs")
    report=dict(target=target,status="SOLVER_OPTIMAL" if result.success else "NO_FEASIBLE_SOLUTION",
                definition="Q/T/M/E raw preference=1, L=l,P=p, normalized by 4+l+p; l,p in [0,1]")
    if result.success:
        l,p=map(float,result.x)
        w={t:(l if t=="L" else p if t=="P" else 1.)/(4+l+p) for t in TERMS}
        report.update(L_scale=l,P_scale=p,weights=w,gains=[float(exact_dot(g,w)) for g in gains],
                      scope="Numerical exploratory boundary; exact feasible schemes below are evaluated independently")
    return report


def rankings(latest, weights):
    result=[]
    for role in ROLES:
        values={str(seed):-float(exact_dot(latest["rows"][f"{role}-seed{seed}"]["terms"],weights)) for seed in (303,304)}
        result.append(dict(role=role,per_seed_U=values,mean_U=statistics.fmean(values.values())))
    result.sort(key=lambda r:(-r["mean_U"],r["role"]))
    return result


def analyze(inventory_path, output, root):
    inventory_path,output,root=map(Path,(inventory_path,output,root))
    inventory=read(inventory_path)
    rows=inventory["rows"]
    latest=latest_study(root)
    schemes={"original_equal":{t:Fraction(1,6) for t in TERMS},
             "readable_new_pair_example":dict(zip(TERMS,map(rational,(.15,.10,.20,.01,.04,.50)))),
             "balanced_readable_new_pair_example":dict(zip(TERMS,map(rational,(.15,.10,.22,.01,.05,.47)))),
             "only_reduce_P_to_0.001_raw_scale":{t:rational(.001 if t=="P" else 1)/rational(5.001) for t in TERMS}}
    for study in latest["studies"]:
        for name in ("maximin","nearest_L1"):
            fit=study[name]
            if fit.get("all_strictly_positive") and (name!="nearest_L1" or fit.get("target_exactly_verified")):
                label=f"new_pair_{name}_floor{study['floor']}_cap{study['cap']}"
                schemes[label]={t:Fraction(v) for t,v in fit["weights_exact"].items()}
    records=[]
    for label,weights in schemes.items():
        assert sum(weights.values())==1 and all(w>=0 for w in weights.values())
        for row in rows:
            known=evaluable(row)
            records.append(dict(scheme=label,label=row["label"],raw_path=row["raw"]["path"],
                                seed=row.get("seed"),stage=row.get("stage"),role=row.get("role"),
                                eligible=row.get("eligible"),complete_for_rescore=known,
                                formal_efficacy_eligible=row.get("formal_efficacy_eligible"),
                                comparison_group_signature=row.get("comparison_group_signature"),
                                excluded_reason=row.get("excluded_reason"),
                                U=-float(exact_dot(row["terms"],weights)) if known else None,
                                U_exact=str(-exact_dot(row["terms"],weights)) if known else None))
    grouped_rankings, pairwise_contrasts = group_rankings(rows, schemes)
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema="gsdeploy.unified_native_weight_rescore.v1",status="COMPLETE_POSTHOC_RESCORE",
                inventory=receipt(inventory_path),source=receipt(__file__),
                all_inventoried_runs=len(rows),evaluable_runs=sum(evaluable(r) for r in rows),
                formal_efficacy_runs=sum(evaluable(r) and r.get("formal_efficacy_eligible") is True for r in rows),
                numeric_comparison_groups=len(set(r["comparison_group_signature"] for r in rows if evaluable(r))),
                group_rankings_file="group-rankings.json", pairwise_contrasts_file="pairwise-contrasts.json",
                scheme_count=len(schemes),all_runs_scored_under_each_same_weight=True,
                schemes={name:dict(weights={t:float(w) for t,w in weights.items()},
                                  weights_exact={t:str(w) for t,w in weights.items()},
                                  latest_pair_gains=[float(exact_dot(g,weights)) for g in latest["paired_gains"]],
                                  fixed_trace_positive_L1_weight_radius_lower_bound=(
                                      float(min(exact_dot(g,weights)/max(abs(rational(g[t])) for t in TERMS)
                                                for g in latest["paired_gains"]))
                                      if all(exact_dot(g,weights)>0 for g in latest["paired_gains"]) else None),
                                  latest_four_candidate_ranking=rankings(latest,weights))
                         for name,weights in schemes.items()},
                latest=latest,only_reduce_L_P=lower_latency_family(latest["paired_gains"]),
                historical_bad_policies_not_required_to_be_rescued=True,
                GPU_runs_added=0,original_runs_changed=False,router_weights_changed=False,
                stable_future_positive_gain_certified=False,physical_full_pool_optimality=False,
                interpretation="Uniform retrospective preference scoring; candidate ranking within compatible contexts only. A positive fixed-route contrast is not a new-weight policy validation.")
    write(output/"analysis.json",result)
    write(output/"rescored-runs.json",records)
    write(output/"group-rankings.json",grouped_rankings)
    write(output/"pairwise-contrasts.json",pairwise_contrasts)
    with (output/"rescored-runs.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    with (output/"weight-schemes.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["scheme",*TERMS,"new303_gain","new304_gain","new_four_candidate_winner"])
        writer.writeheader()
        for name,scheme in result["schemes"].items():
            writer.writerow(dict(scheme=name,**scheme["weights"],new303_gain=scheme["latest_pair_gains"][0],
                                 new304_gain=scheme["latest_pair_gains"][1],
                                 new_four_candidate_winner=scheme["latest_four_candidate_ranking"][0]["role"]))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args();result=analyze(a.inventory,a.output,a.root)
    print(json.dumps({k:result[k] for k in ("all_inventoried_runs","evaluable_runs","scheme_count")},ensure_ascii=False))
