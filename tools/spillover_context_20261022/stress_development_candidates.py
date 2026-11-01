"""Whole-run LOO service-error stress for finite diagnostic search candidates.

Observed whole-run residual ratios scale every service in a scenario together;
this preserves within-plan FIFO effects and avoids independent request noise.
It is a sensitivity envelope, not a predictive distribution or a new utility.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.spillover_context_20261022.development_oracle import DevelopmentOracle, sha


def scales(oracle):
    """Each scale uses a fit that excluded that whole execution run."""
    development = ROOT/'artifacts/static-spillover-study-20261022/local-context-development-v1'
    folds = oracle._read(development/'run-loo-diagnostics.json')
    chosen = next(c for c in folds if c['name'] == oracle.model['name'])
    records = oracle._read(oracle.inventory/'requests.json', oracle.inventory_manifest['files']['requests.json'])
    grouped = defaultdict(list)
    for r in records:
        if r['valid_for_fit'] and (r['family'], r['task'], r['n']) in oracle.domain:
            grouped[r['run_id']].append(r)
    out = []
    for f in chosen['folds']:
        if not f['success']:
            continue
        rr = grouped[f['run_id']]
        beta = f['model']['coefficients']
        predicted = math.fsum(r['baseline_ms']*math.exp(math.fsum(
            r['features'][i]*b for i, b in zip(oracle.columns, beta))) for r in rr)
        actual = math.fsum(r['service_ms'] for r in rr)
        if not rr or predicted <= 0 or actual <= 0:
            raise ValueError('Missing or invalid held-run service totals')
        out.append(dict(run_id=f['run_id'], requests=len(rr),
            held_run_actual_total_ms=actual, leave_whole_run_out_point_total_ms=predicted,
            scale=actual/predicted))
    if len(out) != 19:
        raise ValueError('Expected all 19 frozen whole-run LOO folds')
    return out


class StressOracle(DevelopmentOracle):
    scale = 1.

    def _service(self, *args):
        values, strict, proofs, reasons = super()._service(*args)
        def scaled(table):
            return {k: v*self.scale if v is not None else None for k,v in table.items()}
        return scaled(values), scaled(strict), proofs, reasons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--top-controls', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--top', type=int, default=3)
    args = p.parse_args()
    if args.top <= 0:
        p.error('--top must be positive')
    oracle = StressOracle()
    source = oracle._read(args.top_controls)
    candidates = source['top10'][:args.top] + [c for c in source['controls'].values() if c]
    unique = {}
    for candidate in candidates:
        unique.setdefault(json.dumps(candidate['plan'], sort_keys=True), candidate['plan'])
    observed = scales(oracle)
    scenarios = [dict(run_id='point_candidate', scale=1.)] + observed
    output = []
    for plan in unique.values():
        cases = []
        for scenario in scenarios:
            oracle.scale = scenario['scale']
            runs = []
            for seed in [107,108]:
                d = oracle.evaluate(plan, seed=seed, diagnostic_extrapolation=True)['diagnostic_extrapolation_only']
                runs.append(dict(seed=seed, U=d['U'], terms=d['terms'],
                    planning_feasible=d['resource_feasible'], max_rho=d['max_rho']))
            known = all(r['U'] is not None for r in runs)
            cases.append(dict(scenario=scenario['run_id'], scale=scenario['scale'], runs=runs,
                mean_U=statistics.mean(r['U'] for r in runs) if known else None,
                both_seeds_planning_feasible=all(r['planning_feasible'] is True for r in runs)))
        output.append(dict(plan=plan, cases=cases))
    # Pair each candidate with the best no-text-split control from the search.
    reference = source['controls'].get('no_text_split')
    ref = next((r for r in output if reference and r['plan'] == reference['plan']), None)
    if ref:
        for row in output:
            pairs = list(zip(row['cases'][1:],ref['cases'][1:]))
            differences = [a['mean_U']-b['mean_U'] for a,b in pairs
                           if a['mean_U'] is not None and b['mean_U'] is not None
                           and a['both_seeds_planning_feasible'] and b['both_seeds_planning_feasible']]
            row['delta_from_search_no_text_split'] = dict(
                min=min(differences) if differences else None,
                max=max(differences) if differences else None,
                positive_scenarios=sum(v>1e-10 for v in differences),
                negative_scenarios=sum(v<-1e-10 for v in differences),
                scenarios=len(differences),excluded_not_jointly_planning_feasible=len(pairs)-len(differences),
                scope='Both candidates must pass diagnostic planning gates in both seeds.',
                probabilities_estimated=False)
    oracle.verify_inputs()
    report = dict(schema='gsdeploy.development_whole_run_service_stress.v1',
        scales=observed, candidates=output, formal_U=None, robust_optimality_certified=False,
        utility_weights_changed=False, scenario_selection_uses_gpu108=False,
        limitations=['A global held-run ratio is a diagnostic perturbation, not a mechanistic interference model.',
            'LOO runs differ in deployment and load; scenarios are neither IID predictions nor calibrated probabilities.',
            'Q/M/E candidates are held fixed; their uncertainty remains unresolved.',
            'Original runtime stability remains UNKNOWN; numeric rho/memory/drain gates are development diagnostics.',
            'Stress tests of a short list do not establish robust optimality over the whole search domain.'],
        provenance=oracle.provenance())
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'report.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(dict(
        script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        inputs=oracle.inputs,files={'report.json':sha(path)}),indent=2)+'\n')
    print(json.dumps(dict(candidates=len(output),scenarios=len(observed),
        scale_min=min(r['scale'] for r in observed),scale_max=max(r['scale'] for r in observed),
        output=str(args.output.resolve()))))


if __name__=='__main__':
    main()
