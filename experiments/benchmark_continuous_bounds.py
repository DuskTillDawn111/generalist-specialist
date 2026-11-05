"""Matched-route ablations plus independently run end-to-end DE verification."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsdeploy.continuous import Budget, ContinuousRoutingEvaluator, DERoutingOptimizer
from gsdeploy.continuous.benchmark import make_problem, oracle_from_spec, compact_winner, evaluate_holdout
from gsdeploy.continuous.contracts import digest, load_experiment

ROOT = Path(__file__).resolve().parents[1]


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def snapshot(output):
    files = [p for folder in ('gsdeploy','configs','tests','scripts','experiments','docs')
             for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.md')]
    files += [ROOT/'README.md',ROOT/'pyproject.toml']
    hashes = {}
    for p in sorted(files):
        rel=p.relative_to(ROOT)
        target=output/'source-snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
        hashes[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(output/'source-manifest.json',dict(files=hashes,manifest_sha256=digest(hashes)))


def run(config, output, seeds, outer, seconds, predictions):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    spec,original=load_experiment(config)
    save(output/'configuration.json',dict(spec=spec,seeds=seeds,outer=outer,seconds=seconds,predictions=predictions))
    snapshot(output)
    started=time.perf_counter()
    records=[]
    problems={}
    cases=[(name,load,'nominal',spec['trace_seeds']) for name,load in spec['workloads'].items()]
    cases += [('mixed_robust',spec['workloads']['mixed'],'robust',spec['trace_seeds']),
              ('mixed_multitrace',spec['workloads']['mixed'],'nominal',{'train':[101,211],'validation':[223],'test':[307,311]})]
    for name,load,policy,traces in cases:
        problem=make_problem(replace(original.config,workload=load,search_policy=policy),traces)
        problems[name]=dict(identity=problem.identity,config=problem.config.as_dict(),states=problem.states,
                            traces=[asdict(t) for t in problem.traces])
        save(output/'problems.json',problems)
        for seed in (seeds if name in spec['workloads'] else seeds[:1]):
            common=dict(seconds=seconds,max_predictions=predictions,max_outer=outer)
            evaluator=ContinuousRoutingEvaluator(problem,oracle_from_spec(spec),'enumeration',Budget(**common))
            fit=DERoutingOptimizer(evaluator,seed=seed,**spec.get('de',{})).solve()
            fit['evaluations']=evaluator.history
            reference=evaluator.history
            route_hash=digest([r['routing_shares'] for r in reference])
            variants=[('enumeration_de',evaluator,fit)]
            # Rotate ablation ordering by seed to reduce systematic timing bias.
            modes=[('control',False,False),('bounds',True,False),('warm',False,True),('combined',True,True)]
            shift=seeds.index(seed)%len(modes)
            modes=modes[shift:]+modes[:shift]
            for label,bounds,warm in modes:
                e=ContinuousRoutingEvaluator(problem,oracle_from_spec(spec),'bounded',Budget(**common),
                    inner_bounds=bounds,inner_warm_start=warm)
                for expected in reference:
                    actual=e.evaluate(expected['routing_shares'])
                    if (actual['best_feasible_utility']!=expected['best_feasible_utility'] or
                        actual['certified']!=expected['certified']):
                        raise AssertionError((name,seed,label,'inner mismatch'))
                best=max((r for r in e.history if r['feasible']),key=lambda r:r['best_feasible_utility'],default=None)
                variants.append((label,e,dict(winner=best,status='MATCHED_ROUTES_COMPLETED',
                    evaluations=e.history,budget=e.budget.report(),continuous_optimality=False)))
            e=ContinuousRoutingEvaluator(problem,oracle_from_spec(spec),'bounded',Budget(**common),
                                         inner_warm_start=True)
            end=DERoutingOptimizer(e,seed=seed,**spec.get('de',{})).solve()
            end['evaluations']=e.history
            if digest([r['routing_shares'] for r in e.history])!=route_hash:
                raise AssertionError((name,seed,'end-to-end route sequence mismatch'))
            if compact_winner(end)!=compact_winner(fit):
                raise AssertionError((name,seed,'end-to-end winner mismatch'))
            variants.append(('combined_de',e,end))
            for label,e,raw in variants:
                winner=compact_winner(raw)
                raw['holdout']=evaluate_holdout(problem,e.oracle,winner,seconds,predictions)
                raw.update(problem_sha256=problem.identity,oracle_version=e.oracle.version,
                    route_sequence_sha256=route_hash,evidence_kind=e.oracle.evidence_kind)
                filename=f'{name}-seed{seed}-{label}.json'
                save(output/filename,raw)
                report=raw['budget']
                row=dict(case=name,seed=seed,variant=label,utility=winner['utility'] if winner else None,
                    predictions=report['full_prediction_calls'],leaves=report['predictor_leaf_calls'],
                    seconds=report['elapsed_seconds'],outer=report['outer_evaluations'],
                    bound_seconds=sum(r.get('bound_seconds',0) for r in e.history),
                    safely_pruned=sum(r['safely_pruned_deployments'] for r in e.history),
                    warm_evaluations=sum(r.get('warm_start_evaluated_deployments',0) for r in e.history),
                    status=raw['status'],all_certified=all(r['certified'] for r in e.history),
                    route_sequence_sha256=route_hash,raw_file=filename)
                records.append(row)
                with (output/'results.jsonl').open('a') as f:
                    f.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps(row),flush=True)
    summary=dict(records=records,total_wall_seconds=time.perf_counter()-started,
        all_matched=True,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        evidence_kind='synthetic_fixture',continuous_optimality=False,
        conclusion='Algorithm cost evidence only; no physical GPU validation')
    save(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/continuous_fixture.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,29,47])
    parser.add_argument('--outer',type=int,default=64)
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--predictions',type=int,default=2400)
    args=parser.parse_args()
    run(args.config,args.output,args.seeds,args.outer,args.seconds,args.predictions)
