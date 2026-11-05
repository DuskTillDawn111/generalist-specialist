"""Tiny CPU-only software comparison; never evidence of physical GPU gain."""

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.continuous import (Budget, ContinuousRoutingEvaluator, DERoutingOptimizer,
    ImmutableContinuousProblem, JointMILPContinuousRefinement, SessionPredictorOracle)
from gsdeploy.continuous.multi_candidate import JointDeploymentCandidates, MultiCandidateContinuousRefinement
from gsdeploy.continuous.adaptive_candidate import AdaptiveCandidateRefinementOptimizer
from gsdeploy.continuous.paper_structured import PaperStructuredOptimizer
from gsdeploy.continuous.predictor_adapter import normalize_prediction,source_identity
from gsdeploy.paper_capture import normalize_capture
from gsdeploy.paper_training import fit_config
from gsdeploy.paper_forecast import MODE,PaperFifoSession,canonical
from gsdeploy.profiled_resource_search import ProfiledResourceConfigurationSearch
from tests.test_paper_structured import config,predict,busy_case,resource_candidates,capture_fixture


def baseline(c,kind,options):
    budget=Budget(options['seconds'],options['max_predictions'],options['max_routes'])
    p=ImmutableContinuousProblem(c)
    with SessionPredictorOracle(p) as oracle:
        ex=JointDeploymentCandidates(p,oracle,budget)
        if kind=='Joint':
            raw=ex.solve_one(K=c.route_grid,excluded=[])
        elif kind=='DFS':
            module=oracle.session.module('optimizer');original=module.predict_scenarios
            def traced(*args,**kwargs):
                budget.consume('predictions');budget.counts['predictor_leaves']+=3;t=time.perf_counter()
                try:result=original(*args,**kwargs)
                finally:budget.times['prediction']+=time.perf_counter()-t
                ex.observe(args[1],args[2],result)
                return result
            opt=oracle.session.optimizer('dfs',dfs_time_limit_seconds=options['seconds'])
            with patch.object(module,'predict_scenarios',traced):raw=opt.solve(K=c.route_grid)
        else:
            e=ContinuousRoutingEvaluator(p,oracle,inner='milp_bounds',budget=budget)
            original_candidate=e.candidate
            def observed_candidate(x,routing,*args,**kwargs):
                result=original_candidate(x,routing,*args,**kwargs)
                ex.observe(x,routing,result['traces'][0]['prediction'])
                return result
            e.candidate=observed_candidate
            solved=DERoutingOptimizer(e,seed=options['seed'],popsize=3,maxiter=100).solve()
            raw=dict(**solved,history=e.history)
        return dict(algorithm=kind,raw=raw,winner=ex.best,budget=budget.report(),events=ex.events,
                    predictor_engine_stats=oracle.engine_stats(),status=raw['status'])


def events(raw):
    rows=raw.get('events',raw.get('convergence',[]))
    return [(r.get('seconds',r.get('elapsed_seconds')),r.get('utility')) for r in rows
            if r.get('utility') is not None]


def summarize(name,seed,raw,target):
    w=raw.get('winner');prediction=w['prediction'] if w else {}
    U=w.get('utility',w.get('score')) if w else None
    stats=raw.get('predictor_engine_stats',{}) or {}
    if 'fifo_stats' in raw:
        stats={**stats,'event_engine_calls':raw['fifo_stats']['engine_calls'],
               'fifo_cache_hits':raw['fifo_stats']['fifo_cache_hits']}
    b=raw['budget'];times=b['seconds_by_component']
    return dict(method=name,seed=seed,U=U,total_seconds=b['elapsed_seconds'],
        seconds_by_component={**times,'other':max(0.,b['elapsed_seconds']-sum(times.values()))},
        first_target_seconds=min((t for t,u in events(raw) if t is not None and u>=target-1e-10),default=None),
        full_prediction_queries=b['full_prediction_calls'],milp_solves=b['milp_solve_calls'],
        route_evaluations=b['outer_evaluations'],event_engine_calls=stats.get('event_engine_calls'),
        fifo_cache_hits=stats.get('fifo_cache_hits'),deployment=w.get('deployment') if w else None,
        routing_qps=w.get('routing') if w else None,
        routing_shares={t:{m:q/config().workload[t] for m,q in r.items()} for t,r in w['routing'].items()} if w else None,
        actual_fifo_mapping=prediction.get('request_mapping'),status=raw['status'],
        deployment_domain=raw.get('deployment_domain'),post_seed_milp=raw.get('deployment_milp_calls_after_seed'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[11,19])
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    c=config();save('system-config-CPU-FIXTURE.json',c.as_dict())
    p=ImmutableContinuousProblem(c);save('frozen-trace.json',dict(arrivals=p.train_traces[0].arrivals,seed=p.train_traces[0].seed))
    base=dict(seconds=6.,max_predictions=48,max_routes=24)
    rows=[];target=None
    for seed in args.seeds:
        options={**base,'seed':seed}
        solvers=[('Joint',lambda:baseline(c,'Joint',options)),
            ('DFS',lambda:baseline(c,'DFS',options)),
            ('DE+FixedMILP',lambda:baseline(c,'DE+FixedMILP',options)),
            ('Joint+fixedDE',lambda:JointMILPContinuousRefinement(c,method='de',**options).solve()),
            ('MultiCandidate',lambda:MultiCandidateContinuousRefinement(c,k=3,**options).solve()),
            ('AdaptiveCandidate',lambda:AdaptiveCandidateRefinementOptimizer(c,k=3,**options).solve()),
            ('StructuredCoordinate',lambda:PaperStructuredOptimizer(c,method='coordinate',**options).solve()),
            ('StructuredDE',lambda:PaperStructuredOptimizer(c,method='de',**options).solve())]
        for name,solve in solvers:
            raw=solve();save(f'{name}-{seed}.json',raw)
            if target is None:target=raw['winner']['utility']
            row=summarize(name,seed,raw,target);rows.append(row)
            print(json.dumps({k:row[k] for k in ('method','seed','U','total_seconds','milp_solves','event_engine_calls')},ensure_ascii=False),flush=True)
    save('comparison.json',rows)
    cache=[]
    for enabled in (False,True):
        cc=config();cc.catalog[MODE]['fifo_cache_size']=128 if enabled else 0
        s=PaperFifoSession(cc.catalog);predictions=[];started=time.perf_counter()
        for g in (.7,.8,.9,.6,1.5,1.6,1.7,1.8):
            pred=predict(s,cc,{'g':1,'s':1},{'text':{'g':g,'s':2-g}})
            predictions.append(dict(routing_g=g,U=pred['U'],rho=pred['rho'],Q=pred['Q_norm'],
                mapping=pred['request_mapping'],finishes=pred['forecast']['finishes']))
        cache.append(dict(enabled=enabled,seconds=time.perf_counter()-started,stats=s.stats(),predictions=predictions))
    save('fifo-cache-ablation.json',cache)
    assert cache[0]['predictions']==cache[1]['predictions']
    resource_rows=resource_candidates()
    save('resource-candidate-fixtures.json',[{**r,'config':r['config'].as_dict()} for r in resource_rows])
    resource_result=ProfiledResourceConfigurationSearch(resource_rows,seconds=6,max_predictions=48,max_routes=24,
                                                       joint_seed=False).solve()
    save('resource-decision.json',resource_result)
    c,s,state,queues,future=busy_case()
    continuation=s.forecast_continuation({'g':1},queues,state,now_ns=40_000_000,deadline_ns=1_000_000_000,
        drain_deadline_ns=2_000_000_000,future_arrivals=future,routing={'text':{'g':2.}})
    save('busy-continuation.json',dict(observed=state,queues=queues,future=future,result=continuation))
    entry=c.catalog[MODE]['deployments'][0]
    captures={canonical(entry['deployment']):[capture_fixture(entry,i) for i in range(2)]}
    save('native-capture-CPU-FIXTURE.json',captures)
    normalized={canonical(entry['deployment']):[normalize_capture(r,entry) for r in captures[canonical(entry['deployment'])]]}
    fitted,report=fit_config(c,normalized)
    save('normalized-training.json',normalized);save('fitted-system-config.json',fitted);save('fit-report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    first=[r for r in rows if r['seed']==args.seeds[0]]
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    labels=[r['method'] for r in first]
    for ax,key,title in zip(axes,['total_seconds','milp_solves','event_engine_calls'],
                            ['One-run CPU time (s)','MILP solves','Event engine calls']):
        ax.bar(labels,[r[key] for r in first]);ax.set_title(title);ax.tick_params(axis='x',rotation=65,labelsize=7)
    fig.suptitle('Tiny analytical CPU fixture only; all methods have equal U',fontsize=10)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    summary=dict(CPU_FIXTURE_ONLY=True,GPU_started=False,physical_gain_claim=False,source_identity=source_identity(),
        domain=dict(tasks=1,models=2,legal_deployments=3,requests=2,K=2),budgets=base,seeds=args.seeds,
        comparison_rows=len(rows),same_utility=all(abs(r['U']-target)<1e-9 for r in rows),
        fifo_cache_exact=True,event_cache_calls=[r['stats']['event_engine_calls'] for r in cache],
        resource_winner=resource_result['winner']['configuration_id'],busy_forecast_status=continuation['status'],
        native_eligible=False,independent_accuracy_qualified=False,final_gain_eligible=False,
        timing_scope='single run per method/seed including setup and seed; no statistical speedup or scalability inference')
    save('summary.json',summary)
    files=sorted({*ROOT.glob('gsdeploy/continuous/*.py'),*ROOT.glob('gsdeploy/forecasting/*.py'),
                  *ROOT.glob('gsdeploy/paper_*.py'),ROOT/'gsdeploy/profiled_resource_search.py',
                  ROOT/'gsdeploy/linear_solver.py',ROOT/'gsdeploy/milp.py',ROOT/'gsdeploy/optimizer.py',
                  ROOT/'configs/paper_fifo_cpu_fixture.json',ROOT/'tests/test_paper_structured.py',
                  ROOT/'tests/test_paper_completion.py',Path(__file__),ROOT/'experiments/structured_paper_search.py',
                  ROOT/'experiments/normalize_paper_capture.py'})
    save('source-manifest.json',[dict(path=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files])
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0 if summary['same_utility'] and continuation['status']=='FORECAST_COMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())
