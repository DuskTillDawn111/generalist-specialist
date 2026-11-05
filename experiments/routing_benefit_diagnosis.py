"""Bounded, exact-input diagnostics of the unchanged G/S replay predictor.

The event ledger only observes the nominal FIFO calculation and asserts its
agreement with the original predictor. It never supplies an objective to a
solver, changes a profile, or substitutes an unvalidated prediction.
"""

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsdeploy import SystemConfig
from gsdeploy.calibration_v2 import validated_replay_spec
from gsdeploy.continuous import JointMILPContinuousRefinement
from gsdeploy.continuous.contracts import digest
from gsdeploy.continuous.predictor_adapter import normalize_prediction
from gsdeploy.search_session import SearchSession
from gsdeploy.routing import Router
from gsdeploy.trace import expand_routes, make_trace, trace_digest
from experiments.joint_scalability import ROOT, DEFAULT_SOURCE, save, source_manifest, subset_config, run


def nominal_events(session, deployment, routing):
    """Read-only nominal event accounting, including idle resident workers."""
    c = session.config
    if c.catalog['performance_mode'] != 'calibrated_replay_v2':
        raise ValueError('EVENT_OBSERVER_ONLY_SUPPORTS_VALIDATED_REPLAY_V2')
    module = session.module('calibration_v2')
    spec = module.validated_replay_spec(c.catalog, c.workload)
    trace = make_trace(c.workload, c.horizon_seconds, c.random_seed, c.arrival_process)
    ids = module.replay_input_ids(c.catalog, trace, c.random_seed)
    aliases, rows = expand_routes(deployment, routing)
    router = Router(rows)
    clocks = dict.fromkeys(aliases, 0.)
    events = []
    for i, ((arrival, task), sample) in enumerate(zip(trace, ids)):
        worker = router.choose(task)
        model = aliases[worker]
        service = (module.input_service_ms(spec, model, task, deployment[model], sample)
                   / 1000 * deployment[model] / c.catalog['capacity_factors'][model][str(deployment[model])])
        start = max(arrival, clocks[worker])
        finish = start + service
        clocks[worker] = finish
        events.append(dict(index=i, arrival=arrival, task=task, sample_id=sample,
            worker=worker, model=model, start=start, finish=finish, service_ms=service*1000,
            waiting_ms=(start-arrival)*1000, latency_ms=(finish-arrival)*1000))
    def aggregate(es):
        return dict(requests=len(es), realized_qps=len(es)/c.horizon_seconds,
            mean_service_ms=float(np.mean([e['service_ms'] for e in es])) if es else None,
            mean_waiting_ms=float(np.mean([e['waiting_ms'] for e in es])) if es else None,
            p95_waiting_ms=float(np.percentile([e['waiting_ms'] for e in es],95)) if es else None,
            mean_ms=float(np.mean([e['latency_ms'] for e in es])) if es else None,
            p95_ms=float(np.percentile([e['latency_ms'] for e in es],95)) if es else None,
            busy_seconds=math.fsum(e['service_ms']/1000 for e in es))
    return dict(trace_sha256=trace_digest(trace),
        assignment_sha256=digest([(e['index'],e['arrival'],e['task'],e['sample_id'],e['worker']) for e in events]),
        events=events, per_task={t:aggregate([e for e in events if e['task']==t]) for t in c.workload},
        per_model={m:aggregate([e for e in events if e['model']==m]) for m in deployment},
        per_worker={w:aggregate([e for e in events if e['worker']==w]) for w in aliases},
        per_path={m:{t:aggregate([e for e in events if e['model']==m and e['task']==t])
                        for t in c.workload if t in c.catalog['models'][m]['supported_tasks']} for m in deployment},
        pool_service_ms={m:{t:module.replay_path_statistics(spec,m,t,n)['mean_ms']
                            *n/c.catalog['capacity_factors'][m][str(n)]
                            for t in c.workload if t in c.catalog['models'][m]['supported_tasks']}
                         for m,n in deployment.items()})


def reconcile_events(c, prediction, ledger):
    """Fail rather than publish an observer that differs from the predictor."""
    assert prediction['trace_sha256'] == ledger['trace_sha256']
    es = ledger['events']
    assert len(es) == prediction['offered_requests']
    for task, stats in prediction.get('per_task',{}).items():
        observed = ledger['per_task'][task]
        assert stats['request_count'] == observed['requests']
        for key in ('mean_ms','p95_ms'):
            assert math.isclose(stats[key], observed[key], rel_tol=1e-10, abs_tol=1e-7), (task,key)
    assert sum(e['finish'] <= c.horizon_seconds for e in es) == prediction['within_window_requests']
    assert sum(e['finish'] > c.horizon_seconds+c.drain_seconds for e in es) == prediction['incomplete_after_drain']
    assert all(math.isclose(e['latency_ms'],e['service_ms']+e['waiting_ms'],abs_tol=1e-7) for e in es)


def pure_routing(c, deployment, overrides=None):
    overrides = overrides or {}
    rows = {}
    for task,q in c.workload.items():
        eligible = sorted(m for m in deployment if task in c.catalog['models'][m]['supported_tasks'])
        model = overrides.get(task, 'generalist_2' if 'generalist_2' in eligible else eligible[0])
        if model not in eligible:
            raise ValueError('INVALID_DIAGNOSTIC_ROUTE')
        rows[task] = {model:q}
    return rows


def split_route(c, reference, task, generalist, specialist, ratio):
    if not 0 <= ratio <= 1:
        raise ValueError('INVALID_RATIO')
    routing = copy.deepcopy(reference)
    routing[task] = {m:c.workload[task]*v for m,v in ((generalist,1-ratio),(specialist,ratio)) if v>0}
    return routing


class DiagnosticStudy:
    """Exact prediction-input cache plus separate per-stage call ceilings."""
    def __init__(self, base, out, resume=False):
        self.base,self.out = base,out
        self.sessions = {}
        self.cache = {}
        self.rows = []
        self.counts = {}
        self.hits = {}
        if resume:
            self.rows=json.loads((out/'observations.json').read_text())
            budget=json.loads((out/'prediction-budget.json').read_text())
            self.counts=budget['full_predict_scenarios_calls'];self.hits=budget['exact_input_cache_hits']
            self.cache={r['identity']:json.loads((out/r['raw']).read_text()) for r in self.rows}
        else:
            (out/'points').mkdir()

    def session(self, seed):
        if seed not in self.sessions:
            c = subset_config(self.base, sorted(self.base.catalog['models']), 1, 8., seed=seed)
            self.sessions[seed] = SearchSession(ROOT, c.as_dict())
            save(self.out/f'trace{seed}-config.json',c.as_dict())
        return self.sessions[seed]

    def evaluate(self, stage, ceiling, deployment, routing, *, seed=42, **tags):
        session = self.session(seed)
        c = session.config
        key = digest(dict(config=c.as_dict(),deployment=deployment,routing=routing))
        cached = key in self.cache
        if cached:
            self.hits[stage] = self.hits.get(stage,0)+1
            record = self.cache[key]
        else:
            if self.counts.get(stage,0) >= ceiling:
                raise RuntimeError(f'{stage}_PREDICTION_LIMIT')
            self.counts[stage] = self.counts.get(stage,0)+1
            t = time.perf_counter()
            prediction = session.predict(deployment,routing)
            elapsed = time.perf_counter()-t
            state,u = normalize_prediction(prediction,c.search_policy)
            ledger = None
            if not prediction.get('performance_unknown'):
                ledger = nominal_events(session,deployment,routing)
                reconcile_events(c,prediction,ledger)
            record = dict(identity=key,trace_seed=seed,deployment=deployment,routing=routing,
                seconds=elapsed,state=state,utility=u,prediction=prediction,ledger=ledger,
                observer_reconciled=ledger is not None,
                evidence='unchanged validated predictor; event ledger is read-only nominal accounting')
            save(self.out/'points'/f'{key}.json',record)
            self.cache[key]=record
        p = record['prediction']
        row = dict(stage=stage,**tags,identity=key,trace_seed=seed,cached=cached,
            state=record['state'],utility=record['utility'],reason=p.get('reason'),
            deployment=deployment,routing=routing,seconds=0 if cached else record['seconds'],
            assignment_sha256=(record['ledger'] or {}).get('assignment_sha256'),
            raw=f'points/{key}.json')
        self.rows.append(row)
        return row

    def checkpoint(self):
        save(self.out/'observations.json',self.rows)
        save(self.out/'prediction-budget.json',dict(full_predict_scenarios_calls=self.counts,
            scenario_leaves_per_call=3,exact_input_cache_hits=self.hits,unique_inputs=len(self.cache)))

    def close(self):
        for session in self.sessions.values(): session.close()


def curve_plan():
    g='generalist_2'
    return [dict(name=name,deployment=D,task=t,generalist=gm,specialist=t,overrides=overrides)
        for name,D,t,gm,overrides in [
            ('g2_asr', {g:1,'asr':1},'asr',g,{}),
            ('g2_ocr', {g:1,'ocr':1},'ocr',g,{}),
            ('g2_image', {g:1,'image':1},'image',g,{}),
            ('g2_text', {g:1,'text':1},'text',g,{}),
            ('g2_image_ocr__ocr',{g:1,'image':1,'ocr':1},'ocr',g,{'image':'image'}),
            ('g2_image_ocr__image',{g:1,'image':1,'ocr':1},'image',g,{'ocr':'ocr'}),
            ('g2_ocr_asr__asr',{g:1,'ocr':1,'asr':1},'asr',g,{'ocr':'ocr'}),
            ('g2_ocr_asr__ocr',{g:1,'ocr':1,'asr':1},'ocr',g,{'asr':'asr'}),
            ('g2_image_ocr_asr__asr',{g:1,'image':1,'ocr':1,'asr':1},'asr',g,{'image':'image','ocr':'ocr'}),
            ('g2_image_ocr_asr__ocr',{g:1,'image':1,'ocr':1,'asr':1},'ocr',g,{'image':'image','asr':'asr'}),
            ('g2n2_ocr',{g:2,'ocr':1},'ocr',g,{}),
            ('g2_ocrn2',{g:1,'ocr':2},'ocr',g,{}),
            ('g2_asrn2',{g:1,'asr':2},'asr',g,{}),
            ('g1_ocr',{'generalist_1':1,'ocr':1},'ocr','generalist_1',{}),
            ('g3_ocr',{'generalist_3':1,'ocr':1},'ocr','generalist_3',{}),
        ]]


def scope(base,out):
    rows=[]
    rates=base.workload
    workloads={'original':rates,'low':{t:q*.5 for t,q in rates.items()},
        'high':{t:q*2 for t,q in rates.items()},'all_high':{t:q*4 for t,q in rates.items()},
        'asr_hot':{t:q*(3 if t=='asr' else 1) for t,q in rates.items()},
        'ocr_hot':{t:q*(3 if t=='ocr' else 1) for t,q in rates.items()},
        'neighbor_low':{t:q*.99 for t,q in rates.items()},'neighbor_high':{t:q*1.01 for t,q in rates.items()}}
    folder=out/'scope';folder.mkdir()
    for name,workload in workloads.items():
        raw=base.as_dict();raw['workload']=workload
        save(folder/f'{name}-config.json',raw)
        try:
            validated_replay_spec(base.catalog,workload)
            status='SUPPORTED'
        except ValueError as exc: status=str(exc)
        rows.append(dict(case=name,status=status,config=f'scope/{name}-config.json',workload=workload))
    for i,path in enumerate(sorted((ROOT/'artifacts/static-spillover-study-20261022').glob('transfer*/*config.json'))):
        shutil.copy2(path,folder/f'transfer{i}-config.json')
        try:
            c=SystemConfig.load(path)
            with SearchSession(ROOT,c.as_dict()): pass
            status='SUPPORTED'
        except (ValueError,KeyError) as exc: status=str(exc)
        rows.append(dict(case=str(path.relative_to(ROOT)),status=status,config=f'scope/transfer{i}-config.json'))
    save(out/'scope.json',rows)


def stage_a(study):
    plan=curve_plan()
    save(study.out/'curve-plan.json',plan)
    for curve in plan:
        c=study.session(42).config
        ref=pure_routing(c,curve['deployment'],curve['overrides'])
        for i in range(21):
            ratio=i/20
            study.evaluate('A',650,curve['deployment'],split_route(c,ref,curve['task'],curve['generalist'],curve['specialist'],ratio),
                curve=curve['name'],ratio=ratio,sampling='coarse')
        rows=[r for r in study.rows if r.get('curve')==curve['name'] and r['utility'] is not None]
        if rows:
            best=max(rows,key=lambda r:r['utility'])
            # Predeclared local densification around best coarse point; not
            # a continuous optimum certificate or a new search algorithm.
            for k in range(-9,10):
                ratio=round(best['ratio']+k*.005,10)
                if 0<=ratio<=1:
                    study.evaluate('A',650,curve['deployment'],split_route(c,ref,curve['task'],curve['generalist'],curve['specialist'],ratio),
                        curve=curve['name'],ratio=ratio,sampling='local_best')
            print('A',curve['name'],'best coarse',best['ratio'],best['utility'],flush=True)
        else: print('A',curve['name'],'no valid points',flush=True)
        study.checkpoint()
    # One strictly bounded task-coupling plane: 7x7=49 points.
    D={'generalist_2':1,'ocr':1,'asr':1}
    ref=pure_routing(c,D)
    for i in range(7):
        for j in range(7):
            route=split_route(c,ref,'asr','generalist_2','asr',i/6)
            route=split_route(c,route,'ocr','generalist_2','ocr',j/6)
            study.evaluate('A2D',49,D,route,curve='asr_ocr_2d',asr_ratio=i/6,ocr_ratio=j/6)
    study.checkpoint()


def stage_b(study):
    c=study.session(42).config
    controls=[('g2_only',{'generalist_2':1},{}),
        ('g2n2_only',{'generalist_2':2},{}),
        ('g1_only',{'generalist_1':1},{}),('g3_only',{'generalist_3':1},{}),
        ('specialists_only',{'text':1,'image':1,'ocr':1,'asr':1},{t:t for t in c.workload})]
    for name,D,overrides in controls:
        study.evaluate('B',40,D,pure_routing(c,D,overrides),control=name)
    for curve in curve_plan():
        rows=[r for r in study.rows if r.get('curve')==curve['name'] and r['utility'] is not None]
        if not rows: continue
        best=max(rows,key=lambda r:r['utility'])
        for ratio in sorted({0.,.5,1.,best['ratio']}):
            ref=pure_routing(c,curve['deployment'],curve['overrides'])
            study.evaluate('B',40,curve['deployment'],split_route(c,ref,curve['task'],curve['generalist'],curve['specialist'],ratio),
                curve=curve['name'],ratio=ratio,control='representative')
    study.checkpoint()


def stage_c(study):
    c=study.session(42).config
    for name in ('g2_asr','g2_image_ocr__ocr'):
        curve=next(x for x in curve_plan() if x['name']==name)
        ref=pure_routing(c,curve['deployment'],curve['overrides'])
        for k in range(101):
            ratio=k/10000
            study.evaluate('C',230,curve['deployment'],split_route(c,ref,curve['task'],curve['generalist'],curve['specialist'],ratio),
                curve=name,ratio=ratio,sampling='mapping_endpoint_zoom')
    study.checkpoint()


def stage_d(base,out):
    folder=out/'comparison';folder.mkdir()
    cases=[('full5_n1',['generalist_2','text','image','ocr','asr'],1),
           ('g2_asr_n1',['generalist_2','asr'],1),
           ('g2_ocr_n1',['generalist_2','ocr'],1),
           ('full5_n2',['generalist_2','text','image','ocr','asr'],2)]
    summary=[]
    for name,models,n in cases:
        states={m:list(range(1,n+1)) for m in models}
        c=replace(subset_config(base,models,1,8.,seed=42),milp_max_iterations=200)
        save(folder/f'{name}-config.json',c.as_dict());save(folder/f'{name}-states.json',states)
        jobs=[('joint',K,11) for K in (1,4,8)]+[(a,1,s) for a in ('de_milp','refine_de') for s in (11,29)]
        random.Random(20261101).shuffle(jobs)
        for algorithm,K,seed in jobs:
            config=replace(subset_config(base,models,K,8.,seed=42),milp_max_iterations=200)
            if algorithm=='refine_de':
                raw=JointMILPContinuousRefinement(config,states,seconds=8.,max_predictions=200,max_routes=96,seed=seed).solve()
                w=raw['winner']
                row=dict(case=name,algorithm=algorithm,K=K,seed=seed,seconds=raw['total_seconds'],
                    utility=w['utility'] if w else None,winner=w,predictions=raw['budget']['full_prediction_calls'],
                    milp_calls=raw['budget']['milp_solve_calls'],status=raw['status'],certified=False,
                    certificate_scope='finite-budget heuristic; initial certificate only applies to old grid',raw=raw)
            else:
                from experiments import joint_scalability
                from gsdeploy.continuous import Budget
                # Restrict the existing experimental harness, without editing
                # any optimizer. Original Joint has one prediction per round.
                with patch.object(joint_scalability,'Budget',
                                  side_effect=lambda seconds,_,routes:Budget(seconds,200,routes)):
                    raw=run(config,states,algorithm,8.,96,seed)
                row=dict(case=name,algorithm=algorithm,K=K,seed=seed,seconds=raw['seconds'],utility=raw['utility'],winner=raw['winner'],
                    predictions=raw['metrics']['full_prediction_calls'],milp_calls=raw['metrics'].get('milp_calls',raw['metrics'].get('milp_solve_calls',0)),
                    status=raw['status'],certified=raw['certified'],certificate_scope=raw['raw'].get('certificate_scope'),raw=raw)
            # Post-run observation does not enter optimization wall time.
            if row['winner']:
                from experiments.diagnose_continuous_phase1 import assignment
                row['fifo']=assignment(config,row['winner']['deployment'],row['winner']['routing'])
            assert row['predictions']<=200
            row['config_overrides']=dict(route_grid=K,milp_max_iterations=200)
            row['config_sha256']=digest(config.as_dict())
            save(folder/f'{name}-{algorithm}-K{K}-s{seed}.json',row)
            summary.append({k:row[k] for k in row if k not in {'raw','fifo','winner'}}|dict(
                deployment=w['deployment'] if algorithm=='refine_de' and w else (row['winner'] or {}).get('deployment'),
                routing=(row['winner'] or {}).get('routing')))
            save(out/'comparison.json',summary)
            print('D',name,algorithm,K,seed,row['utility'],row['predictions'],flush=True)


def stage_e(study):
    # Independent traces check endpoints and the discovered interior candidate
    # without selecting a new winner on those holdout traces.
    for curve in curve_plan():
        rows=[r for r in study.rows if r['stage']=='A' and r.get('curve')==curve['name'] and r['utility'] is not None]
        if not rows: continue
        best=max(rows,key=lambda r:r['utility'])
        ratios=sorted({0.,1.,best['ratio']})
        for seed in (43,45):
            c=study.session(seed).config
            ref=pure_routing(c,curve['deployment'],curve['overrides'])
            for ratio in ratios:
                study.evaluate('E',120,curve['deployment'],split_route(c,ref,curve['task'],curve['generalist'],curve['specialist'],ratio),
                    seed=seed,curve=curve['name'],ratio=ratio,sampling='frozen_discovery_ratio_holdout')
    # Explicit idle specialist removal at r=0 under the same route.
    c=study.session(42).config
    for task in ('asr','ocr','image','text'):
        D={'generalist_2':1,task:1}
        route=pure_routing(c,D)
        study.evaluate('E',120,D,route,contrast=f'idle_{task}_resident')
        study.evaluate('E',120,{'generalist_2':1},route,contrast=f'idle_{task}_removed')
    study.checkpoint()


def main(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    before=source_manifest();save(out/'source-manifest.json',before)
    shutil.copy2(ROOT/args.source,out/'frozen-source-config.json')
    save(out/'plan.json',dict(source=args.source,stage_order=['A','B','C','D','E'],
        caps={'A':650,'A2D':49,'B':40,'C':230,'D_per_job':200,'E':120},
        grid=[i/20 for i in range(21)],densify='best coarse +-0.045 in steps 0.005',
        comparison_seconds=8,comparison_routes=96,optimizer_seeds=[11,29],
        trace_discovery=42,trace_holdout=[43,45],no_modified_predictor=True,no_gpu=True))
    base=SystemConfig.load(ROOT/args.source)
    scope(base,out)
    study=DiagnosticStudy(base,out)
    started=time.perf_counter()
    try:
        stage_a(study);stage_b(study)
        save(out/'AB-complete.json',dict(seconds=time.perf_counter()-started,counts=study.counts))
        stage_c(study);stage_d(base,out);stage_e(study)
        assert before==source_manifest()
        study.checkpoint()
        save(out/'complete.json',dict(seconds=time.perf_counter()-started,counts=study.counts,
            exact_cache_hits=study.hits,observations=len(study.rows),unique_inputs=len(study.cache),
            original_sources_unchanged=True,event_ledger_checked_against_original=True))
    finally:
        study.close()


def supplement(args):
    """Targeted endpoint zoom and missing text alternative, with own budget."""
    out=Path(args.output).resolve()
    assert (out/'complete.json').exists()
    assert json.loads((out/'source-manifest.json').read_text())==source_manifest()
    base=SystemConfig.load(out/'frozen-source-config.json')
    study=DiagnosticStudy(base,out,resume=True)
    started=time.perf_counter()
    try:
        c=study.session(42).config
        curve=dict(name='g2_image_ocr_text__text',deployment={'generalist_2':1,'image':1,'ocr':1,'text':1},
            task='text',generalist='generalist_2',specialist='text',overrides={'image':'image','ocr':'ocr'})
        plan=json.loads((out/'curve-plan.json').read_text())
        plan.append(curve);save(out/'curve-plan.json',plan)
        ref=pure_routing(c,curve['deployment'],curve['overrides'])
        for i in range(21):
            ratio=i/20
            study.evaluate('A_extra',300,curve['deployment'],split_route(c,ref,'text','generalist_2','text',ratio),
                curve=curve['name'],ratio=ratio,sampling='coarse')
        valid=[r for r in study.rows if r.get('curve')==curve['name'] and r['utility'] is not None]
        best=max(valid,key=lambda r:r['utility'])
        for k in range(-9,10):
            ratio=round(best['ratio']+k*.005,10)
            if 0<=ratio<=1:
                study.evaluate('A_extra',300,curve['deployment'],split_route(c,ref,'text','generalist_2','text',ratio),
                    curve=curve['name'],ratio=ratio,sampling='local_best')
        # Sub-grid narrow neighborhoods, explicitly separate from coarse scan.
        for name,ratios in [('g2_image_ocr_asr__asr',[.99+i/10000 for i in range(101)]),
                            (curve['name'],[i/10000 for i in range(101)])]:
            p=next(p for p in plan if p['name']==name)
            ref=pure_routing(c,p['deployment'],p['overrides'])
            for ratio in ratios:
                study.evaluate('A_extra',300,p['deployment'],split_route(c,ref,p['task'],p['generalist'],p['specialist'],ratio),
                    curve=name,ratio=ratio,sampling='endpoint_0.0001_zoom')
        rs=[r for r in study.rows if r.get('curve')==curve['name'] and r['utility'] is not None]
        ratio=max(rs,key=lambda r:r['utility'])['ratio']
        for seed in (43,45):
            c=study.session(seed).config
            for value in sorted({0.,1.,ratio}):
                study.evaluate('E_extra',8,curve['deployment'],split_route(c,pure_routing(c,curve['deployment'],curve['overrides']),
                    'text','generalist_2','text',value),seed=seed,curve=curve['name'],ratio=value,sampling='frozen_discovery_ratio_holdout')
        study.checkpoint()
        save(out/'supplement-complete.json',dict(seconds=time.perf_counter()-started,counts=study.counts,
            exact_cache_hits=study.hits,observations=len(study.rows),unique_inputs=len(study.cache),
            purpose='text specialist added to original winning deployment; ASR/text endpoint zoom',
            original_sources_unchanged=source_manifest()==json.loads((out/'source-manifest.json').read_text())))
    finally:study.close()


def slice_comparison(args):
    """Grid and existing continuous DE on the discovered one-task slice."""
    from gsdeploy.route_domain import fraction_rows
    from gsdeploy.continuous import (Budget, ImmutableContinuousProblem, SessionPredictorOracle,
        FixedDeploymentRoutingEvaluator, DERoutingOptimizer)
    out=Path(args.output).resolve()
    base=SystemConfig.load(out/'frozen-source-config.json')
    study=DiagnosticStudy(base,out,resume=True)
    folder=out/'text-slice';folder.mkdir()
    c=study.session(42).config
    D={'generalist_2':1,'image':1,'ocr':1,'text':1}
    ref=pure_routing(c,D,{'image':'image','ocr':'ocr'})
    grid=[]
    for K in (1,4,8):
        for fractions in fraction_rows(('generalist_2','text'),K):
            ratio=float(dict(fractions).get('text',0))
            r=study.evaluate('D_slice',30,D,split_route(c,ref,'text','generalist_2','text',ratio),
                curve='g2_image_ocr_text__text',ratio=ratio,K=K,sampling='complete_fixed_slice_rational_grid')
            grid.append(r)
    save(folder/'grid.json',grid)
    # Only text may vary: the other rows are fixed to their reference model.
    states={m:[1] for m in base.catalog['models']}
    problem=ImmutableContinuousProblem(base,replica_options=states)
    for seed in (11,29,47):
        budget=Budget(8.,200,128)
        with SessionPredictorOracle(problem) as oracle:
            e=FixedDeploymentRoutingEvaluator(problem,oracle,D,budget)
            e.domain.edges={t:(('generalist_2','text') if t=='text' else tuple(ref[t])) for t in base.workload}
            e.domain.dimension=1
            seed_shares={t:{m:q/base.workload[t] for m,q in row.items()} for t,row in ref.items()}
            raw=DERoutingOptimizer(e,seed=seed,popsize=5,maxiter=100,initial_shares=seed_shares).solve()
            raw['history']=e.history
            save(folder/f'de-s{seed}.json',raw)
            w=raw['winner']
            print('slice',seed,w['best_feasible_utility'],w['routing_shares']['text'],flush=True)
    study.checkpoint()
    save(out/'slice-complete.json',dict(counts=study.counts,observations=len(study.rows),unique_inputs=len(study.cache),
        optimizer='unchanged DERoutingOptimizer, only text row allowed to vary',
        per_search_limits=dict(seconds=8,predictions=200,routes=128),
        original_sources_unchanged=source_manifest()==json.loads((out/'source-manifest.json').read_text())))
    study.close()
    slice_audit(args)


def slice_audit(args):
    """Freeze DE winners before checking new traces; observe every DE point."""
    out=Path(args.output).resolve();folder=out/'text-slice'
    study=DiagnosticStudy(SystemConfig.load(out/'frozen-source-config.json'),out,resume=True)
    (folder/'events').mkdir(exist_ok=True)
    audits=[]
    for optimizer_seed in (11,29,47):
        raw=json.loads((folder/f'de-s{optimizer_seed}.json').read_text())
        for i,row in enumerate(raw['history']):
            session=study.session(42)
            ledger=nominal_events(session,row['deployment'],row['routing_qps'])
            prediction=row['prediction']['traces'][0]['prediction']
            reconcile_events(session.config,prediction,ledger)
            save(folder/'events'/f's{optimizer_seed}-{i}.json',dict(deployment=row['deployment'],routing=row['routing_qps'],
                prediction=prediction,ledger=ledger,observer_reconciled=True))
        w=raw['winner'];ratio=w['routing_shares']['text']['text']
        for seed in (42,43,45):
            c=study.session(seed).config
            D=w['deployment'];ref=pure_routing(c,D,{'image':'image','ocr':'ocr'})
            for r in sorted({0.,1.,ratio}) if seed!=42 else [ratio]:
                row=study.evaluate('E_slice' if seed!=42 else 'D_winner',18,D,
                    split_route(c,ref,'text','generalist_2','text',r),seed=seed,
                    curve='g2_image_ocr_text__text',ratio=r,optimizer_seed=optimizer_seed,
                    sampling='frozen_DE_winner_holdout' if seed!=42 else 'DE_winner_FIFO_audit')
                if seed==42:assert math.isclose(row['utility'],w['best_feasible_utility'],abs_tol=1e-12)
                audits.append(row)
    study.checkpoint();save(folder/'winner-audit.json',audits)
    save(out/'slice-audit-complete.json',dict(counts=study.counts,observations=len(study.rows),unique_inputs=len(study.cache),
        audited_DE_points=sum(len(json.loads((folder/f'de-s{s}.json').read_text())['history']) for s in (11,29,47)),
        original_sources_unchanged=source_manifest()==json.loads((out/'source-manifest.json').read_text())))
    study.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=DEFAULT_SOURCE)
    p.add_argument('--output',required=True)
    p.add_argument('--supplement',action='store_true')
    p.add_argument('--slice',action='store_true')
    p.add_argument('--slice-audit',action='store_true')
    args=p.parse_args()
    slice_audit(args) if args.slice_audit else slice_comparison(args) if args.slice else supplement(args) if args.supplement else main(args)
