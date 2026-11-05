"""Read-only diagnostics of the existing finite MILP and frozen replay predictor.

All patches are scoped to a private SearchSession. Ablations weaken only search
relaxations; complete predictions keep the original objective and hard checks.
No GPU samples are created by this experiment.
"""
import argparse
from contextlib import ExitStack, contextmanager
import csv
import gc
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsdeploy import SystemConfig
from gsdeploy.routing import Router
from gsdeploy.trace import expand_routes
from gsdeploy.search_session import SearchSession
from experiments.joint_scalability import (ROOT, DEFAULT_SOURCE, subset_config,
    source_manifest, domain_audit, scope_audit)
from experiments.diagnose_continuous_phase1 import save
from experiments.routing_benefit_diagnosis import (nominal_events,
    reconcile_events, pure_routing, split_route, scope)


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list))
                         else v for k, v in r.items()} for r in rows)


def floor_parts(opt, task, deployment, row, full_floor):
    """Decompose the ORIGINAL route floor, including an exact no-wait replay.

    Used outside timed baseline runs. It never supplies measured service.
    """
    if opt.surrogate is None or opt.task_input_ids is None:
        raise ValueError('FLOOR_DECOMPOSITION_REQUIRES_VALIDATED_PER_INPUT_REPLAY')
    from importlib import import_module
    module = import_module(opt.__class__.__module__.rsplit('.', 1)[0] + '.quality')
    loss, _ = module.task_quality(opt.data, task, deployment, row, opt.trace,
                                  seed=opt.context.get('seed', 42))
    fraction = opt.rates[task] / math.fsum(opt.rates.values())
    q = opt.weights.get('Q', 0) * fraction * loss if loss is not None else 0.
    e = opt.weights.get('E', 0) * (opt.surrogate.route_energy(task, deployment, row)
                                  + fraction * opt.surrogate.constant_energy)
    aliases, routes = expand_routes(deployment, {task: row})
    router = Router(routes)
    stress = 2. if opt.policy == 'robust' else 1.
    values = []
    late = 0
    for arrival, sample in zip(opt.task_offsets[task], opt.task_input_ids[task]):
        model = aliases[router.choose(task)]
        service = stress * opt.surrogate.request_service(model, task, deployment[model], sample)
        service *= deployment[model] / opt.data['capacity_factors'][model][str(deployment[model])]
        values.append(service)
        late += arrival + service / 1000 > opt.context.get('horizon', 120.)
    no_wait = fraction / opt.refs[task] * (
        opt.weights['L'] * float(np.mean(values))
        + opt.weights['P'] * float(np.percentile(values, 95)))
    no_wait += opt.weights['T'] * late / len(opt.trace)
    fifo = full_floor - q - e - no_wait
    if fifo < -1e-9:
        raise AssertionError(('negative FIFO increment', fifo))
    return dict(quality=q, route_energy=e, no_wait_service=no_wait,
                isolated_fifo_increment=max(0., fifo), original_floor=full_floor)


@contextmanager
def observe(module, opt, ablation='baseline'):
    """Restore every wrapper even on errors. Baseline floor is untouched."""
    import scipy.optimize._milp as scipy_milp
    stats = dict(columns_seconds=0., master_seconds=0., highs_seconds=0.,
                 prediction_seconds=0., rounds=[], predictions=[], cuts=[], columns=[])
    saved_choices = {}
    original_floor = opt.floor
    with ExitStack() as stack:
        if ablation != 'baseline':
            def relaxed(task, deployment, row):
                original = original_floor(task, deployment, row)
                if original is None:
                    return None
                if ablation == 'zero_route_floor':
                    return 0.
                parts = floor_parts(opt, task, deployment, row, original)
                removed = {'without_fifo': 'isolated_fifo_increment',
                           'without_delay': None, 'without_quality': 'quality',
                           'without_route_energy': 'route_energy'}[ablation]
                if removed:
                    return max(0., original - parts[removed])
                return parts['quality'] + parts['route_energy']
            stack.enter_context(patch.object(opt, 'floor', relaxed))
        columns = opt.columns
        def collect_columns(choices, *args, **kwargs):
            saved_choices.update(choices)
            t = time.perf_counter()
            value = columns(choices, *args, **kwargs)
            stats['columns_seconds'] += time.perf_counter() - t
            stats['column_generation_complete'] = value is not None
            if value is not None:
                stats['columns'] = value
            return value
        stack.enter_context(patch.object(opt, 'columns', collect_columns))
        backend = scipy_milp._highs_wrapper
        def highs(*args, **kwargs):
            t = time.perf_counter()
            try:
                return backend(*args, **kwargs)
            finally:
                stats['highs_seconds'] += time.perf_counter() - t
        stack.enter_context(patch.object(scipy_milp, '_highs_wrapper', highs))
        master = module.solve_master
        def solve(*args, **kwargs):
            t = time.perf_counter()
            fit = master(*args, **kwargs)
            elapsed = time.perf_counter() - t
            stats['master_seconds'] += elapsed
            states = [(m, n) for m, ns in saved_choices.items() for n in ns]
            incumbent = None if opt.best is None else -opt.best['score']
            dual = getattr(fit, 'mip_dual_bound', None)
            lower = max(opt.lower, float(dual)) if dual is not None and math.isfinite(dual) else opt.lower
            theta = len(states) + len(stats['columns'])
            chosen = [] if fit.x is None else [i for i in range(theta) if fit.x[i] > .5]
            record = dict(round=len(stats['rounds']) + 1, seconds=elapsed,
                variables=len(args[0]), constraints=kwargs['constraints'].A.shape[0],
                nonzeros=int(kwargs['constraints'].A.nnz), solver_status=int(fit.status),
                message=str(fit.message), objective=float(fit.fun) if fit.fun is not None else None,
                dual_bound=float(dual) if dual is not None and math.isfinite(dual) else None,
                incumbent_cost_before=incumbent, lower_cost_after=lower,
                gap_condition=bool(incumbent is not None and lower >= incumbent - opt.tolerance),
                deployment={m: n for i, (m, n) in enumerate(states) if i in chosen and n},
                routing={stats['columns'][i-len(states)]['task']:
                         stats['columns'][i-len(states)]['routing'] for i in chosen if i >= len(states)})
            stats['rounds'].append(record)
            return fit
        stack.enter_context(patch.object(module, 'solve_master', solve))
        predict = module.predict_scenarios
        def prediction(*args, **kwargs):
            t = time.perf_counter()
            p = predict(*args, **kwargs)
            elapsed = time.perf_counter() - t
            stats['prediction_seconds'] += elapsed
            stats['predictions'].append(dict(deployment=args[1], routing=args[2],
                seconds=elapsed, prediction=p))
            return p
        stack.enter_context(patch.object(module, 'predict_scenarios', prediction))
        cut = module.residual_identity_cut
        def feedback(base_row, theta, selected, blocks, residual):
            stats['cuts'].append(dict(round=len(stats['rounds']), residual=residual,
                                      selected=selected, blocks=blocks))
            return cut(base_row, theta, selected, blocks, residual)
        stack.enter_context(patch.object(module, 'residual_identity_cut', feedback))
        yield stats


def run_joint(c, states, ablation='baseline'):
    t = time.perf_counter()
    with SearchSession(ROOT, c.as_dict()) as session:
        opt = session.optimizer('milp')
        setup = time.perf_counter() - t
        with observe(session.module('milp'), opt, ablation) as timing:
            result = opt.solve(K=c.route_grid, replica_options=states)
        total = time.perf_counter() - t
        # Exclusive buckets, never add nested solve_master and HiGHS times.
        timing['master_build_and_other_python_seconds'] = max(0., total - setup
            - timing['columns_seconds'] - timing['master_seconds'] - timing['prediction_seconds'])
        timing['scipy_master_overhead_seconds'] = max(0., timing['master_seconds'] - timing['highs_seconds'])
        fractions = dict(setup=setup, columns=timing['columns_seconds'],
            highs=timing['highs_seconds'], scipy_master=timing['scipy_master_overhead_seconds'],
            prediction=timing['prediction_seconds'], python=timing['master_build_and_other_python_seconds'])
        parts = []
        if ablation == 'baseline':
            for col in timing['columns']:
                parts.append(dict(**col, parts=floor_parts(opt, col['task'], col['deployment'],
                                                           col['routing'], col['floor'])))
        return dict(ablation=ablation, total_seconds=total, setup_seconds=setup,
            exclusive_seconds=fractions, percent={k: 100*v/total for k, v in fractions.items()},
            timing=timing, column_parts=parts, result=result,
            timing_scope='wall includes setup and observers; column decomposition AFTER timed solve; no feedback kernel timer')


def space(c, states):
    """Explicit necessary-filter funnel; evidence coverage is not feasibility."""
    audit = domain_audit(c, states)
    spec = c.catalog['calibrated_replay_v2']
    from gsdeploy.calibration_v2 import _deployment_key
    exact = covered = known = quality_known = 0
    with SearchSession(ROOT, c.as_dict()) as session:
        opt = session.optimizer('milp')
        quality_module = session.module('quality')
        path_quality = {}
        for m, ns in states.items():
            for n in ns:
                for task, rate in c.workload.items():
                    if task in c.catalog['models'][m]['supported_tasks']:
                        loss, _ = quality_module.task_quality(opt.data, task, {m:n}, {m:rate},
                                                             opt.trace, seed=c.random_seed)
                        path_quality[m, task, n] = loss is not None
        for counts in itertools.product(*([0] + states[m] for m in sorted(states))):
            d = {m:n for m, n in zip(sorted(states), counts) if n}
            memory = max(sum(opt.state_memory(m,n) for m,n in d.items()),
                         spec['training_deployment_memory'].get(_deployment_key(d), 0.))
            if memory > c.gpu_memory_mib + 1e-6:
                continue
            exact += 1
            if not all(any(t in c.catalog['models'][m]['supported_tasks'] for m in d) for t in c.workload):
                continue
            covered += 1
            if not all(any((m,t,n) in opt.surrogate.pressure_services for m,n in d.items()) for t in c.workload):
                continue
            known += 1
            quality_known += all(any(path_quality.get((m,t,n), False) for m,n in d.items()) for t in c.workload)
    return dict(**audit, exact_memory_survivors=exact, coverage_survivors=covered,
        per_task_service_evidence_survivors=known, per_task_quality_evidence_survivors=quality_known,
        quality_threshold='NO numerical quality admission threshold; Q is an objective term',
        filter_scope='existence of a known pure path per task; aggregate rho, FIFO drain and complete U remain unchecked')


def full_rank(c, states, out):
    """Exhaustive finite K=1 domain, retaining idle resident models.

    No service/rho pruning: all exact-memory + task-cover plans are predicted.
    This independently checks the master certificate, not continuous optimality.
    """
    if c.route_grid != 1:
        raise ValueError('FULL_RANK_ONLY_K1')
    out.mkdir()
    records = []
    with SearchSession(ROOT, c.as_dict()) as session:
        opt = session.optimizer('milp')
        from gsdeploy.calibration_v2 import _deployment_key
        for counts in itertools.product(*([0]+states[m] for m in sorted(states))):
            d = {m:n for m,n in zip(sorted(states),counts) if n}
            memory = max(sum(opt.state_memory(m,n) for m,n in d.items()),
                         c.catalog['calibrated_replay_v2']['training_deployment_memory'].get(_deployment_key(d), 0.))
            if memory > c.gpu_memory_mib + 1e-6:
                continue
            paths = {t:[m for m in sorted(d) if t in c.catalog['models'][m]['supported_tasks']] for t in c.workload}
            if not all(paths.values()):
                continue
            for models in itertools.product(*paths.values()):
                routing = {t:{m:c.workload[t]} for t,m in zip(paths, models)}
                p = session.predict(d,routing)
                index = len(records)
                save(out/f'plan-{index:04d}.json',dict(deployment=d,routing=routing,prediction=p))
                records.append(dict(index=index,deployment=d,routing=routing,
                    utility=p.get('U'),valid=p.get('eligible',False),reason=p.get('reason'),
                    raw=f'{out.name}/plan-{index:04d}.json',
                    **{f'{k}_contribution':p.get(f'{k}_contribution') for k in 'QTLP ME'.replace(' ','')},
                    safe_memory_mib=p.get('safe_memory_mib'),max_rho=p.get('max_rho')))
    valid = sorted((v for v in records if v['valid']), key=lambda v:-v['utility'])
    best = {}
    for r in valid:
        best.setdefault(json.dumps(r['deployment'],sort_keys=True),r)
    deployments = list(best.values())
    return dict(K=1,scope='complete pure-route finite domain after exact memory and task cover; no continuous certificate',
        full_prediction_calls=len(records),valid_plans=len(valid),unknown_plans=sum('MISSING' in str(r['reason']) for r in records),
        plans=records,ranked_plans=valid,ranked_deployments=deployments,
        gap_first_two_plans=(valid[0]['utility']-valid[1]['utility']) if len(valid)>1 else None,
        gap_first_two_deployments=(deployments[0]['utility']-deployments[1]['utility']) if len(deployments)>1 else None)


def step_ab(base, out):
    manifest = source_manifest()
    save(out/'source-manifest-before.json',manifest)
    pools={3:['generalist_2','image','ocr'],5:['generalist_2','text','image','ocr','asr'],7:sorted(base.catalog['models'])}
    cases=[]
    for pool, n, ks in [(3,1,[1]),(5,1,[1,2,4,8,16]),(7,1,[1,2,8,12,16]),
                        (5,2,[2]),(7,2,[2]),(7,99,[2]),(2,1,[1,8,16])]:
        models = ['generalist_2','image'] if pool==2 else pools[pool]
        states={m:sorted(int(x) for x in base.catalog['capacity_factors'][m] if 0<int(x)<=n) for m in models}
        for k in ks:
            name=f'p{pool}-n{n}-K{k}'
            c=subset_config(base,models,k,8.)
            save(out/f'{name}-config.json',c.as_dict())
            domain=space(c,states)
            save(out/f'{name}-space.json',domain)
            record=run_joint(c,states)
            save(out/f'{name}-joint.json',record)
            cases.append(dict(name=name,pool=pool,n=n,K=k,states=states,
                config=f'{name}-config.json',space=f'{name}-space.json',raw=f'{name}-joint.json'))
            save(out/'cases.json',cases)
            print(name,record['result']['status'],record['result']['metrics'],flush=True)
            del record;gc.collect()
    # Small complete independently enumerated domains.
    for pool in (3,5):
        c=subset_config(base,pools[pool],1,8.)
        states={m:[1] for m in pools[pool]}
        rank=full_rank(c,states,out/f'p{pool}-K1-plans')
        save(out/f'p{pool}-K1-ranking.json',rank)
        write_csv(out/f'p{pool}-K1-plans.csv',rank['plans'])
        write_csv(out/f'p{pool}-K1-deployments.csv',rank['ranked_deployments'])
        print('complete rank',pool,rank['full_prediction_calls'],rank['gap_first_two_deployments'],flush=True)
    c=subset_config(base,pools[5],4,12.)
    save(out/'ablation-config.json',c.as_dict())
    for ablation in ('baseline','without_fifo','without_delay','without_quality','without_route_energy','zero_route_floor'):
        record=run_joint(c,{m:[1] for m in pools[5]},ablation)
        save(out/f'ablation-{ablation}.json',record)
        print('ablation',ablation,record['result']['metrics']['full_prediction_calls'],record['result']['status'],flush=True)
    save(out/'scope-gates-ab.json',scope_audit(base))
    if source_manifest()!=manifest:
        raise AssertionError('CORE_SOURCES_CHANGED')
    save(out/'step-ab-complete.json',dict(source_unchanged=True,cases=len(cases)))


def step_cde(base, out):
    if not (out/'step-ab-complete.json').exists():
        raise RuntimeError('RUN_AB_FIRST')
    before=source_manifest()
    scope(base,out)
    curves=[('g2_ocr',{'generalist_2':1,'ocr':1},'ocr',{}),
            ('g2_asr',{'generalist_2':1,'asr':1},'asr',{}),
            ('winner_text',{'generalist_2':1,'image':1,'ocr':1,'text':1},'text',{'image':'image','ocr':'ocr'})]
    folder=out/'controlled';folder.mkdir()
    observations=[];paths=[];models=[]
    c=subset_config(base,sorted(base.catalog['models']),1,8.)
    save(out/'controlled-config.json',c.as_dict())
    with SearchSession(ROOT,c.as_dict()) as session:
        for name,d,task,overrides in curves:
            ref=pure_routing(c,d,overrides)
            for i in range(21):
                ratio=i/20
                routing=split_route(c,ref,task,'generalist_2',task,ratio)
                p=session.predict(d,routing)
                ledger=nominal_events(session,d,routing)
                reconcile_events(c,p,ledger)
                save(folder/f'{name}-{i:02d}.json',dict(deployment=d,routing=routing,ratio=ratio,
                    prediction=p,ledger=ledger,evidence='frozen validated replay prediction, not new measurement'))
                observations.append(dict(curve=name,ratio=ratio,deployment=d,routing=routing,
                    utility=p.get('U'),valid=p.get('eligible'),reason=p.get('reason'),
                    mean_ms=p.get('pooled_request_mean_ms'),p95_ms=p.get('pooled_request_P95_ms'),
                    assignment_sha256=ledger['assignment_sha256'],
                    **{f'{k}_contribution':p.get(f'{k}_contribution') for k in 'QT LPME'.replace(' ','')}))
                for model, ts in ledger['per_path'].items():
                    for t, stat in ts.items():
                        paths.append(dict(curve=name,ratio=ratio,model=model,task=t,
                            offered_qps=routing[t].get(model,0.),pool_service_ms=ledger['pool_service_ms'][model][t],**stat))
                for model,stat in ledger['per_model'].items():
                    models.append(dict(curve=name,ratio=ratio,model=model,replicas=d[model],
                        offered_qps=sum(r.get(model,0.) for r in routing.values()),rho=p['rho'][model],**stat))
        # Explicit activation families, including valid S-only and measured count states.
        families=[('G-only',{'generalist_2':1},{}),('G+OCR',{'generalist_2':1,'ocr':1},{'ocr':'ocr'}),
          ('winner',{'generalist_2':1,'image':1,'ocr':1},{'image':'image','ocr':'ocr'}),
          ('winner+idle-text',{'generalist_2':1,'image':1,'ocr':1,'text':1},{'image':'image','ocr':'ocr'}),
          ('winner+text-pure',{'generalist_2':1,'image':1,'ocr':1,'text':1},{'image':'image','ocr':'ocr','text':'text'}),
          ('winner+ASR',{'generalist_2':1,'image':1,'ocr':1,'asr':1},{'image':'image','ocr':'ocr','asr':'asr'}),
          ('S-only',{'text':1,'image':1,'ocr':1,'asr':1},{'text':'text','image':'image','ocr':'ocr','asr':'asr'}),
          ('G-two+OCR',{'generalist_2':2,'ocr':1},{'ocr':'ocr'}),
          ('G+OCR-two',{'generalist_2':1,'ocr':2},{'ocr':'ocr'})]
        activation=[]
        for name,d,overrides in families:
            routing=pure_routing(c,d,overrides)
            p=session.predict(d,routing)
            ledger=nominal_events(session,d,routing)
            reconcile_events(c,p,ledger)
            save(folder/f'activation-{name}.json',dict(deployment=d,routing=routing,prediction=p,ledger=ledger))
            es=ledger['events']
            activation.append(dict(name=name,deployment=d,routing=routing,utility=p.get('U'),valid=p.get('eligible'),
                reason=p.get('reason'),safe_memory_mib=p.get('safe_memory_mib'),max_rho=p.get('max_rho'),
                mean_service_ms=float(np.mean([e['service_ms'] for e in es])),
                mean_waiting_ms=float(np.mean([e['waiting_ms'] for e in es])),
                mean_ms=p.get('pooled_request_mean_ms'),p95_ms=p.get('pooled_request_P95_ms'),
                throughput=p.get('predicted_within_window_QPS'),
                **{f'{k}_{suffix}':p.get(f'{k}_{suffix}') for k in 'QTLPME' for suffix in ('norm','contribution')}))
        # Fixed-request fingerprints across all routes distinguish input selection from slowdown.
        spec=session.module('calibration_v2').validated_replay_spec(c.catalog,c.workload)
        service_function=session.module('calibration_v2').input_service_ms
        checks=[]
        for name,d,task,_ in curves:
            for model in ('generalist_2',task):
                values=[service_function(spec,model,task,d[model],s) for s in spec['input_order'][task]]
                matching=[r['pool_service_ms'] for r in paths if r['curve']==name and r['model']==model and r['task']==task]
                assert max(matching)==min(matching)
                checks.append(dict(curve=name,model=model,task=task,input_count=len(values),
                    pool_service_ms=matching[0],pool_service_route_span_ms=max(matching)-min(matching),
                    dependency_signature='spec,model,task,count,input_id; no D/route/load argument'))
    write_csv(out/'controlled-points.csv',observations)
    write_csv(out/'controlled-paths.csv',paths)
    write_csv(out/'controlled-models.csv',models)
    write_csv(out/'activation.csv',activation)
    save(out/'service-invariance.json',checks)
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv'],
                           capture_output=True,text=True,timeout=10)
        hardware=dict(returncode=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr)
    except (OSError,subprocess.TimeoutExpired) as exc:
        hardware=dict(error=str(exc))
    save(out/'hardware-access.json',dict(**hardware,new_gpu_measurements=0,
        reason='Read-only hardware check; no usable profiling environment established; no measurement launch'))
    assert source_manifest()==before
    save(out/'step-cde-complete.json',dict(source_unchanged=True,controlled_predictions=len(observations),
        activation_predictions=len(activation),reconciled_ledgers=len(observations)+len(activation),new_gpu_measurements=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=str(DEFAULT_SOURCE))
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=('ab','cde'),required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve()
    if args.stage=='ab': out.mkdir(parents=True,exist_ok=False)
    base=SystemConfig.load(ROOT/args.source)
    if args.stage=='ab':
        save(out/'source-config.json',base.as_dict())
        step_ab(base,out)
    else: step_cde(base,out)


if __name__=='__main__': main()
