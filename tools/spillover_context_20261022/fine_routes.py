"""Exact formal arrivals/input/SWRR replay with explicit service scenarios.

The FIFO arithmetic screens development proposals only. It is not an activated
predictor: background services and changing interference remain unvalidated.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('_fine_route_exposure', ROOT/'tools/spillover_capacity_20261022/exposure.py')
exposure = importlib.util.module_from_spec(spec); spec.loader.exec_module(exposure)


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values); index = (len(values)-1)*quantile
    lower = math.floor(index); upper = math.ceil(index)
    return values[lower]+(values[upper]-values[lower])*(index-lower)


def cohort(config, samples, rate, seed, share):
    if type(rate) not in (int,float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Positive finite text rate required')
    if type(seed) is not int or seed < 0:
        raise ValueError('Nonnegative integer seed required')
    share = Fraction(share)
    if share < 0 or share > 1:
        raise ValueError('Share outside [0,1]')
    api = exposure.frozen_api()
    rates = dict(text=rate, image=.01, asr=.01, ocr=.01)
    text = {'text': rate*(1-float(share))}
    if share:
        text['generalist_2'] = rate*float(share)
    routing = dict(text=text, image={'generalist_2':.01}, asr={'generalist_2':.01}, ocr={'ocr':.01})
    trace = api['trace'].make_trace(rates, 600, seed, config['arrival_process'])
    aliases, routes = api['trace'].expand_routes(exposure.DEPLOYMENT, routing)
    pool = api['workload'].SamplePool(samples, list(rates), seed)
    router = api['routing'].Router(routes); rows = []
    for index, (arrival, task) in enumerate(trace):
        item = pool.next(task); alias = router.choose(task)
        rows.append(dict(index=index, arrival_seconds=arrival, task=task, pool_id=item['pool_id'],
                         alias=alias, family=aliases[alias]))
    quality, evidence = api['quality'].predict_quality(config['catalog'], exposure.DEPLOYMENT, routing, rates, trace, seed)
    return dict(workload=rates, routing=routing, generalist_share=str(share), seed=seed,
        trace_sha256=api['trace'].trace_digest(trace), rows=rows, historical_Q_loss=quality,
        historical_task_quality_loss={t:v['loss'] for t,v in evidence['per_task'].items()},
        text_input_counts={a:dict(Counter(r['pool_id'] for r in rows if r['task']=='text' and r['alias']==a)) for a in aliases})


def fifo(rows, service_lookup, horizon=600):
    clocks = defaultdict(float); latencies = defaultdict(list); completed = 0; late = 0
    for row in rows:
        service = service_lookup(row)
        if not math.isfinite(service) or service <= 0:
            raise ValueError('Positive finite service required')
        finish = max(row['arrival_seconds'], clocks[row['alias']])+service/1000
        clocks[row['alias']] = finish
        latencies[row['task']].append((finish-row['arrival_seconds'])*1000)
        completed += finish <= horizon
        late += finish > horizon+300
    return dict(per_task={t:dict(count=len(v), mean_ms=statistics.mean(v), p95_ms=percentile(v,.95)) for t,v in latencies.items()},
                completion_fraction=completed/len(rows), beyond300s_drain=late,
                alias_finish_seconds=dict(clocks))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True); p.add_argument('--joint-screen', type=Path, required=True)
    p.add_argument('--baseline-config', type=Path, required=True)
    p.add_argument('--rates', type=float, nargs='+', required=True); p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--shares', nargs='+', default=['0','1/64','1/32','1/16'])
    p.add_argument('--output', type=Path, required=True); args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Never overwrite scenario evidence')
    if any(s in [71,72,73,81,82] for s in args.seeds): raise ValueError('Reserved independent/final seeds')
    packet=json.loads(args.packet.read_text()); config=json.loads(Path(packet['system_config']).read_text())
    joint=json.loads(args.joint_screen.read_text()); vectors=joint['service_vectors']
    baseline_config=json.loads(args.baseline_config.read_text())
    for key in ['models','quality','capacity_factors']:
        if config['catalog'][key]!=baseline_config['catalog'][key]:
            raise ValueError('Historical baseline identity differs: '+key)
    baselines=baseline_config['catalog']['calibrated_replay_v2']['input_baselines']
    # Explicit development sensitivity cases, not fitted interpolation or confidence bounds.
    scenarios=[dict(name='historical_background_full_joint_text', g2_text_joint_multiplier=1.),
               dict(name='historical_background_half_joint_text', g2_text_joint_multiplier=.5),
               dict(name='historical_background_quarter_joint_text', g2_text_joint_multiplier=.25)]
    cases=[]
    for rate in args.rates:
        for seed in args.seeds:
            for share in args.shares:
                case=cohort(config,packet['samples'],rate,seed,share); results=[]
                for scenario in scenarios:
                    def service(row):
                        if row['task']=='text':
                            value=vectors[row['alias']][row['pool_id']]
                            return value*(scenario['g2_text_joint_multiplier'] if row['family']=='generalist_2' else 1)
                        return baselines[row['family']][row['task']]['1']['samples'][row['pool_id']]['median_service_ms']
                    result=fifo(case['rows'],service)
                    total=sum(case['workload'].values()); refs=config['latency_reference_ms']
                    terms={key:sum(case['workload'][t]/total*result['per_task'][t][metric]/refs[t] for t in case['workload'])
                           if set(result['per_task'])==set(case['workload']) else None
                           for key,metric in [('L','mean_ms'),('P','p95_ms')]}
                    result.update(scenario=scenario['name'], descriptive_terms=terms, U=None, formal_rho=None)
                    results.append(result)
                case['scenarios']=results; cases.append(case)
    for case in cases:
        base=next(c for c in cases if c['seed']==case['seed'] and c['workload']==case['workload'] and c['generalist_share']=='0')
        assert [(r['arrival_seconds'],r['task'],r['pool_id']) for r in case['rows']]==[(r['arrival_seconds'],r['task'],r['pool_id']) for r in base['rows']]
        for result, control in zip(case['scenarios'],base['scenarios']):
            result['L_delta_from_no_split']=result['descriptive_terms']['L']-control['descriptive_terms']['L'] if result['descriptive_terms']['L'] is not None else None
            result['P_delta_from_no_split']=result['descriptive_terms']['P']-control['descriptive_terms']['P'] if result['descriptive_terms']['P'] is not None else None
    args.output.mkdir(parents=True,exist_ok=False)
    receipts=[exposure.receipt(p) for p in [args.packet,args.joint_screen,args.baseline_config,Path(packet['system_config']),Path(packet['samples']),Path(__file__),Path(exposure.__file__)]]
    report=dict(schema='gsdeploy.fine_route_development_scenarios.v1',status='UNVALIDATED_SERVICE_SENSITIVITY',
        scenarios=scenarios,cases=cases,inputs=receipts,U=None,predictor_activated=False,
        scope='Exact original Poisson -> SamplePool -> expanded-alias SWRR; deterministic per-alias FIFO under explicit service scenarios.',
        limitations=['Text vectors come from one censored sustained diagnostic, not validated capacity.',
          'Background uses historical standalone services until joint background evidence is available.',
          'Half/quarter text scenarios are sensitivity assumptions, not empirical fits or error bounds.',
          'No M/E prediction or total utility ranking; runtime interference can reverse these differences.',
          'Missing task arrivals stay unknown; no development seed is dropped.'])
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    overview=[dict(rate=c['workload']['text'],seed=c['seed'],share=c['generalist_share'],Q=c['historical_Q_loss'],
        text_G2_inputs=len(c['text_input_counts'].get('generalist_2#0',{})),scenarios=[{k:r[k] for k in ['scenario','L_delta_from_no_split','P_delta_from_no_split']} for r in c['scenarios']]) for c in cases]
    (args.output/'overview.json').write_text(json.dumps(overview,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(dict(inputs=receipts,report=exposure.receipt(args.output/'report.json'),overview=exposure.receipt(args.output/'overview.json')),indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),output=str(args.output.resolve()),U=None)))


if __name__=='__main__':main()
