"""Seal exactly two Poisson-arrival GPU checks of the already frozen predictor.

Same task rates/window/runtime as the scoped validation, unchanged fit and
50% generalist routing. No fitting, rejection sampling, outcome-based seeds,
adaptive workloads or extra runs. All forecasts precede both GPU captures.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import digest
from experiments.replay_activation_aggregate import predict_plan
from gsdeploy.trace import make_trace
from gsdeploy.routing import Router

BASE=ROOT/'artifacts/activation-gpu-20261104/five-model-aggregate-validation-v1'
SEEDS=(6110581,6110582)
TASKS=('asr','image','ocr','text')


def read(p):return json.loads(p.read_text())


def receipt(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    # Reject already-used seeds; never choose replacement seeds after inspecting outcomes.
    for p in (ROOT/'artifacts/activation-gpu-20261104').rglob('plan.json'):
        if read(p).get('seed') in SEEDS:raise ValueError('Predeclared seed already used: '+str(p))
    fit=read(BASE/'fit-frozen.json');original=read(BASE/'aggregate-validation-round1/plan.json')
    context=read(BASE/'aggregate-validation-round1/context.json')
    previous=read(BASE/'protocol.json')
    for r in previous['source_receipts']:
        assert receipt(Path(r['path']))['sha256']==r['sha256']
        assert receipt(Path(r['snapshot']))['sha256']==r['sha256']
    out.mkdir(parents=True,exist_ok=False)
    for name in ('fit-frozen.json','policy-frozen.json'):shutil.copyfile(BASE/name,out/name)
    runtime=read(BASE/'aggregate-validation-round1/sources.json')
    for i,r in enumerate(runtime):
        for key in ('path','snapshot'):assert receipt(Path(r[key]))['sha256']==r['sha256']
        dst=out/'runtime-source-snapshot'/str(i)/Path(r['path']).name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(r['snapshot'],dst);r['snapshot']=str(dst)
    horizon=original['jobs'][0]['duration_ms']/1000
    rate=original['jobs'][0]['design']['task_rate'];rates=dict.fromkeys(TASKS,rate)
    assert math.isclose(horizon*rate,24)
    captures=[];seals=[]
    for index,seed in enumerate(SEEDS,1):
        directory=out/f'poisson-round{index}';directory.mkdir()
        trace=make_trace(rates,horizon,seed,'poisson')
        pools={}
        for t in TASKS:
            g={p['pool_id'] for p in fit['profiles']['profiles'] if (p['family'],p['task'])==('generalist_2',t)}
            s={p['pool_id'] for p in fit['profiles']['profiles'] if (p['family'],p['task'])==(t,t)}
            assert g==s and g
            pools[t]=sorted(g);random.Random(f'{seed}:input:{t}').shuffle(pools[t])
        counts=dict.fromkeys(TASKS,0)
        router=Router({t:{'generalist_2':rate/2,t:rate/2} for t in TASKS})
        requests=[]
        for offset,t in trace:
            requests.append(dict(task=t,alias=router.choose(t)+'#0',pool_id=pools[t][counts[t]%len(pools[t])],offset_ms=offset*1000))
            counts[t]+=1
        plan=copy.deepcopy(original)
        plan.update(run_id=directory.name,seed=seed,split='development_validation',not_for_training=True,
            evaluation_role='prespecified_poisson_prediction_error_only',round=index,budget_seconds=math.ceil(horizon+700),
            input_domain='unchanged profiled input pool; fresh unconditioned Poisson arrivals',unseen_input_generalization=False)
        plan.pop('confirmation_seeds_reserved',None)
        plan['jobs']=[dict(id='poisson-mixed-routing',requests=requests,duration_ms=horizon*1000,
            design=dict(kind='continuous_arrivals',arrival_regime='poisson',arrival_process='poisson',
                task_rates_rps=rates,expected_per_task=horizon*rate,realized_per_task=counts,
                generalist_share=.5,routing_rule='unchanged smooth weighted round robin',
                offered_trace_sha256=digest(trace),unconditioned_counts=True))]
        save(directory/'plan.json',plan);save(directory/'context.json',context);save(directory/'sources.json',runtime)
        prediction=predict_plan(fit,plan)
        save(directory/'prediction-sealed.json',prediction)
        assert prediction['complete'] and not prediction['observed_timestamps_used']
        assert prediction['predicted_episode_board_joules'] is not None
        captures.append(str(directory))
        seals.append(dict(directory=str(directory),seed=seed,realized_counts=counts,
            plan=receipt(directory/'plan.json'),prediction=receipt(directory/'prediction-sealed.json')))
    queue=dict(schema='gsdeploy.native_capture_queue.v1',captures=captures,state=str(out/'capture-state.json'))
    save(out/'queue.json',queue)
    sources=[receipt(Path(r['path'])) for r in previous['source_receipts']]
    for p in [Path(__file__),ROOT/'gsdeploy/trace.py',ROOT/'gsdeploy/routing.py',ROOT/'experiments/run_activation_queue.py',ROOT/'experiments/run_activation_capture.py']:
        sources.append(receipt(p))
    save(out/'protocol.json',dict(schema='gsdeploy.fixed_poisson_predictor_validation.v1',
        prediction_sealed_unix_ns=time.time_ns(),maximum_gpu_episodes=2,predictor_refits=0,prospective_predictions=2,
        fit=receipt(out/'fit-frozen.json'),original_fit=receipt(BASE/'fit-frozen.json'),policy=receipt(out/'policy-frozen.json'),
        captures=captures,predictions=seals,queue=receipt(out/'queue.json'),source_receipts=sources,
        arrival_process='poisson',generator='gsdeploy.trace.make_trace: per-task independent exponential interarrival times',
        same_generator_as_early_static_and_routing=True,old_confirmation_regimes=['stratified_jitter','paired_bursts'],
        task_rates_rps=rates,horizon_seconds=horizon,expected_requests_per_task_per_run=24,
        routing='50% generalist_2 + 50% task specialist, SWRR',residency=plan['aliases'],
        primary_error='100 * abs(predicted_e2e_ms - observed_e2e_ms) / observed_e2e_ms, request-weighted mean/min/max',
        secondary_errors=['absolute relative error of task/run mean latency','absolute relative error of task/run P95 latency',
            'full GPU episode energy relative error','fixed offered-window completion throughput relative error'],
        scope='fresh Poisson arrival histories on the known input pool; two runs, one fixed mixed route; not unseen-input or general deployment qualification',
        stop_rule='two fixed captures once; keep all errors/failures; no retraining, tuning, seed selection or automatic retries'))
    for r in sources:assert receipt(Path(r['path']))==r
    print(json.dumps(dict(output=str(out),episodes=2,counts=[r['realized_counts'] for r in seals],horizon_seconds=horizon,sealed_before_capture=True)))


if __name__=='__main__':main()
