"""Tiny positive-case replay and evidence-labelled capacity scenario design.

No new workload is evaluated by the restricted replay predictor. Capacity
extrapolation is a prospective sizing heuristic, never a qualified GPU law.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from gsdeploy.config import SystemConfig
from gsdeploy.search_session import SearchSession
from experiments.routing_benefit_diagnosis import nominal_events, reconcile_events, pure_routing, split_route

SOURCE=ROOT/'artifacts/static-study-gpu-20261022/search-v2-v5-K8/system-config.json'


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    at=time.perf_counter();c=SystemConfig.load(SOURCE)
    # Ratios selected from archived results before this rerun. Seeds43/45 have
    # already been viewed historically, so are reproducibility, not blind tests.
    ratios=[0.,.5,5/7,.781180101993,.436453564549,1.]
    D={'generalist_2':1,'image':1,'ocr':1,'text':1}
    rows=[]
    for seed in (42,43,45):
        raw=c.as_dict();raw['random_seed']=seed
        with SearchSession(ROOT,raw) as session:
            ref=pure_routing(session.config,D,{'image':'image','ocr':'ocr'})
            for r in ratios:
                route=split_route(session.config,ref,'text','generalist_2','text',r)
                p=session.predict(D,route)
                if not p.get('eligible'): raise ValueError(p.get('reason'))
                ledger=nominal_events(session,D,route);reconcile_events(session.config,p,ledger)
                row=dict(seed=seed,specialist_share=r,U=p['U'],
                    per_task=p['per_task'],rho=p['rho'],
                    text_G_requests=ledger['per_path']['generalist_2']['text']['requests'],
                    text_S_requests=ledger['per_path']['text']['text']['requests'],
                    assignment_sha256=ledger['assignment_sha256'],
                    contributions={k:p[k+'_contribution'] for k in 'QTLPME'})
                rows.append(row)
                save(out/f'trace{seed}-r{r:.12f}.json',dict(summary=row,prediction=p,ledger=ledger))
    # Use actual legacy per-input tables for a transparent *sizing* calculation.
    with SearchSession(ROOT,c.as_dict()) as session:
        cal=session.module('calibration_v2');spec=cal.validated_replay_spec(session.config.catalog,c.workload)
        service=lambda m,t:cal.replay_path_statistics(spec,m,t,1)['mean_ms']/1000
        background=c.workload['text']*service('generalist_2','text')+c.workload['asr']*service('generalist_2','asr')
        g_service=service('generalist_2','ocr');s_service=service('ocr','ocr')
        Cg=(.8-background)/g_service;Cs=.8/s_service
        q=Cs+Cg/2
        lower=1-Cg/q;upper=Cs/q
        # Center and coarse controls declared before any high-load prediction.
        candidate=(lower+upper)/2
        rho=lambda r:{'G':background+q*(1-r)*g_service,'S':q*r*s_service}
        try:
            changed={**c.workload,'ocr':q}
            cal.validated_replay_spec(session.config.catalog,changed)
            gate='SUPPORTED'
        except ValueError as exc:
            gate=str(exc)
    design=dict(status='PROSPECTIVE_SIZING_ONLY_NOT_EXECUTED',
        premise='Legacy mean services assumed transferable only for design; not established at this load',
        deployment={'generalist_2':1,'image':1,'ocr':1},changed_workload=changed,
        G_background_rho=background,legacy_G_ocr_service_seconds=g_service,
        legacy_S_ocr_service_seconds=s_service,G_residual_capacity_rps=Cg,S_capacity_rps=Cs,
        proposed_OCR_qps=q,necessary_share_interval=[lower,upper],
        proposed_interior_share=candidate,
        sizing_rho_at_controls={str(r):rho(r) for r in [0.,.5,candidate,1.]},
        replay_gate=gate,complete_prediction_evaluations_at_changed_load=0,
        physical_feasibility_certified=False,continuous_optimum_certified=False,
        collection_scope='One fixed deployment, one predeclared higher load; train phase/lifecycle laws first; validate endpoints,1:1 and learned interior on identical frozen offered traces. Check memory/energy and full FIFO without dropping overload requests.',
        counterfactual_check='After fixed-D evidence passes, compare all legal alternative deployments/replicas under equal budget; fixed-D gain is not joint-system gain.')
    summary=[]
    for seed in (42,43,45):
        selected=[r for r in rows if r['seed']==seed]
        best_endpoint=max(r['U'] for r in selected if r['specialist_share'] in (0.,1.))
        half=next(r['U'] for r in selected if r['specialist_share']==.5)
        for r in selected:
            summary.append({**r,'gain_over_best_endpoint':r['U']-best_endpoint,'gain_over_half':r['U']-half})
    save(out/'capacity-scenario-design.json',design)
    save(out/'report.json',dict(schema='gsdeploy.continuous_value_case.v1',
        source_config=str(SOURCE),prediction_mode='calibrated_replay_v2',
        new_gpu_measurements=0,full_prediction_bundles=18,rows=summary,
        search_performed=False,seeds_are_previously_viewed=True,
        scope='Reproduction of declared historical points, not fresh discovery or independent validation',
        elapsed_seconds=time.perf_counter()-at))
    paths=[SOURCE,Path(__file__),ROOT/'experiments/routing_benefit_diagnosis.py',ROOT/'gsdeploy/prediction.py',ROOT/'gsdeploy/calibration_v2.py']
    save(out/'sources.json',[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    print(json.dumps(dict(seconds=time.perf_counter()-at,capacity_design=design,
        reproduced=[dict(seed=r['seed'],r=r['specialist_share'],U=r['U'],gain=r['gain_over_best_endpoint'])
                    for r in summary if r['specialist_share'] not in (0.,.5,5/7,1.)]),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
