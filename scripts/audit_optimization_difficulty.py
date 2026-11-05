"""Independently check saved diagnostic identities, bounds, cuts and FIFO ledgers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsdeploy.search_session import SearchSession
from experiments.joint_scalability import ROOT, source_manifest
from experiments.diagnose_continuous_phase1 import save
from experiments.routing_benefit_diagnosis import nominal_events, reconcile_events


def read(path): return json.loads(Path(path).read_text())


def check_prediction(prediction, deployment, routing, config):
    assert set(routing)==set(config['workload'])
    for task,row in routing.items():
        assert math.isclose(sum(row.values()),config['workload'][task],rel_tol=1e-12,abs_tol=1e-12)
        assert all(q>0 and m in deployment and task in config['catalog']['models'][m]['supported_tasks'] for m,q in row.items())
    if prediction.get('U') is not None:
        assert prediction['eligible'] and prediction['constraints_eligible']
        assert prediction['safe_memory_mib']<=config['gpu_memory_mib']+1e-6
        assert prediction['max_rho']<=.8+1e-8 and prediction['incomplete_after_drain']==0
        assert math.isclose(sum(prediction[k+'_contribution'] for k in 'QTLPME'),prediction['U'],abs_tol=1e-12)


def audit(source, report, output):
    assert source_manifest()==read(source/'source-manifest-before.json')
    runs=0;predictions=0;certificates=0;cuts=0;ledgers=0;enumerated=0;events=0
    cases=read(source/'cases.json')
    records=[(source/c['raw'],read(source/c['config'])) for c in cases]
    records.extend((p,read(source/'ablation-config.json')) for p in source.glob('ablation-*.json') if p.name!='ablation-config.json')
    for path,c in records:
        r=read(path);raw=r['result'];t=r['timing'];m=raw['metrics']
        assert len(t['rounds'])==m['milp_calls']
        assert len(t['predictions'])==m['full_prediction_calls']
        assert len(t['cuts'])==m['residual_identity_cuts']
        assert math.isclose(sum(r['exclusive_seconds'].values()),r['total_seconds'],abs_tol=1e-7)
        for v in t['predictions']:
            check_prediction(v['prediction'],v['deployment'],v['routing'],c)
        for hist,cut in zip((h for h in raw['history'] if h.get('candidate_U') is not None and h['reason']=='OK'),t['cuts']):
            assert math.isclose(cut['residual'],max(0.,-hist['candidate_U']-hist['candidate_floor']),abs_tol=1e-12)
        if raw['optimality_certified']:
            assert t['column_generation_complete'] and raw['metrics']['unknown_candidates']==0
            assert raw['optimality_gap']<=1e-8
            assert t['rounds'][-1]['gap_condition'] or t['rounds'][-1]['solver_status']==2
            certificates+=1
        if not t['column_generation_complete']:
            assert not raw['optimality_certified'] and m['milp_calls']==0
        runs+=1;predictions+=len(t['predictions']);cuts+=len(t['cuts'])
    for pool in (3,5):
        rank=read(source/f'p{pool}-K1-ranking.json')
        c=read(source/f'p{pool}-n1-K1-config.json')
        assert len(rank['plans'])==rank['full_prediction_calls']
        for p in rank['plans']:
            saved=read(source/p['raw']);check_prediction(saved['prediction'],saved['deployment'],saved['routing'],c)
            assert saved['prediction']['U']==p['utility']
        assert all(a['utility']>=b['utility'] for a,b in zip(rank['ranked_plans'],rank['ranked_plans'][1:]))
        assert rank['ranked_plans'][0]['utility']==read(source/f'p{pool}-n1-K1-joint.json')['result']['winner']['score']
        enumerated+=len(rank['plans'])
    c=read(source/'controlled-config.json')
    with SearchSession(ROOT,c) as session:
        for path in sorted((source/'controlled').glob('*.json')):
            r=read(path);check_prediction(r['prediction'],r['deployment'],r['routing'],c)
            ledger=nominal_events(session,r['deployment'],r['routing'])
            reconcile_events(session.config,r['prediction'],ledger)
            assert ledger['events']==r['ledger']['events']
            assert ledger['assignment_sha256']==r['ledger']['assignment_sha256']
            ledgers+=1
            events+=len(ledger['events'])
    cert=read(report/'certificate-explanation.json')
    assert cert['other_plans_with_competitive_floor']==0
    assert cert['min_other_plan_floor']>cert['second_master']['objective']
    summary=read(report/'summary.json')
    assert len(summary['figures'])==9 and all((report/p).stat().st_size>1000 for p in summary['figures'])
    assert summary['raw_prediction_bundles']==predictions+enumerated+ledgers
    save(output,dict(status='PASS',core_sources_unchanged=True,run_records=runs,
        certified_runs=certificates,search_prediction_bundles=predictions,identity_cuts_checked=cuts,
        enumerated_plans_checked=enumerated,reconstructed_fifo_ledgers=ledgers,
        reconstructed_fifo_events=events,total_saved_prediction_bundles=predictions+enumerated+ledgers,
        figures=9,audit_full_prediction_calls=0,
        scope='read-only saved-prediction and event reconstruction; no new full predictor runs or physical validation'))
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(Path(a.source).resolve(),Path(a.report).resolve(),Path(a.output).resolve())
