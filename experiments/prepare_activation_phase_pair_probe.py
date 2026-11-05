"""Matched training probe of every two-owner phase combination.

This is the first concurrent training step, not a substitute for larger
rosters, crossed input sizes, FIFO bursts, or independent qualification.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.activation_phase_crossed_design import phase_conditions, align
from experiments.collect_activation_training import POOL, digest, save
from experiments.fit_activation_completion_bands import build_band_template
from experiments.fit_activation_native import load_bundle


def prepare(solo, observer_report, output, seed):
    if any(s in (6110491,6110492) for s in range(seed,seed+3)):
        raise ValueError('Reserved confirmation seeds cannot enter training')
    gate=json.loads(observer_report.read_text())
    if gate.get('observer_repeat_check_passed') is not True or gate.get('issues'):
        raise ValueError('Matched observer gate required')
    bundles=[load_bundle(solo/f'solo-round{i}') for i in (1,2)]
    context=bundles[0]['context'];aliases=context['resident_aliases']
    expected={f+'#0':f for f in ('generalist_2','text','image','ocr','asr')}
    if aliases!=expected or any(b['context']!=context for b in bundles):
        raise ValueError('Matched five-model single-owner baseline context required')
    semantic_receipts=[s for n,s in gate['sources'].items() if n.startswith('semantic-')]
    semantic=[load_bundle(Path(s.rsplit(':sha256:',1)[0]).parent) for s in semantic_receipts]
    if [b['source'] for b in semantic]!=semantic_receipts:raise ValueError('Accepted observer raw receipts changed')
    if len(semantic)!=2 or any(b['context']!=context for b in semantic):
        raise ValueError('Baseline and accepted observer contexts differ')
    repeat=json.loads((solo/'description.json').read_text())
    diagnostic=json.loads((solo/'completion-template-diagnostic.json').read_text())
    receipts=[b['source'] for b in bundles]
    if (repeat['sources']!=receipts or repeat['changed_output_profiles']
            or diagnostic['sources']!=receipts or any(p['maximum']>.2 for p in diagnostic['paths'].values())):
        raise ValueError('Unchanged full-pool outputs and feasible completion baseline required before this phase-only probe')
    content=POOL.read_bytes()
    if hashlib.sha256(content).hexdigest()!=context['input_pool_sha256']:raise ValueError('Frozen input pool changed')
    template,_,_=build_band_template(bundles,[json.loads(line) for line in content.decode().splitlines()])
    roots=template['native_order_law']['fit_diagnostics']
    if not roots or not all(d['accepted'] for d in roots):raise ValueError('Original root-order baseline gates must pass')
    by_path={}
    for profile in template['profiles']:
        by_path.setdefault((profile['family'],profile['task']),[]).append(profile)
    medians={key:sorted(values,key=lambda p:(sum(s['duration_ms']['mean'] for s in p['segments'] if s['kind']=='gpu'),p['pool_id']))[len(values)//2]
             for key,values in by_path.items()}
    conditions=phase_conditions(template,aliases,[2])
    if len(conditions)!=74:raise ValueError('Expected complete 74 two-owner phase/task combinations')
    source=solo/'solo-round1';sources=json.loads((source/'sources.json').read_text())
    for r in sources:
        if any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest()!=r['sha256'] for k in ('path','snapshot')):
            raise ValueError('Frozen execution source drift: '+r['path'])
    output.mkdir(parents=True,exist_ok=False)
    extra=[Path(__file__).resolve(),ROOT/'experiments/activation_phase_crossed_design.py',
        ROOT/'experiments/audit_activation_band_support.py',observer_report,solo/'description.json',
        solo/'quality-description.json',solo/'completion-template-diagnostic.json']
    frozen=[]
    for i,path in enumerate(extra):
        snapshot=output/'design-source-snapshot'/str(i)/path.name;snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(path.read_bytes());frozen.append(dict(path=str(path),snapshot=str(snapshot),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(output/'template.json',template)
    captures=[]
    for round_no in (1,2,3):
        directory=output/f'phase-pair-round{round_no}';directory.mkdir()
        order=copy.deepcopy(conditions);random.Random(seed+round_no-1).shuffle(order)
        ordinals=dict.fromkeys(aliases,0);jobs=[]
        for i,condition in enumerate(order):
            profiles=[medians[c['family'],c['task']] for c in condition]
            job=align(condition,profiles,template,ordinals)
            job['id']='phase-pair:'+digest(condition)[:20];jobs.append(job)
            for c in condition:ordinals[c['alias']]+=1
        # An actual native request separates all idle blocks, so their ages do
        # not merge accidentally. The 90s interval remains untrimmed evidence.
        anchor=medians['image','image']
        for seconds in (1,3,12,30,90):
            jobs.append(dict(id=f'idle-anchor:{seconds}',requests=[dict(alias='image#0',task='image',pool_id=anchor['pool_id'],offset_ms=0.)]))
            jobs.append(dict(id=f'declared-idle:{seconds}',requests=[],duration_ms=seconds*1000.))
        plan=dict(schema='gsdeploy.native_activity_training_plan.v1',mode='overlap',split='training',
            execution_context_sha256=digest(context),run_id=directory.name,seed=seed+round_no-1,
            aliases=aliases,jobs=jobs,budget_seconds=1800,priming='largest_three_per_supported_task',
            probe_only=True,phase_alignment_intent_not_observation=True,
            observer_report_sha256=hashlib.sha256(observer_report.read_bytes()).hexdigest(),
            baseline_sources=receipts,preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            confirmation_seeds_reserved=[6110491,6110492],physical_predictor_qualified=False,formal_utility_qualified=False)
        save(directory/'context.json',context);save(directory/'sources.json',sources);save(directory/'plan.json',plan)
        captures.append(str(directory))
    save(output/'queue.json',dict(schema='gsdeploy.native_capture_queue.v1',state=str(output/'continuation.json'),captures=captures))
    save(output/'design.json',dict(schema='gsdeploy.phase_pair_training_probe.v1',seeds=list(range(seed,seed+3)),
        phase_conditions=len(conditions),requests_per_capture=153,independent_worker_restarts=3,
        source_receipts=frozen,baseline_sources=receipts,training_template_sha256=digest(template),
        selected_input_policy='median solo native duration per path; fixed before any co-run probe result',
        intended_alignment='midpoint of training host bands with planned request ordinal; realized exposure audited separately',
        required_later_steps=['crossed short/medium/long inputs','three/four/five-owner phase exposure',
            'same-owner FIFO bursts','full training replay','frozen independent validation','complete arrival/startup-energy qualification'],
        physical_predictor_qualified=False,formal_utility_qualified=False))
    print(json.dumps(dict(output=str(output),captures=3,phase_conditions=74,requests_per_capture=153)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('solo-base','observer-report','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    prepare(a.solo_base.resolve(),a.observer_report.resolve(),a.output.resolve(),a.seed)
