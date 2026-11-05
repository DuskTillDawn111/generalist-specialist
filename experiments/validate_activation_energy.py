"""Freeze and independently test conditional idle-energy accuracy, not FIFO qualification."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import POOL, prepare, save
from experiments.fit_activation_native import load_bundle
from gsdeploy.energy import integrate_power
from gsdeploy.forecasting.native_energy import NativeIdleAgeEnergy, activity_windows
from gsdeploy.forecasting.native_training import digest, stage_layout, validate_capture

CRITERIA = dict(minimum_independent_runs=2, maximum_whole_run_relative_energy_error=.15,
                maximum_long_idle_window_relative_energy_error=.15,
                all_observed_contexts_supported=True,
                scope='independent energy conditional on actual native spans; not autonomous FIFO energy or formal utility')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('op', choices=['prepare', 'score'])
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--fit', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--seeds', type=int, nargs=2, default=[6110661, 6110662])
    args = p.parse_args(); output = args.output.resolve()
    if args.op == 'prepare':
        if args.fit is None or args.reference is None:
            p.error('prepare requires --fit and --reference')
        fit = json.loads(args.fit.read_text())
        spec = fit['profiles']['native_idle_age_energy']
        if not fit['idle_energy_diagnostic']['accepted'] or len(set(args.seeds)) != 2 or set(args.seeds)&{6110491,6110492}:
            raise ValueError('Accepted energy fit and distinct nonconfirmation seeds required')
        # This protocol deliberately tests energy alone while service is under development.
        NativeIdleAgeEnergy(spec)
        context = json.loads((Path(fit['source_receipts'][0].rsplit(':sha256:',1)[0]).parent/'context.json').read_text())
        if set(context['resident_aliases'].values()) != {'generalist_2','image','ocr'}:
            raise ValueError('This energy protocol covers the original hybrid domain')
        output.mkdir(parents=True, exist_ok=False)
        save(output/'fit-frozen.json',fit); save(output/'criteria-frozen.json',CRITERIA)
        by_task = {}
        for line in POOL.read_text().splitlines():
            row=json.loads(line);by_task.setdefault(row['task_type'],[]).append(row)
        runs=[]
        for round_no,seed in enumerate(args.seeds,1):
            directory=output/f'energy-validation-round{round_no}'
            prepare(directory,args.reference.resolve(),image=True,
                    ocr_replicas=sum(f=='ocr' for f in context['resident_aliases'].values()),mode='overlap',seed=seed)
            plan=json.loads((directory/'plan.json').read_text())
            if plan['execution_context_sha256']!=spec['execution_context_sha256']:
                raise ValueError('Energy validation execution context differs from training')
            rng=random.Random(seed);tasks=list(by_task);rng.shuffle(tasks)
            jobs=[dict(id='heldout_idle_initial',requests=[],duration_ms=22000)]
            for i,task in enumerate(tasks):
                requests=[]
                for alias,family in plan['aliases'].items():
                    owner_task=task if family=='generalist_2' else family
                    inputs=list(by_task[owner_task]);rng.shuffle(inputs)
                    for j,item in enumerate(inputs[:1 if family=='generalist_2' else 5]):
                        requests.append(dict(alias=alias,task=owner_task,pool_id=item['pool_id'],
                                             offset_ms=rng.choice([0.,40.,130.,270.])+j*90.))
                jobs.append(dict(id=f'heldout_busy:{task}',requests=requests))
                jobs.append(dict(id=f'heldout_idle_after:{task}',requests=[],
                                 duration_ms=[12000,28000,18000,26000][i]))
            plan.update(jobs=jobs,split='development_validation',not_for_training=True,
                        energy_only_protocol=True,admission_enabled=False,
                        frozen_fit_sha256=digest(fit),criteria_sha256=digest(CRITERIA))
            save(directory/'plan.json',plan);runs.append(str(directory))
        files=[Path(__file__),ROOT/'experiments/fit_activation_native.py',
               *[ROOT/'gsdeploy/forecasting'/n for n in ('native_energy.py','native_context.py','native_training.py')],
               ROOT/'gsdeploy/energy.py']
        sources=[]
        for i,path in enumerate(files):
            content=path.read_bytes();snapshot=output/'source-snapshot'/str(i)/path.name
            snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
            sources.append(dict(path=str(path),snapshot=str(snapshot),sha256=hashlib.sha256(content).hexdigest()))
        save(output/'protocol.json',dict(schema='gsdeploy.conditional_energy_validation_protocol.v1',
             fit_sha256=digest(fit),criteria_sha256=digest(CRITERIA),runs=runs,prediction_sources=sources,
             source_receipts=fit['source_receipts'],formal_utility_qualified=False))
        print(json.dumps(dict(status='prepared',runs=runs)))
        return
    protocol=json.loads((output/'protocol.json').read_text())
    fit=json.loads((output/'fit-frozen.json').read_text());criteria=json.loads((output/'criteria-frozen.json').read_text())
    if digest(fit)!=protocol['fit_sha256'] or criteria!=CRITERIA or digest(criteria)!=protocol['criteria_sha256']:
        raise ValueError('Frozen conditional energy fit or criteria changed')
    for source in protocol['prediction_sources']:
        if any(hashlib.sha256(Path(source[k]).read_bytes()).hexdigest()!=source['sha256'] for k in ('path','snapshot')):
            raise ValueError('Conditional energy prediction source drift: '+source['path'])
    profiles={p['id']:p for p in fit['profiles']['profiles']}
    lookup={(p['family'],p['task'],p['pool_id']):p for p in profiles.values()}
    law=NativeIdleAgeEnergy(fit['profiles']['native_idle_age_energy']);reports=[];reasons=[];seen=set()
    for directory in map(Path,protocol['runs']):
        bundle=load_bundle(directory);raw=bundle['raw'];plan=bundle['plan']
        if (validate_capture(bundle,allowed_splits=('development_validation',))!=law.spec['execution_context_sha256']
                or plan.get('frozen_fit_sha256')!=digest(fit) or plan.get('criteria_sha256')!=digest(criteria)
                or raw['run_id'] in seen or bundle['source'] in protocol['source_receipts']
                or raw['run_id'] in {r['run_id'] for r in fit['normalized_runs']}):
            raise ValueError('Independent frozen matching energy captures required')
        seen.add(raw['run_id'])
        start,end=raw['observation_start_ns'],raw['observation_end_ns'];spans=[]
        for row in raw['request_records']:
            profile=lookup[(raw['resident_aliases'][row['model']],row['task'],row['pool_id'])]
            spans.extend(dict(alias=row['model'],profile_id=profile['id'],**s) for s in stage_layout(row))
        initial=max(s['end_ns'] for row in raw['priming_request_records'] for s in stage_layout(row) if s['kind']=='gpu')
        windows=activity_windows(profiles,spans,start,end)
        actual=integrate_power(raw['board_samples'],start,end,max_gap_seconds=.25)
        predicted,_=law.integrate(windows,initial)
        error=abs(predicted-actual)/actual;idle=[]
        if error>criteria['maximum_whole_run_relative_energy_error']:
            reasons.append(raw['run_id']+':WHOLE_RUN_ENERGY_ERROR')
        for batch in raw['batches']:
            if batch['request_ids']:continue
            lo,hi=batch['start_ns'],batch['arrival_end_ns']
            prefix=[(a,min(b,lo),x) for a,b,x in windows if a<lo]
            _,state=law.integrate(prefix,initial)
            value,_=law.integrate([(max(a,lo),min(b,hi),x) for a,b,x in windows if a<hi and b>lo],state)
            observed=integrate_power(raw['board_samples'],lo,hi,max_gap_seconds=.25)
            relative=abs(value-observed)/observed
            idle.append(dict(id=batch['id'],observed_joules=observed,predicted_joules=value,relative_error=relative))
            if relative>criteria['maximum_long_idle_window_relative_energy_error']:
                reasons.append(raw['run_id']+':IDLE_WINDOW_ENERGY_ERROR:'+batch['id'])
        reports.append(dict(run_id=raw['run_id'],source=bundle['source'],observed_joules=actual,
                            predicted_joules=predicted,relative_error=error,idle_windows=idle))
    if len(reports)<criteria['minimum_independent_runs']:reasons.append('INSUFFICIENT_INDEPENDENT_RUNS')
    report=dict(schema='gsdeploy.conditional_energy_validation.v1',fit_sha256=digest(fit),criteria=criteria,
                conditional_energy_accuracy_passed=not reasons,reasons=reasons,runs=reports,
                fifo_energy_qualified=False,formal_utility_qualified=False,admission_enabled=False)
    with (output/'validation.json').open('x') as out:json.dump(report,out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
