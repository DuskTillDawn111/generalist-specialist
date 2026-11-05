"""Prepare and score independent captures against an immutable native fit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import prepare,save
from experiments.fit_activation_native import load_bundle
from gsdeploy.forecasting.native_training import digest
from gsdeploy.forecasting.native_validation import DEFAULT_CRITERIA,validate_native_fit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('op',choices=('prepare','score'))
    p.add_argument('--output',required=True,type=Path);p.add_argument('--fit',type=Path)
    p.add_argument('--reference',type=Path);p.add_argument('--seeds',type=int,nargs=2,default=[6110521,6110522])
    args=p.parse_args();output=args.output.resolve()
    if args.op=='prepare':
        if args.fit is None or args.reference is None:p.error('prepare requires --fit and --reference')
        fit=json.loads(args.fit.read_text());output.mkdir(parents=True,exist_ok=False)
        law=fit['profiles']['native_activity_law']
        if (not law['fit_diagnostics'] or any(not g['accepted'] for g in law['fit_diagnostics'])
                or not fit['resource_diagnostic']['accepted']
                or ('native_idle_age_energy' in fit['profiles'] and not fit.get('idle_energy_diagnostic',{}).get('accepted'))
                or any(not g['accepted'] for g in fit['profiles'].get('native_order_law',{}).get('fit_diagnostics',[]))
                or len(set(args.seeds))!=2 or set(args.seeds)&{6110491,6110492}):
            raise ValueError('Accepted training groups/resources and distinct nonconfirmation seeds required')
        families={p['family'] for p in fit['profiles']['profiles']}
        if families-{'generalist_2','text','image','ocr','asr'}:raise ValueError('Expected retained native family domain')
        context=json.loads((Path(fit['source_receipts'][0].split(':sha256:')[0]).parent/'context.json').read_text())
        ocr_count=sum(f=='ocr' for f in context['resident_aliases'].values())
        save(output/'fit-frozen.json',fit);save(output/'criteria-frozen.json',DEFAULT_CRITERIA)
        runs=[]
        for index,seed in enumerate(args.seeds,1):
            directory=output/f'validation-round{index}'
            if context.get('preparation_workflow')=='general_resident_v1':
                from collections import Counter
                from experiments.prepare_activation_resident import prepare_resident
                prepare_resident(directory,args.reference.resolve(),dict(Counter(context['resident_aliases'].values())),mode='overlap',seed=seed)
            elif families=={'generalist_2','image','ocr'}:
                prepare(directory,args.reference.resolve(),image=True,ocr_replicas=ocr_count,mode='overlap',seed=seed)
            else:raise ValueError('No matching collector preparation workflow for this fit')
            plan=json.loads((directory/'plan.json').read_text())
            if plan['execution_context_sha256']!=fit['profiles']['execution_context_sha256']:
                raise ValueError('Validation collector source/residency differs from frozen fit')
            rng=random.Random(seed)
            # Preserve each owner's complete marginal input cohort but change
            # which inputs meet across models. Launch offsets alone cannot test
            # the input-sensitive interference used by continuous routing.
            by_alias={alias:[] for alias in plan['aliases']}
            for job in plan['jobs']:
                for request in job['requests']:by_alias[request['alias']].append(request)
            for alias,requests in by_alias.items():
                if plan['aliases'][alias]=='generalist_2':continue
                inputs=[r['pool_id'] for r in requests];rng.shuffle(inputs)
                for request,pool_id in zip(requests,inputs):request['pool_id']=pool_id
            for job in plan['jobs']:
                # Different request launch alignments expose unseen transitions,
                # while retaining the same pinned inputs and resident state.
                offsets={alias:rng.choice((0.,30.,100.,200.)) for alias in plan['aliases']}
                for request in job['requests']:request['offset_ms']+=offsets[request['alias']]
            plan.update(split='development_validation',not_for_training=True,
                        frozen_fit_sha256=digest(fit),criteria_sha256=digest(DEFAULT_CRITERIA),
                        validation_design='new per-owner launch offsets and cross-model input pairings; owner input marginals preserved')
            save(directory/'plan.json',plan);runs.append(str(directory))
        sources=[]
        files=set((ROOT/'gsdeploy/forecasting').glob('*.py'))
        files.update(ROOT/'gsdeploy'/name for name in ('paper_forecast.py','paper_training.py','paper_capture.py','energy.py','evaluation.py'))
        files.add(Path(__file__))
        for index,path in enumerate(sorted(files)):
            content=path.read_bytes();snapshot=output/'prediction-source-snapshot'/str(index)/path.name
            snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
            sources.append(dict(path=str(path),snapshot=str(snapshot),sha256=hashlib.sha256(content).hexdigest()))
        save(output/'protocol.json',dict(schema='gsdeploy.native_validation_protocol.v1',
            fit_sha256=digest(fit),criteria_sha256=digest(DEFAULT_CRITERIA),runs=runs,
            source_fit=str(args.fit.resolve()),source_file_sha256=hashlib.sha256(args.fit.read_bytes()).hexdigest(),
            confirmation_seeds_reserved=[6110491,6110492],prediction_sources=sources,formal_utility_qualified=False))
        print(json.dumps(dict(status='prepared',runs=runs)))
    else:
        protocol=json.loads((output/'protocol.json').read_text())
        fit=json.loads((output/'fit-frozen.json').read_text());criteria=json.loads((output/'criteria-frozen.json').read_text())
        if digest(fit)!=protocol['fit_sha256'] or digest(criteria)!=protocol['criteria_sha256']:
            raise ValueError('Frozen fit or independent validation criteria changed')
        for source in protocol.get('prediction_sources',[]):
            if any(hashlib.sha256(Path(source[k]).read_bytes()).hexdigest()!=source['sha256'] for k in ('path','snapshot')):
                raise ValueError('Independent validation prediction source changed after protocol freeze: '+source['path'])
        bundles=[load_bundle(Path(directory)) for directory in protocol['runs']]
        for b in bundles:
            if (b['plan'].get('frozen_fit_sha256')!=protocol['fit_sha256']
                    or b['plan'].get('criteria_sha256')!=protocol['criteria_sha256']):
                raise ValueError('Validation captures were not committed to this predictor and criteria')
        report=validate_native_fit(fit,bundles,criteria)
        with (output/'validation.json').open('x') as out:json.dump(report,out,ensure_ascii=False,indent=2);out.write('\n')
        print(json.dumps({k:v for k,v in report.items() if k!='runs'},ensure_ascii=False))


if __name__=='__main__':main()
