"""Prepare controlled pair/triple launch contrasts from training-only anchors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import POOL,prepare,save
from gsdeploy.forecasting.native_training import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',required=True,type=Path);p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--revise-prepared',action='store_true')
    args=p.parse_args();fit=json.loads(args.fit.read_text());template=fit['profiles']
    source=Path(fit['source_receipts'][0].split(':sha256:')[0]).parent
    context=json.loads((source/'context.json').read_text())
    if context['resident_aliases']!={'generalist_2#0':'generalist_2','image#0':'image','ocr#0':'ocr'}:
        raise ValueError('This controlled supplement covers only the measured OCR-1 resident context')
    output=args.output.resolve()
    if args.revise_prepared:
        if any((output/name).exists() for name in ('collector.log','raw.json','progress.json')):
            raise ValueError('A started capture plan cannot be revised')
        for name in ('plan','supplement-design'):
            with (output/(name+'-before-age-contrasts.json')).open('x') as out:out.write((output/(name+'.json')).read_text())
    else:prepare(output,args.reference.resolve(),mode='overlap',seed=args.seed)
    plan=json.loads((output/'plan.json').read_text())
    if plan['execution_context_sha256']!=template['execution_context_sha256']:
        raise ValueError('Supplement collector differs from original measured training context')
    pool=[json.loads(line) for line in POOL.read_text().splitlines()];by_task={}
    for row in pool:by_task.setdefault(row['task_type'],[]).append(row)
    jobs=[];anchors=[]
    for task,inputs in by_task.items():
        candidates=[q for q in template['profiles'] if q['family']=='generalist_2' and q['task']==task]
        candidates.sort(key=lambda q:(sum(s['duration_ms']['mean'] for s in q['segments'] if s['kind']=='gpu'),q['pool_id']))
        for index in (0,len(candidates)//2,len(candidates)-1):
            q=candidates[index];anchor=dict(task=task,pool_id=q['pool_id'],rank=index,
                criterion='training solo native duration; not semantic input difficulty')
            anchors.append(anchor)
            solo_ms=sum(s['duration_ms']['mean'] for s in q['segments'] if s['kind']=='gpu')
            for pattern in ('g_image','g_ocr','g_image_ocr'):
                for lag_name,lag in (('aligned',0.),('fixed150',150.),('solo35',.35*solo_ms),('solo90',.9*solo_ms)):
                    requests=[dict(alias='generalist_2#0',task=task,pool_id=q['pool_id'],offset_ms=0.)]
                    for family in ('image','ocr'):
                        if family in pattern:
                            item=by_task[family][(index+17)%len(by_task[family])]
                            requests.append(dict(alias=family+'#0',task=family,pool_id=item['pool_id'],offset_ms=lag))
                    jobs.append(dict(id=f'controlled:{task}:{index}:{pattern}:{lag_name}',requests=requests))
    # Full image inputs with an OCR permutation different from initial co-runs.
    for i,item in enumerate(by_task['image']):
        ocr=by_task['ocr'][(i*7+13)%len(by_task['ocr'])]
        jobs.append(dict(id=f'image_ocr_permutation:{i}',requests=[
            dict(alias='image#0',task='image',pool_id=item['pool_id'],offset_ms=0.),
            dict(alias='ocr#0',task='ocr',pool_id=ocr['pool_id'],offset_ms=0.)]))
    random.Random(args.seed).shuffle(jobs)
    plan.update(jobs=jobs,diagnostic_supplement=True,training_anchor_fit_sha256=digest(fit),
        design='pair-only plus triple, full image/OCR permutation, fixed and training-solo-scaled launch lags',
        not_formal_utility=True)
    save(output/'plan.json',plan);save(output/'supplement-design.json',dict(
        schema='gsdeploy.native_controlled_supplement.v1',split='training',source_fit=str(args.fit.resolve()),
        source_fit_sha256=digest(fit),anchors=anchors,launch_lag_recipes=['aligned','150ms','0.35*training_solo','0.9*training_solo'],
        no_validation_or_confirmation_data_used=True,admission_enabled=False))
    print(json.dumps(dict(output=str(output),jobs=len(jobs),context=plan['execution_context_sha256'])))


if __name__=='__main__':main()
