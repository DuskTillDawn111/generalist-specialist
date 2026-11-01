"""Freeze a next development pair from complete search, without choosing a seed.

The candidate must be the actual aggregate-search winner. The reference is the
best no-text-split identity with identical residents and background routes.
This tool never substitutes a desired spillover plan if search chose otherwise.
Native admission uses the unchanged frozen development transport contract.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0,str(SOURCE))
from gsdeploy import static_study
from gsdeploy.config import SystemConfig
from gsdeploy.quality import quality_digest


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def same_background(a,b):
    return a['deployment']==b['deployment'] and all(a['routing'][t]==b['routing'][t] for t in ['image','ocr','asr'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aggregate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verified-domain-plans',type=int,required=True)
    args=p.parse_args()
    top=json.loads((args.aggregate/'top-controls.json').read_text())
    rows=[json.loads(line) for line in (args.aggregate/'evaluations.jsonl').read_text().splitlines() if line.strip()]
    if len(rows)!=args.verified_domain_plans or len({r['identity'] for r in rows})!=len(rows):
        raise ValueError('Complete, unique aggregate identity set required')
    if any(len(r['runs'])!=2 or {v['seed'] for v in r['runs']}!={107,108} for r in rows):
        raise ValueError('Both development seed outcomes required at every identity')
    candidate=top['top10'][0]
    winner=candidate['plan']
    alternatives=[r for r in rows if r['diagnostic_feasible'] and r['diagnostic_objective'] is not None
                  and same_background(winner,r['plan']) and len(r['plan']['routing']['text'])==1]
    reference=max(alternatives,key=lambda r:r['diagnostic_objective']) if alternatives else None
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=dict(schema='gsdeploy.search_selected_development_pair_preparation.v1',
        candidate=candidate,matched_no_text_split_reference=reference,
        interpretation='Search-selected extrapolated development proposal; not a validated predictor, full-pool optimum or proven GPU gain.',
        seeds=[109,110,111,112],order=[['candidate','reference'],['reference','candidate'],
            ['reference','candidate'],['candidate','reference']],
        unit='One complete matched run pair; requests are not independent experimental repeats.',
        no_optional_seed_selection=True,independent_validation_seeds_81_82_untouched=True,
        final_confirmation_seeds_71_72_73_untouched=True,
        full_runtime_stability_rule='Unchanged frozen v5; all outcomes retained.',
        further_tuning='Any revision after these new development outcomes is a new proposal requiring a new frozen experiment.',
        gpu_started=False,packet_created=False,
        inputs=[static_study._file(args.aggregate/'top-controls.json'),static_study._file(args.aggregate/'evaluations.jsonl'),
            static_study._file(Path(__file__).resolve())])
    if len(winner['routing']['text'])==1:
        receipt['status']='SEARCH_DID_NOT_SELECT_TEXT_SPLIT_NO_SPILLOVER_EXPERIMENT_FORCED'
        write(args.output/'preparation-receipt.json',receipt);print(receipt['status']);return
    if reference is None:
        receipt['status']='NO_MATCHED_DIAGNOSTIC_FEASIBLE_REFERENCE'
        write(args.output/'preparation-receipt.json',receipt);print(receipt['status']);return
    basepath=ROOT/'artifacts/static-spillover-study-20261022/text185-fine-development-v1/experiment-packet.json'
    base=json.loads(basepath.read_text());raw=copy.deepcopy(base['config']);config=SystemConfig(**raw)
    checks={name:config.validate_plan(**row['plan']) for name,row in [('candidate',candidate),('reference',reference)]}
    receipt.update(unchanged_proxy_native_admission=checks,base_packet=static_study._file(basepath))
    if not all(v['constraints_valid'] for v in checks.values()):
        receipt['status']='SEARCH_PROPOSAL_NOT_ADMITTED_BY_UNCHANGED_NATIVE_PROXY_CONTRACT'
        write(args.output/'preparation-receipt.json',receipt);print(receipt['status']);return
    write(args.output/'predeclared-protocol.json',receipt)
    bindings=base['bindings']+[static_study._file(basepath),*receipt['inputs'],static_study._file(args.output/'predeclared-protocol.json')]
    bindings=sorted({b['path']:b for b in bindings}.values(),key=lambda b:b['path'])
    jobs=[]
    for seed,order in zip(receipt['seeds'],receipt['order']):
        for name in order:
            plan=(candidate if name=='candidate' else reference)['plan']
            job=static_study._job(config,base['workers'],plan,f'search_selected_{name}_seed{seed}',
                'calibration',seed,bindings,30)
            job.update(purpose='prequential_search_selected_development_pair',proxy_constraints=checks[name])
            jobs.append(job)
    packet=dict(schema='gsdeploy.static_experiments.v1',status='PREDECLARED_SEARCH_SELECTED_DEVELOPMENT_PAIRS',
        inputs=base['inputs'],config=raw,workers=base['workers'],bindings=bindings,
        frozen_files_sha256=quality_digest(bindings),jobs=jobs,gpu_started=False,
        discovery_protocol=str((args.output/'predeclared-protocol.json').resolve()),
        execution=dict(policy='All eight serial conditions; no hidden retry or outcome-dependent stopping.',
            runtime_source=str(SOURCE),scheduled=False))
    validation=static_study.validate_packet(packet)
    if not validation['valid']:raise ValueError(str(validation))
    write(args.output/'experiment-packet.json',packet)
    receipt.update(status='PREDECLARED_CPU_VALID_PACKET',packet_created=True,validation=validation,
        packet=static_study._file(args.output/'experiment-packet.json'))
    write(args.output/'preparation-receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],jobs=len(jobs),packet=receipt['packet'])))


if __name__=='__main__':main()
