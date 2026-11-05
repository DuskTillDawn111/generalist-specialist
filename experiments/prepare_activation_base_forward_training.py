"""Prepare fresh training under the new immutable PEFT base-forward observer."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import digest, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context-capture',required=True,type=Path)
    parser.add_argument('--cohort-source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int)
    args=parser.parse_args()
    source=args.context_capture.resolve();cohort_path=args.cohort_source.resolve()/'plan.json'
    context=json.loads((source/'context.json').read_text());pilot=json.loads((source/'plan.json').read_text())
    cohort=json.loads(cohort_path.read_text())
    if (context.get('observer_diagnostic')!='specialists_peft_base_semantic'
            or pilot['execution_context_sha256']!=digest(context) or cohort['split']!='training'
            or cohort['mode'] not in ('solo','overlap') or context['resident_aliases']!=cohort['aliases']
            or args.seed in (6110491,6110492)):
        raise ValueError('Exact base-forward observer and full original training cohort required')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    receipts=[]
    for index,receipt in enumerate(json.loads((source/'sources.json').read_text())):
        content=Path(receipt['snapshot']).read_bytes()
        if (hashlib.sha256(content).hexdigest()!=receipt['sha256']
                or hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest()!=receipt['sha256']):
            raise ValueError('Frozen base-forward executable source changed')
        snapshot=output/'source-snapshot'/str(index)/Path(receipt['path']).name
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
        receipts.append(dict(path=receipt['path'],snapshot=str(snapshot),sha256=receipt['sha256']))
    jobs=copy.deepcopy(cohort['jobs']);random.Random(args.seed).shuffle(jobs)
    plan=dict(schema='gsdeploy.native_activity_training_plan.v1',mode=cohort['mode'],split='training',
        execution_context_sha256=digest(context),run_id=output.name,seed=args.seed,
        aliases=context['resident_aliases'],jobs=jobs,budget_seconds=1800,priming=cohort['priming'],
        confirmation_seeds_reserved=[6110491,6110492],pure_gpu_stage_claim=False,
        independent_run_requirement='fresh separately launched observer training rounds',
        resident_domain=dict(text=1,image=1,ocr=1,asr=1),observer_gate_required=True,
        not_a_fixed_future_decode_template=True,observer_context_source=str(source/'context.json'),
        observer_context_source_sha256=hashlib.sha256((source/'context.json').read_bytes()).hexdigest(),
        cohort_source=str(cohort_path),cohort_source_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(output/'context.json',context);save(output/'sources.json',receipts);save(output/'plan.json',plan)
    (output/'cohort-preparation-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(output=str(output),context=digest(context),jobs=len(jobs),
        requests=sum(len(job['requests']) for job in jobs))))


if __name__=='__main__':main()
