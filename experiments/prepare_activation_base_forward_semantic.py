"""New matched specialist pilots for the explicitly observed PEFT base LM."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import digest, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--semantic',action='store_true')
    args=parser.parse_args()
    command=[sys.executable,str(ROOT/'experiments/prepare_activation_specialist_semantic.py'),
        '--reference',str(args.reference.resolve()),'--output',str(args.output.resolve()),'--seed',str(args.seed)]
    if args.semantic:command.append('--semantic')
    subprocess.run(command,check=True,cwd=ROOT)
    output=args.output.resolve()
    plan=json.loads((output/'plan.json').read_text());context=json.loads((output/'context.json').read_text())
    sources=json.loads((output/'sources.json').read_text())
    observer=ROOT/'tools/predictor_validation_20261027/base_forward_observed_worker.py'
    for source in [Path(__file__).resolve(),observer]:
        content=source.read_bytes();snapshot=output/'base-forward-source-snapshot'/source.name
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
        sources.append(dict(path=str(source),snapshot=str(snapshot),sha256=hashlib.sha256(content).hexdigest()))
    if args.semantic:
        for binding in context['worker_bindings'].values():binding['command'][1]=str(observer)
    sources.sort(key=lambda row:row['path'])
    context.update(source_content=[{key:row[key] for key in ('path','sha256')} for row in sources],
        observer_diagnostic='specialists_peft_base_semantic' if args.semantic else 'specialists_peft_base_control')
    plan.update(execution_context_sha256=digest(context),observer_diagnostic=context['observer_diagnostic'])
    save(output/'context.json',context);save(output/'sources.json',sources);save(output/'plan.json',plan)
    print(json.dumps(dict(output=str(output),context=plan['execution_context_sha256'],not_for_training=True)))


if __name__=='__main__':main()
