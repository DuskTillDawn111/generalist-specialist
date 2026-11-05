"""Freeze fresh independent host-band tests after complete training replay."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.collect_activation_training import save
from experiments.fit_activation_native import load_bundle
from experiments.validate_activation_semantic_bands import score_band_fit
from gsdeploy.forecasting.native_training import digest
from gsdeploy.forecasting.native_validation import DEFAULT_CRITERIA


def verify(output):
    protocol=json.loads((output/'protocol.json').read_text())
    fit=json.loads((output/'fit-frozen.json').read_text())
    criteria=json.loads((output/'criteria-frozen.json').read_text())
    if (digest(fit)!=protocol['fit_sha256'] or digest(DEFAULT_CRITERIA)!=protocol['criteria_sha256']
            or criteria!=DEFAULT_CRITERIA):
        raise ValueError('Frozen independent model or criteria changed')
    for source in protocol['prediction_sources']:
        if any(hashlib.sha256(Path(source[key]).read_bytes()).hexdigest()!=source['sha256'] for key in ('path','snapshot')):
            raise ValueError('Frozen independent band source drift: '+source['path'])
    for name,expected in protocol['plan_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen independent band cohort plan changed')
    return protocol,fit


def prepare(args):
    fit=json.loads(args.fit.read_text());replay=json.loads(args.replay.read_text())
    if (not fit['training_fit_gates_passed'] or not replay.get('training_replay_accuracy_within_threshold')
            or replay.get('training_replay') is not True or replay['fit_sha256']!=digest(fit)
            or any(not group['accepted'] for group in fit['profiles'].get('native_order_law',{}).get('fit_diagnostics',[]))
            or len(set(args.seeds))!=2 or set(args.seeds)&{6110491,6110492}):
        raise ValueError('Complete accepted training replay and distinct unused nonconfirmation seeds required')
    for name,expected in replay['prediction_sources'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Source differs from completed band training replay')
    captures=[load_bundle(Path(source.rsplit(':sha256:',1)[0]).parent) for source in fit['source_receipts']]
    if [bundle['source'] for bundle in captures]!=fit['source_receipts']:
        raise ValueError('Original band training receipts changed')
    source=Path(fit['source_receipts'][0].rsplit(':sha256:',1)[0]).parent
    overlap=next(Path(bundle['source'].rsplit(':sha256:',1)[0]).parent for bundle in captures if bundle['plan']['mode']=='overlap' and len(bundle['plan']['jobs'])>10)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    save(output/'fit-frozen.json',fit);save(output/'criteria-frozen.json',DEFAULT_CRITERIA)
    runs=[]
    for index,seed in enumerate(args.seeds,1):
        directory=output/f'validation-round{index}'
        subprocess.run([sys.executable,str(ROOT/'experiments/prepare_activation_base_forward_training.py'),
            '--context-capture',str(source),'--cohort-source',str(overlap),'--output',str(directory),'--seed',str(seed)],check=True,cwd=ROOT)
        plan=json.loads((directory/'plan.json').read_text())
        if plan['execution_context_sha256']!=fit['profiles']['execution_context_sha256']:
            raise ValueError('Independent collector changed execution context')
        rng=random.Random(seed);by_alias={alias:[] for alias in plan['aliases']}
        for job in plan['jobs']:
            for row in job['requests']:by_alias[row['alias']].append(row)
        for rows in by_alias.values():
            inputs=[row['pool_id'] for row in rows];rng.shuffle(inputs)
            for row,pool_id in zip(rows,inputs):row['pool_id']=pool_id
        for job in plan['jobs']:
            offsets={alias:rng.choice((0.,30.,100.,200.)) for alias in plan['aliases']}
            for row in job['requests']:row['offset_ms']+=offsets[row['alias']]
        plan.update(split='development_validation',not_for_training=True,frozen_fit_sha256=digest(fit),
            criteria_sha256=digest(DEFAULT_CRITERIA),
            validation_design='fresh independent owner-marginal-preserving cross-input pairings and arrival alignments')
        save(directory/'plan.json',plan);runs.append(str(directory))
    files=set(Path(path) for path in replay['prediction_sources'])
    files.update(ROOT/path for path in ['experiments/qualify_activation_semantic_bands.py','experiments/prepare_activation_base_forward_training.py',
        'experiments/fit_activation_native.py','gsdeploy/paper_training.py','gsdeploy/paper_capture.py','gsdeploy/energy.py'])
    receipts=[]
    for index,path in enumerate(sorted(files)):
        content=path.read_bytes();snapshot=output/'prediction-source-snapshot'/str(index)/path.name
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
        receipts.append(dict(path=str(path),snapshot=str(snapshot),sha256=hashlib.sha256(content).hexdigest()))
    save(output/'protocol.json',dict(schema='gsdeploy.independent_band_validation_protocol.v1',runs=runs,
        fit_sha256=digest(fit),criteria_sha256=digest(DEFAULT_CRITERIA),prediction_sources=receipts,
        plan_sha256={str(Path(directory)/'plan.json'):hashlib.sha256((Path(directory)/'plan.json').read_bytes()).hexdigest() for directory in runs},
        source_replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),confirmation_seeds_reserved=[6110491,6110492],formal_utility_qualified=False))


def run(output):
    protocol,fit=verify(output);state_path=output/'continuation.json'
    if state_path.exists():raise FileExistsError(state_path)
    state=dict(status='collecting',completed=[],formal_utility_qualified=False);save(state_path,state)
    for directory in map(Path,protocol['runs']):
        verify(output);state['current']=str(directory);save(state_path,state)
        with (directory/'collector.log').open('x') as log:
            result=subprocess.run([sys.executable,str(ROOT/'experiments/run_activation_capture.py'),'--output',str(directory)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='independent_capture_failed');save(state_path,state);return 1
        state['completed'].append(str(directory));save(state_path,state)
    verify(output);state['status']='scoring';save(state_path,state)
    bundles=[load_bundle(Path(directory)) for directory in protocol['runs']]
    for bundle in bundles:
        if bundle['plan'].get('frozen_fit_sha256')!=protocol['fit_sha256'] or bundle['plan'].get('criteria_sha256')!=protocol['criteria_sha256']:
            raise ValueError('Independent capture not committed to the frozen predictor/criteria')
    report=score_band_fit(fit,bundles,training_replay=False)
    # The nested path diagnostic records the preceding pooled gate, before
    # its additional per-path reason; the top-level verdict includes both.
    report['additional_path_gate']['original_overall_gate_passed']=not any(
        reason!='PER_PATH_FIFO_E2E_ERROR_OR_UNKNOWN' for reason in report['reasons'])
    verify(output)
    with (output/'validation.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    state.update(status='independent_development_accuracy_passed' if report['development_accuracy_passed'] else 'independent_validation_failed',
                 current=None,latency_errors=report['latency_errors'],reasons=report['reasons'])
    save(state_path,state);return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('op',choices=['prepare','run'])
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--fit',type=Path);parser.add_argument('--replay',type=Path)
    parser.add_argument('--seeds',nargs=2,type=int,default=[6111071,6111072]);args=parser.parse_args()
    if args.op=='prepare':
        if args.fit is None or args.replay is None:parser.error('prepare requires fit and replay')
        prepare(args);return 0
    return run(args.output.resolve())


if __name__=='__main__':raise SystemExit(main())
