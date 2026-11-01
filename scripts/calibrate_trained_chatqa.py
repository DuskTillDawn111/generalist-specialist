"""Fresh calibration of the derived QA adapter; never inherits base-model scores."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.profiling.prepare import digest
from gsdeploy.profiling.runner import save


def prepare_and_run(training, output, execute):
    from gsdeploy.backends.compact import CompactAdapter
    from gsdeploy.backends.compact_adapters import verified_qa_adapter
    from gsdeploy.profiling.diagnostics import prepare, run
    from gsdeploy.profiling.admission import quality_gate
    training, output = Path(training).resolve(), Path(output).resolve()
    state_path = training / 'state.json'
    state = json.loads(state_path.read_text())
    if state['status'] != 'TRAINING_COMPLETE_NOT_ADMITTED':
        raise ValueError('The adapter training run must finish successfully before calibration')
    adapter_path = Path(state['adapter_manifest'])
    if digest(adapter_path) != state['adapter_manifest_sha256']:
        raise ValueError('Changed trained adapter identity')
    manifest = json.loads(adapter_path.read_text())
    recipe_path = Path(manifest['recipe'])
    if digest(recipe_path) != manifest['recipe_sha256']:
        raise ValueError('Changed training recipe')
    recipe = json.loads(recipe_path.read_text())
    spec = json.loads(Path(recipe['base_worker']).read_text())
    candidate = copy.deepcopy(spec['candidate'])
    candidate.update(adapter_sha256=state['adapter_manifest_sha256'], state='TRAINED_REPLACEMENT_NOT_ADMITTED',
                     precision='NF4 language/BF16 + train-only QA LoRA SHA256 ' + state['adapter_manifest_sha256'])
    execution = copy.deepcopy(spec['execution'])
    execution['peft_adapter'] = {'manifest': str(adapter_path), 'sha256': state['adapter_manifest_sha256']}
    verified_qa_adapter(execution, candidate)
    policy = json.loads(Path(spec['resources']['protocol']).read_text())
    policy['diagnostic_only'] = True
    adapter = CompactAdapter(policy, json.loads(Path(spec['resources']['classmap']).read_text()))
    study_path = ROOT / 'configs/study_repaired.json'
    study = json.loads(study_path.read_text())
    calibration_path = ROOT / study['calibration_preparation']['manifest']
    if digest(calibration_path) != study['calibration_preparation']['manifest_sha256']:
        raise ValueError('Changed QA calibration manifest')
    entry = json.loads(calibration_path.read_text())['tasks']['text']
    if digest(entry['path']) != entry['sha256']:
        raise ValueError('Changed QA calibration samples')
    samples = [json.loads(line) for line in Path(entry['path']).read_text().splitlines()]
    if any(adapter.build(candidate,sample).get('adapter_sha256') != state['adapter_manifest_sha256'] for sample in samples):
        raise ValueError('Derived adapter identity was lost in the model requests')
    output.mkdir(parents=True, exist_ok=False)
    spec.update(candidate=candidate, execution=execution)
    worker = output / 'template-worker.json'
    save(worker,spec)
    base = output / 'template-manifest.json'
    save(base,{'study':str(study_path),'workers':{'text':{'spec':str(worker),
               'python':'/root/autodl-tmp/gsdeploy-assets/envs/compact-standard/bin/python'}}})
    case = {'id':'qa-trained-native-calibration','parent':None,'factor':None,'candidate':candidate,
            'execution':execution,'policy':policy,'samples':samples}
    packet = output / 'experiments'
    prepare([case],base,packet,extra_sources=[__file__,state_path,adapter_path,recipe_path,
            calibration_path,entry['path'],study_path,Path(spec['resources']['protocol'])])
    receipt = {'status':'CPU_PREPARED_GPU_DISABLED','pool_admitted':False,'gpu_started':False,
               'adapter_sha256':state['adapter_manifest_sha256'],'samples':len(samples)}
    save(output/'state.json',receipt)
    if execute:
        result = run(packet/'experiment-manifest.json',output/'gpu-results')
        if result['status'] != 'COMPLETE_DIAGNOSTICS_NOT_ADMITTED':
            receipt.update(status='CALIBRATION_COLLECTION_FAILED',gpu_started=None)
        else:
            observed = result['results']['qa-trained-native-calibration']
            gate = quality_gate(policy,'text',observed['quality_summary'])
            receipt.update(status='CALIBRATION_PASSED_NOT_ADMITTED' if gate['passed'] else 'CALIBRATION_REJECTED',
                           gpu_started=True,quality_summary=observed['quality_summary'],gate=gate)
        save(output/'state.json',receipt)
        if receipt['status'] == 'CALIBRATION_COLLECTION_FAILED':
            raise RuntimeError('Derived QA calibration failed to collect the complete cohort')
    print(json.dumps(receipt))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    prepare_and_run(args.training,args.output,args.execute)
