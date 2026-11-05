"""New deterministic execution contexts, retaining the original observer gates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import save, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--semantic', action='store_true')
    args = p.parse_args()
    command = [sys.executable, str(ROOT/'experiments/prepare_activation_semantic.py'),
        '--reference', str(args.reference), '--output', str(args.output), '--seed', str(args.seed)]
    if args.semantic:
        command.append('--semantic')
    subprocess.run(command, check=True)
    output = args.output.resolve()
    context = json.loads((output/'context.json').read_text())
    plan = json.loads((output/'plan.json').read_text())
    sources = json.loads((output/'sources.json').read_text())
    worker = ROOT/'tools/predictor_validation_20261027/deterministic_observed_worker.py'
    mode = 'semantic' if args.semantic else 'complete'
    for binding in context['worker_bindings'].values():
        binding['command'][1] = str(worker)
        binding['env'].update(CUBLAS_WORKSPACE_CONFIG=':4096:8', GS_DETERMINISTIC_OBSERVER=mode)
    extra = [worker, Path(__file__).resolve(),
             ROOT/'tools/predictor_validation_20261027/semantic_observed_worker.py']
    existing = {s['path'] for s in sources}
    for path in extra:
        if str(path) in existing:
            continue
        content = path.read_bytes()
        snapshot = output/'deterministic-source-snapshot'/path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
        sources.append(dict(path=str(path), snapshot=str(snapshot), sha256=hashlib.sha256(content).hexdigest()))
    sources.sort(key=lambda s: s['path'])
    context.update(source_content=[{k: s[k] for k in ['path', 'sha256']} for s in sources],
        deterministic_runtime=dict(strict_algorithms=True, warn_only=False, cudnn_benchmark=False,
            cudnn_deterministic=True, CUBLAS_WORKSPACE_CONFIG=':4096:8',
            tf32_policy='unchanged original model profile setting'),
        observer_diagnostic='deterministic_'+mode)
    plan.update(execution_context_sha256=digest(context), observer_diagnostic='deterministic_'+mode,
        deterministic_runtime=context['deterministic_runtime'],
        historical_observer_failure_not_overridden=True, native_eligibility=False,
        changed_runtime_requires_new_templates_and_episode_references=True)
    save(output/'context.json', context)
    save(output/'sources.json', sources)
    save(output/'plan.json', plan)
    print(json.dumps(dict(output=str(output), observer=mode, context=plan['execution_context_sha256'],
                          matched_job_content_sha256=plan['matched_job_content_sha256'])))


if __name__ == '__main__':
    main()
