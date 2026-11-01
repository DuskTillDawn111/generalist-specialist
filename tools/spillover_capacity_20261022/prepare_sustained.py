"""Freeze four sustained-joint development diagnostics without starting a GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1'
BASE = ROOT / 'artifacts/static-spillover-study-20261022/text-sustained-joint-diagnostic-v1'


def runner_module():
    path = Path(__file__).with_name('run_sustained.py')
    spec = importlib.util.spec_from_file_location('sustained_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def balanced_order(text_ids, seed, namespace, repetitions):
    result = []
    for repetition in range(repetitions):
        items = list(text_ids)
        random.Random(f'{seed}:{namespace}:{repetition}').shuffle(items)
        result.extend(items)
    return result


def build_jobs(text_ids):
    jobs = []
    active = {'B':['generalist_2#0'], 'C':['text#0','text#1','generalist_2#0']}
    runner = runner_module()
    for block, seed, states in [(1,103,'BC'), (2,104,'CB')]:
        ids = balanced_order(text_ids, seed, 'sustained-primary', 3)
        for state in states:
            primary, support = {}, {}
            for alias in active[state]:
                primary[alias] = [dict(request_id=f'measurement:{alias}:{i}', task='text', model=alias,
                    role='primary', pool_id=pid, repetition=i//32, sequence=i, arrival_index=i)
                    for i,pid in enumerate(ids)]
                support_ids = balanced_order(text_ids, seed, f'sustained-support:{alias}', 32)
                support[alias] = [dict(request_id=f'support:{alias}:{i}', task='text', model=alias,
                    role='support', pool_id=pid, repetition=i//32, sequence=96+i, arrival_index=96+i,
                    support_sequence=i) for i,pid in enumerate(support_ids)]
            job = dict(job_id=f'block{block}_{state}_seed{seed}', state=state, block=block, seed=seed,
                active_aliases=active[state], input_pool_ids=ids,
                planned_requests=primary, support_requests=support)
            job['job_identity_sha256'] = runner.job_digest(job)
            jobs.append(job)
    return jobs


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    runner = runner_module()
    original_path = OLD / 'experiment-packet.json'
    packet = copy.deepcopy(json.loads(original_path.read_text()))
    # Reuse immutable native identities, including full checkpoint receipts, from
    # the audited first diagnostic. Every original binding is reverified here.
    for path, expected in packet['bindings'].items():
        if runner.digest(path) != expected: raise ValueError('Original binding changed: '+path)
    rows = [json.loads(line) for line in Path(packet['samples']).read_text().splitlines() if line.strip()]
    text_ids = [r['pool_id'] for r in rows if r['task_type']=='text']
    if len(text_ids)!=32 or len(set(text_ids))!=32: raise ValueError('Expected frozen32 text pool')
    coverage = dict(minimum_common_handler_fraction=0.9, minimum_covered_repetitions_per_input=2,
        all_primary_positive_overlap=True, cross_block_per_input_median_ratio_min=0.5,
        cross_block_per_input_median_ratio_max=2.0)
    jobs = build_jobs(text_ids)
    declaration = dict(schema='gsdeploy.sustained_joint_declaration.v1',
        scope='Four development diagnostics, no formal U/capacity/predictor validation or optimum claim',
        seed_usage_check='Existing development protocol/job seed fields checked before new preparation; 103/104 unused',
        seeds=dict(development=[103,104], independent_validation_reserved=[81,82], final_confirmation_reserved=[71,72,73]),
        job_order=['B','C','C','B'],
        primary='96 per active alias, same32 inputs x3, same block order across B/C and aliases',
        support='Independent predeclared balanced cursor per alias; symmetric after own primary completion, at most1024 per alias',
        support_admission_stop='Observe all primary terminal successes before admitting any next support; retain bounded in-flight drain',
        failure_policy='Stop queue on first runtime/resource/transport failure, wall/request timeout or depleted support before all primary complete. No retry or roster extension.',
        deadline='600s measurement hard wall includes primary and support drain; each request300s from admission. Frozen shutdown uses10s shared graceful deadline plus bounded per-process termination/kill and thread joins.',
        coverage_policy=coverage,
        coverage_scope='C all three aliases and32 inputs: >=2 of3 primary fractions>=0.90; every primary fraction>0; based on intersection of actual native handler intervals, not in-flight status.',
        reproducibility_scope='For each C alias/input with >=2 qualified primary, block2/block1 joint-qualified service median ratio in[0.5,2.0]. Separate descriptive drift/output/quality distributions, no causal proof or capacity transfer.',
        support_statistics='All support raw/native timestamps/outputs/errors/resources retained, separate quality and counts; never primary samples, independent repetitions, or utility denominator.',
        unchanged=['seven-model catalog','replicas','quality in equal-weight Q/T/L/P/M/E','rho<=0.8 formal rule','49140MiB and safety factor','formal stability/runtime/SWRR'],
        inherited_identity_packet=dict(path=str(original_path),sha256=runner.digest(original_path)),
        unresolved_after_this_batch=['full-input formal background service/joint activity','open-loop response','new predictor with independent verification','same-domain MILP/DFS and full-pool evidence'],
        no_auto_retry=True, no_following_batch_auto_enqueue=True, U=None)
    BASE.mkdir(parents=True, exist_ok=False)
    save(BASE/'discovery-protocol.json',declaration)
    packet.update(protocol=runner.PROTOCOL, jobs=jobs, support_limit_per_alias=1024,
                  coverage_policy=coverage, gpu_started=False)
    bound = [original_path, Path(__file__).resolve(), Path(__file__).with_name('run_sustained.py').resolve(),
             BASE/'discovery-protocol.json', ROOT/'docs/text_sustained_joint_diagnostic_design_20261022.md']
    for path in bound: packet['bindings'][str(path)] = runner.digest(path)
    packet['bindings_sha256'] = __import__('hashlib').sha256(runner.canonical(packet['bindings'])).hexdigest()
    runner.validate_packet(packet)
    save(BASE/'experiment-packet.json',packet)
    save(BASE/'preparation-receipt.json',dict(packet_sha256=runner.digest(BASE/'experiment-packet.json'),
        jobs=4,planned_primary_requests=768,max_support_per_alias=1024,
        primary_and_support_independently_namespaced=True,
        native_identities_reused_without_modification=True,
        original_packet_sha256=runner.digest(original_path),bound_files=len(packet['bindings']), GPU_started=False))
    print(json.dumps(dict(packet=str(BASE/'experiment-packet.json'),jobs=4,planned_primary_requests=768,GPU_started=False)))


if __name__=='__main__': main()
