"""CPU-only extension of the frozen stage1 packet with two deployment controls.

The search selects the controls; this file freezes their own resident-context
service tables. It changes neither native implementation nor the first packet.
"""
from __future__ import annotations
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.queue_refined_20261023 import admission,protocol,search
from tools.queue_refined_20261023.native_runner import frozen_runtime


def prepare(output,stage1_packet,top_controls):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    original=json.loads(Path(stage1_packet).read_text())
    if not protocol.validate_packet(original)['valid']:raise ValueError('Stage1 packet changed')
    controls=json.loads(Path(top_controls).read_text())['controls']
    selected=[('pureS-text2-image-ocr-asr',controls['pure_specialist']),
              ('mixed-text2-g2-ocr-image',controls['different_deployment'])]
    expected=[{'text':2,'image':1,'ocr':1,'asr':1},{'text':2,'generalist_2':1,'ocr':1,'image':1}]
    oracle=search.DiagnosticOracle([107,108]);contexts=[]
    for (label,row),deployment in zip(selected,expected):
        c=row['candidate']
        if c['deployment']!=deployment or c['kind']!='causal_queue' or row['candidate_id']!=search.identity(c):
            raise ValueError('Search shortlist exact candidate mismatch')
        if row['status']!='COMPLETE' or row['objective'] is None:raise ValueError('Control is not diagnostic feasible')
        if label.startswith('pureS') and c['strategy']!='queue_specialist_only':raise ValueError('Wrong pureS control')
        services,loss,context=oracle.context(c)
        contexts.append(dict(label=label,candidate_id=row['candidate_id'],candidate=c,
            diagnostic_objective=row['objective'],service_ms=services,aggregate_quality_loss=loss,
            resident_reference_context=context,formal_U=None))
    oracle.o.verify_inputs()
    # All imported repository sources, including isolated frozen primitives,
    # are bound; Python/numpy/scipy entry points are also recorded below.
    dependencies={str(Path(__file__).resolve()):protocol.receipt(__file__)}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if not name:continue
        path=Path(name).resolve()
        if path.is_file() and path.is_relative_to(ROOT):dependencies[str(path)]=protocol.receipt(path)
    generation_inputs={p:dict(path=p,sha256=sha) for p,sha in oracle.o.provenance()['inputs'].items()}
    for path in [stage1_packet,top_controls]:generation_inputs[str(Path(path).resolve())]=protocol.receipt(path)
    import numpy,scipy
    context_contract=dict(schema='gsdeploy.causal_native_joint_shortlist_context.v1',
        contexts=contexts,oracle_provenance=oracle.o.provenance(),
        generation_dependencies=dependencies,generation_inputs=generation_inputs,
        software=dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__),
        claim='Frozen development estimates for each exact resident reference context; not native optimality or validated service predictions.',
        GPU_started=False)
    output.mkdir(parents=True,exist_ok=False)
    context_path=output/'frozen-joint-context-contract.json';protocol.write(context_path,context_contract)
    context_receipt=protocol.receipt(context_path)
    sources=copy.deepcopy(original['sources']);sources.update(dependencies)
    sources.update(generation_inputs);sources[context_receipt['path']]=context_receipt
    config=copy.deepcopy(oracle.o.config)
    config['catalog']['performance_mode']='task_replica_proxy';config['catalog']['memory_mode']='standalone_proxy'
    configuration=importlib.import_module(frozen_runtime().__package__.split('.runtime')[0]+'.config')
    old={(j['seed'],j['strategy']):j for j in original['jobs']}
    jobs=[];checks=[];excluded=[]
    guard=copy.deepcopy(original['jobs'][0]['policy_contract']['guard'])
    for seed in [113,114]:
        seedjobs=[]
        for strategy in protocol.STRATEGIES:
            job=copy.deepcopy(old[(seed,strategy)])
            job['label']='mixed-text2-g2-ocr-'+job['label'];job['phase']='development';seedjobs.append(job)
        for context in contexts:
            c=context['candidate']
            contract=protocol.make_policy_contract(c['deployment'],context['service_ms'],context['aggregate_quality_loss'],
                config['weights'],config['latency_reference_ms'],strategy=c['strategy'],models=config['catalog']['models'],
                text_rate=config['workload']['text'],background_routing={t:r for t,r in c['reference_routing'].items() if t!='text'},
                overdue_remaining_ms=c['overdue_remaining_ms'],guard=guard if 'generalist_2' in c['deployment'] else None,
                sources={p:r['sha256'] for p,r in sources.items()})
            identity=dict(schema='gsdeploy.causal_native_joint_shortlist_job_identity.v1',strategy=c['strategy'],seed=seed,
                candidate_id=context['candidate_id'],policy_contract_sha256=protocol.digest(contract),
                base_packet=original['base_packet'],joint_context_contract=context_receipt,
                horizon_seconds=600,warmup_seconds=30,arrival_process='poisson',deployment=c['deployment'],
                measurement_claim='NEW_DEVELOPMENT_NATIVE_OBSERVATION_NOT_INDEPENDENT_CONFIRMATION')
            job=dict(id=protocol.digest(identity),label=f"{context['label']}-{c['strategy']}-seed{seed}",
                strategy=c['strategy'],seed=seed,phase='development',identity=identity,policy_contract=contract,
                nominal_plan=protocol.nominal_plan(contract))
            config['random_seed']=seed;config['horizon_seconds']=600
            envelope=admission.capacity_envelope(configuration.SystemConfig(**config),job['nominal_plan'],contract)
            source_ok,issues=admission.source_checks(contract)
            check=dict(label=job['label'],job_id=job['id'],capacity=envelope,sources_verified=source_ok,source_issues=issues)
            checks.append(check)
            if not envelope['constraints_verified'] or not source_ok:excluded.append(check);continue
            job['capacity_evidence']=envelope;seedjobs.append(job)
        jobs.extend(seedjobs if seed==113 else reversed(seedjobs))
    inputs=copy.deepcopy(original['inputs'])+[protocol.receipt(stage1_packet),protocol.receipt(top_controls),context_receipt]
    inputs.extend(generation_inputs.values())
    inputs=list({r['path']:r for r in inputs}.values())
    packet=dict(schema='gsdeploy.causal_native_packet.v1',status='PREPARED_NOT_EXECUTED',
        base_packet=original['base_packet'],simulation_contract=original['simulation_contract'],
        stage1_packet=protocol.receipt(stage1_packet),joint_context_contract=context_receipt,
        inputs=inputs,sources=sources,jobs=jobs,seeds=[113,114],excluded_controls=excluded,
        shortlist_scope='Five original routing mechanisms plus strongest diagnostic pureS and strongest different resident deployment; not full pool native optimality.',
        worker_binding=original['worker_binding'],no_retries=True,GPU_started=False)
    protocol.write(output/'experiment-packet.json',packet)
    report=dict(schema='gsdeploy.causal_native_joint_shortlist_cpu_check.v1',
        packet=protocol.receipt(output/'experiment-packet.json'),packet_check=protocol.validate_packet(packet),
        added_jobs=checks,excluded_controls=excluded,
        original_jobs_cpu_check=protocol.receipt(Path(stage1_packet).parent/'cpu-check.json'),GPU_started=False)
    protocol.write(output/'cpu-check.json',report)
    return packet,report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage1-packet',type=Path,required=True);p.add_argument('--top-controls',type=Path,required=True);a=p.parse_args()
    packet,report=prepare(a.output,a.stage1_packet,a.top_controls)
    print(json.dumps(dict(packet=protocol.receipt(a.output/'experiment-packet.json'),jobs=len(packet['jobs']),
        excluded=len(packet['excluded_controls']),check=report['packet_check'],GPU_started=False)))
