"""Source-bound multi-domain model composition without rewriting component IDs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from gsdeploy.resources import identity
from .native_collect import receipt,context_identity
from .kernel_sequence_audit_v2 import verify
from .kernel_domain_bridge_review_v3 import load_report

SCHEMA='gsdeploy.composite_training_execution_contract.v2'
MEASUREMENT_FIELDS={'profiler','measurement_domain','worker_variant','affine_observation'}
PACKET_FIELDS={'sources','execution_sources','training_only','collection_budget','inherited_runtime_smoke'}


def physical_projection(context):
    """Strict equality for every nonmeasurement/nonpacket field and source hash.

    Whitelist is explicit. Worker backend, allocation settings, quantization,
    batch, shared-weight support, residency, CPU threads and device all remain.
    No claim of timing equivalence across instrumentation domains is implied.
    """
    result={k:copy.deepcopy(v) for k,v in context.items() if k not in MEASUREMENT_FIELDS|PACKET_FIELDS}
    result['source_content']=[dict(original_path=s['original_path'],sha256=s['frozen']['sha256'])
                             for s in context['execution_sources']]
    return result


def verify_target_runtime(contract,target_context_path):
    """Check exact collector identity plus actual worker/source bytes, CPU only."""
    target_context_path=Path(target_context_path).resolve()
    if receipt(target_context_path)!=contract['target']['context']:
        raise ValueError('Collector context receipt differs from composite target')
    context=json.loads(target_context_path.read_text())
    if (context_identity(context)!=contract['target']['execution_context_sha256'] or
            identity(physical_projection(context))!=contract['physical_projection_sha256']):
        raise ValueError('Collector target execution/physical projection mismatch')
    for source in context['execution_sources']:
        verify(source['frozen'])
        if receipt(source['original_path'])['sha256']!=source['frozen']['sha256']:
            raise ValueError('Actual runtime source differs from target frozen worker/source bytes')
    return dict(known=True,target_context=receipt(target_context_path),
                execution_context_sha256=context_identity(context),actual_source_bytes_verified=True,
                GPU_started=False,model_accuracy_validated=False)


def compose(ordered_path,residual_path,bridge_path,lifecycle_fit_report,target_context_path):
    paths=[Path(p).resolve() for p in (ordered_path,residual_path,bridge_path,lifecycle_fit_report,target_context_path)]
    ordered,residual,bridge,lifecycle,target=[json.loads(p.read_text()) for p in paths]
    if (bridge.get('status')!='PASS' or bridge.get('training_only') is not True
            or lifecycle.get('schema')!='gsdeploy.natural_training_lifecycle_fit.v2' or lifecycle.get('status')!='FITTED'):
        raise ValueError('Passing exact-work bridge and fitted natural lifecycle report required')
    verify(bridge['evaluator'])
    for source in lifecycle['sources']:verify(source)
    for name in ('model','observations','training_index','clock_evidence'):verify(lifecycle[name])
    domains={};projections={};sources=[receipt(p) for p in paths]
    for domain,key in [('light-kernel','light_review'),('metadata','metadata_review'),('natural','natural_review')]:
        verify(bridge[key]);report,plan,raw=load_report(bridge[key]['path'],domain)
        context=json.loads(Path(report['context']['path']).read_text())
        if context_identity(context)!=plan['execution_context_sha256']:
            raise ValueError('Original domain context hash does not match its frozen declaration')
        for execution_source in context['execution_sources']:verify(execution_source['frozen'])
        domains[domain]=dict(execution_context_sha256=plan['execution_context_sha256'],
                            context=report['context'],review=bridge[key])
        projections[domain]=physical_projection(context);sources.append(report['context'])
    if projections['light-kernel']!=projections['metadata'] or projections['light-kernel']!=projections['natural']:
        raise ValueError('Nonmeasurement physical worker/settings/source projections differ')
    light_id=domains['light-kernel']['execution_context_sha256'];natural_id=domains['natural']['execution_context_sha256']
    if (ordered['execution_context_sha256']!=light_id or residual['execution_context_sha256']!=light_id
            or residual['natural_execution_context_sha256']!=natural_id or lifecycle['execution_context_sha256']!=natural_id
            or receipt(paths[0]) not in residual['sources'] or receipt(paths[2]) not in residual['sources']):
        raise ValueError('Component models are not bound to these exact training domains/artifacts')
    captures=[]
    for capture in lifecycle['captures']:
        verify(capture['review']);report,plan,raw=load_report(capture['review']['path'],'natural')
        context=json.loads(Path(report['context']['path']).read_text())
        if context_identity(context)!=natural_id or physical_projection(context)!=projections['natural']:
            raise ValueError('Lifecycle capture context is not physically equivalent to exact bridge')
        if report['raw']!=capture['raw']:raise ValueError('Lifecycle capture raw source differs')
        captures.append(capture['review'])
    if target.get('measurement_domain')!='natural' or physical_projection(target)!=projections['natural']:
        raise ValueError('Target collector context must preserve natural physical worker/settings/sources')
    for execution_source in target['execution_sources']:verify(execution_source['frozen'])
    contract=dict(schema=SCHEMA,domains=domains,
        components=dict(service=dict(ordered=receipt(paths[0]),residual=receipt(paths[1]),original_context_sha256=light_id),
                        lifecycle=dict(model=lifecycle['model'],fit_report=receipt(paths[3]),original_context_sha256=natural_id)),
        target=dict(context=receipt(paths[4]),execution_context_sha256=context_identity(target),measurement_domain='natural'),
        physical_projection_sha256=identity(projections['natural']),
        ignored_measurement_fields=sorted(MEASUREMENT_FIELDS),ignored_packet_fields=sorted(PACKET_FIELDS),
        natural_lifecycle_capture_reviews=captures,scenario_adapter=dict(profiled='mean'),
        sources=sources+[receipt(__file__)],source_contexts_rewritten=False,source_models_rewritten=False,
        natural_timing_transfer='EXPLICIT_UNVALIDATED_MODEL_HYPOTHESIS',
        unseen_input_support=False,physically_validated=False,deployment_qualified=False)
    contract['composite_execution_sha256']=identity(contract)
    return contract


class CompositeComponentAdapter:
    """Forward calls while preserving and checking each original component ID."""
    def __init__(self,component,contract,kind,component_artifact):
        if (contract.get('schema')!=SCHEMA or
                identity({k:v for k,v in contract.items() if k!='composite_execution_sha256'})!=contract.get('composite_execution_sha256')):
            raise ValueError('Unchanged source-bound composite contract required')
        binding=contract['components'][kind]
        if component.execution_context_sha256!=binding['original_context_sha256']:
            raise ValueError('Component original execution identity differs from composition contract')
        expected=binding['residual'] if kind=='service' else binding['model']
        if receipt(component_artifact)!=expected:
            raise ValueError('Component artifact differs from source-bound composition')
        verify(contract['target']['context'])
        self.component=component;self.kind=kind;self.contract=contract
        self.execution_context_sha256=contract['composite_execution_sha256']
        self.original_execution_context_sha256=component.execution_context_sha256
        self.physical_evidence=False;self.physically_validated=False;self.deployment_qualified=False

    def __getattr__(self,name):return getattr(self.component,name)

    def delays(self,row,aliases,scenario):
        mapped=self.contract['scenario_adapter'].get(scenario,scenario)
        result=self.component.delays(row,aliases,mapped)
        return dict(result,composite_execution_sha256=self.execution_context_sha256,
                    original_execution_context_sha256=self.original_execution_context_sha256,
                    scenario_adapter=dict(requested=scenario,component=mapped),
                    natural_timing_validated=False)

    def residual_handoff(self,row,aliases,scenario,*args,**kwargs):
        mapped=self.contract['scenario_adapter'].get(scenario,scenario)
        return self.component.residual_handoff(row,aliases,mapped,*args,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ordered','residual','bridge','lifecycle-fit-report','target-context'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=compose(args.ordered,args.residual,args.bridge,args.lifecycle_fit_report,args.target_context)
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'contract.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(composite=result['composite_execution_sha256'],target=result['target'])))


if __name__=='__main__':main()
