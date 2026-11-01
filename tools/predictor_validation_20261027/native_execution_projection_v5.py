"""Auditable native-execution projection hypothesis across collector wrappers.

Preserves original complete context identities. Matching this deliberately
narrower projection is not equality of full contexts, measured equivalence of
collectors, or natural/profiler timing transfer.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .kernel_baseline_intake_v3 import checked,_verify
from .native_collect import context_identity,receipt,save
from .profiling import digest


FIELDS=('shared_native_execution_sources','worker_bindings','residents','threads','execution',
    'device','measurement_domain','profiler','affine_observation','worker_variant','large_weight_identity_scope')


def native_execution_projection(context,verify_sources=True):
    missing=[key for key in FIELDS+('large_weight_stats',) if key not in context]
    if missing:raise ValueError('Missing explicit native projection fields: '+','.join(missing))
    if context['measurement_domain'] not in ('natural','metadata','light-kernel'):
        raise ValueError('Explicit observed measurement domain required')
    sources=context['shared_native_execution_sources']
    if not sources or len({s['path'] for s in sources})!=len(sources):raise ValueError('Unique explicit native source receipts required')
    if verify_sources:
        for source in sources:_verify(source)
        for binding in context['worker_bindings'].values():
            _verify(dict(path=binding['spec'],sha256=binding['spec_sha256']))
            for task in binding.get('tasks',{}).values():
                _verify(dict(path=task['spec'],sha256=task['spec_sha256']))
        checked(context['large_weight_stats'])
    projection={key:copy.deepcopy(context[key]) for key in FIELDS}
    # Only the outer snapshot location changes. The exact JSON bytes and the
    # original per-weight stat identities inside it remain bound by its SHA.
    projection['large_weight_stats_content_sha256']=context['large_weight_stats']['sha256']
    projection['schema']='gsdeploy.native_execution_projection_hypothesis.v5'
    return dict(schema='gsdeploy.native_execution_projection_identity.v5',
        projection_sha256=digest(projection),original_full_context_sha256=context_identity(context),
        measurement_domain=context['measurement_domain'],projection=projection,
        verified_source_count=len(sources) if verify_sources else 0,
        native_sources_sha256=digest(sources),source_bytes_verified=bool(verify_sources),
        large_weights_rehashed=False,model_weights_loaded=False,GPU_workload_executed=False,
        excluded_context_fields=sorted(set(context)-set(FIELDS)-{'large_weight_stats'}),
        collector_invariance_is_hypothesis=True,full_context_equality_claimed=False,
        natural_profiler_timing_transfer_claimed=False)


def review(context_paths,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    entries=[]
    for path in context_paths:
        source=receipt(path);context=checked(source)
        entries.append(dict(source=source,identity=native_execution_projection(context)))
    report=dict(schema='gsdeploy.native_execution_projection_review.v5',contexts=entries,
        unique_full_context_count=len({e['identity']['original_full_context_sha256'] for e in entries}),
        unique_native_projection_count=len({e['identity']['projection_sha256'] for e in entries}),
        interpretation='Explicit narrower source/configuration projection. All original contexts remain separate; matching projections authorize only a declared new pooling hypothesis.',
        implementation_sources=[receipt(__file__)],GPU_workload_executed=False,model_weights_loaded=False)
    save(output/'report.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--context',action='append',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();review(a.context,a.output)


if __name__=='__main__':main()
