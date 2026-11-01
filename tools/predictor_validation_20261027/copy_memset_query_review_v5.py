"""CPU exact-source resolver review against real CUDA DAG copy/memset nodes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from pathlib import Path

from .copy_memset_baseline_v5 import ExactSourceCopyMemsetHypothesisResolver
from .cuda_dependency_dag_v2 import audit_sqlite
from .kernel_baseline_intake_v3 import checked
from .native_collect import receipt, save


def review(fit_path, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    resolver = ExactSourceCopyMemsetHypothesisResolver(fit_path)
    sources = {}
    for entry in resolver.index.values():
        source = entry['source_sqlite']
        sources[(source['path'], source['sha256'])] = source
    if any('duration_ns' in entry for entry in resolver.index.values()):
        raise ValueError('Measured duration target found in resolver query index')
    counts = Counter()
    buckets = defaultdict(Counter)
    source_reviews = []
    matched = set()
    for source in sources.values():
        dag = audit_sqlite(source['path'])
        if dag['source'] != source:
            raise ValueError('Actual DAG source receipt differs from fitted source')
        local = Counter()
        for node in dag['dag']['nodes']:
            if node['kind'] not in ('copy', 'memset'):
                continue
            result = resolver(node, source)
            counts['whole_trace_copy_memset_nodes'] += 1
            if result.get('observation_id') is None:
                counts['unattributed_nodes_remaining_unknown'] += 1
                if result['numeric_hypothesis_available']:
                    raise ValueError('Unattributed node received a numerical prediction')
                continue
            observation_id = result['observation_id']
            if observation_id in matched:
                raise ValueError('One observation matched more than one DAG node')
            matched.add(observation_id)
            entry = resolver.index[(source['sha256'], node['kind'], int(node['id'].split(':')[-1]))]
            key = entry['family'] + ':' + node['kind']
            label = 'strict_known' if result['strict_known'] else 'numeric_only' if result['numeric_hypothesis_available'] else 'unknown'
            counts[label] += 1
            buckets[key][label] += 1
            local[label] += 1
            altered = dict(node, duration_ns=10**30)
            if resolver(altered, source) != result:
                raise ValueError('Query measured duration affected model prediction')
            changed_kind = copy.deepcopy(node)
            changed_kind['memory_metadata']['bytes'] += 1
            if resolver(changed_kind, source)['numeric_hypothesis_available']:
                raise ValueError('Changed byte identity received a prediction')
            if result['known'] != result['strict_known'] or result['deployment_qualified'] or result['physical_accuracy']:
                raise ValueError('Strict evidence or qualification flags changed')
            if result['measured_target_fallback_used']:
                raise ValueError('Measured target fallback was used')
        source_reviews.append(dict(source_sqlite=source, counts=dict(local)))
    if len(matched) != len(resolver.index):
        raise ValueError('Some fitted observations did not match actual DAG nodes')
    fit = checked(resolver.fit_source)
    report = dict(schema='gsdeploy.copy_memset_exact_source_query_review.v5',
        fit=resolver.fit_source, indexed_observations=len(resolver.index),
        observed_dag_match_count=len(matched), counts=dict(counts),
        family_activity_counts={k:dict(v) for k,v in buckets.items()}, sources=source_reviews,
        query_index_contains_measured_duration_targets=False,
        altered_query_measured_duration_has_no_effect=True,
        changed_byte_identity_rejected=True, original_strict_evidence_preserved=True,
        thresholds=fit['thresholds'], training_source_review_only=True,
        heldout_validation_performed=False, natural_timing_validated=False,
        deployment_qualified=False, GPU_workload_executed=False, model_weights_loaded=False,
        implementation_sources=[receipt(__file__), receipt(Path(__file__).with_name('cuda_dependency_dag_v2.py'))])
    save(output/'report.json', report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    result=review(args.fit,args.output)
    print(dict(indexed_observations=result['indexed_observations'], counts=result['counts'],
               family_activity_counts=result['family_activity_counts']))


if __name__=='__main__':main()
