"""Predeclare an independent risk context; never read its runtime outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .horizon_recovery_domain_v31 import leaf_differences
from .idle_owner_selection_audit_v20 import read
from .owner_lifecycle_mark_v21 import receipt

# These label differences confer no execution equivalence by themselves.
ALLOWED_OMITTED_FIELDS=('execution_nonce','whole57_training_route_id','training_only','validation_scope')


def run(training_context_path,target_context_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    refs=[receipt(__file__),receipt(training_context_path),receipt(target_context_path)]
    train=read(training_context_path);target=read(target_context_path)
    def projection(value):return {key:item for key,item in value.items() if key not in ALLOWED_OMITTED_FIELDS}
    differences=leaf_differences(projection(train),projection(target))
    def source_map(context):
        return {row['original_path']:row['frozen']['sha256'] for row in context['execution_sources']}
    a=source_map(train);b=source_map(target);common=a.keys()&b.keys()
    substantive_keys=('execution','threads','residents','worker_bindings','worker_variant',
        'affine_observation','profiler','phase_scope','task_domain','measurement_domain',
        'monitor_serialization','capture_mode','telemetry_scope','collection_schema')
    after=[receipt(ref['path']) for ref in refs]
    if refs!=after:raise ValueError('Predeclared contexts changed during comparison')
    report=dict(schema='gsdeploy.horizon_target_context_predeclaration.v34',
        status='UNKNOWN_CONTEXT_EXECUTION_PROJECTION_DIFFERS' if differences else 'CONTEXT_COMPATIBLE_WAITING_FOR_READINESS_ONLY_ACTUAL_QUALIFICATION',
        source_receipts_before=refs,source_receipts_after=after,
        allowed_omitted_fields=list(ALLOWED_OMITTED_FIELDS),
        omitted_values={key:dict(training=train.get(key),target=target.get(key)) for key in ALLOWED_OMITTED_FIELDS},
        exact_context_projection_equal=not differences,complete_unexpected_differences=differences,
        difference_counts_by_top_key=dict(Counter(row['path'].split('/')[1] for row in differences)),
        selected_execution_field_comparisons={key:dict(exact_equal=not leaf_differences(train.get(key),target.get(key)),
            differences=leaf_differences(train.get(key),target.get(key),'/'+key)) for key in substantive_keys},
        execution_source_original_path_comparison=dict(training_n=len(a),target_n=len(b),common_n=len(common),
            common_content_SHA_differences=[dict(original_path=key,training_sha256=a[key],target_sha256=b[key]) for key in sorted(common) if a[key]!=b[key]],
            training_only_original_paths=sorted(a.keys()-b.keys()),target_only_original_paths=sorted(b.keys()-a.keys())),
        original_context_SHA_rewritten=False,training_model_refit=False,model_domain_rebound=False,
        target_runtime_records_read=False,target_service_queue_or_labels_read=False,
        actual_runtime_readiness_qualified=False,GPU_started=False,domain_transfer_qualified=False,
        next_step='Preserve target as UNKNOWN; resolve execution contract differences in a separately versioned bridge before any scoring.' if differences else
            'Verify exact target READY payload without service, queue or outcome labels before extending model binding metadata.')
    output.mkdir(parents=True);(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-context',required=True);parser.add_argument('--target-context',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.training_context,args.target_context,args.output)
    print(json.dumps(dict(status=result['status'],difference_counts_by_top_key=result['difference_counts_by_top_key']),indent=2))
