"""Exact ordinal solo-only lifetime reference work for availability scheduling.

Reference nanoseconds are the median of two independently captured, isolated,
same-work/same-ordered-signature kernels. They are not identified SM cycles.
Co-run durations and endpoint ratios are forbidden as service inputs.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
import statistics

from gsdeploy.forecast_errors import UnknownEvidence
from .dag_phase_diagnostic_v8 import load_capture
from .native_collect import receipt
from .paired_training_work_review_v4 import actual_work
from .small_training_domain_bridge_review_v4 import request_kernels
from .static_dag_phase_binding_v8 import fingerprint,kernel_key


class SoloOrdinalReferenceWork:
    def __init__(self, dag_reference, solo_review_references, *, copy_duration_provider=None):
        if len(solo_review_references)!=2 or receipt(dag_reference['path'])!=dag_reference:
            raise ValueError('Exactly two independent solo reviews and unchanged original DAG required')
        dag=json.loads(Path(dag_reference['path']).read_text())
        if dag.get('schema')!='gsdeploy.request_cuda_dag_profiles.v2' or dag.get('training_only') is not True:
            raise ValueError('Exact original training-only DAG required')
        captures=[];sources=[dag_reference,receipt(__file__)];raw_ids=[]
        for reference in solo_review_references:
            report,plan,raw=load_capture(reference,'light-kernel')
            if any(len(j['inputs'])!=1 for j in plan['jobs']):
                raise ValueError('Solo-only jobs required; co-run durations are forbidden')
            windows=sorted((r['worker_response']['start_perf_ns'],r['worker_response']['end_perf_ns']) for r in raw['request_records'])
            if any(b>c for (a,b),(c,d) in zip(windows,windows[1:])):
                raise ValueError('Isolated non-overlapping actual solo handlers required')
            raw_ids.append(report['raw']['sha256']);sources.extend([reference,report['raw'],report['plan'],report['context']])
            records={}
            for alias,owner in report['owners'].items():
                if not owner['request_ids']:continue
                parsed_reference=owner['parsed'];sql_reference=owner['sqlite']
                if receipt(parsed_reference['path'])!=parsed_reference or receipt(sql_reference['path'])!=sql_reference:
                    raise ValueError('Changed exact solo kernel evidence')
                parsed=json.loads(Path(parsed_reference['path']).read_text());sources.extend([parsed_reference,sql_reference])
                for row in raw['request_records']:
                    if row['model']!=alias:continue
                    key=alias,row['task'],row['pool_id']
                    if key in records:raise ValueError('One exact solo observation per capture/cell required')
                    items=request_kernels(parsed,sql_reference['path'],row)
                    records[key]=dict(row=row,items=items,sqlite=sql_reference)
            captures.append(records)
        if len(set(raw_ids))!=2:raise ValueError('Two independent actual raw captures required')
        self.index={};self.groups=[];self.copy_provider=copy_duration_provider
        for profile in dag['profiles']:
            key=profile['alias'],profile['task'],profile['pool_id']
            if key in self.index:raise ValueError('Distinct exact DAG cells required')
            observed=[capture.get(key) for capture in captures]
            if any(r is None for r in observed):raise ValueError('Missing exact independent solo cell')
            left,right=observed;left_items=left['items'];right_items=right['items']
            if not left_items or [(r['group'],r['signature']) for r in left_items] != [(r['group'],r['signature']) for r in right_items]:
                raise ValueError('Exact ordered static signature and semantic membership mismatch')
            if actual_work(left['row']['worker_response'])!=actual_work(right['row']['worker_response']):
                raise ValueError('Actual generated work differs between solo captures')
            if (profile['source_sqlite']!=left['sqlite'] or profile['request_id']!=left['row']['request_id']):
                raise ValueError('Original DAG must bind first exact solo capture, never timing-retargeted')
            by_kernel={kernel_key(n['kernel']):n for n in profile['nodes'] if n['kind']=='gpu'}
            if set(by_kernel)!={kernel_key(r['kernel']) for r in left_items}:
                raise ValueError('Exact original DAG ordinal kernel denominator mismatch')
            index={};sum_work=0.;sum_left=0;sum_right=0;phase_counts=defaultdict(int)
            for ordinal,(a,b) in enumerate(zip(left_items,right_items)):
                node=by_kernel[kernel_key(a['kernel'])]
                if {k:v for k,v in node['kernel'].items() if k!='source_sqlite'}!=a['kernel']:
                    raise ValueError('Original ordinal kernel content differs')
                durations=[x['kernel']['nsys_end_ns']-x['kernel']['nsys_start_ns'] for x in (a,b)]
                if min(durations)<=0:raise ValueError('Positive solo lifetime reference required')
                work=statistics.median(durations);sum_work+=work;sum_left+=durations[0];sum_right+=durations[1]
                index[node['id']]=dict(node_sha256=fingerprint(node),reference_ns=work,ordinal=ordinal,
                    exact_phase=a['group'],two_solo_lifetimes_ns=durations)
                phase_counts[a['group']]+=1
            self.index[key]=index
            self.groups.append(dict(alias=key[0],task=key[1],pool_id=key[2],kernel_count=len(index),
                independent_raw_ids=raw_ids,reference_ns_sum=sum_work,
                solo_lifetime_sums_ns=[sum_left,sum_right],phase_kernel_counts=dict(phase_counts),
                ordered_signature_sha256=fingerprint([(r['group'],r['signature']) for r in left_items])))
        for ref in sources:
            if receipt(ref['path'])!=ref:raise ValueError('Solo reference source changed')
        self.sources=copy.deepcopy(sources)

    def __call__(self,node,row):
        if node['kind']!='gpu':
            if node['kind'] in ('copy','memset') and self.copy_provider is not None:
                return self.copy_provider(node,row)
            raise UnknownEvidence('INDEPENDENT_COPY_REFERENCE_PROVIDER_REQUIRED')
        entry=self.index.get((row['alias'],row['task'],row['pool_id']),{}).get(node['id'])
        if entry is None or entry['node_sha256']!=fingerprint(node):
            raise UnknownEvidence('EXACT_SOLO_ORDINAL_REFERENCE_WORK_REQUIRED')
        return entry['reference_ns']

    def provenance(self):
        return dict(kind='exact_two_capture_solo_ordinal_reference_lifetime',reference_work_domain='SOLO_ONLY_REFERENCE_NS',
            sources=copy.deepcopy(self.sources),groups=copy.deepcopy(self.groups),
            template_count=len(self.index),GPU_node_count=sum(len(v) for v in self.index.values()),
            co_run_lifetimes_used=False,whole_request_multiplier_used=False,
            original_DAG_retimed=False,physical_SM_service_identified=False,
            independent_prediction_accuracy_validated=False,unseen_input_supported=False)
