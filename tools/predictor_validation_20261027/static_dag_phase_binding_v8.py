"""Static semantic phase membership for exact original training DAG kernels.

The original launch API's membership in a source NVTX forward labels its DAG
node. Forecast calls use that immutable label, not future phase wall intervals.
No phase, work, input, signature or source substitution is allowed.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .dag_phase_diagnostic_v8 import load_capture
from .native_collect import receipt


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def kernel_key(kernel):
    return kernel['correlation_id'],kernel['nsys_start_ns'],kernel['nsys_end_ns']


class StaticDagPhaseBinding:
    def __init__(self, dag_reference, light_review_references):
        if receipt(dag_reference['path'])!=dag_reference:
            raise ValueError('Changed original DAG source receipt')
        dag=json.loads(Path(dag_reference['path']).read_text())
        if dag.get('schema')!='gsdeploy.request_cuda_dag_profiles.v2' or dag.get('training_only') is not True:
            raise ValueError('Original training-only exact DAG templates required')
        sources=[copy.deepcopy(dag_reference),receipt(__file__)];observations={}
        for reference in light_review_references:
            report,plan,raw=load_capture(reference,'light-kernel')
            sources.extend([reference,report['raw'],report['plan'],report['context']])
            for alias,owner in report['owners'].items():
                if not owner['request_ids']:continue
                parsed_reference=owner['parsed'];sql_reference=owner['sqlite']
                if receipt(parsed_reference['path'])!=parsed_reference or receipt(sql_reference['path'])!=sql_reference:
                    raise ValueError('Changed original kernel/phase sources')
                parsed=json.loads(Path(parsed_reference['path']).read_text());sources.extend([parsed_reference,sql_reference])
                for row in raw['request_records']:
                    if row['model']!=alias:continue
                    phases=[p for p in parsed['phases'] if p['request_id']==row['request_id']]
                    if not phases:raise ValueError('Missing original measured request phases')
                    key=sql_reference['sha256'],alias,row['task'],row['pool_id'],row['request_id']
                    if key in observations:raise ValueError('Ambiguous original source request')
                    observations[key]=(row,phases)
        self.bindings={};self.node_count=0
        for profile in dag['profiles']:
            key=profile['alias'],profile['task'],profile['pool_id']
            if key in self.bindings:raise ValueError('Distinct exact DAG templates required')
            original=observations.get((profile['source_sqlite']['sha256'],*key,profile['request_id']))
            if original is None:raise ValueError('Exact original DAG request/SQLite review required')
            row,phases=original
            semantic=row['worker_response'].get('semantic_forward_observations',{}).get('records',[])
            phase_kinds={r['id']:r['semantic_phase'] for r in semantic}
            forwards=[p for p in phases if p['segment_id'] in phase_kinds]
            canonical={}
            for phase in phases:
                for kernel in phase['kernels']:
                    identity=kernel_key(kernel)
                    if identity in canonical:raise ValueError('Duplicate canonical phase kernel')
                    canonical[identity]=kernel
            gpu_nodes=[n for n in profile['nodes'] if n['kind']=='gpu']
            if {kernel_key(n['kernel']) for n in gpu_nodes}!=set(canonical):
                raise ValueError('Original DAG GPU denominator differs from canonical phase evidence')
            nodes={}
            for node in gpu_nodes:
                kernel=node['kernel']
                if ('source_sqlite' in kernel and kernel['source_sqlite']!=profile['source_sqlite']):
                    raise ValueError('Canonical kernel source SQLite differs from DAG source')
                if {k:v for k,v in kernel.items() if k!='source_sqlite'}!=canonical[kernel_key(kernel)]:
                    raise ValueError('Original DAG canonical kernel content differs')
                matches=[p for p in forwards if p['nsys_start_ns']<=node['issue_start_ns'] and node['issue_end_ns']<=p['nsys_end_ns']]
                if len(matches)>1:raise ValueError('Ambiguous exact launch semantic membership')
                exact_phase=matches[0]['segment_id'] if matches else '__outside_language_forwards__'
                phase=phase_kinds[exact_phase] if matches else ('ocr_inference' if key[1]=='ocr' else 'outside_language_forwards')
                if phase not in ('prefill','decode','ocr_inference','outside_language_forwards'):
                    raise ValueError('Undeclared semantic phase category')
                nodes[node['id']]=dict(node_sha256=fingerprint(node),phase=phase,exact_phase=exact_phase)
            self.bindings[key]=nodes;self.node_count+=len(nodes)
        for ref in sources:
            if receipt(ref['path'])!=ref:raise ValueError('Static phase source changed during binding')
        self.sources=copy.deepcopy(sources)

    def __call__(self,node,row):
        binding=self.bindings.get((row['alias'],row['task'],row['pool_id']),{}).get(node['id'])
        if binding is None or binding['node_sha256']!=fingerprint(node):
            raise UnknownEvidence('EXACT_ORIGINAL_DAG_PHASE_BINDING_REQUIRED')
        return binding['phase']

    def provenance(self):
        return dict(kind='static_original_DAG_API_to_NVTX_phase_membership',sources=copy.deepcopy(self.sources),
            template_count=len(self.bindings),GPU_node_count=self.node_count,
            future_wall_intervals_used=False,phase_change_resets_grant=False,
            unseen_input_supported=False,physical_service_identified=False)
