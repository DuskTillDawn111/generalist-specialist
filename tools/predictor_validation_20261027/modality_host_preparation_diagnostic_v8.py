"""Four original image/ASR cells: initial API supply and static phase audit."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

from .dag_host_release_v5 import ACTIVITY,structure
from .dag_host_release_v7 import checked
from .dag_phase_diagnostic_v8 import load_capture
import sqlite3
from gsdeploy.resources import identity
from .small_training_domain_bridge_review_v4 import request_kernels
from .trace_export import linux_pid
from .native_collect import receipt,save
from .static_dag_phase_binding_v8 import StaticDagPhaseBinding


def summarize_capture(ref):
    strict, plan, raw = load_capture(ref, 'light-kernel')
    rows = []
    sources = [ref, strict['raw'], strict['plan'], strict['context']]
    for alias, owner in strict['owners'].items():
        if not owner.get('request_ids'):continue
        parsed = checked(owner['parsed'])
        checked_ref = owner['sqlite']
        if receipt(checked_ref['path']) != checked_ref:
            raise ValueError('Changed solo CUDA source')
        sources.extend((owner['parsed'], checked_ref))
        with sqlite3.connect(Path(checked_ref['path']).as_uri()+'?mode=ro', uri=True) as conn:
            conn.row_factory = sqlite3.Row
            runtime = [dict(r) for r in conn.execute('select r.*,s.value as api_name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id')]
            tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
            auxiliary = []
            for table, kind in (('CUPTI_ACTIVITY_KIND_MEMCPY', 'copy'), ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')):
                if table in tables:
                    auxiliary.extend((kind, dict(r)) for r in conn.execute('select * from '+table))
        for row in raw['request_records']:
            if row['model'] != alias:
                continue
            rid = row['request_id']
            phases = [p for p in parsed['phases'] if p['request_id'] == rid]
            pid = phases[0]['pid']
            mapping = parsed['clock_mappings'][str(pid)]
            if mapping.get('valid') is not True:
                raise ValueError('Valid independently observed handler/trace clock mapping required')
            handler = row['worker_response']['start_perf_ns'] - mapping['offset_ns']
            inside = lambda stamp: any(p['nsys_start_ns'] <= stamp <= p['nsys_end_ns'] for p in phases)
            apis = [a for a in runtime if linux_pid(a['globalTid']) == pid and inside(a['start'])]
            apis.sort(key=lambda a: (a['start'], a['end'], a['correlationId']))
            api_by_correlation = {a['correlationId']: a for a in apis}
            kernels = request_kernels(parsed, checked_ref['path'], row)
            activities = [('gpu', k['kernel']['correlation_id'], k['kernel']['nsys_start_ns']) for k in kernels]
            activities.extend((kind, a['correlationId'], a['start']) for kind, a in auxiliary
                              if linux_pid(a['globalPid']) == pid and a['correlationId'] in api_by_correlation)
            activities.sort(key=lambda a: (api_by_correlation[a[1]]['start'], a[2]))
            first = activities[0]
            gpu = next(a for a in activities if a[0] == 'gpu')
            first_api = api_by_correlation[first[1]]
            first_gpu_api = api_by_correlation[gpu[1]]
            before = [a for a in apis if a['start'] < first_api['start']]
            rows.append(dict(alias=alias, task=row['task'], pool_id=row['pool_id'], request_id=rid,
                source_raw=strict['raw'], source_sqlite=checked_ref, clock_mapping=mapping,
                first_runtime_API=dict(name=apis[0]['api_name'], offset_ns=apis[0]['start']-handler),
                first_activity=dict(kind=first[0], api_name=first_api['api_name'],
                    API_entry_offset_ns=first_api['start']-handler, API_return_offset_ns=first_api['end']-handler,
                    activity_start_offset_ns=first[2]-handler),
                first_gpu=dict(API_entry_offset_ns=first_gpu_api['start']-handler,
                    API_return_offset_ns=first_gpu_api['end']-handler, activity_start_offset_ns=gpu[2]-handler),
                preceding_runtime_API=[dict(name=a['api_name'], entry_offset_ns=a['start']-handler,
                    return_offset_ns=a['end']-handler) for a in before],
                complete_runtime_sequence_count=len(apis),
                complete_runtime_name_sequence_sha256=identity([a['api_name'] for a in apis]),
                complete_ordered_gpu_signature_sha256=identity([dict(group=k['group'], signature=k['signature']) for k in kernels]),
                complete_ordered_gpu_count=len(kernels),
                phase_offsets=[dict(id=p['id'], semantic_phase=p['semantic_phase'],
                    start_offset_ns=p['start_ns']-row['worker_response']['start_perf_ns'],
                    end_offset_ns=p['end_ns']-row['worker_response']['start_perf_ns'])
                    for p in row['worker_response'].get('semantic_forward_observations', {}).get('records', [])[:1]]))
    return rows, sources


def analyze(configuration_path,host_path,output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    config=checked(receipt(configuration_path));component=config['components']['original_image_asr']
    host=checked(receipt(host_path));dag=checked(component['dag'])
    if host['dag']!=component['dag']:raise ValueError('Original modality host/DAG source mismatch')
    sources=[receipt(__file__),receipt(configuration_path),receipt(host_path),component['dag'],*host['sources']]
    rows=defaultdict(list)
    for ref in component['solo_reviews']:
        found,refs=summarize_capture(ref);sources.extend(refs)
        for row in found:rows[row['alias'],row['task'],row['pool_id']].append(row)
    natural=defaultdict(list)
    for capture in host['captures']:
        strict,_,raw=load_capture(capture['review'],'natural')
        if strict['raw']!=capture['raw']:raise ValueError('Original modality natural source mismatch')
        sources.extend((capture['review'],strict['raw'],strict['plan'],strict['context']))
        for row in raw['request_records']:
            w=row['worker_response'];semantic=w['semantic_forward_observations']['records']
            natural[row['model'],row['task'],row['pool_id']].append(dict(source_raw=strict['raw'],
                first_semantic_start_offset_ns=semantic[0]['start_ns']-w['start_perf_ns'],
                stages=[dict(id=s['id'],start_offset_ns=s['start_ns']-w['start_perf_ns'],
                    end_offset_ns=s['end_ns']-w['start_perf_ns']) for s in w['observed_stages']]))
    phase=StaticDagPhaseBinding(component['dag'],component['solo_reviews'][:1])
    sources.extend(phase.provenance()['sources'])
    groups={(g['family'],g['task']):g for g in host['groups'] if g['supported']}
    result=[]
    for profile in dag['profiles']:
        key=profile['alias'],profile['task'],profile['pool_id'];light=rows[key]
        if len(light)!=2 or len({r['source_raw']['sha256'] for r in light})!=2:
            raise ValueError('Two independent original isolated modality captures required')
        if len({r['complete_ordered_gpu_signature_sha256'] for r in light})!=1:
            raise ValueError('Original modality exact signature order mismatch')
        features=structure(profile);par=groups[profile['family'],profile['task']]['joint_mean']
        seq=sorted((n for n in profile['nodes'] if n['kind']!='handler_end'),key=lambda n:n['issue_start_ns'])
        first=next(n for n in seq if n['kind'] in ACTIVITY);gpu=next(n for n in seq if n['kind']=='gpu')
        phases=Counter(phase(n,dict(alias=key[0],task=key[1],pool_id=key[2])) for n in seq if n['kind']=='gpu')
        result.append(dict(alias=key[0],task=key[1],pool_id=key[2],independent_light_observations=light,
            independent_natural_observations=natural[key],static_phase_kernel_counts=dict(phases),
            original_host_first_activity_release_ns=round(par['issue_depth_ns_per_step']*features['issue_depth'][first['id']]),
            original_host_first_GPU_release_ns=round(par['issue_depth_ns_per_step']*features['issue_depth'][gpu['id']]),
            original_profile_first_activity_release_ns=first['release_ns'],original_profile_first_GPU_release_ns=gpu['release_ns'],
            preparation_conditioned_host_support='UNKNOWN_NOT_FITTED',
            availability_policy_support='UNKNOWN_UNTIL_MATCHED_MULTIMODAL_TRAINING_CELLS_EXIST'))
    sources=list({r['path']:r for r in sources}.values())
    if any(receipt(r['path'])!=r for r in sources):raise ValueError('Original modality source changed during review')
    report=dict(schema='gsdeploy.original_modality_host_preparation_diagnostic.v8',rows=result,sources=sources,
        static_phase_provenance=phase.provenance(),GPU_started=False,training_only=True,
        exact_template_count=len(result),kernel_count=sum(sum(r['static_phase_kernel_counts'].values()) for r in result),
        image_encoder_vs_audio_encoder_inferred_from_outside_language=False,
        natural_first_CUDA_observed=False,independent_prediction_validation=False,
        interpretation='Original image/ASR phase and solo-source availability verified; host and availability prediction remain explicitly unsupported pending matched training fit')
    save(out/'report.json',report);return report


def main():
    p=argparse.ArgumentParser()
    for name in ('configuration','host','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=analyze(a.configuration,a.host,a.output)
    print(json.dumps(dict(exact_templates=r['exact_template_count'],kernels=r['kernel_count'],report=receipt(Path(a.output)/'report.json'))))

if __name__=='__main__':main()
