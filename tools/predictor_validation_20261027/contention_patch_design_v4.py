"""CPU-only sharded training design; no experiment runner or validation data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .native_collect import receipt,save
from .profiling import RESIDENTS,digest


def jobs(selection,aliases,generalist_task):
    result=[]
    for stratum in (0,3):
        for offset in (0,50,200):
            for order in ([aliases] if offset==0 else [aliases,list(reversed(aliases))]):
                tasks={a:('ocr' if a=='ocr#0' else generalist_task if a=='generalist_2#0' else 'text') for a in aliases}
                job=dict(inputs={a:selection['training'][t][stratum]['pool_id'] for a,t in tasks.items()},
                         tasks=tasks,start_offsets_ms={a:i*offset for i,a in enumerate(order)},
                         input_stratum_index=stratum,split='training',repeat=0,
                         phase='contention-training-patch',validation_view='TRAINING_ONLY_NOT_VALIDATION')
                job['id']=digest(job);result.append(job)
    return result


def design(selection):
    shards=[]
    solo=[]
    for task in ('image','asr'):
        for stratum in (0,3):
            job=dict(inputs={'generalist_2#0':selection['training'][task][stratum]['pool_id']},
                tasks={'generalist_2#0':task},start_offsets_ms={'generalist_2#0':0},
                input_stratum_index=stratum,split='training',repeat=0,
                phase='new-modality-solo-structure',validation_view='TRAINING_ONLY_NOT_VALIDATION')
            job['id']=digest(job);solo.append(job)
    shards.append(dict(name='G2-image-asr-solo',domains=['metadata','light-kernel','natural'],jobs=solo,
        purpose='New modality structure and natural endpoints; no assumption of language-affine coverage.'))
    contexts=[('SS',['text#0','text#1']),('SG',['text#0','generalist_2#0']),
              ('SSG',['text#0','text#1','generalist_2#0']),('SSGO',list(RESIDENTS))]
    for context,aliases in contexts:
        for task in (('text',) if context=='SS' else ('text','image','asr')):
            shard_jobs=jobs(selection,aliases,task)
            shards.append(dict(name=context+'-G2-'+task if context!='SS' else context,
                domains=['light-kernel','natural'],jobs=shard_jobs,
                structure_binding='Reuse only exact source/precision/shape/ordered-launch-signature matches to actual solo metadata. Mismatch UNKNOWN; collect new metadata only for missing structures.'))
    for shard in shards:
        shard['measured_requests_per_domain']=sum(len(j['inputs']) for j in shard['jobs'])
        shard['independent_capture_requirement']=2
    counts={domain:dict(jobs=sum(len(s['jobs']) for s in shards if domain in s['domains']),
        requests=sum(s['measured_requests_per_domain'] for s in shards if domain in s['domains']))
        for domain in ('metadata','light-kernel','natural')}
    return dict(schema='gsdeploy.contention_training_patch_design.v4',status='DESIGN_ONLY_NOT_EXECUTABLE',
        training_only=True,physical_owners=RESIDENTS,shards=shards,counts_per_independent_capture=counts,
        two_capture_measured_requests=2*sum(v['requests'] for v in counts.values()),
        no_gpu_started=True,validation_started=False,
        context_protocol='Freeze complete patch runner before first capture; same-domain context must remain identical across shards and independent captures. Metadata/light/natural remain distinct contexts.',
        clock_requirement='GPU context-switch and kernel data in one system-level session, or explicit bounded cross-session alignment. Median NVTX residual is not a cross-process timing bound.',
        execution_mode_requirement='First verify actual context scheduling. Existing default sharing does not imply MPS or same-SM concurrent kernels. GPU context switch records required for timeslice evidence; SM-level concurrent execution needs additional location-aware evidence.',
        startup_evidence_requirement='New collector must start board telemetry before runtime/model construction, record startup_begin/end and priming boundaries, retain owned and board memory samples through loading. Existing bridge collector starts monitor after runtime startup: startup energy and startup memory are UNKNOWN and may not be filled from active windows.',
        long_decode='NOT_COVERED: current bridge outputs 5 tokens. Need training-only inputs producing longer observed output under unchanged policy; do not force max_new_tokens/min_new_tokens silently.',
        remaining=['long decode','independent held-out prediction validation','whole-horizon queues/P95/delta-U candidate rankings',
                   'admission comparison','verified resource shares','joint batch/shared-weight execution'],
        stop_rules=['Review each shard before next GPU shard; preserve every raw output without retries.',
                    'Keep >=12 GiB disk free throughout; check next-shard measured size estimate before model load.',
                    'If full plan exceeds budget, reduce pending scope or archive verified immutable evidence before continuing; never delete raw evidence implicitly.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();selection=Path(args.selection)
    result=design(json.loads(selection.read_text()));result['selection']=receipt(selection)
    free=shutil.disk_usage(Path(args.output).resolve().parent).free
    # One complete light capture has 8 solo requests and measured retained
    # trace/export footprint. This is a transparent scenario, not a bound.
    per_request=408561879/8
    light_requests=2*result['counts_per_independent_capture']['light-kernel']['requests']
    result['storage_scenario']=dict(free_bytes_at_design=free,free_reserve_bytes=12*2**30,
        light_bytes_per_request_proxy=per_request,proxy_basis='capture1 light retained artifacts:408561879 bytes/8 requests',
        two_capture_light_bytes_estimate=light_requests*per_request,
        metadata_natural_additional_bytes='UNKNOWN_NEW_MODALITY_COST',
        guaranteed_upper_bound=False,full_plan_budget_approved=False,
        next_shard_policy='Start smallest scheduling proof and modality solo shards; re-estimate using actual observed request work and retained bytes.')
    save(args.output,result)
    print(json.dumps(dict(counts=result['counts_per_independent_capture'],two_capture_requests=result['two_capture_measured_requests'],storage=result['storage_scenario'])))


if __name__=='__main__':main()
