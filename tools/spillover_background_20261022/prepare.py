"""Prepare two immutable background-development blocks without starting GPUs."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1'
DESTINATION=ROOT/'artifacts/static-spillover-study-20261022/text-background-joint-diagnostic-v1'


def runner_module():
    spec=importlib.util.spec_from_file_location('_background_diagnostic_runner',Path(__file__).with_name('run.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def balanced(ids,seed,namespace,repeats):
    result=[]
    for repeat in range(repeats):
        values=list(ids);random.Random(f'{seed}:{namespace}:{repeat}').shuffle(values);result.extend(values)
    return result


def build_jobs(pools):
    runner=runner_module();jobs=[]
    for block,seed in [(1,105),(2,106)]:
        primary={a:[] for a in ['text#0','text#1','generalist_2#0','ocr#0']}
        g2_tasks=['image','image','asr'] if block==1 else ['asr','image','image']
        task_orders={t:balanced(pools[t],seed,f'background-primary:{t}',3) for t in ['image','asr','ocr']}
        cursors={'image':0,'asr':0}
        for i,task in enumerate(g2_tasks*96):
            ti=cursors[task];cursors[task]+=1
            primary['generalist_2#0'].append(dict(request_id=f'measurement:generalist_2#0:{i}',model='generalist_2#0',
                task=task,role='primary',pool_id=task_orders[task][ti],sequence=i,arrival_index=i,
                task_sequence=ti,repetition=ti//len(pools[task])))
        for i,pid in enumerate(task_orders['ocr']):
            primary['ocr#0'].append(dict(request_id=f'measurement:ocr#0:{i}',model='ocr#0',task='ocr',role='primary',
                pool_id=pid,sequence=i,arrival_index=i,task_sequence=i,repetition=i//len(pools['ocr'])))
        support={}
        for alias in ['text#0','text#1','generalist_2#0']:
            ids=balanced(pools['text'],seed,f'background-support:{alias}',32)
            support[alias]=[dict(request_id=f'support:{alias}:{i}',model=alias,task='text',role='support',pool_id=pid,
                sequence=len(primary[alias])+i,arrival_index=len(primary[alias])+i,support_sequence=i,repetition=i//32)
                for i,pid in enumerate(ids)]
        # Separate warmup copies never consume measurement primary/support cursors.
        warmup={a:copy.deepcopy(support[a][:32]) for a in ['text#0','text#1']}
        g2_warm=[]
        for i,row in enumerate(primary['generalist_2#0'][:32]):
            g2_warm.append(copy.deepcopy(row))
            if (i+1)%8==0:g2_warm.append(copy.deepcopy(support['generalist_2#0'][(i+1)//8-1]))
        warmup.update({'generalist_2#0':g2_warm,'ocr#0':copy.deepcopy(primary['ocr#0'][:32])})
        job=dict(job_id=f'background_block{block}_seed{seed}',block=block,seed=seed,
            active_aliases=list(primary),planned_requests=primary,support_requests=support,warmup_requests=warmup,
            primary_counts_by_path={'generalist_2/image':192,'generalist_2/asr':96,'ocr/ocr':96},
            generalist_task_pattern=g2_tasks,text_support_spacing=8)
        job['job_identity_sha256']=runner.job_digest(job);jobs.append(job)
    return jobs


def risk_check(config):
    """Work accounting only; historical baseline arithmetic is not capacity."""
    models=config['catalog']['models'];profiles=config['catalog']['task_replica_profiles']
    g2=models['generalist_2']['service_time_ms'];ocr=models['ocr']['service_time_ms']['ocr']
    spec=profiles['text']['text']['2']['mean_service_ms']
    image=192*g2['image']/1000;asr=96*g2['asr']/1000
    inserted=(288-1)//8
    support=inserted*g2['text']/1000
    return dict(schema='gsdeploy.background_diagnostic_cpu_risk.v1',
        historical_baseline_not_capacity=True,generalist_image_service_sum_seconds=image,
        generalist_asr_service_sum_seconds=asr,generalist_inserted_text_support_requests=inserted,
        generalist_inserted_text_baseline_seconds=support,generalist_combined_baseline_seconds=image+asr+support,
        ocr_primary_baseline_seconds=96*ocr/1000,
        specialist1024_balanced_support_baseline_seconds=1024*spec/1000,
        specialist600s_baseline_request_demand=600/(spec/1000),
        stress_scenarios=[dict(all_service_multiplier=factor,
            g2_nominal_service_seconds=factor*(image+asr+support),
            exceeds600s_service_sum=factor*(image+asr+support)>=600) for factor in [1,2,4,10]],
        support_cap_per_alias=1024,measurement_hard_wall_seconds=600,request_timeout_seconds=300,
        admission_decision='BOUNDED_DEVELOPMENT_ATTEMPT_NOT_FEASIBILITY_CERTIFICATE',
        limitations=['No new mixed background service has been measured.',
            'Historical per-input baseline means do not bound current or future service.',
            'Support cap may exhaust before600s; fail immediately, no extension or retry.',
            'A G2 background task and a G2 text support never execute in parallel.',
            'Diagnostic task tape is not formal0.01rps or a validated load/share predictor.'],
        U=None,formal_rho=None,selected_text_rate=None,GPU_started=False)


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def prepare(destination=DESTINATION,check_only=False):
    runner=runner_module();old_path=OLD/'experiment-packet.json';packet=copy.deepcopy(json.loads(old_path.read_text()))
    for path,expected in packet['bindings'].items():
        if runner.digest(path)!=expected:raise ValueError('Immutable original binding changed: '+path)
    rows=[json.loads(line) for line in Path(packet['samples']).read_text().splitlines() if line.strip()]
    pools={t:[r['pool_id'] for r in rows if r['task_type']==t] for t in ['text','image','asr','ocr']}
    if {t:len(v) for t,v in pools.items()}!={'text':32,'image':64,'asr':32,'ocr':32}:raise ValueError('Frozen pools differ')
    config=json.loads(Path(packet['system_config']).read_text());risk=risk_check(config)
    if check_only:return dict(jobs=build_jobs(pools),risk_check=risk,GPU_started=False)
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
    coverage=dict(minimum_common_handler_fraction=.9,minimum_covered_repetitions_per_input=2,
        cross_block_per_input_median_ratio_min=.5,cross_block_per_input_median_ratio_max=2.)
    declaration=dict(schema='gsdeploy.background_joint_diagnostic_declaration.v1',
        scope='Two fresh bounded development diagnostics for background service coverage; no formal utility/load/predictor/optimum',
        seeds=dict(development=[105,106],independent_validation_reserved=[81,82],confirmation_reserved=[71,72,73]),
        resident_deployment=packet['resident_deployment'],full_input_pools={t:len(v) for t,v in pools.items()},
        background_primary_per_block=dict(generalist_image=192,generalist_asr=96,ocr=96),
        generalist_task_tape='block1(image,image,asr)x96; block2(asr,image,image)x96; per-task full pool shuffled independently each repetition',
        generalist_text_support='After each8 completed background primary, before next background primary; after all own primary only while OCR unfinished. Explicit independent balanced cursor.',
        specialist_text_support='Both aliases from common release until all background primary complete; independent predeclared balanced cursors, caps1024 each.',
        ocr_after_primary='Idle but resident, no OCR support; G2/OCR busy versus idle contexts reported separately.',
        stop='First failure pauses serial queue without retry; completion stops new support and drains pending within same600s wall.',
        timeouts=dict(measurement_seconds=600,request_seconds=300,startup_seconds=300,warmup_seconds=30,
            shutdown='Frozen shared graceful10s plus bounded terminate/kill/thread joins'),
        coverage_policy=coverage,
        coverage_scope='Every background path/input/block requires at least2 primary with handler overlap>=.90 with both text specialist handlers simultaneously. G2/OCR activity and task-tape phase contexts reported separately, not inferred from in-flight status.',
        repetition_scope='Per-path/input median using only qualified primary; report cross-block block2/block1 in[.5,2], missing observations and drift. Never substitute support or shutdown-tail output.',
        quality='Native primary by task and support separate; no absolute quality admission, no overwrite of frozen historical quality.',
        resource_correction='New runner samples memory during finally before stopping monitor, including failed attempts; old frozen source/raw untouched.',
        unchanged=['Frozen native worker/checkpoint/quality identities','formal Poisson/SWRR/score/stability','seven-model pool and replica states','rho<=.8 and original capacity factors','49140MiB and safe factor'],
        authorization_scope='Required new development evidence; prior failed sustained cohort stays failed and does not prohibit a new bounded diagnostic.',
        no_auto_retry=True,no_following_batch_auto_enqueue=True,GPU_started=False,U=None)
    write(destination/'discovery-protocol.json',declaration);write(destination/'cpu-risk-check.json',risk)
    packet.update(protocol=runner.PROTOCOL,jobs=build_jobs(pools),support_limit_per_alias=1024,
        generalist_support_every_primary=8,coverage_policy=coverage,diagnostic_only=True,U=None,gpu_started=False)
    bound=[old_path,Path(__file__).resolve(),Path(__file__).with_name('run.py').resolve(),
        destination/'discovery-protocol.json',destination/'cpu-risk-check.json',ROOT/'docs/text_background_joint_coverage_design_20261022.md']
    for path in bound:packet['bindings'][str(path)]=runner.digest(path)
    packet['bindings_sha256']=hashlib.sha256(runner.canonical(packet['bindings'])).hexdigest()
    runner.validate_packet(packet);write(destination/'experiment-packet.json',packet)
    receipt=dict(packet_sha256=runner.digest(destination/'experiment-packet.json'),jobs=2,
        primary_requests_per_block=384,total_primary_requests=768,inserted_G2_text_support_per_completed_tape=35,
        support_limit_per_alias=1024,source_bindings=len(packet['bindings']),GPU_started=False,U=None)
    write(destination/'preparation-receipt.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DESTINATION)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    result=prepare(args.output,args.check_only)
    print(json.dumps(result if not args.check_only else dict(jobs=len(result['jobs']),risk_check=result['risk_check'],GPU_started=False)))


if __name__=='__main__':main()
