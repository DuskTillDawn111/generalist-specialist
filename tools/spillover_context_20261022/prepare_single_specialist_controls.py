"""CPU-only paired text1/G2 half-route control; never schedules or launches GPU."""
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
BASE=ROOT/'artifacts/static-spillover-study-20261022'
DESTINATION=BASE/'text185-single-specialist-controls-v1'
sys.path.insert(0,str(SOURCE))
from gsdeploy import static_study
from gsdeploy.config import SystemConfig
from gsdeploy.quality import quality_digest
from gsdeploy.routing import Router
from gsdeploy.runtime.workload import SamplePool
from gsdeploy.trace import expand_routes,make_trace,trace_digest


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def preview_cohort(config,plan,inputs,seed,phase,duration):
    """Exactly the frozen window's phase-local trace, sample cursor and SWRR."""
    trace=make_trace(config['workload'],duration,seed,config['arrival_process'])
    pool=SamplePool(Path(inputs)/'runtime-inputs.jsonl',list(config['workload']),seed)
    aliases,routes=expand_routes(plan['deployment'],plan['routing']);router=Router(routes)
    rows=[];independent=[];counts=defaultdict(Counter);unique=defaultdict(lambda:defaultdict(set))
    for index,(offset,task) in enumerate(trace):
        sample=pool.next(task);alias=router.choose(task)
        row=dict(request_id=f'{phase}:{index}',arrival_index=index,scheduled_offset_seconds=offset,
            task=task,pool_id=sample['pool_id'],model=alias)
        rows.append(row);independent.append({k:v for k,v in row.items() if k!='model'})
        counts[alias][task]+=1;unique[alias][task].add(sample['pool_id'])
    return dict(phase=phase,duration_seconds=duration,seed=seed,trace_sha256=trace_digest(trace),
        request_count=len(rows),task_counts=dict(Counter(r['task'] for r in rows)),
        route_independent_cohort_sha256=quality_digest(independent),native_routed_cohort_sha256=quality_digest(rows),
        alias_task_counts={a:dict(counts[a]) for a in aliases},
        alias_unique_inputs={a:{t:len(ids) for t,ids in unique[a].items()} for a in aliases},
        request_roster=rows)


def cohort_checks(config,plan,inputs,fine,strong):
    result=[]
    for seed in (107,108):
        phases=[]
        for phase,duration in [('warmup',30),('measurement',600)]:
            current=preview_cohort(config,plan,inputs,seed,phase,duration);peers=[]
            for name,packet in [('fine',fine),('strong_all_specialist',strong)]:
                jobs=[j for j in packet['jobs'] if j['seed']==seed]
                if not jobs:raise ValueError('Missing paired seed: '+name)
                for job in jobs:
                    for key in ('workload','horizon_seconds','arrival_process','drain_seconds'):
                        if job['identity'][key]!=config[key]:raise ValueError('Paired contract changed: '+key)
                    if job['identity']['warmup_seconds']!=30:raise ValueError('Warmup changed')
                    reference=preview_cohort(packet['config'],job['plan'],packet['inputs'],seed,phase,duration)
                    if current['route_independent_cohort_sha256']!=reference['route_independent_cohort_sha256']:
                        raise ValueError('Route-independent task/time/input cohort differs')
                    peers.append(dict(packet=name,job_id=job['id'],trace_sha256=reference['trace_sha256'],
                        route_independent_cohort_sha256=reference['route_independent_cohort_sha256'],match=True))
            current['paired_peers']=peers;phases.append(current)
        result.append(dict(seed=seed,phases=phases))
    return dict(schema='gsdeploy.text185_single_specialist_cohort_check.v1',seeds=result,all_paired_cohorts_match=True,
        policy='Frozen phase-local SamplePool and deterministic expanded-alias SWRR; exact request/input exposure retained, no randomization or ideal fractional substitution.',
        priming_scope='Same largest_three_per_task policy; resident deployment differs, so priming request counts differ and are not claimed identical.',
        GPU_started=False,production_U=None,physical_feasibility='UNKNOWN')


def prepare():
    destination=DESTINATION
    if destination.exists():raise FileExistsError(destination)
    original_path=ROOT/'artifacts/static-study-gpu-20261022/execution-packet-v2.json'
    fine_path=BASE/'text185-fine-development-v1/experiment-packet.json'
    strong_path=BASE/'text185-strong-controls-v1/experiment-packet.json'
    screen_path=BASE/'text185-strong-challengers-cpu-v4/report.json'
    review_path=screen_path.parent/'shortlist-review-v1/review.json'
    original=json.loads(original_path.read_text());fine=json.loads(fine_path.read_text());strong=json.loads(strong_path.read_text())
    screen=json.loads(screen_path.read_text());review=json.loads(review_path.read_text())
    for packet in (original,fine,strong):
        validation=static_study.validate_packet(packet)
        if not validation['valid']:raise ValueError(str(validation))
    if static_study._file(screen_path)!=review['source_report']:raise ValueError('Shortlist source changed')
    candidate=next(c for c in screen['candidates'] if c['id']=='text1_G2_native_half')
    if next(c for c in review['selected'] if c['id']==candidate['id'])!=candidate:raise ValueError('Reviewed candidate changed')
    plan=dict(deployment=dict(text=1,generalist_2=1,ocr=1),routing=dict(text={'text':.925,'generalist_2':.925},
        image={'generalist_2':.01},asr={'generalist_2':.01},ocr={'ocr':.01}))
    if candidate['deployment']!=plan['deployment'] or candidate['routing']!=plan['routing']:raise ValueError('Exact requested candidate differs')
    config_data=copy.deepcopy(fine['config'])
    if config_data!=strong['config']:raise ValueError('Fine/strong configuration differs')
    config=SystemConfig(**config_data)
    if config.workload!=dict(text=1.85,image=.01,asr=.01,ocr=.01) or config.horizon_seconds!=600 or config.arrival_process!='poisson':
        raise ValueError('Original paired workload/horizon/arrival contract differs')
    for key in ('weights','latency_reference_ms','gpu_memory_mib','drain_seconds','arrival_process',
                'energy_reference_watts','startup_amortization_seconds','utility_schema'):
        if config_data[key]!=original['config'][key]:raise ValueError('Original scoring/runtime term changed: '+key)
    for key in ('models','quality','safe_factor','capacity_factors','memory_table_mib','task_replica_profiles'):
        if config_data['catalog'][key]!=original['config']['catalog'][key]:raise ValueError('Original model domain changed: '+key)
    constraints=config.validate_plan(**plan)
    if not constraints['constraints_valid'] or constraints!=candidate['frozen_config_validate_plan']:
        raise ValueError('Formal proxy constraints changed or failed')
    cohort=cohort_checks(config_data,plan,original['inputs'],fine,strong)
    protocol=dict(schema='gsdeploy.text185_single_specialist_controls.v1',
        purpose='Paired strong mixed-deployment development control testing whether a second text specialist replica is necessary; competes with mixed deployments as well as pure-specialist/pure-generalist controls.',
        source_candidate_id=candidate['id'],workload=config.workload,horizon_seconds=600,warmup_seconds=30,
        arrival_process='poisson',drain_seconds=config.drain_seconds,warmup_policy='largest_three_per_task',
        plan=plan,development_seeds=[107,108],execution_order=[107,108],
        comparison_contract='Identical frozen v5 runtime, per-seed route-independent task/time/input cohorts, Q/T/L/P/M/E scoring and stability rules to fine4 and all-specialist strong2. Different native routing and deployment are the treatment.',
        stability='Original rho<=.8, frozen stability classification and eligibility unchanged; formal proxy pass is not native stability proof.',
        routing='Original expanded-alias deterministic SWRR, text .925/.925. Per-alias exact finite input exposure recorded; no ideal fractional or random routing substitute.',
        quality='No absolute quality admission. Actual concurrent native outputs must be rescored; historical shortlist quality is not native quality proof.',
        constraints=constraints,proxy_only=True,actual_resource_preflight_required_before_GPU=True,
        actual_resource_preflight='Same original v5 hardware/checkpoint/quality identity/resource checks as fine4 and strong2; no new loading-bracket or runtime threshold is retroactively imposed.',
        scheduling='PREPARE_ONLY_NOT_QUEUED. Current fine4 plus strong2 remain unchanged. Consider these two only after those runs, if their information is still needed; do not modify the frozen supervisor.',
        no_gpu_started=True,no_auto_execution=True,no_other_challengers_enqueued=True,no_hidden_retry=True,all_attempts_retained=True,
        all_utility_terms_unchanged=True,new_profiles_collected=False,predictor_activated=False,full_pool_optimum=False,
        production_U=None,physical_feasibility='UNKNOWN',reserved_results_read=False,
        interpretation='Jointly changes text replica count and spillover share. A utility advantage challenges necessity of text2 at this workload; it does not isolate a pure replica-count causal effect or prove a global optimum.')
    destination.mkdir(parents=True,exist_ok=False)
    write(destination/'system-config.json',config_data);write(destination/'discovery-protocol.json',protocol)
    write(destination/'cohort-check.json',cohort)
    bindings=static_study._bindings(original['inputs'],SOURCE)
    bindings += [static_study._file(p) for p in (original_path,fine_path,strong_path,screen_path,review_path,Path(__file__).resolve(),
        destination/'system-config.json',destination/'discovery-protocol.json',destination/'cohort-check.json')]
    bindings=sorted({b['path']:b for b in bindings}.values(),key=lambda b:b['path'])
    jobs=[]
    for seed in (107,108):
        label=f'text185_text1_g2half_seed{seed}'
        job=static_study._job(config,original['workers'],plan,label,'calibration',seed,bindings,30)
        job.update(purpose='text185_single_text_specialist_strong_mixed_control_development',proxy_constraints=constraints)
        if job['plan']!=plan:raise ValueError('Generated job differs')
        jobs.append(job)
    packet=dict(schema='gsdeploy.static_experiments.v1',status='PREDECLARED_SINGLE_SPECIALIST_MIXED_CONTROL_DEVELOPMENT',
        inputs=original['inputs'],config=config_data,workers=original['workers'],bindings=bindings,
        frozen_files_sha256=quality_digest(bindings),jobs=jobs,gpu_started=False,
        discovery_protocol=str(destination/'discovery-protocol.json'),cohort_check=str(destination/'cohort-check.json'),
        execution=dict(policy=protocol['scheduling'],runtime_source=str(SOURCE),scheduled=False,auto_execute=False,
            modifies_frozen_supervisor=False))
    validation=static_study.validate_packet(packet)
    if not validation['valid']:raise ValueError(str(validation))
    write(destination/'experiment-packet.json',packet)
    write(destination/'preparation-receipt.json',dict(packet=static_study._file(destination/'experiment-packet.json'),
        validation=validation,source=static_study._file(Path(static_study.__file__)),jobs=2,seeds=[107,108],GPU_started=False,
        gpu_queried=False,scheduled=False,cohort_check=static_study._file(destination/'cohort-check.json'),
        all_paired_cohorts_match=True,actual_resource_preflight_required_before_GPU=True,
        production_U=None,physical_feasibility='UNKNOWN',predictor_activated=False,reserved_results_read=False))
    summaries=[dict(seed=s['seed'],phases=[{k:p[k] for k in ('phase','request_count','task_counts','trace_sha256',
        'route_independent_cohort_sha256','alias_task_counts','alias_unique_inputs')} for p in s['phases']]) for s in cohort['seeds']]
    text='''# 单文本专用副本的强混合对照：仅 CPU 准备

部署 text×1 + generalist_2×1 + OCR×1；text 1.85 rps 各半路由 .925/.925，image/asr 各 .01 到 G2，OCR .01 到 OCR。目标是以真实混合竞争方案检验当前 text×2 是否必要，不能只用纯专或纯通作对照。

沿用 fine4 和强专2 完全相同的冻结 v5、600秒 measurement、30秒 warmup、Poisson、300秒 drain、seed107/108、输入池、SWRR、六项效用和稳定性规则。原正式 proxy max rho=0.5281511663220126，safe planning memory=22265.957576 MiB；这不证明混合实际容量、资源、稳定性或 U。

两 seed 的 warmup/measurement 路由无关 task/time/input cohort 与 fine4、强专2 逐请求一致，SHA、完整 roster、各 alias 请求量与实际 unique input 覆盖见 cohort-check.json。原生 routing 会改变 alias 和输入分配；保留确定性 SWRR，不用理想分数或随机路由替换。priming 沿用 largest_three_per_task，驻留部署改变后 priming 数随之改变，不宣称 priming cohort 相同。

本对照同时改变文本副本数和分流份额，能挑战第二个文本副本在此负载下的必要性，但不能单独归因于副本数量。真实质量、延迟、峰值显存与能耗可能改变排名，需要原生结果；当前 U=null、physical feasibility UNKNOWN，无新画像、模型激活或全池最优声明。

**未排 GPU。** 当前 fine4+强专2 及冻结 supervisor 不变；完成后按信息需要另行决定是否收集本对照。未查询 GPU，未加载模型，未读取保留种子结果。来源、协议、cohort 及新 prepare 工具已绑定到 packet。
'''
    with (destination/'README.zh.md').open('x') as stream:stream.write(text)
    print(json.dumps(dict(destination=str(destination),packet=static_study._file(destination/'experiment-packet.json'),
        validation=validation,cohort_summaries=summaries,scheduled=False,GPU_started=False),ensure_ascii=False))


if __name__=='__main__':prepare()
