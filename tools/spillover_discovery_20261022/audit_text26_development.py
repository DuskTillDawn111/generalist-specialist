#!/usr/bin/env python3
"""Frozen-v5 CPU audit of four text2.6 development plans, including failures."""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'artifacts/static-spillover-study-20261022'
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0,str(SOURCE))
from gsdeploy.calibration import _import_source
from gsdeploy.config import SystemConfig
from gsdeploy.evaluation import evaluate_run
from gsdeploy.quality import quality_digest, measure_quality, sample_totals
from gsdeploy.stability import DEFAULT_THRESHOLDS
from gsdeploy.static_study import validate_packet
from gsdeploy.tasks import score_result, page_ocr_text
from gsdeploy.trace import make_trace, trace_digest, expand_routes
from gsdeploy.routing import Router

RATES=dict(text=2.6,image=.01,ocr=.01,asr=.01)
ROSTER=(('text26_n2_specialist_share_4of7',2,4/7),
        ('text26_n2_specialist_share_1of2',2,.5),
        ('text26_n1_specialist_share_1of2',1,.5),
        ('text26_n3_specialist_share_4of7',3,4/7))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def receipt(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def attempt_directories(runs,label):
    prefix=label+'-attempt'
    return sorted(p for p in runs.glob(prefix+'[0-9]*')
        if p.is_dir() and p.name[len(prefix):].isdigit())
def stats(values):
    values=[float(v) for v in values if type(v) in (int,float) and math.isfinite(v)]
    return {'count':len(values),'mean_ms':float(np.mean(values)) if values else None,
            'p95_ms':float(np.percentile(values,95)) if values else None}


def verify_roster(packet):
    checked=validate_packet(packet)
    if not checked['valid']:raise ValueError('Frozen packet bindings invalid: '+repr(checked))
    if len(packet['jobs'])!=4:raise ValueError('Expected four predeclared jobs')
    for job,(label,n,share) in zip(packet['jobs'],ROSTER):
        routing=dict(text={'text':2.6*share,'generalist_2':2.6*(1-share)},
            image={'generalist_2':.01},asr={'generalist_2':.01},ocr={'ocr':.01})
        plan={'deployment':{'text':n,'generalist_2':1,'ocr':1},'routing':routing}
        if (job['labels']!=[label] or job['phase']!='calibration' or job['seed']!=51
                or job['plan']!=plan or job['identity']['workload']!=RATES
                or job['identity']['horizon_seconds']!=600
                or not job.get('purpose','').startswith('text26_exploratory_development_')):
            raise ValueError('Four-plan development roster changed')
    c=packet['config']
    if c['workload']!=RATES or c['horizon_seconds']!=600 or c['random_seed']!=51:
        raise ValueError('Text2.6 config changed')
    return checked


def identity_checks(raw,job,packet):
    """Failure-path provenance only, with no success/stability override."""
    expected=job['identity'];actual=raw.get('executed_measurement_identity',{})
    checks={
        'prepared_identity_digest':quality_digest(expected)==job['id'],
        'executed_identity_digest':quality_digest(actual)==raw.get('executed_measurement_id'),
        'job_id':raw.get('experiment_job_id')==job['id'],
        'phase':raw.get('experiment_phase')=='calibration',
        'native_backend':raw.get('backend')=='native',
        'plan':raw.get('plan')==job['plan'],
        'identity_fields':all(actual.get(k)==v for k,v in expected.items() if k not in {'hardware_contract','workers'}),
        'hardware_contract':{k:v for k,v in actual.get('hardware_contract',{}).items() if k!='device_uuid'}=={k:v for k,v in expected['hardware_contract'].items() if k!='device_uuid'},
    }
    worker_checks={}
    for model,worker in expected['workers'].items():
        value=copy.deepcopy(actual.get('workers',{}).get(model,{}))
        value.get('env',{}).pop('GS_CHECKPOINT_FILES_JSON',None)
        worker_checks[model]=value==worker
    checks['workers']=all(worker_checks.values())
    config=raw.get('config',{});transport=raw.get('transport_config',{})
    checks['config_fields']=all(config.get(k)==packet['config'].get(k) for k in (
        'workload','weights','latency_reference_ms','gpu_memory_mib','horizon_seconds','random_seed',
        'arrival_process','drain_seconds','energy_reference_watts','startup_amortization_seconds','utility_schema'))
    checks['frozen_catalog_fields']=all(config.get('catalog',{}).get(k)==packet['config']['catalog'].get(k)
        for k in ('models','quality','safe_factor','capacity_factors','memory_table_mib','task_replica_profiles'))
    checks['transport']=all(transport.get(k)==expected[v] for k,v in (
        ('measurement_seconds','horizon_seconds'),('random_seed','seed'),('arrival_process','arrival_process'),
        ('warmup_seconds','warmup_seconds'),('drain_timeout_seconds','drain_seconds')))
    checks['priming']=raw.get('priming_policy')==expected['warmup_policy'] and raw.get('priming_size_metric')==expected['warmup_size_metric']
    preflight=raw.get('preflight',{});gpu=preflight.get('gpu',{});context=config.get('catalog',{}).get('measurement_context',{})
    checks['checkpoint_context']=actual.get('runtime_checkpoint_sha256')==context.get('runtime_checkpoint_sha256')
    checks['GPU_context']=actual.get('hardware_contract',{}).get('device_uuid')==gpu.get('uuid')==context.get('gpu_uuid')
    devices={w.get('env',{}).get('CUDA_VISIBLE_DEVICES','0') for w in expected['workers'].values()}
    checks['single_GPU_preflight']=(preflight.get('ready') is True and preflight.get('gpu_checked') is True
        and actual.get('hardware_contract',{}).get('single_gpu') is True
        and isinstance(preflight.get('device'),str) and ',' not in preflight['device']
        and preflight['device'] not in {'','-1'} and devices=={preflight['device']}
        and isinstance(gpu.get('memory_mib'),(int,float)) and gpu['memory_mib']>=config.get('gpu_memory_mib',float('inf')))
    window=raw.get('measurement_window',{});records=raw.get('request_records',[])
    checks['declared_trace_hash']=window.get('trace_sha256')==expected['trace_sha256']
    try:
        trace=make_trace(RATES,600,51,packet['config']['arrival_process'])
        aliases,routes=expand_routes(job['plan']['deployment'],job['plan']['routing']);router=Router(routes)
        checks['cohort_and_routing']=(len(records)==len(trace) and trace_digest(trace)==window.get('trace_sha256')
            and all(r.get('task')==task and r.get('request_id')==f'measurement:{i}'
                and r.get('model')==router.choose(task)
                and r.get('scheduled_time_ns')==window['start_time_ns']+round(offset*1e9)
                for i,((offset,task),r) in enumerate(zip(trace,records))))
    except (KeyError,TypeError,ValueError):checks['cohort_and_routing']=False
    return {'verified':all(checks.values()),'checks':checks,'worker_checks':worker_checks,
            'scope':'Identity/provenance checks only; failure-path audit never grants utility eligibility'}


def request_quality(raw,record):
    if record.get('status')!='completed':return {'status':'UNKNOWN_INCOMPLETE'}
    response=record.get('worker_response',{})
    if response.get('success') is not True:return {'status':'UNKNOWN_UNSUCCESSFUL_OUTPUT'}
    try:
        task=record['task'];sample=raw['config']['catalog']['quality']['references'][task][record['pool_id']]
        result=dict(response);result['task']=response.get('task',response.get('task_type'))
        if task=='ocr':result['prediction']=page_ocr_text(result['prediction'])
        numerator,denominator=sample_totals(task,score_result(sample,result))
        return {'status':'MEASURED_OUTPUT','error_numerator':numerator,'reference_denominator':denominator,
                'loss':numerator/denominator}
    except (KeyError,TypeError,ValueError) as exc:return {'status':'UNKNOWN','error':str(exc)}


def analyze(path,job,packet,packet_path,out):
    raw=json.loads(path.read_text());source={'id':job['id'],**receipt(path),
        'packet_path':str(packet_path.resolve()),'packet_sha256':sha(packet_path)}
    identity=identity_checks(raw,job,packet)
    strict=None;strict_error=None
    try:
        imported,utility,strict=_import_source(source,{job['identity']['runtime_code_sha256']})
        if imported!=raw:raise ValueError('Strict imported raw differs')
    except (ValueError,KeyError,TypeError) as exc:
        strict_error={'type':type(exc).__name__,'message':str(exc)}
    try:evaluation=evaluate_run(SystemConfig(**raw['config']),raw)
    except (ValueError,KeyError,TypeError) as exc:
        evaluation={'utility':{'U':None,'eligible':False},'evaluation_error':{'type':type(exc).__name__,'message':str(exc)}}
    utility=dict(evaluation['utility']);stability=evaluation.get('stability',{})
    allowed=(strict is not None and identity['verified'] and utility.get('eligible') is True
        and stability.get('stability_state_v2') in {'steady','recoverable_congested'})
    if not allowed:utility['U']=None;utility['eligible']=False
    measured_Q,quality=measure_quality(raw['config']['catalog'],raw,RATES)
    records=raw.get('request_records',[]);summary={};request_rows=[]
    for record in records:
        response=record.get('worker_response',{});q=request_quality(raw,record)
        request_rows.append({'request_id':record.get('request_id'),'task':record.get('task'),
            'alias':record.get('model'),'family':raw.get('replica_to_family',{}).get(record.get('model')),
            'sample_id':record.get('sample_id',record.get('pool_id')),'status':record.get('status'),
            **{k:record.get(k) for k in ('scheduled_offset_seconds','scheduled_time_ns','actual_submit_time_ns',
                'completion_time_ns','worker_latency_ms','worker_queue_wait_ms','e2e_latency_ms','schedule_delay_ms')},
            'worker_service_ms':response.get('service_ms'),'worker_start_unix_ns':response.get('start_ns'),
            'worker_end_unix_ns':response.get('end_ns'),'output_success':response.get('success'),
            'generated_tokens':response.get('generated_tokens'),'request_sha256':response.get('request_sha256'),
            'prediction':json.dumps(response.get('prediction'),ensure_ascii=False),
            'quality_status':q['status'],'quality_error_numerator':q.get('error_numerator'),
            'quality_reference_denominator':q.get('reference_denominator'),'quality_loss':q.get('loss')})
    for grouping in ('task','path','alias'):
        keys=sorted({r['task'] if grouping=='task' else r['model'] if grouping=='alias'
            else raw['replica_to_family'][r['model']]+'/'+r['task'] for r in records})
        summary[grouping]={}
        for key in keys:
            subset=[r for r in records if (r['task'] if grouping=='task' else r['model'] if grouping=='alias'
                else raw['replica_to_family'][r['model']]+'/'+r['task'])==key]
            summary[grouping][key]={'offered':len(subset),'completed':sum(r.get('status')=='completed' for r in subset),
                'failed_or_incomplete':sum(r.get('status')!='completed' for r in subset),
                **{k:stats([r.get(k) for r in subset]) for k in ('worker_latency_ms','worker_queue_wait_ms','e2e_latency_ms','schedule_delay_ms')},
                'worker_service_ms':stats([r.get('worker_response',{}).get('service_ms') for r in subset])}
    name=job['labels'][0]+'__'+path.parent.name
    csv_path=out/(name+'__requests.csv')
    if request_rows:
        with csv_path.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(request_rows[0]));writer.writeheader();writer.writerows(request_rows)
    result={'id':name,'label':job['labels'][0],'attempt':path.parent.name,'source':source,
        'status':raw.get('status'),'raw_errors':raw.get('errors',[]),'identity':identity,
        'strict_import_passed':strict is not None,'strict_import_receipt':strict,'strict_import_error':strict_error,
        'plan':raw.get('plan'),'workload':RATES,'seed':51,'horizon_seconds':600,
        'stability':stability,'stability_evidence':evaluation.get('stability_evidence'),
        'utility':utility,'utility_comparison_allowed':allowed,'evaluation_error':evaluation.get('evaluation_error'),
        'actual_output_quality':{'Q_norm':measured_Q,'evidence':quality},'request_summaries':summary,
        'request_csv':receipt(csv_path) if request_rows else None,'measurement_window':raw.get('measurement_window'),
        'memory_evidence':{k:v for k,v in raw.get('memory_evidence',{}).items() if k!='samples'},
        'energy_evidence':{k:v for k,v in raw.get('energy_evidence',{}).items() if k!='samples'},
        'diagnostic_U_ranking_forbidden':not allowed,'failed_attempt_preserved':not allowed}
    return result,raw


def comparisons(results,raws):
    output=[]
    for a,b in itertools.combinations(results,2):
        ra,rb=raws[a['id']],raws[b['id']]
        aa,bb=ra.get('request_records',[]),rb.get('request_records',[])
        same=(len(aa)==len(bb)>0 and all((x.get('task'),x.get('sample_id',x.get('pool_id')),x.get('scheduled_offset_seconds'))
            ==(y.get('task'),y.get('sample_id',y.get('pool_id')),y.get('scheduled_offset_seconds')) for x,y in zip(aa,bb)))
        allowed=a['utility_comparison_allowed'] and b['utility_comparison_allowed'] and same
        delta=b['utility']['U']-a['utility']['U'] if allowed else None
        output.append({'a':a['id'],'b':b['id'],'direction':'b minus a','same_ordered_task_sample_schedule':same,
            'same_trace_sha256':ra.get('measurement_window',{}).get('trace_sha256')==rb.get('measurement_window',{}).get('trace_sha256'),
            'utility_comparison_allowed':allowed,'delta_U':delta,
            'delta_six_contributions':{k:b['utility'].get(k+'_contribution')-a['utility'].get(k+'_contribution') for k in 'QTLPME'} if allowed else None,
            'reason':'Valid native same-trace development comparison' if allowed else 'UNKNOWN: invalid utility, instability, identity failure or unmatched cohort; no diagnostic-U comparison',
            'scope':'Limited predeclared four-plan development comparison only; not full-pool optimum'})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development',type=Path,default=STUDY/'text26-development-v1')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if not args.check_only and args.output is None:parser.error('--output required unless --check-only')
    if any('validation' in p or 'confirmation' in p for p in args.development.parts):raise ValueError('Holdout directory forbidden')
    packet_path=args.development/'experiment-packet.json';missing=[];attempts={}
    if not packet_path.is_file():missing.append(str(packet_path.resolve()))
    for label,n,s in ROSTER:
        folders=attempt_directories(args.development/'runs',label)
        attempts[label]=[p/'run.json' for p in folders if (p/'run.json').is_file()]
        if not folders:missing.append(str((args.development/'runs'/(label+'-attempt1')/'run.json').resolve()))
        missing.extend(str((p/'run.json').resolve()) for p in folders if not (p/'run.json').is_file())
    readiness={'schema':'text26_development_audit.v1','status':'INCOMPLETE' if missing else 'FILES_PRESENT_NOT_AUDITED',
        'missing_files':missing,'existing_attempt_files':{k:[str(p.resolve()) for p in v] for k,v in attempts.items()},
        'raw_contents_read':False,'gpu_started':False,'production_math_changed':False,
        'validation_results_read':False,'confirmation_results_read':False}
    if args.check_only:print(json.dumps(readiness,ensure_ascii=False,indent=2));return 2 if missing else 0
    args.output.mkdir(parents=True,exist_ok=False)
    report=readiness;code=2 if missing else 0
    if packet_path.is_file():
        try:
            assert Path(sys.modules['gsdeploy.calibration'].__file__).resolve().is_relative_to(SOURCE)
            packet=json.loads(packet_path.read_text());validation=verify_roster(packet)
            before={str(p):sha(p) for p in SOURCE.rglob('*.py')}
            results=[];raws={}
            for job in packet['jobs']:
                for path in attempts[job['labels'][0]]:
                    try:
                        result,raw=analyze(path,job,packet,packet_path,args.output)
                        results.append(result);raws[result['id']]=raw
                    except Exception as exc:
                        results.append({'id':job['labels'][0]+'__'+path.parent.name,'label':job['labels'][0],
                            'source':receipt(path),'status':'FAILED_ATTEMPT_AUDIT','utility':{'U':None,'eligible':False},
                            'utility_comparison_allowed':False,'strict_import_passed':False,'stability':{},
                            'error_type':type(exc).__name__,'error':str(exc),'failed_attempt_preserved':True})
            report={'schema':'text26_development_audit.v1','status':'INCOMPLETE' if missing else 'ALL_PREDECLARED_ATTEMPTS_AUDITED',
                'scope':'Four concrete native development plans; no full-pool optimum or independent validation claim',
                'packet':receipt(packet_path),'packet_validation':validation,'runtime':str(SOURCE),
                'stability_thresholds':DEFAULT_THRESHOLDS,'stability_code':receipt(SOURCE/'gsdeploy/stability.py'),
                'conditions':results,'same_trace_comparisons':comparisons([r for r in results if r['id'] in raws],raws),
                'missing_files':missing,
                'valid_utility_attempts':sum(r['utility_comparison_allowed'] for r in results),
                'invalid_attempts_preserved':sum(not r['utility_comparison_allowed'] for r in results),
                'production_math_changed':False,'raw_contents_read':True,'gpu_started':False,
                'validation_results_read':False,'confirmation_results_read':False,
                'frozen_sources_unchanged':all(sha(p)==s for p,s in before.items()),
                'full_pool_optimality_claim':False,'diagnostic_U_used_for_ranking':False}
            assert report['frozen_sources_unchanged']
        except Exception as exc:
            report={**readiness,'status':'FAILED_PACKET_OR_AUDIT_CONTRACT','error_type':type(exc).__name__,'error':str(exc),'raw_contents_read':True};code=3
    save(args.output/'audit.json',report)
    lines=['# Text2.6 四计划原生开发审计','',f"状态：{report['status']}",'']
    if 'conditions' in report:
        lines+=['text2.6 rps，image/OCR/ASR背景各.01，seed51/H600；原稳定性阈值和冻结v5评价保持。所有尝试保留。','',
            '| 计划/尝试 | 状态 | 稳定性 | 严格导入 | 可比较U |','|---|---|---|---|---:|']
        for r in report['conditions']:lines.append(f"| {r['id']} | {r['status']} | {r['stability'].get('stability_state_v2','UNKNOWN')} | {r['strict_import_passed']} | {r['utility']['U']} |")
        lines+=['','仅两个条件均严格有效、通过原稳定性评价且同输入trace时，才比较正式实测U。失败/稳定性不通过的U为空；其诊断算术不得参与排名。逐请求CSV包含实际模型输出、真实质量分子/分母、服务/等待/端到端；任务/路径/别名的均值和p95及完整六项见audit.json。','',
            '此处只检验n2份额4/7、n2份额1/2、n1份额1/2和n3份额4/7四个具体计划；不能称全池最优。','']
    else:lines+=['未完成全部输入或合同失败；缺失列表/错误见audit.json，不补造结果。','']
    (args.output/'README.zh.md').write_text('\n'.join(lines))
    save(args.output/'manifest.json',{'script':receipt(__file__),'files':{p.name:sha(p) for p in args.output.iterdir() if p.is_file()},'gpu_started':False})
    print(json.dumps({'status':report['status'],'output':str(args.output.resolve()),'missing_files':missing},ensure_ascii=False))
    return code


if __name__=='__main__':sys.exit(main())
