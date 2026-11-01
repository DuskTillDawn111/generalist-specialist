#!/usr/bin/env python3
"""Retrospective finite-cohort phase diagnosis; no fit, U, rank or causal claim."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('_joint_capacity_analysis',Path(__file__).with_name('analyze.py'))
a=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(a)


def phase(start,end,first,last):
    if end<=first:return 'before_first_specialist_handler_finish'
    if start>=last:return 'after_both_specialist_handlers_finish'
    if start>=first and end<=last:return 'between_specialist_handler_finishes'
    return 'crosses_specialist_finish_boundary'


def finite_number(value):return type(value) in (int,float) and math.isfinite(value)


def row_summary(rows):
    return dict(count=len(rows),unique_inputs=len({r['pool_id'] for r in rows}),
        native_service_ms=a.stats([r['service_ms'] for r in rows]),
        service_sum_seconds=math.fsum(r['service_ms'] for r in rows)/1000,
        handler_ms=a.stats([r['handler_ms'] for r in rows]),
        handler_minus_service_ms=a.stats([r['handler_ms']-r['service_ms'] for r in rows]),
        paired_same_input_repeat_C_over_B=a.stats([r['paired_service_ratio'] for r in rows]),
        generated_tokens=a.stats([r['generated_tokens'] for r in rows]),
        input_tokens=a.stats([r['input_tokens'] for r in rows]),
        exact_prediction_matches=sum(r['prediction_matches_B'] for r in rows),
        generated_token_matches=sum(r['generated_tokens_match_B'] for r in rows),
        input_token_matches=sum(r['input_tokens_match_B'] for r in rows),
        cache_performed=sum(r['cache_reclamation'].get('performed') is True for r in rows),
        B_cache_performed=sum(r['B_cache_reclamation'].get('performed') is True for r in rows),
        cache_before_mib=a.stats([r['cache_reclamation'].get('unused_reserved_before_mib') for r in rows]),
        cache_after_mib=a.stats([r['cache_reclamation'].get('unused_reserved_after_mib') for r in rows]),
        B_cache_before_mib=a.stats([r['B_cache_reclamation'].get('unused_reserved_before_mib') for r in rows]),
        B_cache_after_mib=a.stats([r['B_cache_reclamation'].get('unused_reserved_after_mib') for r in rows]),
        all_active_fraction=a.stats([r['activity']['all_active_aliases_busy_fraction'] for r in rows]),
        any_other_fraction=a.stats([r['activity']['any_other_alias_busy_fraction'] for r in rows]),
        quality_native_accuracy=a.stats([1-r['quality']['loss'] for r in rows if r['quality']['status']=='AVAILABLE']))


def paired_row(row,reference,start,first,last):
    response=row['worker_response'];base=reference['worker_response']
    s,e=row['worker_start_unix_ns'],row['worker_end_unix_ns']
    host=row['original_record'];hostref=reference['original_record']
    return dict(request_id=row['request_id'],pool_id=row['pool_id'],alias=row['model'],sequence=row['sequence'],repetition=row['repetition'],
        start_elapsed_seconds=(s-start)/1e9,end_elapsed_seconds=(e-start)/1e9,
        start_unix_ns=s,end_unix_ns=e,phase=phase(s,e,first,last),service_ms=row['service_ms'],handler_ms=row['handler_ms'],
        B_service_ms=reference['service_ms'],B_handler_ms=reference['handler_ms'],
        paired_service_ratio=row['service_ms']/reference['service_ms'],
        generated_tokens=response.get('generated_tokens'),B_generated_tokens=base.get('generated_tokens'),
        input_tokens=response.get('input_tokens'),B_input_tokens=base.get('input_tokens'),
        generated_tokens_match_B=response.get('generated_tokens')==base.get('generated_tokens'),
        input_tokens_match_B=response.get('input_tokens')==base.get('input_tokens'),
        prediction_matches_B=response.get('prediction')==base.get('prediction'),
        prediction_sha256=a.digest(response.get('prediction')),B_prediction_sha256=a.digest(base.get('prediction')),
        raw_output_matches_B=response.get('raw_output')==base.get('raw_output'),
        native_success=response.get('success'),B_native_success=base.get('success'),
        quality=row['quality'],B_quality=reference['quality'],cache_reclamation=response.get('cache_reclamation') or {},
        B_cache_reclamation=base.get('cache_reclamation') or {},activity=row['activity'],
        original_record_sha256=row['original_record_sha256'],B_original_record_sha256=reference['original_record_sha256'],
        ledger=row['ledger'],B_ledger=reference['ledger'],
        host_completion_time_ns=host.get('completion_time_ns'),host_submit_time_ns=host.get('actual_submit_time_ns'),
        B_host_completion_time_ns=hostref.get('completion_time_ns'))


def diagnose_block(C,B,A):
    rows=C['request_diagnostics'];g2='generalist_2#0';specialists=['text#0','text#1']
    if any(r['service_status']!='AVAILABLE' for r in rows):raise ValueError('This complete phase diagnosis requires all C service rows')
    start=C['finite_cohort_timing']['measurement_window']['start_unix_ns']
    finishes={alias:max(r['worker_end_unix_ns'] for r in rows if r['model']==alias) for alias in specialists}
    first=min(finishes.values());last=max(finishes.values())
    reference={(r['model'],r['pool_id'],r['repetition']):r for r in B['request_diagnostics']}
    paired=[]
    for row in rows:
        if row['model']!=g2:continue
        key=(g2,row['pool_id'],row['repetition']);base=reference[key]
        if base['sequence']!=row['sequence'] or base['service_status']!='AVAILABLE':raise ValueError('Same input/repeat/sequence B pair differs')
        paired.append(paired_row(row,base,start,first,last))
    by_phase={p:row_summary([r for r in paired if r['phase']==p]) for p in sorted({r['phase'] for r in paired})}
    by_overlap={label:row_summary([r for r in paired if (r['activity']['all_active_aliases_busy_ns']>0)==positive])
        for label,positive in [('positive_all_active_overlap',True),('zero_all_active_overlap',False)]}
    by_repeat={str(k):dict(**row_summary([r for r in paired if r['repetition']==k]),
        phase_counts=dict(Counter(r['phase'] for r in paired if r['repetition']==k))) for k in range(3)}
    inputs=[]
    for pool_id in sorted({r['pool_id'] for r in paired}):
        seq=sorted([r for r in paired if r['pool_id']==pool_id],key=lambda r:(r['service_ms'],r['repetition']))
        if len(seq)!=3:raise ValueError('Expected complete three repeats')
        median=seq[1];tail=[r for r in seq if r['phase']=='after_both_specialist_handlers_finish']
        common=[r for r in seq if r['activity']['all_active_aliases_busy_ns']>0]
        full_common=[r for r in seq if r['activity']['all_active_aliases_busy_fraction']>=.95]
        bmedian=sorted(r['B_service_ms'] for r in seq)[1]
        inputs.append(dict(pool_id=pool_id,all_three_median_service_ms=median['service_ms'],B_three_median_service_ms=bmedian,
            median_C_over_B=median['service_ms']/bmedian,median_source_repetition=median['repetition'],median_source_phase=median['phase'],
            median_source_all_active_fraction=median['activity']['all_active_aliases_busy_fraction'],
            tail_observations=len(tail),positive_all_active_observations=len(common),at_least_95pct_all_active_observations=len(full_common),
            repetition0_C_over_B=next(r['paired_service_ratio'] for r in seq if r['repetition']==0),
            common_only_service_ms=a.stats([r['service_ms'] for r in common]),tail_only_service_ms=a.stats([r['service_ms'] for r in tail]),
            ordered_by_service=[{k:r[k] for k in ['repetition','service_ms','B_service_ms','paired_service_ratio','phase','start_elapsed_seconds','end_elapsed_seconds','activity','generated_tokens','prediction_matches_B','cache_reclamation']} for r in seq]))
    coverage=[]
    for alias in C['aliases']:
        for pool_id in sorted({r['pool_id'] for r in rows if r['model']==alias}):
            seq=[r for r in rows if r['model']==alias and r['pool_id']==pool_id]
            coverage.append(dict(alias=alias,pool_id=pool_id,total=len(seq),
                positive_all_active=sum(r['activity']['all_active_aliases_busy_ns']>0 for r in seq),
                at_least_95pct_all_active=sum(r['activity']['all_active_aliases_busy_fraction']>=.95 for r in seq),
                fractions=[dict(repetition=r['repetition'],all_active=r['activity']['all_active_aliases_busy_fraction'],any_other=r['activity']['any_other_alias_busy_fraction']) for r in seq]))
    # Divide each observed handler interval around the last specialist finish,
    # retaining a boundary-crossing request intact in its own descriptive group.
    before_work=sum(max(0,min(r['end_unix_ns'],last)-r['start_unix_ns']) for r in paired)/1e9
    after_work=sum(max(0,r['end_unix_ns']-max(r['start_unix_ns'],last)) for r in paired)/1e9
    before_completed=sum(r['end_unix_ns']<=last for r in paired)
    tail_begin=next((r for r in paired if r['start_unix_ns']>=last),None)
    adjacent=[]
    for before,after in zip(paired,paired[1:]):
        adjacent.append(dict(before_sequence=before['sequence'],after_sequence=after['sequence'],
            before_end_seconds=before['end_elapsed_seconds'],after_start_seconds=after['start_elapsed_seconds'],
            service_after_over_before=after['service_ms']/before['service_ms'],
            phase_before=before['phase'],phase_after=after['phase'],same_input=before['pool_id']==after['pool_id']))
    specialist_pairs=[]
    aindex={(r['model'],r['pool_id'],r['repetition']):r for r in A['request_diagnostics']}
    for alias in specialists:
        seq=[r for r in rows if r['model']==alias]
        ratios=[r['service_ms']/aindex[(alias,r['pool_id'],r['repetition'])]['service_ms'] for r in seq]
        specialist_pairs.append(dict(alias=alias,C_over_A_same_input_repeat=a.stats(ratios),
            prediction_matches=sum(r['worker_response'].get('prediction')==aindex[(alias,r['pool_id'],r['repetition'])]['worker_response'].get('prediction') for r in seq)))
    return dict(block=C['block'],C_job=C['job_id'],B_job=B['job_id'],A_job=A['job_id'],
        measurement_start_unix_ns=start,specialist_handler_finish_elapsed_seconds={k:(v-start)/1e9 for k,v in finishes.items()},
        specialist_host_cohort_completion_seconds={k:C['finite_cohort_timing']['aliases'][k]['release_to_alias_completion_seconds'] for k in specialists},
        G2_host_cohort_completion_seconds=C['finite_cohort_timing']['aliases'][g2]['release_to_alias_completion_seconds'],
        B_G2_host_cohort_completion_seconds=B['finite_cohort_timing']['aliases'][g2]['release_to_alias_completion_seconds'],
        C_G2_service_sum_seconds=sum(r['service_ms'] for r in paired)/1000,
        B_G2_service_sum_seconds=sum(r['B_service_ms'] for r in paired)/1000,
        G2_observed_handler_seconds_before_last_specialist_finish=before_work,
        G2_observed_handler_seconds_after_last_specialist_finish=after_work,
        G2_requests_completed_before_last_specialist_finish=before_completed,
        first_fully_tail_request={k:tail_begin[k] for k in ['sequence','repetition','start_elapsed_seconds','service_ms']} if tail_begin else None,
        all_G2=row_summary(paired),by_phase=by_phase,by_overlap=by_overlap,by_repetition=by_repeat,
        per_input_median_provenance=inputs,median_provenance_summary=dict(
            source_phases=dict(Counter(r['median_source_phase'] for r in inputs)),
            source_repetitions=dict(Counter(r['median_source_repetition'] for r in inputs)),
            tail_observation_counts=dict(Counter(r['tail_observations'] for r in inputs)),
            positive_common_observation_counts=dict(Counter(r['positive_all_active_observations'] for r in inputs)),
            at_least95pct_common_observation_counts=dict(Counter(r['at_least_95pct_all_active_observations'] for r in inputs)),
            median_C_over_B=a.stats([r['median_C_over_B'] for r in inputs])),
        per_input_activity_coverage=coverage,activity_coverage_by_alias={alias:dict(
            total_inputs=sum(r['alias']==alias for r in coverage),
            positive_common_counts=dict(Counter(r['positive_all_active'] for r in coverage if r['alias']==alias)),
            at_least95pct_common_counts=dict(Counter(r['at_least_95pct_all_active'] for r in coverage if r['alias']==alias))) for alias in C['aliases']},
        specialist_same_input_repeat_comparisons=specialist_pairs,
        G2_request_pairs=paired,top_G2_requests_by_service=sorted(paired,key=lambda r:r['service_ms'],reverse=True)[:10],
        adjacent_G2_timing=adjacent,U=None,diagnostic_only=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',required=True,type=Path)
    p.add_argument('--queue-progress',required=True,type=Path);p.add_argument('--analysis',required=True,type=Path)
    p.add_argument('--audit',type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError('New diagnostic directory required')
    packet,config,workers,refs,scorer,total,metadata=a.inspect_packet(args.packet)
    queue_receipt=a.file_receipt(args.queue_progress);waiting,registered=a.registered_paths(packet,[args.queue_progress],metadata['packet']['sha256'])
    if waiting or len(registered)!=6:raise ValueError('All six registered terminal receipts required')
    manifest_path=args.analysis/'manifest.json';manifest=json.loads(manifest_path.read_text());receipts=[metadata['packet'],queue_receipt,a.file_receipt(manifest_path),a.file_receipt(__file__),a.file_receipt(Path(a.__file__))]
    for name,sha in manifest['files'].items():
        item=a.file_receipt(args.analysis/name)
        if item['sha256']!=sha:raise ValueError('Bound analysis file changed: '+name)
        receipts.append(item)
    report=json.loads((args.analysis/'report.json').read_text())
    if report['status']!='ALL_SIX_DEVELOPMENT_DIAGNOSTICS_COMPLETE' or report['U'] is not None:raise ValueError('Expected complete diagnostic report')
    audit=None
    if args.audit:
        audit=json.loads(args.audit.read_text());receipts.append(a.file_receipt(args.audit))
        if audit.get('status')!='PASS' or audit.get('terminal_raw_files_read')!=6:raise ValueError('Expected corrected all-six PASS audit')
    details={};raw_evidence=[]
    for job,registration,path in registered:
        raw,receipt=a.read_final(path,registration['raw']['sha256']);receipts.append(receipt)
        detail=json.loads((args.analysis/(job['job_id']+'.json')).read_text())
        if detail['raw']['sha256']!=receipt['sha256'] or raw['job']!=job:raise ValueError('Analysis/raw identity differs')
        index={r['request_id']:r for r in raw['request_records']}
        for row in detail['request_diagnostics']:
            if a.digest(index[row['request_id']])!=row['original_record_sha256'] or index[row['request_id']]!=row['original_record']:raise ValueError('Analysis request differs from registered raw')
        if not detail['runtime_execution_identity']['verified']:raise ValueError('Runtime identity not verified')
        details[(job['block'],job['state'])]=detail
        raw_evidence.append(dict(job_id=job['job_id'],raw=receipt,telemetry_errors=raw.get('telemetry_errors'),
            telemetry_samples=raw.get('telemetry_samples'),warmup_window=raw.get('warmup_window'),
            energy_scope=raw.get('energy_evidence',{}).get('scope'),resource_valid=raw.get('memory_evidence',{}).get('resource_valid')))
    blocks=[diagnose_block(details[(b,'C')],details[(b,'B')],details[(b,'A')]) for b in [1,2]]
    for item in receipts:
        if a.file_receipt(item['path'])['sha256']!=item['sha256']:raise ValueError('Diagnostic input changed')
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,data):(args.output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    for b in blocks:save(f"block{b['block']}.json",b)
    save('report.json',dict(schema='gsdeploy.joint_service_phase_diagnostic.v1',blocks=blocks,
        scope=__doc__,U=None,ranking_performed=False,fitted_parameters=False,causal_effect_identified=False,
        audit_note='Independent all-six audit v2 PASS receipt supplied separately; historical v1 failure from protocol SHA assertion retained. This tool diagnoses service phases and does not replace that audit.',audit=audit,
        cache_scope='Cache reclamation performed/unused bytes observed; no independent reclamation-duration timer, no attribution of slowdown to cache.',
        limitations=['Phase boundaries are observed specialist handler finishes, not randomized treatments.',
            'Different repetition/timephase/order/warmup histories are confounded; no causal interference, thermal or cache attribution.',
            'All-three median can be dominated by two requests after specialists finish.',
            'Positive common activity is a weak coverage indicator;95% threshold is descriptive, not a predeclared acceptance test.',
            'No new experiment, predictor, capacity/stability boundary, U ranking or validation certificate.']))
    save('raw_context.json',raw_evidence)
    write_markdown(args.output,blocks,audit)
    plot(args.output,blocks)
    save('manifest.json',dict(schema='gsdeploy.joint_service_phase_diagnostic.manifest.v1',inputs=receipts,
        outputs={path.name:a.file_receipt(path)['sha256'] for path in args.output.iterdir() if path.is_file()},U=None))
    print(json.dumps(dict(status='JOINT_SERVICE_PHASE_DIAGNOSIS_COMPLETE',output=str(args.output.resolve()),blocks=2,U=None)))


def write_markdown(output,blocks,audit):
    lines=['# C共同活动与尾段服务诊断','','全部六次已登记终态均按SHA读取，并与analysis-v1逐请求原始记录核对。独立审计v2已PASS，v1错误断言历史保留；审计通过不表示持续共同活动或开放容量已验证。U=null，不拟合、不排序、不推断因果。','',
        '|block|G2 B完成秒|G2 C完成秒|最后专用handler结束秒|之前完成G2数|首个纯尾段sequence|32输入中位数来源尾段数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for b in blocks:
        phases=b['median_provenance_summary']['source_phases'];lines.append(f"|{b['block']}|{b['B_G2_host_cohort_completion_seconds']:.3f}|{b['G2_host_cohort_completion_seconds']:.3f}|{max(b['specialist_handler_finish_elapsed_seconds'].values()):.3f}|{b['G2_requests_completed_before_last_specialist_finish']}|{b['first_fully_tail_request']['sequence']}|{phases.get('after_both_specialist_handlers_finish',0)}|")
    lines.extend(['','|block/组|请求数|service均值ms|service中位ms|同输入同轮C/B比中位|与B输出相同|与B生成token相同|cache reclaim执行|','|---|---:|---:|---:|---:|---:|---:|---:|'])
    for b in blocks:
        for label,s in [('all',b['all_G2']),*[(f'repeat{k}',v) for k,v in b['by_repetition'].items()],*b['by_overlap'].items(),*b['by_phase'].items()]:
            lines.append(f"|{b['block']}/{label}|{s['count']}|{s['native_service_ms']['mean']:.3f}|{s['native_service_ms']['p50']:.3f}|{s['paired_same_input_repeat_C_over_B']['p50']:.3f}|{s['exact_prediction_matches']}/{s['count']}|{s['generated_token_matches']}/{s['count']}|{s['cache_performed']}/{s['count']}|")
    lines.extend(['','每输入三次服务中位数的具体来源、各轮/相位、B配对值、输出hash/token/cache、共同活动覆盖在block JSON完整保留。95%共同活动比例仅用于描述，不能事后变成验收门槛。',
        '', '全cohort算术均值能看出总服务工作量增加；全cohort总体中位数和先取每输入三次中位数再跨条件比，可能掩盖仅发生在最初一轮/共同活动时的慢段。两种统计不能互相替代，也不能用尾段中位数校准持续共同活动模型。',
        '', '缓存reclamation仅记录是否执行及前后unused reserved MiB，没有独立耗时，不能由performed字段给缓存归因。相同输入、prediction和token可以排除这些已观测字段差异，但不能排除调度、共享GPU、warmup、频率/温度或其他未记录状态。',
        '', '下一动作：先停止把C整段三次中位数迁移到open-loop；完成审计修正并核对首轮慢段的时间账本/telemetry。若要识别持续共同活动服务，需要另行冻结能在各alias输入上保持充分共同活动覆盖的新诊断和跨块重复门槛；不得从本报告直接拟合share/rate耦合或选高负载。没有共同支持区间则停止开放负载六点方案。'])
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def plot(output,blocks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for col,b in enumerate(blocks):
        rows=b['G2_request_pairs'];colors=['#b45309','#0369a1','#059669']
        for repetition in range(3):
            seq=[r for r in rows if r['repetition']==repetition]
            axes[0,col].scatter([r['start_elapsed_seconds'] for r in seq],[r['paired_service_ratio'] for r in seq],s=18,color=colors[repetition],label=f'repetition {repetition}')
            axes[1,col].scatter([r['start_elapsed_seconds'] for r in seq],[r['activity']['all_active_aliases_busy_fraction'] for r in seq],s=18,color=colors[repetition])
            axes[2,col].scatter([r['activity']['all_active_aliases_busy_fraction'] for r in seq],[r['paired_service_ratio'] for r in seq],s=18,color=colors[repetition])
        for ax in axes[:2,col]:
            for alias,stamp in b['specialist_handler_finish_elapsed_seconds'].items():ax.axvline(stamp,color='#555555',ls='--',lw=.8)
            ax.grid(alpha=.15)
        axes[0,col].set_yscale('log');axes[0,col].axhline(1,color='#888888',ls=':',lw=.8);axes[0,col].set_title(f"Block {b['block']}: G2 C / B, same input and repetition")
        axes[0,col].legend(fontsize=8);axes[1,col].set_xlabel('C handler start seconds after measurement release');axes[1,col].set_ylim(-.04,1.04)
        axes[0,col].set_xlabel('C handler start seconds after measurement release')
        axes[2,col].set_xlabel('All active aliases busy / G2 handler');axes[2,col].set_yscale('log');axes[2,col].axhline(1,color='#888888',ls=':',lw=.8);axes[2,col].grid(alpha=.15)
    axes[0,0].set_ylabel('Native service time ratio (log scale)');axes[1,0].set_ylabel('All active aliases busy / G2 handler');axes[2,0].set_ylabel('Native service time ratio (log scale)')
    fig.text(.06,.025,'Dashed lines: observed specialist final handler ends. Phase, order and repetition are confounded; no causal attribution.\nRetrospective handler overlap is not GPU utilization or a predictor. Two blocks are development diagnostics; U=null.',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,1));fig.savefig(output/'joint-service-phases.png',dpi=180);fig.savefig(output/'joint-service-phases.pdf');plt.close(fig)


if __name__=='__main__':main()
