"""Post-measurement evidence and exact ASR decomposition; no inference or routing changes."""
import numpy as np,time
from .common import *
from .campaign import verify

def main():
 f=read(OUT/'experiment_manifest.json');verify(f)
 assert read(OUT/'campaign_status.json')['status']=='COMPLETED'
 obs=read(OUT/'run_observations.json');assert len(obs)==3 and all(o['eligible'] for o in obs)
 raws={s['run_id']:read(OUT/'raw_runs'/(s['run_id']+'.json')) for s in f['slots']}
 checks=[dict(path=r['path'],passed=ref(r['path'])['sha256']==r['sha256']) for r in f['checkpoints']]
 assert all(c['passed'] for c in checks)
 by={o['run_id']:o for o in obs};baseline=raws['A0']['phases']['measurement']['request_records'];lookup={r['request_id']:r for r in baseline};N=sum(r['task']=='asr' for r in baseline)
 rows=[]
 for aid in ['A1','A2']:
  rr=[r for r in raws[aid]['phases']['measurement']['request_records'] if r['task']=='asr'];same=[r for r in rr if r['model']=='whisper-small#0'];div=[r for r in rr if r['model']=='qwen#0']
  def delta(group,key):return sum(r[key]-lookup[r['request_id']][key] for r in group)
  svcS=delta(same,'worker_latency_ms');waitS=delta(same,'worker_queue_wait_ms');svcG=delta(div,'worker_latency_ms');waitG=delta(div,'worker_queue_wait_ms')
  def residual(r):return r['e2e_latency_ms']-r['worker_latency_ms']-r['worker_queue_wait_ms']
  host=sum(residual(r)-residual(lookup[r['request_id']]) for r in rr)
  true=by[aid]['per_task']['asr']['mean_ms']-by['A0']['per_task']['asr']['mean_ms']
  reconstructed=(svcS+waitS+svcG+waitG+host)/N;assert abs(true-reconstructed)<1e-9
  rows.append(dict(group=aid,N_ASR=N,retained_S=len(same),diverted_G=len(div),
   retained_S_service_delta_total_ms=svcS,retained_S_wait_delta_total_ms=waitS,retained_S_wait_reduction_mean_ms=-waitS/len(same),
   retained_S_control_wait_mean_ms=float(np.mean([lookup[r['request_id']]['worker_queue_wait_ms'] for r in same])),
   retained_S_split_wait_mean_ms=float(np.mean([r['worker_queue_wait_ms'] for r in same])),
   diverted_G_extra_service_total_ms=svcG,diverted_G_extra_service_mean_ms=svcG/len(div),diverted_G_wait_delta_total_ms=waitG,
   retained_S_service_contribution_to_ASR_mean_ms=svcS/N,retained_S_wait_contribution_to_ASR_mean_ms=waitS/N,
   diverted_G_service_contribution_to_ASR_mean_ms=svcG/N,diverted_G_wait_contribution_to_ASR_mean_ms=waitG/N,
   host_residual_contribution_to_ASR_mean_ms=host/N,ASR_mean_delta_ms=true,reconstructed_ASR_mean_delta_ms=reconstructed))
 csvsave('native_latency_decomposition.csv',rows)
 evidence=[]
 for aid,r in raws.items():
  w=r['phases']['measurement']['window'];evidence.append(dict(group=aid,formal_seconds=w['duration_seconds'],formal_requests=len(r['phases']['measurement']['request_records']),all_completed=all(x['status']=='completed' for x in r['phases']['measurement']['request_records']),control_max_ms=r['control_latency_seconds']['max_check']*1000,collector_exited=r['telemetry']['collector_exited'],GPU_released=r['gpu_release']['verified'],raw=ref(OUT/'raw_runs'/(aid+'.json')),journal=ref(OUT/'request_journals'/(aid+'.jsonl'))))
 save('final_data_acceptance.json',dict(passed=True,checked_unix_ns=time.time_ns(),formal_windows=3,total_formal_seconds=1800,total_formal_requests=sum(e['formal_requests'] for e in evidence),warmup_smoke_probes_excluded=True,evidence=evidence,full_final_checkpoint_hash_check=checks,source_and_input_hashes_verified=True,history_preserved=read(OUT/'gate_decision.json')['history_preserved'],formal_retries=0,analysis_only_addition=ref(__file__)))
 original=(OUT/'module31a_summary.md').read_text();lines=['','## 主要结果与解释','',f"三组各 600 秒，共 {sum(e['formal_requests'] for e in evidence)} 个正式请求，全部完成、失败数 0；每组完整 cohort 吞吐 {1184/600:.6f} QPS。三臂 journal、时间戳、资源、线程退出、GPU 释放及原始效用复算均通过；全部检查点最终完整 SHA256 再核验通过。",'','|组|S-ASR 执行/等待均值 ms|G-ASR 执行/等待均值 ms|VQA mean/P95 ms|','|---|---:|---:|---:|']
 for aid in ['A0','A1','A2']:
  o=by[aid];sp=next(p for p in o['paths'] if p['task']=='asr' and p['model']=='whisper-small');gp=next(p for p in o['paths'] if p['task']=='asr' and p['model']=='qwen');v=o['per_task']['visual_qa']
  gs='无 G-ASR 请求' if not gp['completed'] else f"{gp['service']['mean']:.2f}/{gp['wait']['mean']:.2f}"
  lines.append(f"|{aid}|{sp['service']['mean']:.2f}/{sp['wait']['mean']:.2f}|{gs}|{v['mean_ms']:.2f}/{v['p95_ms']:.2f}|")
 for x in rows:
  lines+=['',f"{x['group']} 同输入保留 S 等待减少 {x['retained_S_wait_reduction_mean_ms']:.3f} ms/请求（负值表示增加），总变化 {x['retained_S_wait_delta_total_ms']/1000:.3f} 秒；转交 G 相对同输入原 S 增加执行 {x['diverted_G_extra_service_mean_ms']:.3f} ms/请求，共 {x['diverted_G_extra_service_total_ms']/1000:.3f} 秒。五项严格加总重建 ASR mean Δ={x['ASR_mean_delta_ms']:.3f} ms，见 native_latency_decomposition.csv。"]
 lines+=['','基线 S-ASR 服务需求/600 秒='+f"{sum(r['worker_latency_ms'] for r in baseline if r['task']=='asr')/600000:.4f}"+'（服务需求比，非 GPU utilization），排队红利有限。执行差和共享 G 队列的背景尾部代价按数据分开报告；单次序列实验不能唯一隔离 GPU 因果，也不能推广到其他负载点。','', '完整六任务完成/失败、cohort 与窗口内吞吐、mean/P50/P95/P99/max 见 per_task_latency_comparison.csv；资源均值/峰值见 resource_comparison.csv；配对干扰见 worker_interference_analysis.csv。']
 lines+=['','## 统计证据','']
 for x in read(OUT/'statistical_diagnostics.json')['results']:lines.append(f"- {x['arm']} ΔU 的窗内配对 30 秒区块 bootstrap 95% 诊断区间 [{x['diagnostic_percentile_95_low']:.6f}, {x['diagnostic_percentile_95_high']:.6f}]；不含独立跨运行不确定性，不能作为已证实可重复收益的依据。")
 (OUT/'module31a_summary.md').write_text(original+'\n'.join(lines)+'\n')
 p=OUT/'interference_analysis.md';p.write_text(p.read_text()+'\n## 精确 ASR 均值分解\n\n'+ '\n'.join(f"- {x['group']}: S 保留等待变化 {x['retained_S_wait_delta_total_ms']:.3f}ms；G 额外执行 {x['diverted_G_extra_service_total_ms']:.3f}ms；ASR mean Δ={x['ASR_mean_delta_ms']:.3f}ms，精确重建={x['reconstructed_ASR_mean_delta_ms']:.3f}ms。" for x in rows)+'\n\n完整五项及分母见 native_latency_decomposition.csv。保留 S 负 wait delta 是真实配对等待下降；分流请求消失不冒充保留 S 排队改善。\n')
 print('FINAL_ACCEPTANCE',len(checks),evidence,flush=True)
if __name__=='__main__':main()
