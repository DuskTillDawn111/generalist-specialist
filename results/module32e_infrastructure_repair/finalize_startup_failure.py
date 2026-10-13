"""Engineering gate/report only. No utility computations or historical writes."""
from m32e.isolation import install,VIOLATIONS
install()
from m32e.common import *
def write(name,value):(OUT/name).write_text(value)
def probe_label(ctrl,key):
 value=ctrl.get(key)
 return "未测量（启动失败，0个检查样本）" if value is None else f"{value*1000:.6f}ms"
def main():
 e=read(OUT/'cpu_regression/routing_equivalence_report.json');c=read(OUT/'cpu_regression/CPU_runtime_acceptance.json')
 p=read(OUT/'history_write_protection_tests.json');b=read(OUT/'control_probe_attribution.json')
 comparisons=[read(OUT/'history_hashes'/n) for n in ['before_CPU_vs_after_CPU.json','before_GPU_vs_after_GPU.json','before_CPU_vs_after_GPU.json'] if (OUT/'history_hashes'/n).exists()]
 a=read(OUT/'gpu_preflight_acceptance.json') if (OUT/'gpu_preflight_acceptance.json').exists() else dict(passed=False,checks={})
 hf=all(x['passed'] for x in comparisons) and len(comparisons)==3
 checks=dict(history_protection=p['passed'] and hf,control_CPU=b['passed'],router_CPU=c['passed'],same_state_semantics=e['passed'] and e['decisions']==1381,
  GPU_preflight=a['passed'],GPU_history_verified=hf)
 if not checks['history_protection']:gate='HISTORY_WRITE_ISOLATION_FAILED'
 elif not b['passed']:gate='CONTROL_PROBE_UNRESOLVED'
 elif not c['passed'] or not e['passed']:gate='ROUTER_RUNTIME_REGRESSION'
 elif not a['passed']:gate='GPU_PREFLIGHT_FAILED'
 else:gate='INFRASTRUCTURE_READY_FOR_VALIDATION'
 out=dict(module='Module32E',gate=gate,checks=checks,engineering_valid=all(checks.values()),U=None,utility_validated=False,formal_GPU_runs=0,
  GPU_engineering_attempts=len(list((OUT/'run_attempts').glob('*.json'))),no_automatic_long_window=True,no_optimizer_changes=True,no_history_Gate_changes=hf,
  prior32D_byte_exact_progress_restore=False,prior32D_exception=ref(OUT/'prior_incident_record.json'),
  control_threshold_ms=10,router_P95_threshold_ms=1,policy_source_unchanged=e['policy_source_byte_identical'],
  historical_outlier_uniquely_attributed=False,evidence=[ref(OUT/n) for n in ['history_write_protection_tests.json','control_probe_attribution.json','cpu_regression/CPU_runtime_acceptance.json','cpu_regression/routing_equivalence_report.json']])
 save('module32e_gate.json',out)
 save('routing_equivalence_report.json',e)
 for n in ['router_performance_benchmark.csv','router_performance_by_300s.csv']:(OUT/n).write_bytes((OUT/'cpu_regression'/n).read_bytes())
 rows=c['results'];cpu='\n'.join(f"| {r['history_multiplier']}× | {r['offered']} | {r['history_records']} | {r['p50_ms']:.6f} | {r['p95_ms']:.6f} | {r['p99_ms']:.6f} | {r['max_ms']:.6f} | {r['control_check_ms']['max_ms']:.6f} |" for r in rows)
 write('cpu_engineering_regression.md',f'''# CPU工程回归

1381历史决策全部字段完全一致，浮点容差0；政策、增量runtime源码与32D逐字节一致。collector生命周期覆盖重复/迟到响应、队列溢出、worker EXIT和排空超时；phase切换测试证明旧快照不消耗新phase完成事件。

| 保留历史规模 | 请求 | 全部archive | 路由P50 ms | P95 ms | P99 ms | max ms | 外层控制max ms |
|---|---:|---:|---:|---:|---:|---:|---:|
{cpu}

每档使用真实32C完整2400秒3419请求/1070ASR生命周期，50倍加速约48秒墙钟；独立sender/collector线程，0.5秒sensor、10秒实际journal/fsync，无GPU。测量完整runtime锁获取、投影、decide、enqueue；另在计时外校验full-archive projection等价。它不是2400秒墙钟soak，不证明长GPU运行性能。所有离群点完整保留，P95门槛≤1ms不变。8个300秒逻辑分段及锁等待/持有、浅层archive和active内存均在CSV/JSON保存。增量active只跟活跃请求规模相关，审计archive仍保留历史；不删除策略状态。性能验收没有将max路由耗时冒充P95，也未删除30ms级离群点。

首次缓存兼容测试的dir_fd审计误判保留在pre_GPU_dirfd_correction，修复路径解析并加保护测试后重跑全部CPU门槛。尚无GPU数据用于该修复或策略调参。
''')
 rawpath=OUT/'raw_runs/E0_DYNAMIC.json'
 if rawpath.exists():
  raw=read(rawpath);samples=raw['telemetry']['samples'];ss=[x for x in samples if x['phase']=='measurement']
  resource={k:stats(x[k] for x in ss) for k in ['temperature_gpu_c','sm_clock_mhz','memory_clock_mhz','power_draw_w','power_limit_w','owned_used_mib']} if ss else {}
  state=dict(phases=dict(collections.Counter(x['phase'] for x in samples)),measurement=resource,
   thermal_slowdown_samples=sum(x['hw_thermal_slowdown']!='Not Active' or x['sw_thermal_slowdown']!='Not Active' for x in samples),
   worker_guards=[read(OUT/'work/control_guards'/f"{v['pid']}_exit.json") for v in raw['worker_identities'].values() if (OUT/'work/control_guards'/f"{v['pid']}_exit.json").exists()],
   clock_semantics='NVML actual SM/memory clocks and thermal flags. Mean temperature alone is not a throttling test.',
   pair_environment_validation=False,prelaunch=raw.get('baseline_GPU'),release_verified=raw['gpu_release']['verified'])
  save('gpu_environment_report.json',state)
  write('gpu_environment_report.md','# GPU工程环境报告\n\n'+json.dumps(state,ensure_ascii=False,indent=2)+'\n\n本轮一臂工程验收，无配对温差/频率可比性结论，无正式U。完整运行前预热、温控、native worker日志、请求/控制时间和逐0.5秒GPU采样见原始数据。\n')
 details=a.get('details',{});route=details.get('router') or {};ctrl=a.get('control',{})
 summary=f'''# Module32E总结

Gate：`{gate}`。

- 历史写入保护：{'通过' if checks['history_protection'] else '未通过'}。18项保护测试覆盖native子进程、历史输出误用、符号链接及dir_fd；全部历史输入{comparisons[0]['files'] if comparisons else 0}文件完整SHA、大小、mtime及增删清单核对。32D原进度不能逐字节恢复的旧差异仍明确保留，本轮未伪称修复历史字节。
- 路由开销：CPU P95 {rows[0]['p95_ms']:.6f}/{rows[1]['p95_ms']:.6f}ms；真实GPU P95 {route.get('p95_ms') if route.get('p95_ms') is not None else '未测量'}。原≤1ms门槛保持。1381次历史决策一致；真实GPU同状态因果回放结果见gpu_preflight_acceptance.json。
- 控制探针：CPU修复验证{'通过' if b['passed'] else '未通过'}；真实GPU最大check {probe_label(ctrl,'max_check')}、最大publish {probe_label(ctrl,'max_publish')}。原所有样本严格小于10ms门槛保持。旧281请求复制锁路径复现超过10ms；历史19.36ms缺乏逐检查owner/时序，无法唯一归因。
- GPU预检：{'通过' if a['passed'] else '未通过'}，首个PARSeq worker在CUDA初始化时Error304退出，无请求进入预热或测量窗口，0次控制检查和0次在线路由计时。不能判定GPU路由/控制/质量/六任务稳定性/驻留显存通过。启动遥测、worker错误、清洁退出与GPU释放证据完整保留；未执行过的600秒计划轨迹仍在冻结清单中，不伪造完整运行。GPU初始化与新隔离机制的兼容性尚未确证，也未逐syscall归因，见GPU_startup_failure_audit.json。原验收器假设journal存在产生的次级FileNotFoundError独立保留，由新增事后报告器正确记为启动失败；不改动已经冻结的执行源码。

只尝试一次GPU启动；原计划600秒工程预检、300秒真实S-only调理、30秒warmup均未执行。冻结的是原32B开发轨迹，不是独立复现。U没有计算，不能推出正效用、长窗口收益或可集成正式优化器。保留全部CPU版本/异常/离群点和GPU结果。{'可供后续独立长窗口验证使用；本轮已经停止，不自动启动。' if gate=='INFRASTRUCTURE_READY_FOR_VALIDATION' else '本轮停止；不重试GPU、不放宽门槛，不启动长窗口。'}正式部署优化器、历史实验、模型权重及历史Gate均不修改。
'''
 write('module32e_summary.md',summary);save('module32e_summary.json',dict(gate=gate,checks=checks,GPU=a,CPU=c,U=None))
 assert not VIOLATIONS
 files=[ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and 'work' not in p.relative_to(OUT).parts and p.name!='artifact_manifest.json']
 save('artifact_manifest.json',dict(at_unix_ns=time.time_ns(),files=files,history_unchanged_since32E_baseline=hf,old32D_byte_restore_not_claimed=True))
 print('MODULE32E_FINISHED',gate,flush=True)
if __name__=='__main__':main()
