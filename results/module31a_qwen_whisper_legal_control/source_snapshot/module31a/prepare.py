import copy,time,shutil,subprocess,collections,sys,os,ast
from .common import *
from replica_scaling.telemetry import snapshot
from module21.trajectory import choose

def main():
 assert not (OUT/'experiment_manifest.json').exists(),'Never overwrite a frozen manifest'
 f=copy.deepcopy(read(OLD/'frozen_gpu_manifest.json'))
 stress=read(OLD/'telemetry_stress_result_v2.json');assert stress['passed'] and stress['elapsed_seconds']>1997.196239099
 gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<=128
 assert all(gpu[k]==f['GPU'][k] for k in ['uuid','name','driver','total_mib'])
 historical={}
 for p in (ROOT/'results').rglob('*'):
  if p.is_file() and OUT not in p.parents:historical[str(p)]=[p.stat().st_size,p.stat().st_mtime_ns]
 save('protected_history_metadata.json',historical)
 save('protected_core_hashes.json',[ref(p) for folder in [ROOT/'module20',ROOT/'module30'] for p in folder.glob('*.py')]+[ref(p) for p in (ROOT/'results').rglob('*gate*.json') if OUT not in p.parents])
 slots=[]
 for aid,arm in [('A0','s_only'),('A1','split05'),('A2','split10')]:
  s=copy.deepcopy(next(s for s in f['slots'] if s['candidate']=='C1' and s['arm']==arm))
  s.update(run_id=aid,stage='31A_ONLY',experiment_group=aid)
  for label,r in s['trajectories'].items():
   assert ref(r['path'])['sha256']==r['sha256']
   dest=OUT/'trajectories'/Path(r['path']).name;dest.parent.mkdir(exist_ok=True);shutil.copy2(r['path'],dest);s['trajectories'][label]=ref(dest)
  events=read(s['trajectories']['measurement']['path'])['events']
  counts=collections.Counter(e['task'] for e in events);g=sum(e['task']=='asr' and choose(e,s['routing'],s['deployment'])=='qwen#0' for e in events)
  s['frozen_offered_counts']=dict(counts);s['actual_planned_G_asr']=g;s['actual_planned_G_fraction']=g/counts['asr']
  assert all(counts[t]>=72 for t in counts if t!='asr') and (aid=='A0' or g>=20)
  for e in events+read(s['trajectories']['warmup']['path'])['events']:
   val=e['item']['sample'].get('input')
   if isinstance(val,str) and val.startswith('/'):assert Path(val).is_file(),val
  slots.append(s)
 f['slots']=slots;f['plan']['registry']={k:v for k,v in f['plan']['registry'].items() if k in slots[0]['deployment']}
 f['plan']['models']={k:v for k,v in f['plan']['models'].items() if k in slots[0]['deployment']}
 f.update(module='Module31A Qwen/Whisper legal same-version control',GPU=gpu,frozen_unix_ns=time.time_ns(),maximum_formal_runs=3,
  formal_seconds=600,source_manifest=ref(OLD/'frozen_gpu_manifest_v2.json'),telemetry_version='SafeMonitorV2',Stage_D=False,scope='31A only',
  formal_order=['A0','A1','A2'],smoke_seconds=30,smoke_order=['A0','A1','A2'],warmup_excluded=True,
  failure_policy={'technical_error':'INVALID; retain evidence; diagnose before same-config rerun','no_result_selective_retries':True,'prelaunch_failure':'retain evidence; same-config preparation only'},
  statistical_policy={'independent_repetitions':1,'positive_U_is_candidate_only':True,'bootstrap':'paired 30-second arrival blocks, 2000 replicates, seed 310031; within-window diagnostic only; no independent repeatability claim','original_holdout_support_rule':read(ROOT/'results/module20_utility_optimizer/utility_definition.json')['holdout_support_rule']})
 git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
 f['git']={'commit':git.stdout.strip() if git.returncode==0 else None,'available':git.returncode==0,'error':git.stderr.strip(),'fallback':'content-addressed source snapshots; no invented commit'}
 quality=list(__import__('csv').DictReader((OLD/'quality_admission_provenance.csv').open()))
 csvsave('quality_admission_provenance.csv',[r for r in quality if r['run_id'].startswith('C1')]);assert all(r['passed']=='True' for r in quality if r['run_id'].startswith('C1'))
 checkpoint_dirs=[ROOT/'models/qwen2.5-omni-7b',ROOT/'models/whisper-small',ROOT/'models/candidate_pool_v2/roberta_sst2',ROOT/'models/candidate_pool_v2/convnext_tiny']
 entry=next(e for e in read(ROOT/'results/module14_high_quality_heavy/ocr_gapfill_shortlist.json')['entries'] if e['model_id']=='baudm/parseq:parseq')
 cpfiles=sorted({p.resolve() for d in checkpoint_dirs for p in d.rglob('*') if p.is_file() and '.cache' not in p.parts}|{Path(entry['weight_path']).resolve()})
 print('HASH_CHECKPOINTS',len(cpfiles),sum(p.stat().st_size for p in cpfiles),flush=True)
 f['checkpoints']=[ref(p) for p in cpfiles];f['parseq_identity']=entry
 dirs=['module31a','module30','module20','module21','module22','module25','module26','module11b','module14b','joint_deploy','replica_scaling','performance_model','candidate_pool_audit','benchmark/scripts']
 srcs=sorted({p for d in dirs for p in (ROOT/d).rglob('*.py') if '__pycache__' not in p.parts}|{ROOT/'results/module28_dynamic_split/launchers/parseq.py'})
 srcs+=sorted((ROOT/'models/module14b'/('parseq-'+entry['implementation_commit'])).rglob('*.py'))
 f['sources']=[]
 for p in srcs:
  ast.parse(p.read_text());r=ref(p);dest=OUT/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);r['snapshot']=str(dest);f['sources'].append(r)
 f['environment']={}
 for py in sorted({r['python'] for r in f['plan']['registry'].values()}):
  probe=subprocess.run([py,'-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))'],text=True,capture_output=True,check=True)
  f['environment'][py]=json.loads(probe.stdout)
 refs=[ROOT/'results/module20_utility_optimizer/utility_definition.json',ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json',OLD/'selected_pool_catalog.json',ROOT/'results/module11_replica_and_cost/frozen_samples.jsonl']
 refs += [Path(r['path']) for r in f['source_refs']]
 f['frozen_inputs']=[ref(p) for p in sorted(set(refs))]
 f['utility_definition']=read(refs[0]);f['utility_reference']=read(refs[1])
 # Source/sample/model integrity checked before every run. Checkpoint hash cache relies only on unchanged size/mtime after initial full hash.
 f['checkpoint_metadata']={r['path']:[Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns] for r in f['checkpoints']}
 save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f)
 shutil.copy2(OLD/'selected_pool_catalog.json',OUT/'selected_pool_catalog.json')
 for aid,s in zip(['A0','A1','A2'],slots):
  tr=read(s['trajectories']['measurement']['path']);tr['events']=[e for e in tr['events'] if e['offset_seconds']<30];tr['duration']=30.
  save('trajectories/smoke_'+aid+'.json',tr)
 save('preflight.json',dict(passed=True,counts=slots[0]['frozen_offered_counts'],formal_G_counts=[s['actual_planned_G_asr'] for s in slots],GPU=gpu,git_available=git.returncode==0,checkpoint_files=len(cpfiles),same_offered_trace=True,quality_unchanged=True))
 (OUT/'reuse_audit.md').write_text('''# Module30 审计与复用\n\nQwen C1 s_only/split05/split10 实际 ASR G/S 为 0/595、28/567、53/542，实际比例 0%、4.705882%、8.907563%。前两窗初版遥测 check 共用采样长锁，合法 U=NULL；split10 使用 v2，U=-1.2243815965，跨版本 ΔU 禁止。旧数据不参与本次效用对照。\n\n本次复用 C1 seed=301001 的六任务输入/到达/路由 uniform 轨迹。ASR=1 QPS，Text/Image/OCR/LID/VQA=0.2 QPS。Text→RoBERTa-base SST-2 FP32，Image→ConvNeXt-tiny FP32，OCR→PARSeq v1.0.0 严格加载 FP32，LID→Whisper-small FP32，VQA→Qwen2.5-Omni-7B BF16；ASR 未分流部分→Whisper-small，分流部分→Qwen。五副本单卡、34567.5534 MiB 预算不变。检查点全文件 SHA256、脚本快照、依赖、质量准入和精确轨迹见 manifest。\n\n复用 SafeMonitorV2；依赖的 SafeMonitor.sample 仅作串行 I/O 和 runtime.lock 一致快照，SafeMonitorV2.check/set_active 覆盖旧控制方法。通过的 2118.546294 秒长压与故障注入证据均有哈希。GPU 型号、UUID、驱动、总显存与原冻结一致。\n\n效用：module20.optimizer.components 与 module26.common.arithmetic 独立复算；Q=0、T=.25、L=.3、P=.3、M=.15；六任务按冻结到达率加权，参考延迟不变，完整 cohort/600s 和窗口内吞吐不混淆，预热/path probes 不计入。原稳定性、质量、资源、+0.01 门槛不变。SLO 使用历史稳定/完整/排空、资源与质量约束；未新增绝对时延阈值。单组三窗仅探索，原独立统计支持规则不能由一次配对轨迹满足。\n\nGit 元数据不存在，commit=NULL，源码内容 SHA256 和副本代替身份冻结，明确报告此复现限制。Module14F–30 结果及 Gate、权重、正式优化器原文件不修改，Stage D/31B–31E 不执行。\n''')
 print('PREPARED',slots[0]['frozen_offered_counts'],flush=True)
if __name__=='__main__':main()
