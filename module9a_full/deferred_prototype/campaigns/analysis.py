"""Frozen-identity validation and statistical model comparison."""
import csv,itertools,time
from collections import defaultdict,Counter
import numpy as np
from scipy.stats import binomtest
from .common import *
from deployment_core.interference.validation import metrics,slowdown_bootstrap,paired_model_bootstrap

def raw_arrays(r):
 m,k,j,l=[r[key] for key in ['target_model','target_task','co_model','co_task']];base=[];pair=[];refs=[];overlaps=[]
 p=load(OUT/'protocol_identity.json');items=p['states'][f'{m}__{k}__none']['target_items'];ids=[v['pool_id'] for v in items]
 for rep in range(1,4):
  arrays=[]
  for key in [f'{m}__{k}__none',f'{m}__{k}__{j}__{l}']:
   path=OUT/'raw_runs'/f'{key}__rep{rep}.json';a=load(path);assert a['status']=='COMPLETE' and a['gpu_release']['verified'];assert [v['pool_id'] for v in a['target_records']]==ids;assert [v['sample_index'] for v in a['target_records']]==list(range(len(ids)))
   state=p['states'][key];assert [v['pool_id'] for v in a['warmup_records']]==[v['item']['pool_id'] for v in state['warmup_sequence']]
   assert all(v['scheduled_offset_seconds']==i*7 for i,v in enumerate(a['target_records']));assert all(v['outstanding_co_requests']>=64 for v in a['co_backlog_trace']) if state['co_model'] else True
   assert len({v['pid'] for v in a['ready'].values()})==len(a['ready'])
   arrays.append([v['worker_latency_ms'] for v in a['target_records']]);refs.append({'path':str(path),'sha256':sha(path)})
   if state['co_model']:overlaps.extend(v['active_overlap_ratio'] for v in a['target_records'])
  base.append(arrays[0]);pair.append(arrays[1])
 assert min(overlaps)>=.9
 return np.array(base),np.array(pair),refs,overlaps

def ground_truth(r):
 a,b,refs,overlaps=raw_arrays(r);seed=int(digest({'bootstrap_seed':94002,'identity':r['identity']})[:8],16);v=slowdown_bootstrap(a,b,seed=seed)
 return {**r,**v,'standalone_mean_ms':float(a.mean()),'pair_mean_ms':float(b.mean()),'repetitions':3,'requests_per_repetition':30,'minimum_active_overlap':min(overlaps),'mean_active_overlap':float(np.mean(overlaps)),'raw_refs':refs}

def freeze_baseline():
 assert not (OUT/'I1_calibration_freeze.json').exists()
 manifest=load(OUT/'pair_validation_manifest.json');ground=[ground_truth(r) for r in manifest['calibration_pairs']];group=defaultdict(list)
 for r in ground:group[(r['target_model'],r['co_model'])].append(r['mu_real'])
 means={m+'<-'+j:float(np.mean(v)) for (m,j),v in group.items()}
 save('I1_calibration_freeze.json',{'frozen_at_utc':now(),'source':'new protocol-identical calibration identities only','baseline_mu_by_directional_model_pair':means,'ground_truth':ground,'validation_results_seen':False,'aggregation':'equal weight mean of unique calibration pair pooled aggregate mu; no target/co task distinction','calibration_raw_refs':[ref for r in ground for ref in r['raw_refs']]})
 csvsave('calibration_ground_truth.csv',ground)
 print(json.dumps({'event':'I1_BASELINE_FROZEN','means':means}),flush=True)

def task_ordering(rows):
 out=[]
 for a,b in itertools.combinations(rows,2):
  if (a['target_model'],a['co_model'],a['co_task'])!=(b['target_model'],b['co_model'],b['co_task']) or a['target_task']==b['target_task']:continue
  m=a['target_model'];roles=load(OUT/'catalog.json')['roles']
  if roles[m]!='generalist':continue
  aa,ab,_,_=raw_arrays(a);ba,bb,_,_=raw_arrays(b);rng=np.random.default_rng(int(digest(sorted([a['identity'],b['identity']]))[:8],16));B=10000;blocks=rng.integers(0,3,size=(B,3));ia=rng.integers(0,30,size=(B,3,30));ib=rng.integers(0,30,size=(B,3,30));ap=ab[blocks[:,:,None],ia].mean(axis=(1,2))/aa[blocks[:,:,None],ia].mean(axis=(1,2));bp=bb[blocks[:,:,None],ib].mean(axis=(1,2))/ba[blocks[:,:,None],ib].mean(axis=(1,2));ci=np.quantile(ap-bp,[.025,.975]);actual=a['mu_real']-b['mu_real'];pred=a['mu_hat']-b['mu_hat'];resolved=bool(ci[0]>0 or ci[1]<0)
  out.append({'target_model':m,'task_a':a['target_task'],'task_b':b['target_task'],'co_model':a['co_model'],'co_task':a['co_task'],'split':'VALIDATION','predicted_Delta_mu':pred,'real_Delta_mu':actual,'real_Delta_mu_CI95':ci.tolist(),'real_ordering_CI_resolved':resolved,'ordering_correct':bool(np.sign(pred)==np.sign(actual)) if pred!=0 else False,'predicted_ordering_tie':pred==0,'task_effect_CI_excludes_zero':resolved,'prediction_significance_claimed':False,'evidence_use':'held-out pair task contrast'})
 csvsave('task_conditioned_ordering.csv',out,fields=['target_model','task_a','task_b','co_model','co_task','split','predicted_Delta_mu','real_Delta_mu','real_Delta_mu_CI95','real_ordering_CI_resolved','ordering_correct','predicted_ordering_tie','task_effect_CI_excludes_zero','prediction_significance_claimed','evidence_use'])
 return out

def development_sanity():
 # Historical mixture is NOT a single co-task path identity. Match the mixture
 # according to separately profiled standalone per-task service demand (not pair fit).
 from deployment_core.interference.routing_conditioned import InterferenceEstimator
 estimator=InterferenceEstimator.from_directory(OUT);p=load(ROOT/'results/module9a2_repeatability/protocol_identity.json');out=[]
 for task in ['asr','visual_qa']:
  key=f'qwen__{task}__whisper-small';s=p['states'][key];co_tasks=s['co_tasks'];weights={k:estimator.s0[('whisper-small',k)] for k in co_tasks};total=sum(weights.values());weights={k:v/total for k,v in weights.items()}
  predicted=sum(weights[k]*estimator.pairs[(('qwen',task),('whisper-small',k))] for k in co_tasks)/estimator.s0[('qwen',task)];run=[]
  for rep in range(1,6):
   a=load(ROOT/f'results/module9a2_repeatability/raw_runs/qwen__{task}__none__rep{rep}.json');b=load(ROOT/f'results/module9a2_repeatability/raw_runs/{key}__rep{rep}.json');run.append(np.mean([r['worker_latency_ms'] for r in b['target_records']])/np.mean([r['worker_latency_ms'] for r in a['target_records']])-1)
  out.append({'target_model':'qwen','task':task,'co_model':'whisper-small','co_tasks':co_tasks,'co_task_mixture':weights,'mu_hat':predicted,'mean_historical_run_mu':float(np.mean(run)),'split':'DEVELOPMENT_ONLY','headline_validation':False,'history_protocol_matches_new_warmup':False})
 delta=out[0]['mu_hat']-out[1]['mu_hat'];actual=out[0]['mean_historical_run_mu']-out[1]['mean_historical_run_mu'];save('development_sanity.json',{'paths':out,'predicted_Delta_ASR_minus_VQA':delta,'historical_Delta_ASR_minus_VQA':actual,'correct_ordering':bool(np.sign(delta)==np.sign(actual)),'headline_validation':False,'calibration_for_coefficients':False})
 return out

def final():
 manifest=load(OUT/'pair_validation_manifest.json');baseline=load(OUT/'I1_calibration_freeze.json');verify(baseline['calibration_raw_refs']);ground=[ground_truth(r) for r in manifest['validation_pairs']]
 for r in ground:r['I0_mu']=0.;r['I1_mu']=baseline['baseline_mu_by_directional_model_pair'][r['target_model']+'<-'+r['co_model']];r['I2_mu']=r['mu_hat']
 csvsave('pair_ground_truth.csv',baseline['ground_truth']+ground);csvsave('validation_pair_errors.csv',[{'identity':r['identity'],'mu_real':r['mu_real'],'I0_mu':r['I0_mu'],'I1_mu':r['I1_mu'],'I2_mu':r['I2_mu'],'I0_AE':abs(r['mu_real']-r['I0_mu']),'I1_AE':abs(r['mu_real']-r['I1_mu']),'I2_AE':abs(r['mu_real']-r['I2_mu'])} for r in ground])
 roles=load(OUT/'catalog.json')['roles'];scopes={'all':ground,'Generalist_target':[r for r in ground if roles[r['target_model']]=='generalist'],'Specialist_target':[r for r in ground if roles[r['target_model']]=='specialist']};metric={scope:{method:metrics([r[method+'_mu'] for r in rr],[r['mu_real'] for r in rr]) for method in ['I0','I1','I2']} for scope,rr in scopes.items()};save('pair_validation_metrics.json',{'evidence':'VALIDATION_IDENTITIES_ONLY','metrics':metric,'primary_statistical_comparison':'pair-identity paired bootstrap of absolute errors, I2-I1'})
 save('baseline_comparison.json',{'baselines':{'I0':'NO_INTERFERENCE','I1':'TASK_AGNOSTIC_MODEL_PAIR, calibration-only directional modelpair average','I2':'TASK_PATH_KERNEL, Roomie-style context-compatible engineering approximation'},'heldout':metric,'I1_source_freeze_sha256':sha(OUT/'I1_calibration_freeze.json'),'validation_refit':False})
 comparison=paired_model_bootstrap([r['mu_real'] for r in ground],[r['I1_mu'] for r in ground],[r['I2_mu'] for r in ground]);comparison['validation_identities']=[r['identity'] for r in ground];save('paired_bootstrap_model_comparison.json',comparison)
 ordering=task_ordering(ground);resolved=[r for r in ordering if r['real_ordering_CI_resolved']];wrong=sum(not r['ordering_correct'] for r in resolved);pvalue=float(binomtest(wrong,len(resolved),.5,alternative='greater').pvalue) if resolved else None
 save('ordering_accuracy.json',{'new_validation_contrasts':len(ordering),'correct':sum(r['ordering_correct'] for r in ordering),'accuracy':float(np.mean([r['ordering_correct'] for r in ordering])) if ordering else None,'CI_resolved_contrasts':len(resolved),'CI_resolved_wrong':wrong,'one_sided_systematic_reversal_sign_test_p':pvalue,'systematic_direction_error':pvalue is not None and pvalue<.05,'ties_not_counted_correct':True,'caveat':'small bounded validation set, limited independent task contrasts'})
 development_sanity()
 print(json.dumps({'metrics':metric['all'],'bootstrap':comparison,'ordering':ordering}),flush=True)
if __name__=='__main__':final()
