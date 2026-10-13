import sys,os,json,csv,copy,time,collections
from pathlib import Path
ROOT=Path('/root/autodl-tmp/generalist-specialist');OUT=ROOT/'results/module32_heavy_specialist'
sys.path.insert(0,str(ROOT))
from module30.common import read,ref,json_native
from module31a.common import save as oldsave,csvsave as oldcsv
import module31a.common as oldcommon
oldcommon.OUT=OUT
save=oldsave;csvsave=oldcsv
S='openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1'
OCR='baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt'
def configure():
 from module30 import gpu,telemetry,report
 from module31a import harness,campaign
 for m in [gpu,telemetry,report,harness,campaign]:
  m.OUT=OUT;m.save=save;m.csvsave=csvsave
 harness.configure=configure;campaign.configure=configure
 return harness,report

def trajectory(rates,duration,seed,label):
 from module21.trajectory import arrivals
 from joint_deploy.measurement.load_generator import SamplePool
 pool=SamplePool(ROOT/'results/module11_replica_and_cost/frozen_samples.jsonl',list(rates),seed)
 ev=arrivals(rates,duration,seed,label)
 for e in ev:e['item']=pool.next(e['task'])
 name=f'trajectories/{label}_{seed}.json';save(name,dict(duration=duration,seed=seed,label=label,events=ev));return ref(OUT/name)

def slot(f,rid,q,p,duration,seed,stage):
 rates=dict(text=.2,image=.2,ocr=.2,asr=q,lid=.2,visual_qa=.2)
 routing=dict(text={'roberta_sst2':.2},image={'convnext_tiny':.2},ocr={OCR:.2},lid={'qwen':.2},visual_qa={'qwen':.2},asr={S:q*(1-p)})
 if p:routing['asr']['qwen']=q*p
 deployment={m:1 for m in ['qwen',S,'roberta_sst2','convnext_tiny',OCR]}
 return dict(run_id=rid,candidate='C1',arm='s_only' if p==0 else 'split05' if p==.05 else 'split10' if p==.1 else 'G_probe',stage=stage,seed=seed,repetition=1,target_task='asr',Generalist='qwen',Specialist=S,deployment=deployment,lambda_rps=rates,routing=routing,measurement_seconds=duration,safe_memory_budget_mib=34567.5534,frozen_G_fraction=p,trajectories={'warmup':trajectory(rates,30,seed,'warmup_'+rid),'measurement':trajectory(rates,duration,seed,'measurement_'+rid)})
