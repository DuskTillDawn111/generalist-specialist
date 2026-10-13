"""CPU preflight and immutable artifact/environment lock before any new GPU run."""
import os,subprocess
from module14.common import *
import pynvml as n

def main():
 assert not (OUT/'execution_freeze.json').exists()
 pre=read(OUT/'cpu_preflight_results.json');dl=read(OUT/'download_manifest.json');assert pre['completed'] and dl['completed']
 assert pre['all_success'] and dl['all_success'],'Repair minimal compatibility before locking; no checkpoint replacement'
 for model in dl['models']:
  for r in model['files']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 for r in read(OUT/'discovery_freeze_manifest.json')['files']+read(OUT/'dataset_input_hashes.json')['files']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 runtimes={}
 for name,python in [('torch',ROOT/'envs/module13b_transformers/bin/python'),('Paddle_original_runtime',Path('/root/miniconda3/envs/ocr_env/bin/python')),('Image_original_runtime',ROOT/'envs/specialist_env/bin/python')]:
  r=subprocess.run([str(python),'-m','pip','freeze','--all'],text=True,capture_output=True,check=True);runtimes[name]={'python':str(python),'pip_freeze_all':r.stdout.splitlines()}
 n.nvmlInit();h=n.nvmlDeviceGetHandleByIndex(0);hardware={'name':n.nvmlDeviceGetName(h),'uuid':n.nvmlDeviceGetUUID(h),'total_vram_mib':n.nvmlDeviceGetMemoryInfo(h).total/1048576,'driver':n.nvmlSystemGetDriverVersion(),'processes':[p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(h)]};assert not hardware['processes'];n.nvmlShutdown()
 save('local_environment.json',{'at_utc':now(),'hardware':hardware,'standard24GB_claim':False,'runtimes':runtimes,'cpu_preflight':pre,'dtype':'float32','no_old_environment_packages_modified':True})
 implementations=[ref(p) for p in sorted((ROOT/'module14').glob('*.py'))]+[ref(ROOT/'module13b'/name) for name in ['campaign.py','download.py','analyze.py']]
 save('execution_freeze.json',{'at_utc':now(),'before_any_Module14_GPU_result':True,'implementation':implementations,'environment':ref(OUT/'local_environment.json'),'baseline_artifact_provenance':ref(OUT/'baseline_artifact_provenance.json'),'discovery_manifest':ref(OUT/'discovery_freeze_manifest.json'),'download_manifest':ref(OUT/'download_manifest.json'),'CPU_preflight':ref(OUT/'cpu_preflight_results.json'),'Paddle_native_loader_compatibility':ref(OUT/'cpu_compatibility_audit.json'),'selection_rule':read(OUT/'measurement_protocol_freeze.json')['selection_rule'],'all_model_GS_deployment_outcomes_ignored':True,'no_artificial_memory_or_latency_scaling':True});print('EXECUTION_LOCKED',hardware,flush=True)
if __name__=='__main__':main()
