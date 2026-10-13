"""Lock execution implementation/environment after CPU preflight, before GPU."""
import datetime,hashlib,json,os,subprocess,sys,math
from pathlib import Path
import pynvml as n
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/module13_real_specialists'
def read(p):return json.loads(p.read_text())
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def write(name,d):(OUT/name).write_text(json.dumps(d,indent=2)+'\n')
def main():
    assert not (OUT/'module13b_execution_freeze.json').exists(),'Do not overwrite execution freeze'
    downloads=read(OUT/'download_manifest.json');pre=read(OUT/'cpu_preflight_results.json')
    assert downloads['completed'] and pre['completed']
    manifest=read(OUT/'freeze_manifest.json')
    # The 13A artifact manifest is immutable; accept its existing schema.
    refs=manifest.get('files',manifest.get('artifacts',manifest.get('references')))
    if isinstance(refs,dict):refs=list(refs.values())
    for r in refs:assert ref(Path(r['path']))['sha256']==r['sha256'],r['path']
    data_inputs=read(OUT/'dataset_input_hashes.json')
    for r in data_inputs['files']:assert ref(Path(r['path']))['sha256']==r['sha256'],r['path']
    versions={}
    for name,python in [('transformers',ROOT/'envs/module13b_transformers/bin/python'),('speechbrain',ROOT/'envs/module13b_speechbrain/bin/python'),('unchanged_reference_runtime',ROOT/'envs/specialist_env/bin/python')]:
        r=subprocess.run([str(python),'-m','pip','freeze','--all'],env={**os.environ,'OMP_NUM_THREADS':'1'},text=True,capture_output=True,check=True)
        versions[name]={'python':str(python),'pip_freeze_all':r.stdout.splitlines()}
    n.nvmlInit();h=n.nvmlDeviceGetHandleByIndex(0);m=n.nvmlDeviceGetMemoryInfo(h)
    hardware={'name':n.nvmlDeviceGetName(h),'uuid':n.nvmlDeviceGetUUID(h),'total_vram_mib':m.total/1048576,'driver':n.nvmlSystemGetDriverVersion(),'GPU_processes_before_lock':[p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(h)],'clock_graphics_mhz':n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_GRAPHICS),'clock_memory_mhz':n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_MEM),'power_limit_w':n.nvmlDeviceGetPowerManagementLimit(h)/1000,'temperature_c':n.nvmlDeviceGetTemperature(h,n.NVML_TEMPERATURE_GPU)};n.nvmlShutdown()
    assert not hardware['GPU_processes_before_lock']
    environment={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hardware':hardware,'versions':versions,'cpu_preflight':pre,'actual_frozen_audio_files_SHA_verified':len(data_inputs['files']),'dtype':'float32','device':'cuda:0 for measurements; CPU for preflight','standard_24GB_claim':False,'reference_environment_changed':False,'environment_install_logs':[str(p) for p in OUT.glob('speechbrain*install*.log')]}
    write('local_environment.json',environment)
    freeze={'at_utc':environment['at_utc'],'frozen':True,'before_any_new_GPU_result':True,'13A_immutable_manifest':ref(OUT/'freeze_manifest.json'),'implementation':[ref(p) for p in sorted((ROOT/'module13b').glob('*.py'))],'environment':ref(OUT/'local_environment.json'),'cpu_preflight':ref(OUT/'cpu_preflight_results.json'),'checkpoint_list':ref(OUT/'measurement_shortlist.json'),'measurement_protocol':ref(OUT/'measurement_protocol_freeze.json'),'selection_rule':ref(OUT/'specialist_selection_rule.json'),'safe_factor':read(ROOT/'results/module8f1a_memory_breakpoints/memory_safety_factor.json')['safety_factor'],
      'latest_user_measurement_scope':'All nine frozen candidate paths measured for quality and descriptive standalone s0/M, including numeric quality failures; selection still only admitted paths. No 13A file changed.',
      'regime_catalog_policy':{'primary':'REPLACE_TASK_EXCLUSIVE_OLD_IDENTITIES_WITH_SELECTED_NEW_WINNERS_PRESERVE_SHARED_CAPABILITIES','reason':'Keep the complete original EFFICIENCY regime unchanged as comparator. In BALANCED, remove an old Specialist identity only when all its admitted tasks have selected new winners. BASELINE_RETAINED tasks preserve their entire original pool. A shared retained old checkpoint keeps every frozen capability, including ASR if retained for LID. A selected new checkpoint keeps all quality-admitted frozen measured paths, never an unmeasured path. Generalists and OCR/Image remain unchanged. All old models participated in selection; no dominating baseline may be bypassed.', 'new_replica_counts':[0,1],'unknown_new_g_gt1':'NOT_AVAILABLE; no extrapolation or transfer of old-family replica curves','old_measured_replica_evidence':'unchanged for retained identities','shared_checkpoint_residency':'one identity; if admitted for multiple measured paths use maximum observed per-path mean idle as measured resident envelope, never sum duplicate weights','matched_count_diagnostic':'Also replay both regimes with all searched-task Specialist identities restricted to observed0/1. Keep Generalist and OCR/Image bounds unchanged. Report separately; transitions caused only by unequal evidence bounds cannot establish SUPPORTED gate.'},
      'prediction_validation_policy':'Compare global and task-local sensitivity guidance with actual joint gamma/mu; no exact breakpoint equivalence claimed. All baseline retained means no new transition test, not evidence that sensitivity mathematics is false.',
      'gate_proximity_policy':{'near_log_distance_max':math.log(1.05),'scale_source':'5% relative width of frozen Module12 first memory transition bracket mu=(1.00,1.05]','meaning':'13B gate reporting only; exact13A target segments and selection-distance formula are unchanged. Within5% multiplicative L-infinity distance is operationally NEAR, literal membership reported separately. No measured values adjusted. SUPPORTED additionally requires real structural change persisting in matched-count diagnostic.'},
      'prohibitions':['No full replica frontier','No low-QPS or multi-hotspot campaign','No Task-Rich','No Module9 mutation','No baseline mutation','No artificial delays or allocation','No new shortlist model','No selection after replay outcomes']}
    write('module13b_execution_freeze.json',freeze);print('EXECUTION_LOCKED',hardware)
if __name__=='__main__':main()
