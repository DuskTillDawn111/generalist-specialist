import sys,shutil,time
from pathlib import Path
OUT=Path('/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist');sys.path[:0]=[str(OUT),str(OUT.parents[1])]
from m32.common import *
f=read(OUT/'experiment_manifest.json');assert not f['slots']
raw=read(OUT/'raw_runs/B_S.json');assert not raw['phases'] and not raw.get('path_probe_records')
a=OUT/'initial_ready_failure';a.mkdir(exist_ok=False)
for p in list(OUT.iterdir()):
 if p.is_file() and p.name not in ['recover_startup.py','protected_before.json']:
  shutil.copy2(p,a/p.name)
for name in ['raw_runs','run_attempts','thermal_prelaunch','request_journals','stage_B']:
 p=OUT/name
 if p.exists():shutil.move(p,a/name)
for name in ['campaign_failure.json','campaign.log']:
 (OUT/name).unlink()
old=ROOT/'results/module26_positive_probe/launchers/whisper_large_v3.py'
shutil.copy2(OUT/'launchers/whisper_large_v3.py',a/'failed_launcher.py')
shutil.copy2(old,OUT/'launchers/whisper_large_v3.py')
f['startup_preparation_revision']=dict(at_unix_ns=time.time_ns(),failed_manifest=ref(a/'experiment_manifest.json'),failed_raw=ref(a/'raw_runs/B_S.json'),reason='Use existing Module26 task-READY adapter; inference handler/settings unchanged',formal_windows_started=0,capacity_windows_started=0,legacy_adapter=ref(old))
f['sources']=[ref(r['path']) for r in f['sources']];f['sources'].append(ref(OUT/'recover_startup.py'))
save('experiment_manifest.json',f);save('frozen_gpu_manifest.json',f);save('startup_preparation_revision.json',f['startup_preparation_revision'])
print('READY_METADATA_RECOVERED')
