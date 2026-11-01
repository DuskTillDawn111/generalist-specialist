"""Predeclare new-load validation without observing any validation output."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/generalist-specialist-project')
OLD = ROOT / 'artifacts/static-study-gpu-20261022'
BASE = ROOT / 'artifacts/static-spillover-study-20261022'
SOURCE = BASE / 'predictor-source-v1'
sys.path.insert(0, str(SOURCE))
from gsdeploy.config import SystemConfig
from gsdeploy.replay_transfer import prepare_transfer, declare_validation

def main():
    output = BASE / 'transfer-validation-asr18-v1'
    output.mkdir(exist_ok=False)
    rates = dict(text=.05, image=.05, ocr=.05, asr=1.8)
    raw = prepare_transfer(OLD / 'surrogate-v2/system-config.json', rates,
        source_manifest_path=OLD / 'surrogate-v2/manifest.json')
    (output / 'unvalidated-config.json').write_text(json.dumps(raw, indent=2)+'\n')
    # Native collection is independently declared from the frozen v5 modules.
    helper = output / 'prepare_native_packet.py'
    helper.write_text('''import copy,json,sys
from pathlib import Path
ROOT=Path('/root/autodl-tmp/generalist-specialist-project')
OLD=ROOT/'artifacts/static-study-gpu-20261022'
OUT=ROOT/'artifacts/static-spillover-study-20261022/transfer-validation-asr18-v1'
SOURCE=OLD/'execution-source-v5'
sys.path.insert(0,str(SOURCE))
from gsdeploy.config import SystemConfig
from gsdeploy.static_study import _bindings,_file,_job,validate_packet
from gsdeploy.quality import quality_digest
original=json.loads((OLD/'execution-packet-v2.json').read_text())
raw=copy.deepcopy(original['config'])
raw.update(workload=dict(text=.05,image=.05,ocr=.05,asr=1.8),horizon_seconds=1800,random_seed=42,route_grid=8)
raw['catalog']['performance_mode']='task_replica_proxy'
raw['catalog']['memory_mode']='standalone_proxy'
raw['catalog']['measurement_context']={**raw['catalog'].get('measurement_context',{}),'gpu_uuid':'GPU-8a262fe0-598c-46b7-c953-41d9da84a47e'}
bindings=_bindings(original['inputs'],SOURCE)+[_file(OUT/'unvalidated-config.json'),_file(__file__)]
jobs=[]
for seed,share in [(61,1.),(62,2/3)]:
 c=SystemConfig(**{**raw,'horizon_seconds':600,'random_seed':seed})
 route=dict(text={'generalist_2':.05},image={'generalist_2':.05},ocr={'ocr':.05},asr={'asr':1.8*share})
 if share<1:route['asr']['generalist_2']=1.8*(1-share)
 plan={'deployment':dict(asr=2,generalist_2=1,ocr=1),'routing':route}
 assert c.validate_plan(**plan)['constraints_valid']
 job=_job(c,original['workers'],plan,'validation_asr18_seed'+str(seed),'calibration',seed,bindings,30)
 job['purpose']='replay_transfer_independent_validation'
 jobs.append(job)
p=dict(schema='gsdeploy.static_experiments.v1',status='NEW_LOAD_VALIDATION_PREDECLARED',inputs=original['inputs'],config=raw,workers=original['workers'],bindings=bindings,frozen_files_sha256=quality_digest(bindings),gpu_started=False,jobs=jobs)
assert validate_packet(p)['valid']
(OUT/'experiment-packet.json').write_text(json.dumps(p,indent=2)+'\\n')
''')
    subprocess.run([sys.executable, str(helper)], check=True, cwd=ROOT)
    protocol = declare_validation(SystemConfig(**raw), output / 'experiment-packet.json')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    manifest = {'files': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
                'fitted_on_new_observations':False, 'validation_data_observed_before_declaration':False,
                'predictor_source':str(SOURCE), 'gpu_started':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'TWO_NEW_LOAD_HOLDOUTS_PREDECLARED','seeds':[61,62],'shares':[1.,2/3],
        'output':str(output),'old_v2_validation_transferred':False}))

if __name__=='__main__':
    main()
