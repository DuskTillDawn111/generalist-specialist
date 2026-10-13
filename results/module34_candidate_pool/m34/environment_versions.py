from .common import *
import subprocess,platform

def main():
 f=read(OUT/'frozen_gpu_manifest_v2.json');by={}
 for m,r in f['registry_original'].items():by.setdefault(r['python'],[]).append(m)
 code="import importlib.metadata as m,json,sys;out={'python':sys.version,'executable':sys.executable};\nfor p in ['torch','numpy','transformers','triton','torchaudio','librosa','Pillow','timm','openai-whisper']:\n try:out[p]=m.version(p)\n except m.PackageNotFoundError:out[p]=None\nprint(json.dumps(out))"
 rows=[]
 for py,models in by.items():
  env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(OUT/'work/tmp')};p=subprocess.run([py,'-c',code],capture_output=True,text=True,env=env,check=True);rows.append(dict(models=models,versions=json.loads(p.stdout)))
 save('software_environment_versions.json',dict(captured_unix_ns=time.time_ns(),platform=platform.platform(),CPU_platform=platform.processor(),machine=platform.machine(),packages=rows,historical_full_package_versions='Not complete in14F profile manifest; do not infer full historical equality from same environment paths. No package installs or environment changes were performed by34.',native_registry=f['registry_original'],GPU=f['GPU']))
 print('ENVIRONMENT_VERSIONS',len(rows))
if __name__=='__main__':main()
