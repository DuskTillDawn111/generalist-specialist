"""Hash loaded-model assets once, retain frozen revision identity."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/quality_workload_audit'
def main():
 p=OUT/'model_asset_hashes.json'
 if p.exists():return
 manifest=json.loads((OUT/'quality_evaluation_manifest.json').read_text())
 d={}
 for name,m in manifest['models'].items():
  folder=Path(m['local_directory'])
  files={}
  for asset in sorted(folder.rglob('*')):
   if asset.is_file() and '__pycache__' not in asset.parts and '.cache' not in asset.parts:
    h=hashlib.sha256()
    with asset.open('rb') as stream:
     for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    files[str(asset)]={'sha256':h.hexdigest(),'bytes':asset.stat().st_size}
  d[name]={'frozen_source_revision':m['source_revision'],'files':files}
  print(name,'hashed',len(files),'files',flush=True)
 p.write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':main()
