"""Freeze a common full-page OCR image budget without cropping or prediction use."""
import json
from pathlib import Path
from PIL import Image, ImageOps
from gsdeploy.profiling.evaluation import read_rows
from gsdeploy.profiling.prepare import digest, write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/four-task-small-pool-20261015'

def main():
 for name in ['evaluation-v5', 'calibration-v5']:
  if (BASE/name).exists():raise FileExistsError(BASE/name)
 policy_path=ROOT/'configs/compact_protocol.json'
 policy=json.loads(policy_path.read_text())
 policy['tasks']['ocr']['common_input_preparation']={'max_image_side':2048,'resampling':'Pillow LANCZOS',
      'output':'lossless PNG after RGB/EXIF normalization','full_page':True,'crop':False,
      'gold':'unchanged verified native line order','reason':'fixed CPU and visual input budget before model output comparison'}
 policy['native_input_limits']={'qwen_image_min_pixels':3136,'qwen_image_max_pixels':1003520,
    'qwen_max_visual_tokens':1280,'all_models_receive_identical_prepared_page_bytes':True}
 policy['state']='CPU_PROTOCOL_FROZEN_GPU_UNVALIDATED'
 policy['notes'][0]='Inputs, decoding, scoring, quality gates and utility reference scales frozen before compact model outputs; GPU validation pending.'
 policy_path.write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n')
 for oldname,newname,manifestname in [('evaluation-v4','evaluation-v5','evaluation-status.json'),('calibration-v4','calibration-v5','sample-manifests.json')]:
  old=BASE/oldname;new=BASE/newname;new.mkdir()
  report=json.loads((old/manifestname).read_text());all_rows=[]
  for task,entry in report['tasks'].items():
   rows=read_rows(entry['path'])
   if task=='ocr':
    pages=new/'pages';pages.mkdir()
    for row in rows:
     source=Path(row['input']);assert digest(source)==row['input_sha256']
     with Image.open(source) as im:
      image=ImageOps.exif_transpose(im).convert('RGB');original_size=list(image.size)
      image.thumbnail((2048,2048),Image.Resampling.LANCZOS)
      target=pages/(row['id']+'.png');image.save(target)
     row.update(source_input=str(source),source_input_sha256=row['input_sha256'],source_image_size=original_size,
        input=str(target),input_sha256=digest(target),prepared_image_size=list(image.size),
        input_preparation='whole_page_rgb_exif_max_side_2048_lanczos_lossless_png_v1')
   report['tasks'][task].update(write(new/(task+'.jsonl'),rows));all_rows.extend(rows)
   if task=='asr' and newname.startswith('calibration'):write(new/'speech.jsonl',rows)
  report.update(predecessor_manifest_sha256=digest(old/manifestname),protocol_sha256=digest(policy_path))
  if newname.startswith('evaluation'):
   report['protocol_revision_note']='Same 1600 source IDs and gold references as v3/v4; all OCR pages share declared full-page resize, no crop. No compact model outputs used.'
  else:write(new/'samples.jsonl',all_rows)
  (new/manifestname).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 study_path=ROOT/'configs/study.json';study=json.loads(study_path.read_text())
 for key,folder,manifest in [('quality_preparation','evaluation-v5','evaluation-status.json'),('calibration_preparation','calibration-v5','sample-manifests.json')]:
  p=BASE/folder/manifest;study[key].update(manifest=str(p.relative_to(ROOT)),manifest_sha256=digest(p))
 study_path.write_text(json.dumps(study,ensure_ascii=False,indent=2)+'\n')
 print('Whole-page input budgets frozen for quality200/calibration32; all source IDs and gold retained.')

if __name__=='__main__':main()
