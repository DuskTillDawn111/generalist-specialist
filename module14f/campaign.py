"""Only the four witness-backed OCR x3 states; exact reused 11B runner."""
from .common import *
from module11b import closed_loop

def main():
 f=read(OUT/'profiling_freeze.json');verify(read(OUT/'input_freeze.json')['implementation']);closed_loop.OUT=OUT
 for slot in f['slots']:
  p=OUT/'raw_runs'/(slot['run_id']+'.json')
  if p.exists():
   r=read(p);assert r['slot']==slot and r['gpu_release']['verified'];print('REUSE',slot['run_id'],r['valid_session'],flush=True);continue
  print('START',slot['run_id'],slot['model'],flush=True);r=closed_loop.run(slot,p)
  print('FINISH',slot['run_id'],r['valid_session'],r.get('phases',{}).get('closed_loop_capacity',{}).get('throughput_rps'),r['errors'],flush=True)
  assert r['gpu_release']['verified'],'Unreleased GPU; stop'
 print('X3_CAMPAIGN_COMPLETE',flush=True)
if __name__=='__main__':main()
