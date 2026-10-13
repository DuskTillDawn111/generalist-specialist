"""Run exact 11B closed-loop runner against a new local freeze; no old writes."""
import os
from .common import *
from module11b import closed_loop

def main():
 manifest=read(OUT/'replica_measurement_manifest.json');freeze=read(OUT/'input_freeze.json')
 verify(freeze['implementation'])
 closed_loop.OUT=OUT
 for slot in manifest['slots']:
  output=OUT/'raw_runs'/(slot['run_id']+'.json')
  if output.exists():
   raw=read(output);assert raw['slot']==slot and raw['gpu_release']['verified']
   print('REUSE_OBSERVATION',slot['run_id'],raw['valid_session'],flush=True);continue
  print('START',slot['run_id'],slot['model'],flush=True)
  raw=closed_loop.run(slot,output)
  print('FINISH',slot['run_id'],'valid',raw['valid_session'],'throughput',raw.get('phases',{}).get('closed_loop_capacity',{}).get('throughput_rps'),'errors',raw['errors'],flush=True)
  assert raw['gpu_release']['verified'],'GPU release failed; stop campaign'
 print('CAPACITY_CAMPAIGN_COMPLETE',flush=True)
if __name__=='__main__':main()
