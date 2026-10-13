import numpy as np,collections,math
from .common import *
from module26.common import arithmetic,cohort
from module25.common import utility_inputs

def assess(raw):
 _,report=configure();o=report.observation(raw)
 ph=raw.get('phases',{}).get('measurement');checks={'valid_session':raw.get('valid_session',False),'durable_journal':o['journal_audit']['passed'],'nonblocking_control':raw.get('control_nonblocking',False)}
 if ph:
  w=ph['window'];ss=[s for s in raw['telemetry']['samples'] if w['start_time_ns']<=s['time_ns']<=w['end_time_ns']]
  ts=[s['time_ns'] for s in ss];un=[s['unix_ns'] for s in ss]
  checks.update(exact_frozen_window=w['duration_seconds']==raw['slot']['measurement_seconds'] and w['end_time_ns']-w['start_time_ns']==round(raw['slot']['measurement_seconds']*1e9),
   trace_matched=o['matched_trace'],native_timestamps=o['native_timings_consistent'],full_cohort=o['offered']==o['completed'],
   sensor_values=bool(ss) and all(all(s.get(k) is not None and np.isfinite(s[k]) for k in ['time_ns','unix_ns','used_mib','owned_used_mib','gpu_util_percent','memory_util_percent','worker_CPU_core_percent','host_RSS_bytes','temperature_gpu_c','sm_clock_mhz']) for s in ss),
   monotonic_timestamps=all(b>a for a,b in zip(ts,ts[1:])) and all(b>a for a,b in zip(un,un[1:])),
   no_external_GPU=all(not s['foreign_pids'] for s in ss),
   telemetry_coverage=bool(ts) and (ts[0]-w['start_time_ns'])/1e9<2 and (w['end_time_ns']-ts[-1])/1e9<2 and max(np.diff(ts)/1e9,default=0)<5,
   bounded_drain=w['drain_seconds']<=60.1,
   unix_perf_alignment=bool(ss) and max(abs((s['unix_ns']-w['start_unix_ns'])-(s['time_ns']-w['start_time_ns']))/1e9 for s in ss)<1)
  d,refs=utility_inputs();v=arithmetic(raw,read(OUT/'selected_pool_catalog.json'),refs,d['weights']);o['raw_arithmetic']=v
  checks['utility_recomputed']=o.get('components',{}).get('U') is None or abs(o['components']['U']-v['raw_arithmetic_U'])<1e-12
  o['telemetry_max_gap_seconds']=max(np.diff(ts)/1e9,default=None)
 o['validation']=checks;o['technical_valid']=all(checks.values());o['eligible']=bool(o['eligible'] and o['technical_valid'])
 if not o['eligible']:
  o['U']=None
  if 'components' in o:o['components']['U']=None
 o['window_status']='VALID' if o['technical_valid'] else 'INVALID'
 return o
