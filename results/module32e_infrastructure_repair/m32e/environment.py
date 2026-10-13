"""Read-only GPU operating-state monitoring, replacing the original sensor query."""
import subprocess
def thermal():
 fields=['temperature.gpu','clocks.current.sm','clocks.current.memory','pstate','power.draw','power.limit','clocks_event_reasons.active','clocks_event_reasons.hw_thermal_slowdown','clocks_event_reasons.sw_thermal_slowdown','clocks_event_reasons.sw_power_cap','fan.speed']
 v=subprocess.check_output(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
 out=dict(temperature_gpu_c=float(v[0]),sm_clock_mhz=float(v[1]),memory_clock_mhz=float(v[2]),pstate=v[3].strip(),power_draw_w=float(v[4]),power_limit_w=float(v[5]),clock_event_reasons_active=v[6].strip(),hw_thermal_slowdown=v[7].strip(),sw_thermal_slowdown=v[8].strip(),sw_power_cap=v[9].strip(),fan_percent=float(v[10]))
 return out
