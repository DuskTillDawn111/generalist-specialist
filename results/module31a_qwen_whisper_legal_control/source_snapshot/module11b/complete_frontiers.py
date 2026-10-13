"""Sequential next-count evidence only for the original breakpoint models."""
import fcntl,subprocess
from .common import *
from .campaign import closed_loop_frontier
from .analyze_replica import main as analyze,calibration

def main():
    lock=(OUT/'GPU_execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(OUT/'measurement_plan.json');c,hw,roles,base=inputs();initial=plan['required_initial_frontier']
    assert (OUT/'closed_loop_initial_status.json').exists()
    assert not subprocess.run(['pgrep','-af','[n]vidia-cuda-mps'],capture_output=True,text=True).stdout.strip()
    history=read(OUT/'continuation_requirements.json')['rows'] if (OUT/'continuation_requirements.json').exists() else []
    # Round-robin next frontiers. Each run is measured and analyzed before selecting
    # another count; no unmeasured g(x) is inserted anywhere.
    while True:
        analyze();scaling=read(OUT/'replica_scaling.json');cal=calibration()
        pending=[r for r in initial if roles[r['model']]=='specialist' and r['model'] not in scaling['terminal_frontiers']]
        if not pending:break
        for original in pending:
            analyze();scaling=read(OUT/'replica_scaling.json');cal=calibration();m=original['model']
            if m in scaling['terminal_frontiers']:continue
            x=cal.evidence_bounds[m]+1
            if x==original['required_x']:
                raise RuntimeError(f'Initial closed-loop frontier incomplete or low-load unavailable without confirmed terminal: {m}/{x}')
            required={**original,'current_max_measured_x':x-1,'required_x':x,
                'reason':original['reason']+['SEQUENTIAL_EMPIRICAL_STOPPING_AT_SAME_BREAKPOINT_SPECIALIST_FRONTIER'],
                'preceding_empirical_point':next(p for p in scaling['new_points'] if p['model']==m and p['x']==x-1)}
            if not any(r['model']==m and r['required_x']==x for r in history):
                history.append(required);save('continuation_requirements.json',{'rows':history,'policy':'original breakpoint contexts/models only; stop first measured terminal'})
                csvsave('replica_evidence_required.csv',initial+history)
            closed_loop_frontier(required,cal.capacity_factors[m][x-1],tag='closed_loop_continuation')
            analyze()
    save('replica_completion_status.json',{'status':'COMPLETE','at_utc':now(),
        'terminal_models':read(OUT/'replica_scaling.json')['terminal_frontiers'],
        'Generalist_scope':'required 2G counts completed; no counts beyond two total G added',
        'formal_hotspot_GPU_runs':0})
if __name__=='__main__':main()
