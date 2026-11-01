"""Apply prospectively frozen flow thresholds, without fitting or GPU work."""
import argparse
import json
import math
from pathlib import Path

from .native_flow import checked,receipt,save


def assess(plan,report):
    limits=plan['validation_thresholds'];checks=[]
    def add(name,value,threshold,minimum=False):
        known=type(value) in (int,float) and math.isfinite(value)
        checks.append(dict(name=name,value=value,threshold=threshold,
            operator='>=' if minimum else '<=',passed=(value>=threshold if minimum else value<=threshold) if known else None))
    wanted=len(plan['candidates']);got=report['candidate_coverage']['observed']
    add('candidate_coverage',got/wanted,limits['complete_candidate_coverage'],True)
    predicted={r['candidate_id']:r for r in report['candidates']}
    for candidate in plan['candidates']:
        name=candidate['id'];row=predicted.get(name,{})
        add(name+':request_coverage',row.get('known',0)/len(plan['trace']),limits['complete_request_coverage'],True)
        errors=row.get('timing_errors',{})
        for metric,key in [('service','service_mare_max'),('e2e','e2e_mare_max')]:
            add(name+':'+metric+'_mare',errors.get(metric,{}).get('mean_absolute_relative_error'),limits[key])
        add(name+':queue_mae',row.get('queue_error',{}).get('mae'),limits['queue_depth_mean_absolute_error_max_requests'])
        p=row.get('prediction_metrics') or {};o=row.get('observed_metrics') or {}
        for task in plan['workload_rps']:
            pv=p.get('per_task',{}).get(task,{}).get('p95_ms')
            ov=o.get('per_task',{}).get(task,{}).get('p95_ms')
            error=abs(pv-ov)/ov if pv is not None and ov is not None and ov>0 else None
            add(name+':'+task+'_p95_are',error,limits['each_task_p95_absolute_relative_error_max'])
    pair=report['paired_delta_coverage'];rank=report['rank_coverage']
    add('paired_coverage',pair['known']/pair['requested'] if pair['requested'] else None,1.,True)
    add('rank_coverage',rank['known']/rank['requested'] if rank['requested'] else None,1.,True)
    add('delta_sign_accuracy',report.get('delta_sign_accuracy'),limits['paired_delta_sign_accuracy_min'],True)
    add('rank_accuracy',report.get('rank_accuracy'),limits['candidate_pair_rank_accuracy_min'],True)
    status='INCOMPLETE' if any(r['passed'] is None for r in checks) else 'PASS' if all(r['passed'] for r in checks) else 'FAIL'
    return dict(schema='gsdeploy.native_flow_frozen_gate.v1',status=status,checks=checks,
        failed_checks=[r['name'] for r in checks if r['passed'] is False],
        unknown_checks=[r['name'] for r in checks if r['passed'] is None],
        scope=limits['scope'],final_four_groups_enabled=False,
        independent_units=1,confidence_interval_calibrated=False,
        claim='Predictor pilot gate only; original stability/resource qualification and repeated final confirmation remain required')


def run(plan_path,report_path,output):
    plan=json.loads(Path(plan_path).read_text());report=json.loads(Path(report_path).read_text())
    sealed=json.loads(checked(report['sealed']).read_text())
    for ref in [*report.get('raw_sources',[]),report['report_source'],*sealed['sources'].values(),
                *sealed.get('training_sources',[]),*sealed.get('resource_sources',[]),
                *sealed.get('execution_sources',[]),*sealed.get('quality_sources',[]),
                *sealed.get('predictor_sources',[]),sealed['tool_source'],sealed['metric_source']]:
        checked(ref)
    if sealed['sources']['plan']!=receipt(plan_path):
        raise ValueError('Gate plan differs from pre-measurement sealed plan')
    result=assess(plan,report);result.update(plan=receipt(plan_path),report=receipt(report_path),tool=receipt(__file__))
    save(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    p.add_argument('--report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.plan,a.report,a.output);print(json.dumps(dict(status=r['status'],failed=r['failed_checks'],unknown=r['unknown_checks'])))
