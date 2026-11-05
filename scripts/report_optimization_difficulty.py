"""Summarize independently recorded diagnostics and plot nine evidence categories."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.optimization_difficulty_diagnosis import write_csv
from experiments.joint_scalability import ROOT, source_manifest
from experiments.diagnose_continuous_phase1 import save


def read(path): return json.loads(Path(path).read_text())
def csv_rows(path):
    with Path(path).open(newline='') as stream: return list(csv.DictReader(stream))
def number(value): return float(value) if value not in ('',None) else np.nan


def report(source, out):
    out.mkdir(parents=True,exist_ok=False)
    if not read(source/'step-cde-complete.json')['source_unchanged']:
        raise AssertionError('incomplete C/D/E')
    runs=[];rounds=[];spaces=[];ablations=[]
    for case in read(source/'cases.json'):
        raw=read(source/case['raw']);r=raw['result'];t=raw['timing'];m=r['metrics']
        spaces.append(dict(name=case['name'],**read(source/case['space'])))
        runs.append(dict(name=case['name'],pool=case['pool'],replicas=case['n'],K=case['K'],
            total_seconds=raw['total_seconds'],status=r['status'],certified=r['optimality_certified'],
            utility=(r['winner'] or {}).get('score'),full_prediction_calls=m['full_prediction_calls'],
            milp_calls=m['milp_calls'],column_generation_complete=t['column_generation_complete'],
            task_columns=m['task_columns'],column_memory_pruned=m['column_memory_pruned'],
            column_pressure_pruned=(spaces[-1]['raw_task_count_columns']-m['task_columns']-m['column_memory_pruned'])
                if t['column_generation_complete'] else None,
            variables=t['rounds'][0]['variables'] if t['rounds'] else None,
            first_constraints=t['rounds'][0]['constraints'] if t['rounds'] else None,
            **raw['exclusive_seconds'],unknown=m['unknown_candidates'],infeasible=m['infeasible_cuts'],
            feedback_cuts=m['feedback_cuts'],identity_exclusions=m['evaluated_identity_exclusions'],
            winner_deployment=(r['winner'] or {}).get('deployment'),winner_routing=(r['winner'] or {}).get('routing')))
        rounds.extend(dict(name=case['name'],**v) for v in t['rounds'])
        del raw
    for path in sorted(source.glob('ablation-*.json')):
        if path.name == 'ablation-config.json':
            continue
        a=read(path);r=a['result']
        ablations.append(dict(ablation=a['ablation'],full_prediction_calls=r['metrics']['full_prediction_calls'],
            milp_calls=r['metrics']['milp_calls'],status=r['status'],certified=r['optimality_certified'],
            utility=(r['winner'] or {}).get('score'),total_seconds=a['total_seconds'],
            columns_seconds=a['timing']['columns_seconds'],highs_seconds=a['timing']['highs_seconds'],
            prediction_seconds=a['timing']['prediction_seconds'],
            timing_warning='ablation floor decomposition is diagnostic Python work; compare counts, not acceleration'))
    write_csv(out/'runs.csv',runs);write_csv(out/'master-rounds.csv',rounds)
    write_csv(out/'search-spaces.csv',spaces);write_csv(out/'ablations.csv',ablations)
    # Independent K1 complete-domain floor ranking. Hard-rejected columns are
    # counted explicitly, not assigned invented finite lower bounds.
    bounds=[]
    config=read(source/'p5-n1-K1-config.json');c=config['catalog'];s=c['calibrated_replay_v2']
    raw=read(source/'p5-n1-K1-joint.json')
    rank=read(source/'p5-K1-ranking.json');winner=rank['ranked_plans'][0]
    for plan in rank['plans']:
        cols=[]
        for task,row in plan['routing'].items():
            found=next((v for v in raw['column_parts'] if v['task']==task and v['routing']==row
                        and all(plan['deployment'].get(m)==n for m,n in v['deployment'].items())),None)
            if found is None: break
            cols.append(found)
        if len(cols)!=len(config['workload']):
            bounds.append(dict(index=plan['index'],master_path_present=False,utility=plan['utility']))
            continue
        memory=config['weights']['M']*sum(c['safe_factor']*s['memory_multiplier']*c['memory_table_mib'][m][str(n)]
                                         for m,n in plan['deployment'].items())/config['gpu_memory_mib']
        state_energy=config['weights']['E']*sum(plan['deployment'].values())*(s['power_coefficients'][2]
            +s['startup_joules_per_worker']/config['startup_amortization_seconds'])/config['energy_reference_watts']
        parts={k:sum(col['parts'][k] for col in cols) for k in ('quality','route_energy','no_wait_service','isolated_fifo_increment')}
        floor=memory+state_energy+sum(parts.values())
        record=dict(index=plan['index'],deployment=plan['deployment'],routing=plan['routing'],
            master_path_present=True,valid=plan['valid'],utility=plan['utility'],memory=memory,
            state_energy=state_energy,**parts,floor_cost=floor,
            residual=(-plan['utility']-floor) if plan['valid'] else None,
            floor_can_beat_incumbent=floor < -winner['utility']-1e-8)
        if plan['valid'] and floor > -plan['utility']+1e-8: raise AssertionError('floor exceeds full cost')
        bounds.append(record)
    write_csv(out/'complete-plan-bounds.csv',bounds)
    competing=[b for b in bounds if b['master_path_present'] and b['index']!=winner['index']]
    best_other=min(competing,key=lambda b:b['floor_cost'])
    certificate=dict(first_master=raw['timing']['rounds'][0],second_master=raw['timing']['rounds'][1],
        history=raw['result']['history'],cuts=raw['timing']['cuts'],
        gap_first_two_deployments=rank['gap_first_two_deployments'],
        exhaustive_K1_predictions=rank['full_prediction_calls'],exhaustive_K1_valid=rank['valid_plans'],
        min_other_plan_floor=best_other['floor_cost'],best_other_plan=best_other,
        other_plan_floor_margin_over_incumbent=best_other['floor_cost']+winner['utility'],
        other_plans_with_competitive_floor=sum(b['floor_can_beat_incumbent'] for b in competing),
        complete_plan_bound_checks=sum(b['master_path_present'] and b.get('valid',False) for b in bounds))
    save(out/'certificate-explanation.json',certificate)
    # Evidence coverage inventories: actual calibration jobs versus mathematical
    # transfers to unmeasured combinations remain separate columns.
    all_config=read(source/'source-config.json');spec=all_config['catalog']['calibrated_replay_v2']
    packets={};conditions=[]
    for condition in spec['protocol']['conditions']:
        path=condition['packet_path']
        if path not in packets: packets[path]=read(path)
        job=next(j for j in packets[path]['jobs'] if j['id']==condition['id'])
        identity=job['identity']
        conditions.append(dict(id=condition['id'],purpose=condition['purpose'],
            deployment=identity['deployment'],routing=identity['routing'],workload=identity['workload'],
            horizon_seconds=identity['horizon_seconds'],seed=identity['seed'],
            packet_path=path,source=next((x['path'] for x in spec['sources'] if x['id']==condition['id']),None)))
    write_csv(out/'measured-calibration-conditions.csv',conditions)
    paths=[]
    for model,md in all_config['catalog']['models'].items():
        for task in md['supported_tasks']:
            for n in sorted(map(int,all_config['catalog']['capacity_factors'][model])):
                if n==0: continue
                samples=spec['input_baselines'].get(model,{}).get(task,{}).get(str(n),{}).get('samples',{})
                ids=spec['input_order'][task]
                present=sum(i in samples for i in ids)
                kinds={k:sum(v.get('identity_kind')==k for v in samples.values())
                       for k in ('raw_request_observation','historical_profile_input_binding')}
                paths.append(dict(model=model,task=task,replicas=n,inputs=len(ids),present=present,
                    complete=present==len(ids),**kinds,
                    scope='standalone per-input baseline; not dynamic mixed-GPU validation'))
    write_csv(out/'path-coverage.csv',paths)
    coverage=[]
    for family,replicas,route_range,formal,scope_text in [
        ('G2-only','G2:1','all tasks to G2','yes when hard feasible','actual development condition; new traces remain predictions'),
        ('G2+OCR / G2+ASR / G2+Text','one each','split task r in [0,1]','yes when hard feasible','sparse measured conditions; continuous routes transferred by frozen predictor'),
        ('G2+image+OCR','one each','legal task simplexes incl. exact endpoints','yes when hard feasible','two heldout mixed routes; no dense dynamic service validation'),
        ('four Specialists','one each','pure supported task per Specialist','yes when hard feasible','actual development condition; no G-free splitting'),
        ('legal 3/5/7 model pools','catalog states only','K1..16 finite grids / supported simplexes','conditional','validate all routed paths, memory, rho and drain; not every combination measured'),
        ('G1/G3 or high replica counts','catalog states only','known routed paths only','conditional','standalone transfer; not present in the ten mixed calibration conditions'),
        ('baseline pools','catalog states','low/high/hotspot or task subset','no','V2_REPLAY_INPUT_CONTEXT_OR_WORKLOAD_CHANGED'),
        ('route_load_replay_v1','none validated','no formal routes','no','ROUTE_LOAD_MODEL_NOT_IMPLEMENTED')]:
        accepted=formal.startswith('yes') or formal=='conditional'
        coverage.append(dict(model_family=family,replicas=replicas,tasks='text/image/ocr/asr',
            workload='each 0.0748360842705697 qps' if accepted else 'changed or unimplemented',
            routes=route_range,colocation_service='validated global-ratio transfer, not route-dependent' if accepted else 'unavailable',
            FIFO='implemented with frozen service; hard feasibility checked per plan' if accepted else 'no formal U',
            formal_joint_utility=formal,route_load_dynamic_interference='not implemented',evidence_scope=scope_text))
    write_csv(out/'predictor-coverage-table.csv',coverage)
    gates=read(source/'scope.json')+read(source/'scope-gates-ab.json')
    write_csv(out/'coverage-gates.csv',gates)
    save(out/'predictor-evidence.json',dict(mode='calibrated_replay_v2',
        spec_status=spec['status'],version_sha256=spec['version_sha256'],
        measurement_context=spec['measurement_context'],validation=spec['validation'],
        calibration_conditions=len(conditions),development=len(spec['development_ids']),holdout=len(spec['validation_ids']),
        service_multiplier=spec['service_multiplier'],memory_multiplier=spec['memory_multiplier'],
        power_coefficients=spec['power_coefficients'],startup_joules_per_worker=spec['startup_joules_per_worker'],
        active_route_load_service=False,route_load_replay_v1='UNIMPLEMENTED',
        distinct_workloads_measured=len({json.dumps(x['workload'],sort_keys=True) for x in conditions}),
        hardware_access=read(source/'hardware-access.json')))
    # Plot titles deliberately identify frozen predictions, never measurements.
    def finish(name):
        plt.tight_layout();plt.savefig(out/name,dpi=170);plt.close()
    chosen=[next(s for s in spaces if s['name']==n) for n in
            ('p3-n1-K1','p5-n1-K1','p7-n1-K1','p5-n2-K2','p7-n2-K2','p7-n99-K2')]
    x=np.arange(len(chosen));plt.figure(figsize=(10,4))
    for offset,key,label in [(-.25,'raw_deployment_combinations','Raw Cartesian'),
          (0,'exact_memory_survivors','Memory survivors'),(.25,'coverage_survivors','Memory + coverage')]:
        plt.bar(x+offset,[s[key] for s in chosen],width=.25,label=label)
    plt.yscale('log');plt.xticks(x,[s['name'] for s in chosen]);plt.ylabel('Deployment vectors (log)')
    plt.title('1. Effective deployment domain; necessary filters');plt.legend();finish('01-deployment-filters.png')
    plt.figure(figsize=(9,4));top=rank['ranked_deployments'][:8]
    plt.bar(np.arange(len(top)),[-r['utility'] for r in top]);plt.xticks(np.arange(len(top)),
        ['+'.join(r['deployment']) for r in top],rotation=25,ha='right',fontsize=8)
    plt.ylabel('Cost = -U (lower is better)');plt.title(f'2. Complete K1 deployment ranking; gap={rank["gap_first_two_deployments"]:.6f}')
    finish('02-utility-gaps.png')
    plt.figure(figsize=(8,4))
    for pool in (2,5,7):
        rr=[r for r in runs if r['pool']==pool and r['replicas']==1]
        plt.plot([r['K'] for r in rr],[r['full_prediction_calls'] for r in rr],'o-',label=f'{pool} models')
        for r in rr:
            if not r['certified']: plt.annotate('uncertified',(r['K'],r['full_prediction_calls']),fontsize=7)
    plt.xlabel('K');plt.ylabel('Full prediction bundles');plt.title('3. Prediction calls, frozen replay');plt.legend();finish('03-K-predictions.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for pool in (2,5,7):
        rr=[r for r in runs if r['pool']==pool and r['replicas']==1]
        axs[0].plot([r['K'] for r in rr],[r['columns'] for r in rr],'o-',label=f'{pool} models')
        axs[1].plot([r['K'] for r in rr],[r['highs'] for r in rr],'o-',label=f'{pool} models')
    axs[0].set(xlabel='K',ylabel='seconds',title='4a. Route-column construction')
    axs[1].set(xlabel='K',ylabel='seconds',title='4b. HiGHS backend')
    for ax in axs:ax.legend()
    finish('04-K-construction-highs.png')
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    winning=next(b for b in bounds if b['index']==winner['index'])
    keys=['memory','state_energy','quality','route_energy','no_wait_service','isolated_fifo_increment','residual']
    axs[0].bar(np.arange(len(keys)),[winning[k] for k in keys]);axs[0].set_xticks(np.arange(len(keys)),keys,rotation=35,ha='right',fontsize=8)
    axs[0].set(title='5a. Winner cost floor + exact residual',ylabel='Weighted cost')
    axs[1].bar(np.arange(len(ablations)),[r['full_prediction_calls'] for r in ablations]);axs[1].set_xticks(np.arange(len(ablations)),
        [r['ablation'] for r in ablations],rotation=35,ha='right',fontsize=8)
    for i,r in enumerate(ablations):
        if not r['certified']:
            axs[1].annotate('budget stop',(i,r['full_prediction_calls']),ha='center',fontsize=7)
    axs[1].set(title='5b. Relaxation ablations; same full predictor',ylabel='Full prediction calls')
    finish('05-bounds-and-ablations.png')
    controlled=csv_rows(source/'controlled-paths.csv')
    curve_tasks={'g2_ocr':'ocr','g2_asr':'asr','winner_text':'text'}
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,(curve,task) in zip(axs,curve_tasks.items()):
        for model in ('generalist_2',task):
            rr=[r for r in controlled if r['curve']==curve and r['task']==task and r['model']==model]
            ax.plot([number(r['ratio']) for r in rr],[number(r['pool_service_ms']) for r in rr],label=f'{model} pool')
            ax.plot([number(r['ratio']) for r in rr],[number(r['mean_service_ms']) for r in rr],'--',label=f'{model} assigned')
        ax.set(title=curve,xlabel='Specialist share',ylabel='service ms');ax.legend(fontsize=7)
    fig.suptitle('6. Frozen services: pool vs routed input subset');finish('06-route-service.png')
    controlled_models=csv_rows(source/'controlled-models.csv')
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,(curve,task) in zip(axs,curve_tasks.items()):
        for model in ('generalist_2',task):
            rr=[r for r in controlled_models if r['curve']==curve and r['model']==model]
            ax.plot([number(r['ratio']) for r in rr],[number(r['mean_waiting_ms']) for r in rr],'o-',label=model)
        ax.set(title=curve,xlabel='Specialist share',ylabel='mean FIFO wait ms');ax.legend(fontsize=7)
    fig.suptitle('7. FIFO changes under identical fixed service law');finish('07-route-waits.png')
    activation=csv_rows(source/'activation.csv');plt.figure(figsize=(12,5));bottom=np.zeros(len(activation))
    for k in 'QTLPME':
        v=np.array([-number(r[f'{k}_contribution']) for r in activation]);plt.bar(np.arange(len(v)),v,bottom=bottom,label=k);bottom+=v
    plt.yscale('symlog',linthresh=.5);plt.xticks(np.arange(len(activation)),[r['name'] for r in activation],rotation=25,ha='right')
    plt.ylabel('Weighted costs = -contributions (symlog)');plt.title('8. Predicted activation cost components');plt.legend(ncol=6)
    finish('08-activation-components.png')
    counts={}
    for r in gates:
        # Duplicate gate probes across A and C are attempts, not distinct workloads.
        counts[r['status']]=counts.get(r['status'],0)+1
    plt.figure(figsize=(11,4));plt.barh(np.arange(len(counts)),list(counts.values()))
    plt.yticks(np.arange(len(counts)),list(counts),fontsize=8);plt.xlabel('Gate probe attempts (A and C; duplicates retained)')
    plt.title('9. Coverage gate acceptance and rejection');finish('09-coverage-gates.png')
    summary=dict(runs=len(runs),ablations=len(ablations),spaces=len(spaces),certificate=certificate,
        raw_prediction_bundles=sum(r['full_prediction_calls'] for r in runs)+sum(r['full_prediction_calls'] for r in ablations)
            +sum(read(source/f'p{p}-K1-ranking.json')['full_prediction_calls'] for p in (3,5))
            +read(source/'step-cde-complete.json')['controlled_predictions']+read(source/'step-cde-complete.json')['activation_predictions'],
        figures=sorted(p.name for p in out.glob('*.png')),core_sources_unchanged=source_manifest()==read(source/'source-manifest-before.json'))
    save(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='certificate'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();report(Path(a.source).resolve(),Path(a.output).resolve())
