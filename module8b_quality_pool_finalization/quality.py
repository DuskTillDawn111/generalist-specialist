"""Freeze user-selected historical SLA plus semantic-only VQA/LID admission."""
import csv
from .common import *
def main():
 with (QUALITY/'quality_matrix.csv').open() as f:quality=list(csv.DictReader(f))
 scores={(r['task'],r['model']):r for r in quality}
 scenario=ROOT/'benchmark/profiles/placement_scenario.json'
 history=read(scenario)['quality_presets']
 executed=ROOT/'benchmark/results/placement/placement_specialist_baseline_g1.json'
 assert read(executed)['quality_preset']=='specialist_baseline'
 policies=[]
 for task in TASKS:
  h=history['specialist_baseline'].get(task)
  p={'task':task,'metric':h['metric'] if h else 'accuracy','direction':h['op'] if h else 'higher_is_better_report_only',
     'historical_threshold_available':h is not None,'historical_threshold':h['value'] if h else None,
     'policy_status':'HISTORICAL_THRESHOLD' if h else 'SEMANTIC_VALIDITY_ONLY',
     'quality_threshold_status':'SPECIFIED_HISTORICAL' if h else 'UNSPECIFIED',
     'admission_rule':f"capability verified AND quality measured under frozen semantics AND {h['metric']} {h['op']} {h['value']} using full-precision score" if h else
       'capability verified AND quality measured with valid frozen task/input/prompt/decoding/scoring semantics; no numeric accuracy threshold; per-sample mistakes or invalid answers remain counted as wrong, not an invented path exclusion rule',
     'source':{'user_selection':'2026-10-03: 历史 SLA + VQA/LID 语义准入（推荐）',
               'historical_presets':ref(scenario),'historical_executed_placement':ref(executed) if h else None,
               'task_definition':ref(ROOT/'results/candidate_pool_v2/task_definitions.json')},
     'measured_metric_key':'top1_accuracy' if task=='image' else h['metric'] if h else 'accuracy',
     'metric_alias_note':'Historical accuracy is the same closed-set top-1 accuracy stored as top1_accuracy in Module 8A' if task=='image' else None,
     'secondary_report_metrics':['cer'] if task=='ocr' else []}
  policies.append(p)
 policy={'module':'8B','frozen':True,'selected_historical_preset':'specialist_baseline','tasks':policies,
         'selection_source':'Explicit user response before planning: historical SLA + VQA/LID semantic admission',
         'alternative_historical_presets_preserved':history,
         'new_numeric_thresholds_created':False,'policy_scope':'per task-model path; remove failed routing edges, not whole model identities',
         'semantic_only_scope':'VQA/LID admitted for verified semantics without numeric quality SLO. FINAL_READY, if true, applies to this declared policy and does not certify a future accuracy SLO.'}
 write(OUT/'quality_policy.json',policy,freeze=True)
 pool=read(POOL_OUT/'candidate_pool_v2_1.json');models={m['id']:m for m in pool['models'] if m['id'] in pool['formal_model_ids']}
 vqa=read(QUALITY/'generalist_vqa_revalidation.json')['results']
 rows=[]
 for name,model in models.items():
  for p in policies:
   task=p['task'];verified=task in model['capabilities_verified']
   s=scores.get((task,name));source=ref(QUALITY/'quality_matrix.csv');q=None
   if name in vqa and task=='visual_qa':
    r=vqa[name]
    assert r['status']=='VERIFIED'
    q=r['quality']['accuracy'];measured=r['quality']['n']==200
    source=ref(QUALITY/'generalist_vqa_revalidation.json')
   elif s and s['evaluation_status']=='COMPLETE':
    q=float(s[p['measured_metric_key']]);measured=int(s['n'])==200
   else:measured=False
   semantic_valid=verified and measured
   if not verified:admitted=False;reason='CAPABILITY_NOT_VERIFIED';blocked_quality=False
   elif not measured:admitted=False;reason='QUALITY_NOT_MEASURED';blocked_quality=False
   elif p['policy_status']=='HISTORICAL_THRESHOLD':
    admitted=(q>=p['historical_threshold']) if p['direction']=='>=' else (q<=p['historical_threshold'])
    reason='HISTORICAL_THRESHOLD_PASS' if admitted else 'HISTORICAL_THRESHOLD_FAIL'
    blocked_quality=not admitted
   else:
    admitted=semantic_valid;reason='SEMANTIC_VALIDITY_ONLY_NO_NUMERIC_THRESHOLD';blocked_quality=False
   rows.append({'task':task,'model':name,'capability_status':model['task_compatibility'][task],
                'capability_verified':verified,'quality_measured':measured,'semantic_validity_verified':semantic_valid,
                'quality_metric':p['metric'],'measured_quality':q,'historical_threshold':p['historical_threshold'],
                'quality_admissible':admitted,'planning_admissible':admitted,'blocked_by_quality_threshold':blocked_quality,
                'admission_reason':reason,'policy_status':p['policy_status'],'source':source,
                'secondary_cer':float(s['cer']) if s and s.get('cer') else None,
                'invalid_output_count':int(s['invalid_output_count']) if s and s.get('invalid_output_count') else vqa[name]['quality']['invalid_output_count'] if task=='visual_qa' and name in vqa else None})
 assert sum(r['capability_verified'] for r in rows)==27
 assert sum(r['quality_measured'] for r in rows)==27
 admission={'module':'8B','frozen':True,'policy_source':ref(OUT/'quality_policy.json'),'candidate_pool_source':ref(POOL_OUT/'candidate_pool_v2_1.json'),
            'rows':rows,'verified_paths':27,'measured_paths':27,'admitted_paths':sum(r['quality_admissible'] for r in rows),
            'quality_threshold_excluded_paths':[{'task':r['task'],'model':r['model'],'quality':r['measured_quality'],'threshold':r['historical_threshold']} for r in rows if r['blocked_by_quality_threshold']],
            'no_whole_model_elimination':'Models retain all verified capabilities in v2.1; planning restricts routing to admitted edges.'}
 write(OUT/'quality_admission_matrix.json',admission,freeze=True)
 print('Frozen quality policy:',len(admission['quality_threshold_excluded_paths']),'paths excluded by unchanged historical thresholds;',admission['admitted_paths'],'admitted paths')
if __name__=='__main__':main()
