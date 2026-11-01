"""Independent old/new interval arithmetic and actual Fraction cross-check."""
import json
import hashlib
import sqlite3
from collections import defaultdict
from bisect import bisect_right
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027.context_survival_v12_4 import ContextEnvelopeSurvival,SCHEMA,clip_episodes
from tools.predictor_validation_20261027.profiling import digest
from tests import test_context_survival_v12_independent_review as old_counterexamples
from tests.test_context_survival_v12_numeric_independent_review import independently_query,merged

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'artifacts/predictor-context-survival-v12_4-20261028-training01/model.json'


class NewSurvivalIndependentTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_exact_half_survival_has_median_after_five_small_events(self):
        # Known v12_4 failure: exact risk denominators alone do not make the
        # accumulated float survival a safe threshold comparison.
        rows=[old_counterexamples.interval(c,str(i),0,i if i<=5 else 10,i<=5)
              for c in ('a','b') for i in range(1,11)]
        model=dict(schema=SCHEMA,training_only=True,episodes=rows);model['model_sha256']=digest(model)
        fit=ContextEnvelopeSurvival(model,allow_unvalidated_survival_hypothesis=True,allow_cpu_fixture=True)
        exact=independently_query(rows,0)
        self.assertEqual(exact[-1],5)
        self.assertEqual(fit.query('held',old_counterexamples.F,0)['median_residual_ns'],5)

    def test_all_four_prior_statistical_counterexamples_execute_on_new_provider(self):
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(old_counterexamples.ContextSurvivalIndependentTests)
        with patch.object(old_counterexamples,'ContextEnvelopeSurvival',ContextEnvelopeSurvival),patch.object(old_counterexamples,'SCHEMA',SCHEMA):
            result=unittest.TestResult();suite.run(result)
        self.assertEqual(result.testsRun,4);self.assertEqual(result.failures,[]);self.assertEqual(result.errors,[])

    def test_clip_preserves_restore_age_censors_boundaries_and_no_overlap(self):
        self.assertEqual(clip_episodes(0,100,20,80,[30,60]),[
            dict(start_ns=20,end_ns=30,age_entry_ns=20,age_exit_ns=30,save_observed=False),
            dict(start_ns=30,end_ns=60,age_entry_ns=30,age_exit_ns=60,save_observed=False),
            dict(start_ns=60,end_ns=80,age_entry_ns=60,age_exit_ns=80,save_observed=False)])
        first=clip_episodes(0,100,20,50,[]);second=clip_episodes(0,100,50,100,[])
        self.assertEqual(first[0]['age_exit_ns'],second[0]['age_entry_ns']);self.assertTrue(second[0]['save_observed'])
        for cuts in ([60,30],[30,30]):
            with self.assertRaises(ValueError):clip_episodes(0,100,20,80,cuts)
        self.assertEqual(clip_episodes(0,100,100,110,[]),[])

    def test_actual_32_supported_cells_match_independent_exact_risksets(self):
        model=json.loads(MODEL.read_text());provider=ContextEnvelopeSurvival(model,allow_unvalidated_survival_hypothesis=True,allow_cpu_fixture=True)
        checked=0
        for (owner,_),rows in sorted(provider.groups.items(),key=lambda item:str(item[0])):
            if len({r['capture'] for r in rows})<2:continue
            selected=None
            for age in sorted({0,*[r['age_entry_ns'] for r in rows]}):
                expected=independently_query(rows,age)
                if expected is not None:selected=(age,expected);break
            if selected is None:continue
            age,(horizon,mass,tail,mean,median)=selected;actual=provider.query(owner,rows[0]['frontier'],age)
            self.assertEqual(actual['restricted_residual_horizon_ns'],horizon);self.assertEqual(actual['median_residual_ns'],median)
            self.assertEqual(len(actual['residual_event_mass']),len(mass))
            for observed,(t,p) in zip(actual['residual_event_mass'],mass):
                self.assertEqual(observed['residual_ns'],t);self.assertAlmostEqual(observed['probability'],float(p),delta=1e-11)
            self.assertAlmostEqual(actual['unresolved_tail_probability'],float(tail),delta=1e-11)
            self.assertAlmostEqual(actual['restricted_mean_residual_ns'],float(mean),delta=max(1e-4,float(mean)*1e-10))
            checked+=1
            if checked==32:break
        self.assertEqual(checked,32)

    def test_all_original_sqlite_envelopes_cover_every_job_resident_and_original_age(self):
        model=json.loads(MODEL.read_text());checked=0;frontier_checks=0;clipped=0
        for audit in model['training_audits']:
            capture=audit['raw']['sha256'];report=json.loads(Path(audit['report']['path']).read_text())
            raw=json.loads(Path(audit['raw']['path']).read_text())
            sql=next(r['path'] for r in report['sources'] if r['path'].endswith('.sqlite'))
            parsed=json.loads(Path(next(r['path'] for r in report['sources'] if Path(r['path']).name=='parsed-measured.json')).read_text())
            with sqlite3.connect('file:'+sql+'?mode=ro',uri=True) as connection:
                tags=dict(connection.execute('select id,name from ENUM_GPU_CTX_SWITCH'))
                opened={};envelopes=defaultdict(list);issues=[]
                for vm,gpu,context,pid,clock,tag in connection.execute('select vmId,gpuId,contextId,globalPid,timestamp,tag from GPU_CONTEXT_SWITCH_EVENTS order by timestamp,seqNo'):
                    key=(vm,gpu,context,pid)
                    if tags[tag]=='RESTORE_START':
                        if key in opened:issues.append(opened[key])
                        opened[key]=clock
                    else:
                        self.assertEqual(tags[tag],'SAVE_END')
                        if key in opened:envelopes[key].append((opened.pop(key),clock))
                        else:issues.append(clock)
                issues.extend(opened.values())
            mapping={w['alias']:tuple(w['context']) for w in report['controlled_temporal_attribution']['creation_windows']}
            self.assertEqual(len(mapping),4)
            windows=defaultdict(list)
            for phase in parsed['phases']:windows[phase['request_id']].append((phase['nsys_start_ns'],phase['nsys_end_ns']))
            jobs=defaultdict(list)
            for request in raw['request_records']:jobs[request['profile_job_id']].append(request)
            jobwindows={job:(min(a for r in rows for a,b in windows[r['request_id']]),max(b for r in rows for a,b in windows[r['request_id']])) for job,rows in jobs.items()}
            observations=json.loads(Path(report['kernel_ready_observations']['path']).read_text());phase_ranges=defaultdict(list)
            for node in observations['rows']:
                ready=node['certified_ready_bound_ns'];start=node['observed_start_ns']
                phase_ranges[node['request_id'],node['phase']].append((start if ready is None else min(ready,start),node['observed_end_ns']))
            indexes={key:merged(intervals) for key,intervals in phase_ranges.items()}
            pending={job:merged((a,b) for (rid,_),pieces in indexes.items() if rid in {r['request_id'] for r in rows} for a,b in pieces) for job,rows in jobs.items()}
            actual=defaultdict(list);episodes=[r for r in model['episodes'] if r['capture']==capture]
            valid={owner:set(envelopes[key]) for owner,key in mapping.items()}
            for row in episodes:
                owner=row['held_owner'];a=row['original_envelope_start_ns'];b=row['original_envelope_end_ns'];job=row['job_id']
                self.assertIn((a,b),valid[owner]);lo,hi=jobwindows[job];cliplo,cliphi=max(a,lo),min(b,hi)
                self.assertEqual(row['original_request_clip_start_ns'],cliplo);self.assertEqual(row['original_request_clip_end_ns'],cliphi)
                start=a+row['age_entry_ns'];end=a+row['age_exit_ns']
                self.assertTrue(cliplo<=start<end<=cliphi);self.assertEqual(row['save_observed'],end==b)
                expected_id=hashlib.sha256(json.dumps(dict(run=capture,owner=owner,start=a,end=b),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                self.assertEqual(row['envelope_id'],expected_id)
                actual[job,owner,a,b].append((start,end));checked+=1
            expected={};pending_starts={job:[a for a,b in pieces] for job,pieces in pending.items()}
            for job,(lo,hi) in jobwindows.items():
                for owner,key in mapping.items():
                    for a,b in envelopes[key]:
                        if any(a<=issue<=b for issue in issues):continue
                        left=max(a,lo);right=min(b,hi)
                        if left>=right:continue
                        starts=pending_starts[job]
                        begin=max(0,bisect_right(starts,left)-1);finish=bisect_right(starts,right)
                        pieces=merged((max(left,c),min(right,d)) for c,d in pending[job][begin:finish] if max(left,c)<min(right,d))
                        if pieces:expected[job,owner,a,b]=pieces;clipped+=int(a<lo or hi<b)
            self.assertEqual(set(actual),set(expected))
            for key,pieces in actual.items():
                ordered=sorted(pieces)
                self.assertTrue(all(left[1]<=right[0] for left,right in zip(ordered,ordered[1:])))
                self.assertEqual(merged(pieces),expected[key])
            for row in [episodes[(len(episodes)-1)*i//31] for i in range(32)]:
                for age in (row['age_entry_ns'],row['age_exit_ns']-1):
                    clock=row['original_envelope_start_ns']+age;frontier=[]
                    for request in jobs[row['job_id']]:
                        phases=[]
                        for (rid,phase),pieces in indexes.items():
                            if rid!=request['request_id']:continue
                            at=bisect_right(pieces,(clock,float('inf')))-1
                            if at>=0 and pieces[at][0]<=clock<pieces[at][1]:phases.append(phase)
                        if phases:frontier.append([request['model'],request['task'],request['pool_id'],sorted(phases)])
                    self.assertEqual(sorted(frontier),row['frontier']);frontier_checks+=1
        self.assertEqual(checked,len(model['episodes']));self.assertEqual(frontier_checks,128);self.assertGreater(clipped,0)


if __name__=='__main__':unittest.main()



