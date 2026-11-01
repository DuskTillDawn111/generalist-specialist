"""Fraction arithmetic cross-check on deterministic actual source-model cells.

The separate review rederives all episodes from original sources. These tests
use the frozen real episodes to cross-check numeric queries independently.
"""
from fractions import Fraction
from collections import defaultdict
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import unittest

from tools.predictor_validation_20261027.context_survival_v12 import ContextEnvelopeSurvival

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'artifacts/predictor-context-survival-v12-20261028-training01/model.json'


def merged(intervals):
    result=[]
    for a,b in sorted(intervals):
        if result and a<=result[-1][1]:result[-1]=(result[-1][0],max(result[-1][1],b))
        else:result.append((a,b))
    return result


def independently_query(rows,age):
    captures={r['capture'] for r in rows}
    horizons=[]
    for capture in captures:
        pieces=merged((r['age_entry_ns'],r['age_exit_ns']) for r in rows if r['capture']==capture)
        containing=[b for a,b in pieces if a<=age<b]
        if not containing:return None
        horizons.append(containing[0])
    limit=min(horizons)
    weights={capture:Fraction(1,len(captures)*len({r['envelope_id'] for r in rows if r['capture']==capture})) for capture in captures}
    times=sorted({r['age_exit_ns'] for r in rows if r['save_observed'] and age<r['age_exit_ns']<=limit})
    survival=Fraction(1);area=Fraction(0);last=age;mass=[];median=None
    for t in times:
        risk=sum((weights[r['capture']] for r in rows if r['age_entry_ns']<t<=r['age_exit_ns']),Fraction(0))
        events=sum((weights[r['capture']] for r in rows if r['save_observed'] and r['age_exit_ns']==t),Fraction(0))
        area+=survival*(t-last);probability=survival*events/risk;survival-=probability
        mass.append((t-age,probability));last=t
        if median is None and survival<=Fraction(1,2):median=t-age
    area+=survival*(limit-last)
    return limit-age,mass,survival,area,median


class ActualSurvivalNumericIndependentTests(unittest.TestCase):
    def test_32_supported_actual_cells_match_exact_fraction_product_limit(self):
        model=json.loads(MODEL.read_text())
        # Provenance is checked in the independent full source rederive audit;
        # CPU opt-in here avoids repeating it for a numeric-only comparison.
        provider=ContextEnvelopeSurvival(model,allow_unvalidated_survival_hypothesis=True,allow_cpu_fixture=True)
        checked=0
        for (owner,_),rows in sorted(provider.groups.items(),key=lambda item:str(item[0])):
            if len({r['capture'] for r in rows})<2:continue
            candidates=sorted({0,*[r['age_entry_ns'] for r in rows]})
            selected=next(((age,independently_query(rows,age)) for age in candidates if independently_query(rows,age) is not None),None)
            if selected is None:continue
            age,expected=selected;horizon,mass,tail,mean,median=expected
            actual=provider.query(owner,rows[0]['frontier'],age)
            self.assertEqual(actual['restricted_residual_horizon_ns'],horizon)
            self.assertEqual(actual['median_residual_ns'],median)
            self.assertEqual(len(actual['residual_event_mass']),len(mass))
            for observed,(t,p) in zip(actual['residual_event_mass'],mass):
                self.assertEqual(observed['residual_ns'],t)
                self.assertAlmostEqual(observed['probability'],float(p),delta=1e-11)
            self.assertAlmostEqual(actual['unresolved_tail_probability'],float(tail),delta=1e-11)
            self.assertAlmostEqual(actual['restricted_mean_residual_ns'],float(mean),delta=max(1e-4,float(mean)*1e-10))
            self.assertAlmostEqual(sum(m['probability'] for m in actual['residual_event_mass'])+actual['unresolved_tail_probability'],1.,delta=1e-11)
            checked+=1
            if checked==32:break
        self.assertEqual(checked,32)


    def test_all_episode_envelope_bindings_and_64_source_frontiers_match_original_observations(self):
        model=json.loads(MODEL.read_text());checked_endpoints=0;nonpending=0
        for audit in model['training_audits']:
            capture=audit['raw']['sha256'];report=json.loads(Path(audit['report']['path']).read_text())
            raw=json.loads(Path(audit['raw']['path']).read_text())
            samples={}
            for sample in report['samples']:
                value=dict(run=capture,job=sample['job_id'],owner=sample['alias'],start=sample['envelope_start_ns'],end=sample['envelope_end_ns'])
                eid=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                samples[eid]=sample
            episodes=[r for r in model['episodes'] if r['capture']==capture]
            for row in episodes:
                sample=samples[row['envelope_id']]
                self.assertEqual(row['original_envelope_start_ns'],sample['envelope_start_ns'])
                self.assertEqual(row['original_envelope_end_ns'],sample['envelope_end_ns'])
                self.assertEqual(row['held_owner'],sample['alias'])
                self.assertEqual(row['save_observed'],row['age_exit_ns']==sample['envelope_end_ns']-sample['envelope_start_ns'])
                self.assertLessEqual(row['age_exit_ns'],sample['envelope_end_ns']-sample['envelope_start_ns'])
            observed=json.loads(Path(report['kernel_ready_observations']['path']).read_text())
            phase_ranges=defaultdict(list)
            for node in observed['rows']:
                lo=node['observed_start_ns'] if node['certified_ready_bound_ns'] is None else min(node['observed_start_ns'],node['certified_ready_bound_ns'])
                phase_ranges[node['request_id'],node['phase']].append((lo,node['observed_end_ns']))
            indexes={key:merged(intervals) for key,intervals in phase_ranges.items()}
            jobs=defaultdict(list)
            for request in raw['request_records']:jobs[request['profile_job_id']].append(request)
            selected=[episodes[(len(episodes)-1)*i//31] for i in range(32)]
            for episode in selected:
                if episode['held_owner'] not in {r[0] for r in episode['frontier']}:nonpending+=1
                for age in (episode['age_entry_ns'],episode['age_exit_ns']-1):
                    clock=episode['original_envelope_start_ns']+age;frontier=[]
                    for row in jobs[episode['job_id']]:
                        phases=[]
                        for (rid,phase),intervals in indexes.items():
                            if rid!=row['request_id']:continue
                            at=bisect_right(intervals,(clock,float('inf')))-1
                            if at>=0 and intervals[at][0]<=clock<intervals[at][1]:phases.append(phase)
                        if phases:frontier.append([row['model'],row['task'],row['pool_id'],sorted(phases)])
                    self.assertEqual(sorted(frontier),episode['frontier'])
                    checked_endpoints+=1
        self.assertEqual(checked_endpoints,128)
        self.assertGreater(nonpending,0)


if __name__=='__main__':unittest.main()
