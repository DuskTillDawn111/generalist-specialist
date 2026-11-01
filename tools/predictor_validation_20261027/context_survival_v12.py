"""Age-conditional empirical context-envelope support from training sources.

SAVE_END is an envelope endpoint, not an SM service quantum. Frontier changes
right-censor episodes; later entries are left-truncated. Product-limit estimates
therefore assume conditional exchangeability and noninformative censoring.
Neither assumption nor out-of-sample accuracy is established by this module.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .context_envelope_diagnostic_v8 import IntervalUnion
from .dag_availability_fit_v8 import stable_key
from .multi_native_composite_v5 import checked
from .native_collect import receipt,save
from .profiling import digest
from .training_source_guard_v11 import availability_training_sources

SCHEMA = 'gsdeploy.training_context_envelope_survival_episodes.v12'


def derive(report_references):
    audits = availability_training_sources(report_references)
    episodes = [];sources = [receipt(__file__)];counts = Counter()
    for audit in audits:
        report = checked(audit['report']);raw = checked(audit['raw']);run = audit['raw']['sha256']
        observation = checked(report['kernel_ready_observations'])
        sources.extend([audit['report'],*report['sources'],report['kernel_ready_observations']])
        ranges = defaultdict(list)
        for row in observation['rows']:
            ready = row['certified_ready_bound_ns'];start = row['observed_start_ns']
            lo = min(ready,start) if ready is not None else start
            ranges[row['request_id'],row['phase']].append((lo,row['observed_end_ns']))
        indexes = {key:IntervalUnion(value) for key,value in ranges.items()}
        jobs = defaultdict(list)
        for row in raw['request_records']:
            jobs[row['profile_job_id']].append(row)
        cuts_by_job = {}
        for job,rows in jobs.items():
            ids = {r['request_id'] for r in rows}
            cuts_by_job[job] = sorted({t for (rid,_),index in indexes.items() if rid in ids
                                      for a,b in index.intervals for t in (a,b)})
        def frontier_at(job,clock):
            state = []
            for row in jobs[job]:
                phases = []
                for (rid,phase),index in indexes.items():
                    if rid != row['request_id']:
                        continue
                    i = bisect_right(index.starts,clock)-1
                    if i >= 0 and clock < index.ends[i]:
                        phases.append(phase)
                if phases:
                    state.append([row['model'],row['task'],row['pool_id'],sorted(phases)])
            return sorted(state)
        seen = set()
        for sample in report['samples']:
            job = sample['job_id'];owner = sample['alias']
            a,b = sample['envelope_start_ns'],sample['envelope_end_ns']
            if type(a) is not int or type(b) is not int or a >= b:
                raise ValueError('Complete positive integer source context envelope required')
            key = (job,owner,a,b)
            if key in seen:
                raise ValueError('Duplicate observed training context envelope')
            seen.add(key);all_cuts = cuts_by_job[job]
            cuts = [a,*all_cuts[bisect_right(all_cuts,a):bisect_right(all_cuts,b-1)],b]
            eid = digest(dict(run=run,job=job,owner=owner,start=a,end=b))
            for lo,hi in zip(cuts,cuts[1:]):
                state = frontier_at(job,lo)
                if not state:
                    counts['empty_ready_episode_retained_as_no_query_support'] += 1
                    continue
                if owner not in {r[0] for r in state}:
                    counts['held_owner_not_pending_preserved'] += 1
                episodes.append(dict(capture=run,envelope_id=eid,held_owner=owner,frontier=state,
                    age_entry_ns=lo-a,age_exit_ns=hi-a,save_observed=hi==b,
                    original_envelope_start_ns=a,original_envelope_end_ns=b,job_id=job))
                counts['observed_save' if hi==b else 'frontier_change_censored'] += 1
    for ref in sources:
        checked_ref = receipt(ref['path'])
        if checked_ref != ref:
            raise ValueError('Context survival source changed during extraction')
    result = dict(schema=SCHEMA,training_only=True,training_audits=audits,sources=sources,
        episodes=episodes,counts=dict(counts),
        unit='RESTORE_START_to_SAVE_END_envelope_age_ns_not_SM_service',
        current_held_context_not_replaced_by_sole_ready_owner=True,
        frontier_change_is_censoring_hypothesis=True,conditional_exchangeability_validated=False,
        independent_prediction_accuracy_validated=False,GPU_started=False)
    result['model_sha256'] = digest(result)
    return result


class ContextEnvelopeSurvival:
    """No tail extrapolation, no constant quantum, no owner-selection policy."""
    def __init__(self, model, *, allow_unvalidated_survival_hypothesis=False,
                 allow_single_capture_training_diagnostic=False,allow_cpu_fixture=False):
        if (allow_unvalidated_survival_hypothesis is not True or model.get('schema') != SCHEMA
                or model.get('training_only') is not True
                or model.get('model_sha256') != digest({k:v for k,v in model.items() if k!='model_sha256'})):
            raise ValueError('Source-bound training survival hypothesis required')
        self.cpu_fixture = allow_cpu_fixture is True
        if not self.cpu_fixture:
            audits = model.get('training_audits',[])
            regenerated = derive([row['report'] for row in audits])
            if regenerated != model:
                raise ValueError('Survival model differs from complete source episode rederivation')
        self.required = 1 if allow_single_capture_training_diagnostic is True else 2
        self.groups = defaultdict(list)
        seen = set()
        intervals = defaultdict(list)
        for row in model['episodes']:
            a,b = row['age_entry_ns'],row['age_exit_ns']
            if (type(a) is not int or type(b) is not int or not 0<=a<b
                    or type(row['save_observed']) is not bool):
                raise ValueError('Valid delayed-entry/right-censor training interval required')
            key = (row['capture'],row['envelope_id'],a,b)
            if key in seen:
                raise ValueError('Duplicate training survival interval')
            seen.add(key)
            intervals[row['capture'],row['envelope_id']].append((a,b))
            self.groups[row['held_owner'],stable_key(row['frontier'])].append(dict(row))
        for windows in intervals.values():
            ordered = sorted(windows)
            if any(before[1]>after[0] for before,after in zip(ordered,ordered[1:])):
                raise ValueError('One observed envelope cannot enter overlapping survival risk episodes')
        self.model_sha256 = model['model_sha256']

    def query(self,held_owner,pending_frontier,context_age_ns):
        if type(context_age_ns) is not int or context_age_ns < 0:
            raise ValueError('Nonnegative integer actual restore age required')
        age = context_age_ns
        rows = self.groups.get((held_owner,stable_key(pending_frontier)),[])
        captures = sorted({r['capture'] for r in rows})
        if len(captures) < self.required:
            raise UnknownEvidence('CONTEXT_SURVIVAL_INDEPENDENT_CAPTURE_SUPPORT_UNAVAILABLE')
        support_ends = []
        for capture in captures:
            intervals = sorted((r['age_entry_ns'],r['age_exit_ns']) for r in rows if r['capture']==capture)
            end = age;entered = False
            for a,b in intervals:
                if a <= age < b:
                    entered = True
                if a <= end and b > end:
                    end = b
                elif a > end:
                    break
            if not entered or end <= age:
                raise UnknownEvidence('CONTEXT_SURVIVAL_AGE_OUTSIDE_EACH_CAPTURE_RISK_SUPPORT')
            support_ends.append(end)
        limit = min(support_ends)
        envelopes = {c:{r['envelope_id'] for r in rows if r['capture']==c} for c in captures}
        weight = {c:1/(len(captures)*len(envelopes[c])) for c in captures}
        times = sorted({r['age_exit_ns'] for r in rows
                        if r['save_observed'] and age<r['age_exit_ns']<=limit})
        survival = 1.;mass = [];last = age;restricted_mean = 0.;median = None
        for t in times:
            risk = sum(weight[r['capture']] for r in rows if r['age_entry_ns']<t<=r['age_exit_ns'])
            events = sum(weight[r['capture']] for r in rows if r['save_observed'] and r['age_exit_ns']==t)
            if risk <= 0 or events > risk+1e-12:
                raise ValueError('Invalid delayed-entry product-limit risk set')
            restricted_mean += survival*(t-last);last = t
            probability = survival*events/risk
            survival = max(0.,survival-probability)
            mass.append(dict(residual_ns=t-age,probability=probability))
            if median is None and survival <= .5:
                median = t-age
        restricted_mean += survival*(limit-last)
        return dict(status='UNVALIDATED_CONDITIONAL_ENVELOPE_SURVIVAL',model_sha256=self.model_sha256,
            held_owner=held_owner,context_age_ns=age,capture_count=len(captures),
            residual_event_mass=mass,unresolved_tail_probability=survival,
            restricted_residual_horizon_ns=limit-age,restricted_mean_residual_ns=restricted_mean,
            median_residual_ns=median,complete_finite_residual_distribution=survival==0.,
            tail_filled_with_zero=False,context_age_reset=False,constant_quantum_used=False,
            pending_work_and_waiting_age_conditioned=False,
            conditional_exchangeability_is_hypothesis=True,noninformative_censoring_is_hypothesis=True,
            physical_SM_service_identified=False,independent_prediction_accuracy_validated=False,
            next_owner_policy_identified=False,cpu_fixture=self.cpu_fixture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',action='append',required=True);parser.add_argument('--output',required=True)
    args = parser.parse_args();out = Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    result = derive([receipt(p) for p in args.report]);save(out/'model.json',result)
    print(json.dumps(dict(episodes=len(result['episodes']),counts=result['counts'],model=receipt(out/'model.json'))))


if __name__ == '__main__':
    main()
