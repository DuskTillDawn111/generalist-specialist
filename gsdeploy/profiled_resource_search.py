"""USHER-inspired finite resource/configuration decisions with full FIFO U.

Each allowed control configuration needs its OWN execution-bound profiles.
No MPS-percent-to-speed formula, occupancy inference or resource fake is used.
This closes the CPU decision-to-launch-packet path; physical realization and
accuracy still require the corresponding platform and measured evidence.
"""

import copy
import math
import time

from .config import SystemConfig
from .continuous.contracts import digest
from .continuous.paper_structured import PaperStructuredOptimizer
from .paper_forecast import MODE, canonical, validate_spec
from .resources import identity, prepare_registry, validate_resources


class ProfiledResourceConfigurationSearch:
    def __init__(self,candidates,*,seconds=12.,max_predictions=96,max_routes=96,seed=11,
                 method='coordinate',joint_seed=True):
        if not candidates or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('Explicit allowed configurations and positive time required')
        if any(type(v) is not int or v<1 for v in (max_predictions,max_routes)):
            raise ValueError('Positive shared evaluation budgets required')
        self.candidates=copy.deepcopy(candidates)
        self.seconds,self.max_predictions,self.max_routes=seconds,max_predictions,max_routes
        self.seed,self.method,self.joint_seed=seed,method,joint_seed
        ids=set();reference=None;execution_bindings={}
        for candidate in self.candidates:
            if (set(candidate)!={'id','config','profile_bindings'} or not isinstance(candidate['id'],str)
                    or not candidate['id'].strip() or candidate['id'] in ids):
                raise ValueError('Distinct configuration ids, configs and execution bindings required')
            ids.add(candidate['id'])
            c=candidate['config']
            if not isinstance(c,SystemConfig) or c.catalog.get('performance_mode')!=MODE:
                raise ValueError('Explicit stage predictor configuration required')
            validate_spec(c.catalog)
            # All original objective, pool, workload, trace and capacity-factor
            # definitions stay identical. Only measured execution laws and
            # explicit resource control differ between allowed configurations.
            objective=c.as_dict()
            objective.pop('catalog')
            objective.update(models=c.catalog['models'],capacity_factors=c.catalog['capacity_factors'],
                utility_normalization=c.catalog.get('utility_normalization'),
                quality=c.catalog.get('quality'),safe_factor=c.catalog['safe_factor'],
                input_order=c.catalog[MODE]['input_order'],scenarios=c.catalog[MODE]['scenarios'],
                original_capacity={canonical(e['deployment']):e['capacity_service_ms']
                                   for e in c.catalog[MODE]['deployments']})
            key=digest(objective)
            if reference is not None and key!=reference:
                raise ValueError('Resource candidates must share objective, pool, workload, trace and original constraints')
            reference=key
            entries=c.catalog[MODE]['deployments']
            if set(candidate['profile_bindings'])!={canonical(e['deployment']) for e in entries}:
                raise ValueError('Execution bindings must cover every profiled deployment')
            for entry in entries:
                binding=candidate['profile_bindings'][canonical(entry['deployment'])]
                aliases={f'{m}#{i}' for m,n in entry['deployment'].items() for i in range(n)}
                if 'resources' not in entry:
                    raise ValueError('Resource search requires explicit per-resident control configurations')
                resource=validate_resources(entry['resources'],aliases)
                if (set(binding)!={'execution_context_sha256','resource_configuration_sha256','evidence'}
                        or binding['execution_context_sha256']!=entry['profiles']['execution_context_sha256']
                        or binding['resource_configuration_sha256']!=identity(resource)
                        or not binding['evidence']):
                    raise ValueError('Requested resource configuration must match its own profile binding')
                context=binding['execution_context_sha256']
                execution=(canonical(entry['deployment']),identity(resource))
                if context in execution_bindings and execution_bindings[context]!=execution:
                    raise ValueError('Different resident/control configurations cannot reuse an execution context')
                execution_bindings[context]=execution
        self.comparison_identity=reference

    def solve(self):
        started=time.perf_counter();deadline=started+self.seconds
        predictions=routes=milp=engines=0;history=[];best=None
        for index,candidate in enumerate(self.candidates):
            seconds=deadline-time.perf_counter()
            if seconds<=0 or predictions>=self.max_predictions or routes>=self.max_routes:break
            left=len(self.candidates)-index
            raw=PaperStructuredOptimizer(candidate['config'],seconds=seconds/left,
                max_predictions=max(1,(self.max_predictions-predictions)//left),
                max_routes=max(1,(self.max_routes-routes)//left),seed=self.seed,
                method=self.method,joint_seed=self.joint_seed).solve()
            predictions+=raw['budget']['full_prediction_calls'];routes+=raw['budget']['outer_evaluations']
            milp+=raw['budget']['milp_solve_calls'];engines+=raw['fifo_stats']['engine_calls']
            history.append(dict(configuration_id=candidate['id'],result=raw))
            if raw['winner'] and (best is None or raw['winner']['utility']>best['winner']['utility']):
                best=dict(configuration_id=candidate['id'],winner=copy.deepcopy(raw['winner']))
        packet=None
        if best:
            chosen=next(c for c in self.candidates if c['id']==best['configuration_id'])
            entry=next(e for e in chosen['config'].catalog[MODE]['deployments'] if e['deployment']==best['winner']['deployment'])
            resources=copy.deepcopy(entry['resources'])
            aliases=set(resources['allocations'])
            # This contains only resource bindings. The caller must supply its
            # actual command/backend registry to prepare_launch_registry().
            prepared=prepare_registry({a:dict(env={}) for a in aliases},resources)
            packet=dict(resources=resources,profile_binding=chosen['profile_bindings'][canonical(entry['deployment'])],
                        requested_worker_bindings=prepared,runnable_registry=False,
                        effective_isolation_verified=False,
                        launch_blockers=sorted({b for row in prepared.values() for b in row['resource_binding']['launch_blockers']}))
        return dict(algorithm='ProfiledResourceConfigurationSearch',winner=best,execution_packet=packet,
            history=history,configuration_count=len(self.candidates),evaluated_configurations=len(history),
            comparison_identity=self.comparison_identity,elapsed_seconds=time.perf_counter()-started,
            counts=dict(full_predictions=predictions,routes=routes,milp=milp,event_engine_calls=engines),
            status='FINITE_CONFIGURATION_SEARCH_FINISHED' if len(history)==len(self.candidates) else 'BUDGET_LIMIT',
            optimality_certified=False,native_eligible=False,independent_accuracy_qualified=False,
            effective_isolation_verified=False,final_gain_eligible=False)

    @staticmethod
    def prepare_launch_registry(result,registry):
        packet=result.get('execution_packet')
        if not packet:raise ValueError('No valid selected execution configuration')
        return prepare_registry(registry,packet['resources'])
