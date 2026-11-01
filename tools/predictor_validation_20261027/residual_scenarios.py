"""Training-only joint mixed-wall residual scenarios for phase alignment.

A vector is sampled once per request and reused across its phases. It is a
statistical work proxy, not a physical work measurement or calibrated bound.
The same request/scenario uniform is used across candidate routes. No single
phase quantile is substituted for the quantile of full-cohort E2E or utility.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .background_statistical import BackgroundStatisticalProvider
from .profiling import digest

SCHEMA='gsdeploy.joint_mixed_wall_phase_residual_scenarios.v1'


def target_key(family,task,pool_id):
    return json.dumps([family,task,pool_id],ensure_ascii=False,separators=(',',':'))


def adapt_reconstructed_vectors(reconstruction, *, reconstruction_basis):
    """Adapt the pure reconstruction API without changing any split or source."""
    if reconstruction.get('schema')!='gsdeploy.frozen_model_observed_joint_phase_work.v1':
        raise ValueError('Explicit frozen-model work reconstruction required')
    if not reconstruction_basis:
        raise ValueError('Explicit reconstruction basis required')
    return [{**copy.deepcopy(row), 'known':row['complete_known_vector'] is True,
             'reconstruction_basis':copy.deepcopy(reconstruction_basis)}
            for row in reconstruction['requests']]


def fit_vectors(records,*,base_model_sha256,execution_context_sha256,sources,minimum_vectors=3):
    """Fit only externally reconstructed TRAINING vectors, never endpoint labels.

    Callers should reconstruct from cross-fitted training models to avoid
    understating fitted residual variability. This function does not perform
    cross-fitting and records the supplied reconstruction basis explicitly.
    """
    if not records or not sources or minimum_vectors<3:
        raise ValueError('Training vectors, sources and at least three repeats required')
    for source in sources:
        p=Path(source['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Changed residual training source')
    populations={};seen=set();unusable=[]
    for row in records:
        if row.get('split')!='training' or row.get('not_for_training',False):
            raise ValueError('Validation residuals must never enter fitting')
        identity=row['trace_id'],row['request_id']
        if identity in seen:raise ValueError('Duplicate residual observation')
        seen.add(identity)
        vector=row.get('work_by_phase')
        if row.get('known') is not True or not vector or any(type(v) not in (int,float) or
                not math.isfinite(v) or v<=0 for v in vector.values()):
            unusable.append(list(identity));continue
        key=target_key(row['family'],row['task'],row['pool_id'])
        populations.setdefault(key,[]).append({'source_identity':list(identity),'work_by_phase':copy.deepcopy(vector),
            'reconstruction_basis':row.get('reconstruction_basis','UNSPECIFIED_NOT_CROSS_FIT_CERTIFIED')})
    document=dict(schema=SCHEMA,base_model_sha256=base_model_sha256,execution_context_sha256=execution_context_sha256,
        source_receipts=list(sources),minimum_vectors=minimum_vectors,populations=populations,
        training_observations=len(records),unusable_observations=unusable,
        definition=dict(sampling='One empirical joint phase vector per request; common uniform across candidates',
            dependence='Request draws are independent; run-level and inter-request dependence are not modeled',
            centering='Uncentered multiplicative residual vectors; conditional calibration is an independent validation task',
            identification='Mixed wall service residual, not uniquely identified physical work',
            unknown='Missing input/phase population or too few complete vectors is UNKNOWN',
            endpoint='Full scenario rollout computes empirical task E2E P95 and six-term utility',
            uncertainty_calibrated=False,validation_data_used=False))
    document['model_sha256']=digest(document);return document


class JointScenarioProvider:
    def __init__(self,base,document,scenario_index,*,namespace='joint-wall-work-scenarios-v1'):
        # This provider's service() reads immutable profile baselines. Generic
        # providers may reuse scaled plan durations and would scale twice.
        if type(base) is not BackgroundStatisticalProvider:
            raise TypeError('Joint residual scaling requires BackgroundStatisticalProvider')
        unsigned=copy.deepcopy(document);sha=unsigned.pop('model_sha256',None)
        if document.get('schema')!=SCHEMA or sha!=digest(unsigned):
            raise ValueError('Changed explicit joint residual scenario model')
        if base.document['model_sha256']!=document['base_model_sha256']:
            raise ValueError('Residual model belongs to a different point model')
        if base.execution_context_sha256!=document['execution_context_sha256']:
            raise ValueError('Scenario execution identity mismatch')
        if type(scenario_index) is not int or scenario_index<0:
            raise ValueError('Nonnegative independent scenario index required')
        self.base,self.document,self.index,self.namespace=base,copy.deepcopy(document),scenario_index,namespace
        self.execution_context_sha256=base.execution_context_sha256
        self.lookup=base.lookup;self.profiles=base.profiles;self.selected={}

    def _vector(self,row,aliases):
        target=target_key(aliases[row['alias']]['family'],row['task'],row['pool_id'])
        key=row['request_id'],target
        if key not in self.selected:
            population=self.document['populations'].get(target,[])
            if len(population)<self.document['minimum_vectors']:
                raise UnknownEvidence('INSUFFICIENT_JOINT_RESIDUAL_INPUT_POPULATION:'+target)
            # Do not include candidate or target: identical requests use common
            # uniforms, with each route mapping into its own empirical population.
            seed=digest([self.namespace,self.index,row['request_id']])
            uniform=int(seed,16)/(1<<256)
            self.selected[key]=population[min(len(population)-1,int(uniform*len(population)))]
        return self.selected[key]['work_by_phase']

    def plan(self,row,aliases,scenario):
        plan=copy.deepcopy(self.base.plan(row,aliases,scenario));vector=self._vector(row,aliases)
        phases={s['id'] for s in plan['segments']}
        if set(vector)!=phases:
            raise UnknownEvidence('INCOMPLETE_JOINT_RESIDUAL_PHASE_VECTOR')
        for segment in plan['segments']:
            segment['duration_ms']*=vector[segment['id']]
        return plan

    def service(self,row,active,aliases,scenario):
        result=self.base.service(row,active,aliases,scenario)
        if not result.get('known'):return result
        vector=self._vector(row,aliases)
        if row['segment_id'] not in vector:
            return {'known':False,'reason':'UNKNOWN_JOINT_RESIDUAL_PHASE'}
        factor=vector[row['segment_id']]
        return {**result,'service_ms':result['service_ms']*factor,'sampled_work_factor':factor,
            'scenario_index':self.index,'joint_phase_sampling':True,'uncertainty_calibrated':False,
            'scope':'Empirical joint mixed-wall residual scenario, not physical work or an uncertainty guarantee'}

    def power(self,*args):return self.base.power(*args)
    def memory(self,*args):return self.base.memory(*args)
