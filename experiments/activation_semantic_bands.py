"""Experimental host-call bands and a phase-aware adapter for the shared FIFO.

Band boundaries are derived observations, never invented PHASE messages. Their
GPU kind retains the original mixed host/transfers/CUDA meaning. Prediction
uses training durations and predicted transitions, never actual decode length.
This module is isolated from the frozen forecasting entry points.
"""
import copy
import json
import math

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.native_context import NativeContextLaw, categories, counts
from gsdeploy.forecasting.native_energy import NativeIdleAgeEnergy
from gsdeploy.forecasting.native_order import NativeOrderLaw, identity as order_identity
from gsdeploy.forecasting.native_training import digest, stage_layout
from gsdeploy.forecasting.profiles import ProfileProvider
from gsdeploy.forecasting.semantic_profiles import split_semantic_native

SCHEMA='gsdeploy.derived_host_call_band_model.v1'


def band_layout(row, family):
    """Two large language/OCR bands; image retains its whole native span."""
    coarse=stage_layout(row)
    # Validate all nested records, coverage and same-stream event completion.
    # The returned CUDA intervals are not used to place host-clock boundaries.
    split_semantic_native(row)
    events=row['worker_response']['semantic_phase_events']
    result=[]
    for span in coarse:
        if span['kind']!='gpu':
            result.append(dict(span))
            continue
        contained=[event for event in events
                   if span['start_ns']<=event['host_start_ns']<event['host_end_ns']<=span['end_ns']]
        if family=='image':
            if not any(event['component']=='image_features' for event in contained):
                raise UnknownEvidence('UNKNOWN_IMAGE_NATIVE_BAND')
            parts=[('image_native',span['start_ns'],span['end_ns'])]
        else:
            component='ocr_detection' if family=='ocr' else 'language' if family=='generalist_2' else 'model_forward'
            forwards=[event for event in contained if event['component']==component]
            if not forwards:
                raise UnknownEvidence('UNKNOWN_EXPECTED_NATIVE_BAND_COMPONENT:'+component)
            first=min(forwards,key=lambda event:event['host_start_ns'])
            boundary=max(event['host_end_ns'] for event in forwards) if family=='ocr' else first['host_end_ns']
            if family=='ocr':
                recognition=[event for event in contained if event['component']=='ocr_recognition']
                if not recognition or min(event['host_start_ns'] for event in recognition)<boundary:
                    raise UnknownEvidence('UNKNOWN_ORDERED_DETECTION_RECOGNITION_BANDS')
            if not span['start_ns']<boundary<span['end_ns']:
                raise UnknownEvidence('UNKNOWN_POSITIVE_HOST_CALL_BANDS')
            roles=('detection','recognition_and_tail') if family=='ocr' else ('first_forward','continuation')
            parts=[(roles[0],span['start_ns'],boundary),(roles[1],boundary,span['end_ns'])]
        for role,left,right in parts:
            result.append(dict(id=span['id']+'.band.'+role,kind=span['kind'],start_ns=left,end_ns=right,
                native_root_id=span['id'],native_band_role=role,derived_host_boundary=True,
                mixed_host_transfers_CUDA=True))
    if (result[0]['start_ns']!=coarse[0]['start_ns'] or result[-1]['end_ns']!=coarse[-1]['end_ns']
            or any(left['end_ns']!=right['start_ns'] for left,right in zip(result,result[1:]))):
        raise ValueError('Derived host bands do not reconstruct the full observed handler')
    return result


def virtual_profiles(template):
    """Phase labels are regression categories, not extra deployed models."""
    profiles=[]
    mapping={}
    for profile in template['profiles']:
        for segment in profile['segments']:
            if segment['kind']!='gpu':
                continue
            role=segment['native_band_role']
            ident=digest([profile['id'],segment['id']])
            mapping[profile['id'],segment['id']]=ident
            profiles.append(dict(id=ident,family=profile['family']+'::'+role,
                task=profile['task'],pool_id=profile['pool_id'],execution=copy.deepcopy(profile['execution']),
                original_profile_id=profile['id'],original_family=profile['family'],native_root_id=segment['native_root_id'],
                segments=[dict(id='native_band',kind='gpu',duration_ms=copy.deepcopy(segment['duration_ms']),
                               demand=dict.fromkeys(('sm','dram','l2'),None))]))
    return profiles,mapping


def virtual_order_law(template):
    """Reuse a declared root coefficient across bands, with explicit identity."""
    order=template.get('native_order_law')
    if order is None:
        return None
    result=copy.deepcopy(order)
    groups=[]
    for group in order['groups']:
        roles={segment['native_band_role'] for profile in template['profiles']
               if (profile['family'],profile['task'])==(group['family'],group['task'])
               for segment in profile['segments'] if segment['kind']=='gpu'
               and segment['native_root_id']==group['segment_id']}
        for role in sorted(roles):
            groups.append(dict(copy.deepcopy(group),family=group['family']+'::'+role,segment_id='native_band'))
    result['groups']=groups
    return result


class BandContextProvider(ProfileProvider):
    """Explicit experimental phase-category progress and coarse aggregate energy."""
    def __init__(self,document,context,*,allow_cpu_fixture=False):
        spec=document.get('derived_host_call_band_model')
        if (not spec or spec.get('schema')!=SCHEMA or spec.get('status')!='FITTED_UNVALIDATED'
                or spec.get('execution_context_sha256')!=context or spec.get('physical_bound') is not False
                or spec.get('admission_enabled') is not False
                or spec.get('decode_stopping_policy')!='training_mean_band_work_no_future_tokens'):
            raise ValueError('Matched explicit unvalidated host-band model required')
        if document.get('contexts') or 'context_interpolation' in document:
            raise ValueError('Choose host-band progress instead of mixing an exact context table')
        base={key:copy.deepcopy(value) for key,value in document.items()
              if key not in ('derived_host_call_band_model','native_activity_law','native_order_law','native_idle_age_energy')}
        # The common provider intentionally accepts only its original segment
        # schema. Keep derived provenance separately, rather than weakening it.
        self.band_metadata={(profile['id'],segment['id']):{
            key:copy.deepcopy(value) for key,value in segment.items()
            if key not in {'id','kind','duration_ms','demand'}}
            for profile in document['profiles'] for segment in profile['segments']}
        for profile in base['profiles']:
            profile['segments']=[{key:segment[key] for key in ('id','kind','duration_ms','demand')}
                                 for segment in profile['segments']]
        super().__init__(base,context,allow_cpu_fixture=allow_cpu_fixture)
        virtual,self.mapping=virtual_profiles(document)
        self.phase_law=NativeContextLaw(spec['progress_law'],{profile['id']:profile for profile in virtual},context)
        if self.phase_law.temporal_progress is not None:
            raise ValueError('Initial host-band model has constant progress per observed phase roster')
        self.order=NativeOrderLaw(document['native_order_law'],context) if document.get('native_order_law') else None
        self.native_order_law=self.order
        order=document.get('native_order_law')
        virtual_order=virtual_order_law(document)
        if (spec.get('native_root_order_law_sha256')!=(order_identity(order) if order else None)
                or spec['progress_law'].get('native_order_law_sha256')!=(order_identity(virtual_order) if virtual_order else None)):
            raise ValueError('Band progress differs from its declared shared root-order law')
        self.energy=NativeIdleAgeEnergy(document['native_idle_age_energy'])
        if self.energy.spec.get('execution_context_sha256')!=context:
            raise ValueError('Band and energy execution contexts differ')
        self.names=categories(self.profiles)
        if self.energy.spec['categories']!=[list(name) for name in self.names]:
            raise ValueError('Band model aggregate energy categories differ')
        self.requires_energy_history=True
        self.requires_native_call_age=False
        self.peak_memory=spec.get('observed_peak_memory_mib')
        if not isinstance(self.peak_memory,(int,float)) or not math.isfinite(self.peak_memory) or self.peak_memory<=0:
            raise ValueError('Empirical observed peak memory required')

    def plan(self,row,aliases,scenario):
        result=super().plan(row,aliases,scenario)
        for segment in result['segments']:
            segment.update(copy.deepcopy(self.band_metadata[result['profile_id'],segment['id']]))
        if self.order is not None:
            profile=self.profiles[result['profile_id']]
            for segment in result['segments']:
                if segment['kind']=='gpu':
                    segment['duration_ms']*=self.order.factor(profile,dict(id=segment['native_root_id']),row.get('native_order'))
        return result

    def energy_interval(self,active,left_ns,right_ns,last_native_end_ns):
        # Bands partition each complete native root; aggregate family counts
        # and the busy-to-idle timestamp are unchanged by that partition.
        return self.energy.integrate([(left_ns,right_ns,counts(active,self.profiles,self.names))],last_native_end_ns)

    def _context(self,active,aliases,scenario):
        try:
            return super()._context(active,aliases,scenario)
        except UnknownEvidence:
            pass
        if scenario!='mean':
            raise UnknownEvidence('UNKNOWN_HOST_BAND_SCENARIO:'+scenario)
        converted={}
        for alias,member in active.items():
            profile=self.profiles[member['profile_id']]
            segment=next(segment for segment in profile['segments'] if segment['id']==member['segment_id'])
            if segment['kind']=='gpu':
                converted[alias]=dict(profile_id=self.mapping[profile['id'],segment['id']],segment_id='native_band')
        progress={alias:(self.phase_law.progress(alias,converted[alias],converted) if alias in converted else 1.)
                  for alias in active}
        return dict(progress=progress,board_watts=None,peak_memory_mib=self.peak_memory),json.dumps(dict(
            schema=SCHEMA,physical_bound=False,independent_accuracy_qualified=False,
            native_scope='derived host-call bands, mixed host/transfers/CUDA',
            future_token_count_used=False),sort_keys=True)
