"""Explicit host-provider selection for every original execution component.

Selection is frozen per exact component before execution. UNKNOWN never triggers
another provider; independent image/ASR host evidence is not an input fallback.
"""
from __future__ import annotations
from gsdeploy.forecast_errors import UnknownEvidence
from .multi_native_composite_v5 import checked
from .dag_host_release_v7 import DagHostReleaseProvider,SCHEMA as EXACT_SCHEMA

SCHEMA='gsdeploy.explicit_component_host_supply_selection.v8'


class ComponentHostSupply:
    def __init__(self,document,adapter,aliases,public_bindings,*,scenario,expected_composite):
        if (document.get('schema')!=SCHEMA or set(document)!={'schema','composite','components'}
                or document['composite']!=expected_composite
                or checked(document['composite'])['composite_execution_sha256']!=adapter.execution_context_sha256):
            raise ValueError('Exact unchanged multi-native composite host selection required')
        entries=document['components']
        if (not isinstance(entries,list) or len(entries)!=len(adapter.components)
                or {e['component_id'] for e in entries}!=set(adapter.components)):
            raise ValueError('Explicit complete nonduplicate host component selection required')
        self.adapter=adapter;self.providers={};self.selections={}
        for selection in entries:
            cid=selection['component_id'];entry=adapter.components[cid]['entry'];spec=entry['spec']
            keys={key for key,route in adapter.routes.items() if route==cid}
            if not keys:raise ValueError('Host component must have an explicit exact-template domain')
            if any(key not in public_bindings or public_bindings[key]['component_id']!=cid for key in keys):
                raise ValueError('Source-verified public binding differs from host component route')
            model=checked(selection['model'])
            kind=selection['provider_kind']
            if kind=='exact_input_v7':
                if (set(selection)!={'component_id','provider_kind','model','allow_identification_hypothesis','allow_exact_input_conditioning_hypothesis'}
                        or selection['allow_identification_hypothesis'] is not True
                        or selection['allow_exact_input_conditioning_hypothesis'] is not True):
                    raise ValueError('Exact-input host requires explicit identification and conditioning hypotheses')
                if (model.get('schema')!=EXACT_SCHEMA or model['source_host']!=spec['host_supply']
                        or model['dag']!=spec['dag']
                        or model['light_execution_context_sha256']!=entry['original_light_context_sha256']
                        or model['execution_context_sha256']!=entry['original_natural_context_sha256']):
                    raise ValueError('Exact-input host original component source identity mismatch')
                support={(g['alias'],g['task'],g['pool_id']) for g in model['groups'] if g['supported']}
                if support!=keys:raise UnknownEvidence('EXACT_INPUT_HOST_COMPONENT_DOMAIN_NOT_COMPLETELY_SUPPORTED')
                provider=DagHostReleaseProvider(model,entry['original_natural_context_sha256'],
                    {k:adapter.plans[k] for k in keys},aliases,scenario=scenario,
                    allow_identification_hypothesis=True,allow_exact_input_conditioning_hypothesis=True,
                    public_bindings={k:public_bindings[k] for k in keys})
            elif kind=='original_component_host_v5':
                if (set(selection)!={'component_id','provider_kind','model','allow_identification_hypothesis','allow_cross_input_pooling_hypothesis'}
                        or selection['allow_identification_hypothesis'] is not True
                        or selection['allow_cross_input_pooling_hypothesis'] is not True):
                    raise ValueError('Original component host requires explicit identification and pooling hypotheses')
                if (selection['model']!=spec['host_supply'] or model.get('schema')!='gsdeploy.dag_host_supply_training_hypothesis.v5'
                        or model['dag']!=spec['dag']
                        or model['execution_context_sha256']!=entry['original_natural_context_sha256']):
                    raise ValueError('Original host selection differs from source-reverified component')
                # MultiNativeProvider has already rederived this exact model,
                # lifecycle and original context from its bound training raws.
                provider=adapter.components[cid]['release']
            else:raise ValueError('Explicit supported component host provider kind required')
            self.providers[cid]=provider;self.selections[cid]=selection.copy()

    def __call__(self,node,row):
        key=row['alias'],row['task'],row['pool_id']
        if key not in self.adapter.routes:raise UnknownEvidence('NO_EXACT_COMPONENT_HOST_ROUTE')
        return self.providers[self.adapter.routes[key]](node,row)

    def provenance(self,row):
        key=row['alias'],row['task'],row['pool_id']
        if key not in self.adapter.routes:raise UnknownEvidence('NO_EXACT_COMPONENT_HOST_ROUTE')
        cid=self.adapter.routes[key];entry=self.adapter.components[cid]['entry'];selection=self.selections[cid]
        return dict(component_id=cid,provider_kind=selection['provider_kind'],model=selection['model'],
            original_light_context_sha256=entry['original_light_context_sha256'],
            original_natural_context_sha256=entry['original_natural_context_sha256'],
            selection_changed_after_unknown=False,physical_host_identification=False)
