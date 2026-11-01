"""Isolated whole-block resource adapter; native v3 rederivation stays mandatory.

Only the selected eight-donor/two-block model supplies predictions. The complete
model and all three saved folds are independently rederived before selection.
This remains an empirical training hypothesis, not resource qualification.
"""
import copy

from . import twoFIFO_resource_provider_v3 as resource
from .stage_response_features_v1 import checked, receipt
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA='gsdeploy.twoFIFO_resource_fold_provider.v1'
require=resource.require


class FoldResourceTrainingProvider(resource.ResourceTrainingProvider):
    def __init__(self,model_path,target_plan,target_context,*,heldout_block_id,
                 allow_training_resource_hypothesis=False,scenario='mean'):
        require(allow_training_resource_hypothesis is True,'Explicit training resource hypothesis opt-in required')
        require(scenario in ('mean','max'),'Explicit empirical resource scenario required')
        model_ref=receipt(model_path);doc=checked(model_ref)
        require(doc.get('schema')==resource.SCHEMA,'Exact v3 resource model required')
        campaign=checked(doc['campaign'])
        require(campaign.get('schema')==resource.intake.campaign_v2.SCHEMA,
            'Only clean v3 complete native campaign is accepted')
        for ref in doc['sources']+doc['training_reviews']:
            require(receipt(ref['path'])==ref,'Resource source or native review changed')
        tables,sources=resource.load_complete(doc['campaign']['path'],[r['path'] for r in doc['training_reviews']])
        actual=resource.fit_tables(tables)
        require(len(tables)==12 and len(actual['training_block_ids'])==3,
            'Exactly twelve attempts in three whole blocks required')
        require(all(doc.get(k)==v for k,v in actual.items()),'Full resource law must rederive from native polls')
        require(all(ref in doc['sources'] for ref in sources),'Complete rederived source closure required')
        expected=[]
        for block in actual['training_block_ids']:
            donor=resource.fit_tables([t for t in tables if t['capture_id']!=block])
            require(len(donor['training_block_ids'])==2 and len(donor['training_attempt_ids'])==8,
                'Fold requires exactly two whole donor blocks and eight attempts')
            expected.append(dict(heldout_block_id=block,training_model=donor,
                heldout_attempt_ids=[t['attempt_id'] for t in tables if t['capture_id']==block],
                independent_prediction_accuracy_validated=False))
        require(doc.get('whole_block_leaveout')==expected,'All three whole-block resource folds must rederive exactly')
        require(heldout_block_id in actual['training_block_ids'],'Exact planned heldout block required')
        selected=next(f for f in expected if f['heldout_block_id']==heldout_block_id)
        identity=resource.physical_execution_identity(target_plan,target_context)
        require(identity==selected['training_model']['physical_execution_identity'],
            'Target physical worker/config/priming axes differ')
        self.document=copy.deepcopy(selected['training_model'])
        self.scenario=scenario;self.target_plan=copy.deepcopy(target_plan)
        self.target_context=copy.deepcopy(target_context);self.physical_execution_identity=identity
        self.fold_binding=dict(schema=SCHEMA,full_native_model=model_ref,heldout_block_id=heldout_block_id,
            training_block_ids=copy.deepcopy(self.document['training_block_ids']),
            training_attempt_ids=copy.deepcopy(self.document['training_attempt_ids']),
            heldout_attempt_ids=copy.deepcopy(selected['heldout_attempt_ids']),
            selected_fold_model_sha256=resource.digest(self.document),
            all_three_folds_rederived=True,complete_twelve_native_attempt_gate=True,
            startup_recomputed_from_only_eight_donor_attempts=True,
            sources=strict_references([sources,implementation_sources(__file__)]),
            resource_prediction_qualified=False,main_queue_provider_eligible=False,
            final_gain_eligible=False,validation_labels_read=False,GPU_started=False)
        require(all(receipt(ref['path'])==ref for ref in self.fold_binding['sources']),
            'Fold adapter source changed during construction')


def predict_timeline(timeline,provider,*,target_context,execution_binding):
    require(isinstance(provider,FoldResourceTrainingProvider),'Explicit fold resource provider required')
    result=resource.predict_timeline(timeline,provider,target_context=target_context,
        execution_binding=execution_binding)
    result['resource_fold_binding']=copy.deepcopy(provider.fold_binding)
    return result
