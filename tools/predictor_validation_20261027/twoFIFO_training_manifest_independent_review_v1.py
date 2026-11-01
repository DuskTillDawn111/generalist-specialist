"""Read-only review: accepted counterexamples are explicit review failures."""
import argparse
import copy
import json
from pathlib import Path

from . import twoFIFO_training_manifest_v1 as manifest
from .stage_response_features_v1 import receipt
from .implementation_source_closure_v27 import strict_references,implementation_sources


def run(path,output):
    original_ref=receipt(path);original=json.loads(Path(path).read_text())
    baseline=manifest.validate(original);counterexamples=[]
    def check(name,mutant,reason):
        try:result=manifest.validate(mutant);accepted=True;error=None
        except Exception as exc:result=None;accepted=False;error=repr(exc)
        counterexamples.append(dict(name=name,accepted_by_current_validator=accepted,
            expected='REJECT_AT_COLLECTION_PREFLIGHT',reason=reason,validation_result=result,error=error,
            mutant=mutant))
    mutant=copy.deepcopy(original)
    for block in mutant['capture_blocks']:block['capture_id']='training-block-1'
    check('duplicate_capture_identity',mutant,'Three declared blocks must have distinct identities; count=3 does not establish independent repeated captures.')
    mutant=copy.deepcopy(original)
    mutant['capture_blocks'][0]['cases'][1]['case_id']=mutant['capture_blocks'][0]['cases'][0]['case_id']
    check('duplicate_case_identity',mutant,'A missing/duplicated condition is not the declared fixed sixteen-case matrix.')
    mutant=copy.deepcopy(original);block=mutant['capture_blocks'][0]
    replacement=copy.deepcopy(block['cases'][0]);replacement['case_id']=block['cases'][1]['case_id']
    for i,row in enumerate(replacement['arrivals']):row['request_id']='review-mutant:'+str(i)
    replacement['public_arrivals_sha256']=manifest.digest(replacement['arrivals']);block['cases'][1]=replacement
    check('nonrepeated_public_cohort_and_offered',mutant,'Changing one block from 100 to 97 offered and changing its cohort must not validate as the fixed repeated matrix.')
    mutant=copy.deepcopy(original);case=mutant['capture_blocks'][1]['cases'][0]
    case['arrivals'][0]['arrival_ns']+=12345;case['public_arrivals_sha256']=manifest.digest(case['arrivals'])
    check('cross_capture_arrival_schedule_changed',mutant,'A self-consistent updated digest does not establish the same normalized public cohort across captures.')
    mutant=copy.deepcopy(original);mutant['inputs']['text']['short']['exact_input_sha256']='0'*64
    check('input_content_claim_not_rederived',mutant,'Input declaration must be rederived from declared context and actual prepared input bytes; checking unchanged source files alone does not bind the claim.')
    sources=strict_references([[original_ref],original['sources'],[original['prepared_pool']],implementation_sources(__file__)])
    for source in sources:manifest.require(receipt(source['path'])==source,'Frozen source changed during independent review')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[]
    for i,row in enumerate(counterexamples):
        mutant=row.pop('mutant');target=output/('counterexample-'+str(i)+'.json')
        target.write_text(json.dumps(mutant,indent=2)+'\n');row['reproducible_manifest']=receipt(target);records.append(row)
    report=dict(schema='gsdeploy.twoFIFO_training_manifest_independent_review.v1',
        status='COLLECTION_PREFLIGHT_CHANGES_REQUIRED' if any(r['accepted_by_current_validator'] for r in records) else 'COUNTEREXAMPLES_REJECTED',
        sources=sources,original_validation=baseline,counterexamples=records,
        frozen_sources_modified=False,collector_execution_review_complete=False,
        dispatcher_findings=dict(
            sequential_arrival_enqueue_has_RESULT_READY_gate=False,
            all_offered_checkpointed_before_first_enqueue=True,
            fixed_epoch_measurement_plus_drain_preserved=True,
            late_RESULT_separately_flagged=True,
            actual_initial_state_or_full_READY_history_enforced_here=False,
            startup_monitor_cleanup_supervisor_implemented_here=False),
        required_next_checks=['Distinct independent worker startup per capture block, exactly one startup charge within its lifecycle.',
            'Validate case IDs and normalized public arrivals against frozen definitions; repeat all conditions for each exact input in all captures.',
            'Rebind input task/content/size and prepared pool to frozen context; require source receipts and manifest identity before GPU launch.',
            'Sidecar must judge actual macro-phase processed exposure and denominators, never intended case labels.',
            'Original sender pipelines without waiting for RESULT/READY; same-owner FIFO and all late/error/incomplete rows retained.',
            'Case initial state and prior READY/notification history explicit; RESULT completion does not prove worker READY or empty station.',
            'Fixed measurement/drain deadlines; last live-owned sample and monitor end precede worker close, with owned cleanup verified.',
            'Supervisor exceptions/interruption retain all offered and NOT_STARTED cases, stop monitor, then clean owned processes.'],
        validation_labels_read=False,GPU_started=False,main_queue_provider_eligible=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return receipt(output/'report.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(run(a.manifest,a.output)))
