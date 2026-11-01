"""Exact training support audit and minimal prospective intervention plan."""
import argparse
import json
from pathlib import Path
import numpy as np

from . import macro_joint_response_fit_v1 as base
from . import phase_heterogeneous_response_training_v1 as model
from .implementation_source_closure_v27 import strict_references,implementation_sources


def run(training,association,output):
    training_ref=base.receipt(training);report=base.checked(training_ref)
    association_ref=base.receipt(association);assoc=base.checked(association_ref)
    docs=[];sources=[training_ref,association_ref,*report['sources']]
    for source in report['sources']:
        if Path(source['path']).name.startswith('capture') and Path(source['path']).name.endswith('-macros.json'):
            doc=base.checked(source)
            if doc.get('schema')=='gsdeploy.composable_response_macro_training.v1':docs.append(doc)
    base.require(len(docs)==3,'Three complete original macro documents required')
    candidate=next(c for c in report['candidates'] if c['candidate']=='own_family_phase_count_global_same')
    rows=model.prepared(docs,candidate['features']);failures=[]
    for fold in candidate['whole_capture_leaveout']:
        fit=fold['training_fit']
        if fit['numerical_qualification']:continue
        heldout=fold['heldout_capture_sha256'];train=[r for r in rows if r['capture']!=heldout]
        zeros=[feature for j,feature in enumerate(fit['features']) if all(np.all(row['matrix'][:,j]==0) for row in train)]
        affected=[]
        for row in train:
            if row['category']=='ocr:entry':
                affected.append(dict(capture=row['capture'],diagnostic_request_id=row['request_id'],
                    public_group=list(row['group']),peer_count_path=row['matrix'][:,fit['features'].index('count@ocr:entry')].tolist()))
        roots=[dict(objective=r['mean_nll'],coefficients=r['solution'],
            root_success=r['success'],stationary_strict_local_minimum=r['stationary_strict_local_minimum']) for r in fit['gradient_root_attempts']]
        failures.append(dict(heldout_capture=heldout,joint_response_jacobian=fit['joint_response_jacobian'],
            nuisance_adjusted_rank=fit['nuisance_adjusted_derivative_rank'],features=len(fit['features']),
            exact_all_training_piecewise_zero_columns=zeros,affected_training_OCR_entry_rows=affected,
            coefficient_roots=roots,root_max_disagreement=fit['root_max_solution_disagreement'],
            training_reference_sigma=fit['lognormal_reference_sigma'],
            heldout_conditioned_evaluation=fold['evaluation']['summary'],
            conclusion='Training response coefficient unidentifiable, with arbitrary donor-to-heldout extrapolation; optimizer success flags do not qualify it.'))
    matrix=[
        dict(priority=1,own=['ocr#0:entry'],peer=['text#0:prefill','text#0:decode','generalist_2#0:prefill','generalist_2#0:decode'],
            peer_counts=[0,1,2],required='Each exact OCR public input in every independent capture, with peer BEGIN processed before OCR dispatch and realized exposure logged.'),
        dict(priority=2,own=['ocr#0:ocr_inference','generalist_2#0:prefill','text#0:prefill','text#1:prefill'],
            peer=['language_prefill','language_decode','ocr_inference'],peer_counts=[0,1,2,3],
            required='Cross own family/phase with peer family/phase and count; vary peer phase at fixed count and count at fixed peer composition whenever physically reachable.'),
        dict(priority=3,own=['text#0:decode','text#1:decode','generalist_2#0:decode'],
            peer=['language_prefill','language_decode','ocr_inference'],peer_counts=[0,1,2,3],
            required='Retain the complete request vector including final decode, entry and READY tail; joint request residuals, not independent stage draws.'),
    ]
    plan=dict(status='CPU_MANIFEST_REQUIREMENTS_NO_COLLECTION_STARTED',matrix=matrix,
        public_inputs='Repeat fixed exact public short/medium/long inputs per language owner and both existing OCR inputs in each condition; frozen generation cap/batch/precision.',
        start_order='Own-first, peer-first and simultaneous; predeclare offsets using training-only pilot phase lengths and log processed BEGIN, actual occurrence and notification clocks.',
        nonattainment='Retain offered requests and failed intended-phase overlap. Record realized overlap and failure reason; never discard to manufacture support.',
        replication='At least three independently restarted training captures, each containing every identifiable condition; rotate condition order/seed within each capture. Reserve independent validation captures untouched.',
        input_confounding='Each exact input must repeat across overlap conditions in every capture; never equate input ID or context/job ID with interference condition.',
        unreachable='Do not request same OCR inference with only one OCR owner, or visible ready_tail after native-END clear. These are structural absent states, not missing samples.',
        additional_busy_matrix='Combine with actual two-FIFO sender/worker busy bursts, same-owner HOL, mixed owners and fixed window/drain while retaining complete offered denominators.',
        profiling_clocks='Collect START, BEGIN/H/query/native-END, RESULT occurrence/reader/dequeue/processing, READY occurrence/reader and stdin/write/worker-release clocks; keep notification availability distinct from work completion.')
    result=dict(schema='gsdeploy.phase_response_identifiability_training_audit.v1',
        status='EXACT_FAILED_FOLD_SUPPORT_AND_TARGETED_TRAINING_PLAN',sources=strict_references([sources,implementation_sources(__file__)]),
        failures=failures,structural22_failure=next(c for c in report['candidates'] if c['candidate']=='own_family_phase_count_and_same'),
        targeted_training_plan=plan,paper_transfer_boundary=dict(
            usable_static_structure='USHER/Roomie motivate separating operator/phase resource demand and complementarity; static projection FLOP/logical-byte descriptors provide partial structural covariates only.',
            missing_physical_inputs='Attention/KV, quantized realized cost, measured DRAM/cache, bandwidth/SM concurrency, actual shared-weight cache backend and shared batching remain unqualified.',
            empirical_response_slopes='Current global/family-phase free coefficients are empirical conditional corrections, not the physical USHER/Roomie model and not measured interference reduction.',
            next_candidate_rule='Use the fixed structural descriptors plus identified correction on controlled crossed captures; no context-ID template, no arbitrary extra response coefficients to hide missing support.'),
        generated_heterogeneous_improvement_proven=False,main_queue_provider_eligible=False,final_gain_eligible=False,
        validation_labels_read=False,GPU_started=False)
    for source in result['sources']:base.require(base.receipt(source['path'])==source,'Identifiability source changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return base.receipt(output/'report.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--association',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.training,a.association,a.output)))
