"""Independent CPU-only acceptance of original-FIFO collection preparation.

This report certifies control/data contracts, never native training or phase
coverage. It replays the earlier frozen counterexamples without changing them.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from . import twoFIFO_training_semantic_preflight_v2 as strict
from . import twoFIFO_owned_collect_v2 as collect
from . import twoFIFO_owned_training_review_v2 as review
from .stage_response_features_v1 import receipt
from .implementation_source_closure_v27 import implementation_sources, strict_references


ROOT=Path(__file__).resolve().parents[2]
EARLIER=(
    ROOT/'artifacts/predictor-twoFIFO-manifest-independent-review-v1-20261028-cpu01/report.json',
    ROOT/'artifacts/predictor-twoFIFO-OCR-manifest-independent-review-v2-20261028-cpu01/report.json')
TESTS=('tests.test_twoFIFO_owned_collect_v2',
       'tests.test_twoFIFO_owned_training_review_v2',
       'tests.test_original_sender_late_write_record_independent_v1')


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    sources=strict_references([implementation_sources(collect.__file__),
        implementation_sources(review.__file__),[receipt(__file__)],
        [receipt(ROOT/(name.replace('.','/')+'.py')) for name in TESTS]])
    base,entry=strict.load_frozen()
    expected_hashes=('4e7793e1b349dc3305cad61a61f9933e562a81316d8871833afbc2ba0680918f',
                     'ff2319c6a0b538d9f9b005ea6179ae4afb95119450f001cde2a90f746990776f')
    assert tuple(receipt(p)['sha256'] for p in (strict.BASE,strict.ENTRY))==expected_hashes
    strict.validate(base,entry)
    mutants=[]
    for kind,path in enumerate(EARLIER):
        old=json.loads(path.read_text())
        for counterexample in old['counterexamples']:
            ref=counterexample['reproducible_manifest'];assert receipt(ref['path'])==ref
            altered=json.loads(Path(ref['path']).read_text())
            args=(altered,entry) if kind==0 else (base,altered)
            try:
                strict.validate(*args)
            except (ValueError,KeyError) as exc:
                mutants.append(dict(name=counterexample['name'],counterexample=ref,
                    status='REJECTED_AT_SEMANTIC_PREFLIGHT',error=str(exc)))
            else:
                raise AssertionError('Counterexample accepted: '+counterexample['name'])
    assert len(mutants)==10
    blocks=collect.shard_cases(base,entry);appearances={};block_summaries=[]
    for block in blocks:
        rows=[r for shard in block['shards'] for r in shard]
        assert len(rows)==50 and len({r['case_id'] for r in rows})==50
        assert sum(r['offered'] for r in rows)==182
        assert [len(s) for s in block['shards']]==[13,13,13,11]
        for shard_index,shard in enumerate(block['shards'],1):
            for position,case in enumerate(shard):
                appearances.setdefault(case['case_id'],[]).append(dict(
                    capture_id=block['capture_id'],shard_index=shard_index,position_in_attempt=position))
        block_summaries.append(dict(capture_id=block['capture_id'],cases=len(rows),
            measured_offered=182,shard_case_counts=[len(s) for s in block['shards']]))
    assert len(appearances)==50
    assert all(len(rows)==3 and len({r['shard_index'] for r in rows})==3
               and len({r['position_in_attempt'] for r in rows})==3 for rows in appearances.values())
    # Execute the actual collector's CPU control path, then inspect the actual
    # boundary snapshots with the separately implemented raw reviewer.
    from tests.test_twoFIFO_owned_collect_v2 import CollectorTests
    CollectorTests.setUpClass()
    raw,saved,runtime,_=CollectorTests().collect_fake()
    fake_plan=dict(cases=copy.deepcopy(base['capture_blocks'][0]['cases'][:2]))
    histories=review.inspect_histories(fake_plan,raw)
    assert saved==raw and runtime.closed and raw['kind']==raw['backend']=='CPU_TEST_ONLY'
    assert raw['training_feature_extraction_eligible'] is False
    assert raw['main_queue_provider_eligible'] is False
    history_mutants=[]
    for name in ('drop_prior_row','rewrite_known_response','claim_empty_initial_state'):
        changed=copy.deepcopy(raw);snapshot=changed['case_histories'][0]['prior_history_snapshot']
        if name=='drop_prior_row':snapshot['request_records'].pop()
        elif name=='rewrite_known_response':snapshot['request_records'][0]['model']='unresident'
        else:snapshot['empty_initial_state_assumed']=True
        try:review.inspect_histories(fake_plan,changed)
        except ValueError as exc:history_mutants.append(dict(name=name,status='REJECTED',error=str(exc)))
        else:raise AssertionError('History mutant accepted: '+name)
    completed=subprocess.run([sys.executable,'-m','unittest',*TESTS,'-v'],cwd=ROOT,
        text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (output/'tests.log').write_text(completed.stdout)
    assert completed.returncode==0,completed.stdout
    assert all(receipt(r['path'])==r for r in sources),'Implementation changed during independent review'
    report=dict(schema='gsdeploy.twoFIFO_owned_independent_review.v2',
        status='PASS_CPU_PREPARATION_AND_CONTRACTS_NOT_NATIVE_TRAINING',
        source_closure=sources,frozen_manifests=[receipt(strict.BASE),receipt(strict.ENTRY)],
        original_counterexamples=mutants,original_counterexamples_rejected=10,
        block_design=block_summaries,case_rotation=appearances,
        independent_owned_attempts=12,startups_required=12,priming_per_attempt=18,
        total_priming_offered=216,total_measured_offered=546,total_case_attempts=150,
        fixed_measurement_ns=10_000_000_000,fixed_drain_ns=60_000_000_000,
        actual_collector_CPU_path_histories=histories,history_counterexamples=history_mutants,
        CPU_fake_kind='CPU_TEST_ONLY',CPU_fake_native_eligible=False,
        CPU_tests=dict(command=[sys.executable,'-m','unittest',*TESTS,'-v'],returncode=completed.returncode,
            log=receipt(output/'tests.log')),
        global_admission_cap_added=False,frozen_sources_modified=False,GPU_started=False,
        validation_labels_read=False,actual_phase_coverage='UNKNOWN_NOT_COLLECTED',
        decode_phase_coverage='UNKNOWN_NOT_COLLECTED',full_and_leaveout_design_rank='UNKNOWN_NOT_COLLECTED',
        native_supervisor_collection_tested=False,main_queue_provider_eligible=False,
        final_gain_eligible=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return receipt(output/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    print(json.dumps(run(parser.parse_args().output)))
