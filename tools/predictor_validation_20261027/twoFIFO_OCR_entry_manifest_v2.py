"""CPU-only targeted OCR-entry support grid; never claim scheduled phase support."""
import argparse
import copy
import json
from pathlib import Path

from .twoFIFO_training_manifest_v1 import SCHEMA as BASE_SCHEMA,digest,require
from .stage_response_features_v1 import receipt
from .implementation_source_closure_v27 import implementation_sources,strict_references

SCHEMA='gsdeploy.original_twoFIFO_OCR_entry_support_manifest.v2'


def build_cases(base):
    peer_pool=base['inputs']['text']['long']['pool_id'];blocks=[]
    for block in base['capture_blocks']:
        planned=[]
        for size in ('short','long'):
            ocr_pool=base['inputs']['ocr'][size]['pool_id']
            def add(tag,peers,lag_ms,direction):
                ocr_at=lag_ms*1_000_000 if direction=='peer_first' else 0
                peer_at=lag_ms*1_000_000 if direction=='OCR_first' else 0
                case_id='OCR-entry-'+size+'-'+tag+'-'+direction+'-'+str(lag_ms)+'ms'
                rows=[dict(owner='ocr#0',task='ocr',pool_id=ocr_pool,arrival_ns=ocr_at)]
                rows.extend(dict(owner=owner,task='text',pool_id=peer_pool,arrival_ns=peer_at) for owner in peers)
                rows.sort(key=lambda row:(row['arrival_ns'],row['owner']))
                for i,row in enumerate(rows):row.update(request_id=block['capture_id']+':'+case_id+':'+str(i),arrival_index=i)
                planned.append(dict(case_id=case_id,intended_stratum='OCR_entry_processed_language_peer_count_support',
                    OCR_pool_id=ocr_pool,peer_owners=list(peers),planned_offset_ms=lag_ms,planned_direction=direction,
                    expected_phase_is_label=False,offered=len(rows),arrivals=rows,public_arrivals_sha256=digest(rows)))
            add('own',(),0,'own')
            for lag in (1,20,50,100):
                add('text',('text#0',),lag,'peer_first')
                add('G2',('generalist_2#0',),lag,'peer_first')
                add('text-G2',('text#0','generalist_2#0'),lag,'peer_first')
                add('text-G2',('text#0','generalist_2#0'),lag,'OCR_first')
        blocks.append(dict(capture_id=block['capture_id'],cases=planned))
    return blocks


def validate(manifest):
    require(manifest['schema']==SCHEMA and len(manifest['capture_blocks'])==3,'Three training blocks required')
    ids=set();offered=0
    for block in manifest['capture_blocks']:
        require(len(block['cases'])==34,'Both exact OCR inputs must repeat every fixed condition in every block')
        for case in block['cases']:
            require(case['public_arrivals_sha256']==digest(case['arrivals']) and len(case['arrivals'])==case['offered'],
                'Complete immutable offered cohort required')
            for row in case['arrivals']:
                require(row['request_id'] not in ids and 0<=row['arrival_ns']<manifest['measurement_ns'],
                    'Unique bounded public arrival required')
                ids.add(row['request_id']);offered+=1
    require(all(receipt(r['path'])==r for r in manifest['sources']),'Targeted manifest source changed')
    return dict(case_attempts=102,total_offered=offered,offered_per_capture=82,
        actual_OCR_entry_n0_and_npositive_support='UNKNOWN_NOT_COLLECTED',
        full_and_leaveout_design_rank='UNKNOWN_NOT_COLLECTED',GPU_started=False)


def prepare(base_path,response_design_path,output):
    base_ref=receipt(base_path);base=json.loads(Path(base_path).read_text());require(base['schema']==BASE_SCHEMA,'Frozen v1 base required')
    response_ref=receipt(response_design_path);response=json.loads(Path(response_design_path).read_text())
    sources=strict_references([base['sources'],[base_ref,response_ref],response.get('sources',[]),implementation_sources(__file__)])
    manifest=copy.deepcopy(base)
    manifest.update(schema=SCHEMA,status='TARGETED_CPU_SIDECAR_NOT_COLLECTED',sources=sources,base_manifest=base_ref,
        response_identifiability_report=response_ref,capture_blocks=build_cases(base),
        replaces_base_matrix=False,extra_natural_training_offered=246,
        specific_failure='OCR entry peer-count column zero in all eight rows across one leaveout training pair; rank59/60 despite full-three-capture rank60.',
        acceptance=dict(each_capture_each_exact_OCR_input_requires=['positive-duration OCR entry exposure at processed peer count0',
                'positive-duration OCR entry exposure at processed peer count>0',
                'measured processed language BEGIN still visible during OCR entry, retaining phase/request identity'],
            desired_crosses=['peer_count1 with text prefill/decode','peer_count1 with G2 prefill/decode',
                'peer_count2 with text+G2 semantic phase composition'],
            rank_qualification='Rebuild the complete response design from raw clocks; full fit and every whole-capture leaveout need rank/curvature/score qualification.',
            failed_offset_policy='Retain all offered and mark support missing; no retiming, replacement or unreported retry.',
            scheduled_offset_is_observed_support=False,
            structural_zero_states_remain_unavailable=['OCR same-phase peer with only one OCR owner','same visible READY-tail after CLEAR']),
        resources_match='Within each exact input and peer roster, preserve resident/input multiset across offsets; compare coarser full-handler pooling against actual phase composition.',
        automatic_GPU_execution=False,validation_labels_read=False,GPU_started=False,main_queue_provider_eligible=False)
    audit=validate(manifest);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for name,data in [('manifest',manifest),('validation',audit)]:
        with (output/(name+'.json')).open('x') as stream:json.dump(data,stream,ensure_ascii=False,indent=2);stream.write('\n')
    return receipt(output/'manifest.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base-manifest',required=True)
    parser.add_argument('--response-design',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.base_manifest,args.response_design,args.output)))
