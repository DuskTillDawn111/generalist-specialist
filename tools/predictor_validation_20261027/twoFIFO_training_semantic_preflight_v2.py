"""Strict CPU semantic reconstruction for the frozen natural FIFO designs."""
import copy
import hashlib
import json
from pathlib import Path

from .stage_response_features_v1 import receipt
from . import twoFIFO_training_manifest_v1 as base
from . import twoFIFO_OCR_entry_manifest_v2 as entry

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/predictor-twoFIFO-training-design-v1-20261028-cpu01/manifest.json'
ENTRY=ROOT/'artifacts/predictor-twoFIFO-OCR-entry-design-v2-20261028-cpu01/manifest.json'
CONTEXT=ROOT/'artifacts/predictor-device-progress-paired-v6-20261028-training01/training/natural/context.json'
POOL=ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl'
BASE_SHA='4e7793e1b349dc3305cad61a61f9933e562a81316d8871833afbc2ba0680918f'
ENTRY_SHA='ff2319c6a0b538d9f9b005ea6179ae4afb95119450f001cde2a90f746990776f'
CAPTURES=['training-block-'+str(i) for i in (1,2,3)]
require=base.require


def checked(ref):
    require(set(ref)=={'path','sha256'} and receipt(ref['path'])==ref,'Immutable receipt changed: '+str(ref))
    return json.loads(Path(ref['path']).read_text())


def source_catalog(context,pool_path=POOL):
    """Verify actual unchanged bytes for all sixteen declared training inputs."""
    declared=context['declared_inputs'];wanted={r['pool_id'] for r in declared}
    require(len(declared)==16 and len(wanted)==16,'Exact sixteen-input training catalog required')
    rows={}
    with Path(pool_path).open() as stream:
        for line in stream:
            if any(pool_id in line for pool_id in wanted):
                row=json.loads(line)
                if row['pool_id'] in wanted:
                    require(row['pool_id'] not in rows,'Duplicated public pool identity')
                    rows[row['pool_id']]=row
    require(set(rows)==wanted,'Complete declared training pool required')
    for item in declared:
        row=rows[item['pool_id']];sample=row['sample'];task=item['task']
        require(row['task_type']==task,'Declared task differs from source pool')
        if task=='text':
            exact=hashlib.sha256(json.dumps({k:sample[k] for k in ('context','question')},
                sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            size=len(sample['context'])+len(sample['question'])
        else:
            exact=receipt(sample['input'])['sha256']
            # The original public catalog defines ASR size as duration_s;
            # content authentication is the actual media SHA above. Its .wav
            # suffix is not an encoding guarantee (some files contain FLAC).
            size=sample['duration_s'] if task=='asr' else Path(sample['input']).stat().st_size
        require(exact==item['exact_input_sha256'] and abs(size-item['input_size'])<1e-9,
                'Declared exact input SHA/size differs from actual source: '+item['pool_id'])
    return rows


def expected_inputs(context,pool):
    selected={}
    for task in ('text','ocr'):
        choices=sorted((r for r in context['declared_inputs'] if r['task']==task),key=lambda r:r['input_size'])
        selected[task]={}
        for size,item in zip(('short','medium','long'),(choices[0],choices[1],choices[-1])):
            row=dict(item,task=task,size_measure='text_input_characters' if task=='text' else 'source_image_bytes')
            row['sample_JSON_UTF8_bytes']=len(json.dumps(pool[item['pool_id']]['sample'],ensure_ascii=False).encode())
            row['wire_byte_measure']='Conditional UTF-8 text encoding; actual full envelope and encoding must be recorded at future send.'
            selected[task][size]=row
    return selected


def expected_base_blocks(inputs):
    definitions=base.cases(inputs);blocks=[]
    for capture_id in CAPTURES:
        cases=[]
        for definition in definitions:
            case=copy.deepcopy(definition)
            for i,row in enumerate(case['arrivals']):row.update(request_id=capture_id+':'+case['case_id']+':'+str(i),arrival_index=i)
            case.update(offered=len(case['arrivals']),public_arrivals_sha256=base.digest(case['arrivals']))
            cases.append(case)
        blocks.append(dict(capture_id=capture_id,cases=cases))
    return blocks


def compare_blocks(actual,expected):
    require(len(actual)==3 and [b['capture_id'] for b in actual]==CAPTURES,'Unique exact three capture IDs required')
    for block,wanted in zip(actual,expected):
        rows=block['cases'];ids=[c['case_id'] for c in rows]
        require(len(ids)==len(set(ids)) and set(ids)=={c['case_id'] for c in wanted['cases']},'Missing/extra/duplicate semantic case')
        expected_by_id={c['case_id']:c for c in wanted['cases']}
        require(all(c==expected_by_id[c['case_id']] for c in rows),'Case semantics/arrivals/IDs differ from reconstructed design')


def validate(manifest,ocr_manifest,*,context_path=CONTEXT,pool_path=POOL):
    """Accept case ordering changes only; rehashed semantic mutants still fail."""
    require(manifest['schema']==base.SCHEMA and ocr_manifest['schema']==entry.SCHEMA,'Exact new design schemas required')
    for design in (manifest,ocr_manifest):
        require(design['split']=='training' and design['resident_owners']==list(base.OWNERS),'Original training owner roster required')
        require(design['measurement_ns']==base.MEASUREMENT_NS and design['drain_ns']==base.DRAIN_NS,'Fixed10s/60s required')
        require(type(design['max_pending_per_model']) is int and design['max_pending_per_model']>=18,'Sufficient original host queue required')
        require(design['runtime_policy']==dict(admission=None,batching=False,weight_reuse=False,
            enqueue_all_without_waiting_for_RESULT_or_READY=True),'Original natural FIFO policy required')
        for ref in design['sources']:require(receipt(ref['path'])==ref,'Source drift')
        require(receipt(design['prepared_pool']['path'])==design['prepared_pool'],'Prepared pool receipt changed')
        require(receipt(context_path) in design['sources'] and receipt(pool_path) in design['sources'],'Frozen source context/pool binding missing')
    require(ocr_manifest['base_manifest']==dict(path=str(BASE),sha256=BASE_SHA),'Exact frozen v1 base receipt required')
    checked(ocr_manifest['base_manifest'])
    context=json.loads(Path(context_path).read_text())
    for src in context.get('execution_sources',[]):
        require(receipt(src['frozen']['path'])==src['frozen']
            and receipt(src['original_path'])['sha256']==src['frozen']['sha256'],'Original frozen execution source drift')
    pool=source_catalog(context,pool_path)
    inputs=expected_inputs(context,pool)
    for design in (manifest,ocr_manifest):
        require(design['inputs']==inputs,'Exact public input declarations differ from actual frozen context/pool')
        selected={v['pool_id'] for group in inputs.values() for v in group.values()}
        prepared=[json.loads(line) for line in Path(design['prepared_pool']['path']).read_text().splitlines()]
        require(len(prepared)==6 and {r['pool_id']:r for r in prepared}=={k:pool[k] for k in selected},'Prepared pool content differs')
    blocks=expected_base_blocks(inputs);compare_blocks(manifest['capture_blocks'],blocks)
    reconstructed=dict(inputs=inputs,capture_blocks=blocks)
    compare_blocks(ocr_manifest['capture_blocks'],entry.build_cases(reconstructed))
    return dict(status='CPU_SEMANTIC_PREFLIGHT_PASS',capture_blocks=3,cases_per_block=50,
        offered_per_block=182,total_offered=546,priming_per_attempt=18,GPU_started=False)


def load_frozen():
    manifest=checked(dict(path=str(BASE),sha256=BASE_SHA));ocr=checked(dict(path=str(ENTRY),sha256=ENTRY_SHA))
    validate(manifest,ocr)
    return manifest,ocr
