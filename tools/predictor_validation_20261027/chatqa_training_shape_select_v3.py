"""Select one authentic additional ChatQA training input using CPU tokenizer only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import chatqa_public_shapes_v3 as public_shapes
from .kernel_baseline_intake_v3 import checked, _verify
from .native_collect import receipt, save
from .smoke_verifier_v3 import load_cpu_adapter


def prepare(context_path, prior_selection_path, reservation_path, output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    context_source=receipt(context_path);context=checked(context_source)
    catalog_source=next(s['frozen'] for s in context['execution_sources']
        if s['original_path'].endswith('/runtime-inputs.jsonl'))
    _verify(catalog_source)
    texts=[r for r in (json.loads(line) for line in Path(catalog_source['path']).read_text().splitlines())
        if r['sample']['task']=='text']
    split_source=receipt(prior_selection_path);splits=checked(split_source)
    reservation_source=receipt(reservation_path);reservation=checked(reservation_source)
    if reservation['split']!='prospective_validation_reserved':raise ValueError('Explicit prospective reservation required')
    _verify(reservation['source'])
    if reservation['source']['sha256']!=catalog_source['sha256']:raise ValueError('Reserved domain uses a different catalog')
    reserved={r['pool_id'] for r in reservation['inputs'] if r['task']=='text'}
    excluded={r['pool_id'] for split in splits.values() for r in split['text']}|reserved
    if not excluded<={r['pool_id'] for r in texts}:raise ValueError('Excluded text input missing from immutable catalog')
    excluded_content={public_shapes.public_content_sha(r['sample']) for r in texts if r['pool_id'] in excluded}
    eligible=[r for r in texts if r['pool_id'] not in excluded and
        public_shapes.public_content_sha(r['sample']) not in excluded_content]
    tokenized,token_sources=public_shapes.tokenize_public(context,output,[r['sample'] for r in eligible])
    ranking=[dict(pool_id=r['pool_id'],public_content_sha256=public_shapes.public_content_sha(r['sample']),
        **tokenized[public_shapes.public_content_sha(r['sample'])]) for r in eligible]
    odd=sorted((r for r in ranking if r['input_tokens']%2 and 414<r['input_tokens']<4514),
        key=lambda r:(abs(r['input_tokens']-1025),r['input_tokens'],r['pool_id']))
    if not odd:raise ValueError('No genuine eligible odd input inside observed shape bracket')
    selected=odd[0];row=next(r for r in eligible if r['pool_id']==selected['pool_id'])
    save(output/'selected-input.json',row)
    binding=context['worker_bindings']['text'];spec=checked(dict(path=binding['spec'],sha256=binding['spec_sha256']))
    adapter=load_cpu_adapter(binding,spec)
    save(output/'selected-native-request.json',adapter.build(spec['task_candidates']['text'],row['sample']))
    result=dict(schema='gsdeploy.chatqa_new_shape_training_selection.v3',status='PREPARED_CPU_ONLY',
        purpose='ADDITIONAL_TRAINING_NOT_HELDOUT_VALIDATION',context=context_source,catalog=catalog_source,
        prior_split_selection=split_source,runtime_validation_reservation=reservation_source,
        reserved_runtime_validation_pool_ids=sorted(reserved),excluded_pool_ids=sorted(excluded),
        excluded_public_content_sha256=sorted(excluded_content),
        selection_rule='Exact1025 if available, otherwise nearest odd token count strictly inside observed414..4514 bracket; deterministic tie by length and pool ID',
        selected=selected,selected_input=receipt(output/'selected-input.json'),
        selected_native_request=receipt(output/'selected-native-request.json'),eligible_tokenized_inputs=ranking,
        tokenizer=token_sources,implementation_sources=[receipt(__file__),receipt(public_shapes.__file__)],
        known_prior_training_scope='Four text IDs in bound prior selection; no claim of global project-history novelty',
        GPU_workload_executed=False,model_weights_loaded=False,new_shape_execution_validated=False,
        validation_outputs_or_timings_consumed=False)
    save(output/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('context','prior-selection','reservation','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();result=prepare(args.context,args.prior_selection,args.reservation,args.output)
    print(json.dumps(result['selected']))


if __name__=='__main__':main()
