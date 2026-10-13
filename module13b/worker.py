"""Frozen task adapters on the unchanged project READY/RESULT/BYE timer."""
import argparse
import ast
import copy
import importlib.metadata
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'joint_deploy'))
sys.path.insert(0,str(ROOT/'benchmark/scripts'))
import torch
torch.set_num_threads(1)
torch.manual_seed(0)
original_tf32_policy=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
from worker_protocol import run_worker

parser=argparse.ArgumentParser()
parser.add_argument('--model-id',required=True)
parser.add_argument('--task',required=True,choices=['asr','lid','text'])
parser.add_argument('--directory',required=True)
parser.add_argument('--cpu-preflight',action='store_true')
args=parser.parse_args()
directory=Path(args.directory)
device=torch.device('cpu' if args.cpu_preflight else 'cuda:0')
is_sb=args.model_id.startswith('speechbrain/')

if is_sb:
    # Reuse the exact frozen source functions without importing the otherwise
    # unused Transformers runtime into the independent SpeechBrain environment.
    import soundfile as sf
    import types
    source=ast.parse((ROOT/'benchmark/scripts/eval_whisper_librispeech.py').read_text())
    nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ['normalize_asr','levenshtein_words','load_audio']]
    namespace={'re':re,'sf':sf}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<unchanged frozen audio/scoring functions>','exec'),namespace)
    asr=types.SimpleNamespace(**{n.name:namespace[n.name] for n in nodes})
    from speechbrain.inference.classifiers import EncoderClassifier
    model=EncoderClassifier.from_hparams(source=str(directory),savedir=str(ROOT/'models/module13b/speechbrain_runtime_cache'),
         overrides={'pretrained_path':str(directory)},run_opts={'device':str(device)})
    model.eval()
    for module in model.mods.values():module.float()
    assert len(model.hparams.label_encoder.ind2lab)==107
    model.hparams.label_encoder.expect_len(107)
    processor=None
else:
    import eval_whisper_librispeech as asr
    import eval_whisper_voxlingua as lid
    from transformers import (AutoProcessor,AutoTokenizer,AutoModelForSequenceClassification,
        AutoModelForSpeechSeq2Seq,AutoFeatureExtractor,AutoModelForAudioClassification,AutoModelForCTC)
    common={'local_files_only':True,'trust_remote_code':False}
    def load(cls):
        # This historical SST-2 artifact has a stale _num_labels=3 config,
        # while its frozen trained classification head contains exactly two
        # rows. Reconcile the constructor with those existing binary weights;
        # strict loading below still rejects any missing/mismatched head.
        repair={'num_labels':2} if args.model_id=='textattack/facebook-bart-large-SST-2' else {}
        loaded,info=cls.from_pretrained(directory,dtype=torch.float32,output_loading_info=True,**common,**repair)
        allowed={'wav2vec2.masked_spec_embed'} if 'wav2vec2-large' in args.model_id else set()
        assert set(info.get('missing_keys',[]))<=allowed,info
        assert not info.get('mismatched_keys'),info
        # The historical Wav2Vec2 checkpoint omits this training-only masking
        # vector. Native eval() never applies SpecAugment; no inference weights
        # or classifier head may be newly initialized.
        loaded._module13b_loading_info=json.loads(json.dumps(info,default=lambda x:sorted(x) if isinstance(x,set) else repr(x)))
        return loaded
    if args.task=='text':
        processor=AutoTokenizer.from_pretrained(directory,**common)
        model=load(AutoModelForSequenceClassification)
        assert model.config.num_labels==2
    elif args.model_id.startswith('facebook/mms-lid'):
        processor=AutoFeatureExtractor.from_pretrained(directory,**common)
        model=load(AutoModelForAudioClassification)
        required={'ara','deu','eng','spa','fas','fra','jpn','rus','tur','cmn'}
        assert required<=set(model.config.id2label.values())
    elif 'wav2vec2-large' in args.model_id:
        processor=AutoProcessor.from_pretrained(directory,**common)
        model=load(AutoModelForCTC)
    else:
        processor=AutoProcessor.from_pretrained(directory,**common)
        model=load(AutoModelForSpeechSeq2Seq)
        for conf in [model.config,model.generation_config]:
            if hasattr(conf,'forced_decoder_ids'):conf.forced_decoder_ids=None
        if args.task=='lid':
            assert callable(model.detect_language)
            assert all(f'<|{x}|>' in model.generation_config.lang_to_id for x in ['ar','de','en','es','fa','fr','ja','ru','tr','zh'])
    model.eval().to(device)

# SpeechBrain import enables TF32 globally. The13A protocol explicitly locks
# original PyTorch defaults, as used by the unchanged baseline runtime.
torch.backends.cuda.matmul.allow_tf32=original_tf32_policy[0]
torch.backends.cudnn.allow_tf32=original_tf32_policy[1]

def sync():
    if device.type=='cuda':torch.cuda.synchronize()

versions={}
for package in ['torch','transformers','tokenizers','numpy','soundfile','speechbrain','torchaudio','hyperpyyaml','huggingface-hub','sentencepiece','safetensors','protobuf']:
    try:versions[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:pass
params=list(model.parameters())
metadata={'model_id':args.model_id,'task':args.task,'directory':str(directory),
    'device':str(device),'dtype':str(params[0].dtype),'parameter_count':sum(p.numel() for p in params),
    'framework_versions':versions,'attention':getattr(getattr(model,'config',None),'_attn_implementation','native SpeechBrain'),
    'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32,
    'threads':torch.get_num_threads(),'cuda_initialized':torch.cuda.is_initialized(),'load_success':True,
    'native_num_labels':107 if is_sb else getattr(getattr(model,'config',None),'num_labels',None),
    'checkpoint_loading_info':getattr(model,'_module13b_loading_info',None),
    'random_seed':0}
if args.cpu_preflight:
    if args.task=='text':
        tokens=processor('An ordinary English sentence.',return_tensors='pt',truncation=True)
        metadata['tokenizer_sanity_shape']={k:list(v.shape) for k,v in tokens.items()}
    if is_sb:
        decoded=model.hparams.label_encoder.decode_ndim(torch.arange(107))
        codes={str(s).split(':')[0].strip().lower() for s in decoded}
        assert set(['ar','de','en','es','fa','fr','ja','ru','tr','zh'])<=codes
        metadata['speechbrain_all107_label_mapping_verified']=True
    assert not torch.cuda.is_initialized()
    print('CPU_PREFLIGHT|'+json.dumps(metadata),flush=True)
    sys.exit(0)

mapping={'ara':'ar','deu':'de','eng':'en','spa':'es','fas':'fa','fra':'fr','jpn':'ja','rus':'ru','tur':'tr','cmn':'zh'}

@torch.inference_mode()
def handler(request):
    assert request['task_type']==args.task
    sample=request['sample']
    if args.task=='text':
        inputs=processor(sample['input'],return_tensors='pt',truncation=True)
        inputs={k:v.to(device) for k,v in inputs.items()}
        pred_id=int(model(**inputs).logits.argmax(-1).item())
        prediction='positive' if pred_id==1 else 'negative'
        reference=sample['label'].strip().lower()
        result={'prediction':prediction,'reference':reference,'prediction_id':pred_id,'correct':prediction==reference}
    elif args.task=='asr':
        audio,sr=asr.load_audio(sample['input'])
        if 'wav2vec2-large' in args.model_id:
            inputs=processor(audio,sampling_rate=sr,return_tensors='pt',padding=True)
            inputs={k:v.to(device) for k,v in inputs.items()}
            ids=model(**inputs).logits.argmax(-1)
            text=processor.batch_decode(ids)[0];long_form=False
        else:
            inputs,long_form=asr.prepare_inputs(processor,audio,sr,float(sample['duration_s']),device)
            ids=asr.generate_transcription(model,inputs,long_form)
            text=processor.batch_decode(ids,skip_special_tokens=True)[0]
        sync()
        prediction=asr.normalize_asr(text)
        reference=asr.normalize_asr(sample.get('normalized_label',sample['label']))
        n=len(reference.split());errors=asr.levenshtein_words(reference.split(),prediction.split())
        result={'prediction':prediction,'reference':reference,'word_errors':errors,'reference_words':n,'wer':errors/n if n else 0.,'long_form':long_form}
    elif is_sb:
        audio,sr=asr.load_audio(sample['input'])
        signal=torch.from_numpy(audio).unsqueeze(0).to(device)
        _,_,_,labels=model.classify_batch(signal)
        prediction=str(labels[0]).split(':')[0].strip().lower()
        reference=sample['label'].strip().lower()
        sync()
        result={'prediction':prediction,'reference':reference,'correct':prediction==reference,'native_label':str(labels[0]),'native_class_count':107}
    elif args.model_id.startswith('facebook/mms-lid'):
        audio,sr=asr.load_audio(sample['input'])
        inputs=processor(audio,sampling_rate=sr,return_tensors='pt')
        inputs={k:v.to(device) for k,v in inputs.items()}
        index=int(model(**inputs).logits.argmax(-1).item())
        native=model.config.id2label[index];prediction=mapping.get(native,native).strip().lower()
        reference=sample['label'].strip().lower();sync()
        result={'prediction':prediction,'reference':reference,'correct':prediction==reference,'native_label':native,'native_class_count':model.config.num_labels}
    else:
        # Same complete acoustic frontend/detect_language/scorer as reference.
        result=lid.run_one(model=model,processor=processor,sample=sample,device=device)
    sync()
    return dict(result,model=args.model_id)

sync()
# Reset maxima before the timed region, without introducing a control request
# into the frozen worker protocol or adding GPU allocations.
torch.cuda.reset_peak_memory_stats()
run_worker(args.model_id,handler,metadata)
sync()
print('MEMORY_END|'+json.dumps({'allocated_current_mib':torch.cuda.memory_allocated()/1048576,
    'reserved_current_mib':torch.cuda.memory_reserved()/1048576,
    'allocated_peak_mib':torch.cuda.max_memory_allocated()/1048576,
    'reserved_peak_mib':torch.cuda.max_memory_reserved()/1048576}),flush=True)
