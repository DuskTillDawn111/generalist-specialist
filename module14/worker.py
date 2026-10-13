"""Native FP32 Image/OCR specialists on unchanged worker request timer."""
import argparse,ast,importlib.metadata,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'joint_deploy'));sys.path.insert(0,str(ROOT/'benchmark/scripts'))
from worker_protocol import run_worker
p=argparse.ArgumentParser();p.add_argument('--model-id',required=True);p.add_argument('--cpu-preflight',action='store_true');a=p.parse_args()
OUT=ROOT/'results/module14_high_quality_heavy';protocol=json.loads((OUT/'measurement_protocol_freeze.json').read_text());entry=next(e for e in protocol['entries'] if e['model_id']==a.model_id);directory=Path(entry['directory']);is_paddle=entry['backend']=='paddle';device='cpu' if a.cpu_preflight else 'gpu:0' if is_paddle else 'cuda:0'
from PIL import Image
if is_paddle:
 import paddle
 import eval_paddleocr_iiit5k as formal
 from paddleocr import TextRecognition
 paddle.set_device(device)
 # Same constructor/default inference backend as original OCR reference. HPI
 # is explicitly false (native default), and no extra pipeline stages exist.
 model=TextRecognition(model_name=a.model_id.split('/')[-1],model_dir=str(directory),device=device,enable_hpi=False)
 def sync():
  if not a.cpu_preflight:paddle.device.synchronize()
 param_count=None;attention='native Paddle static recognition';dtype=paddle.get_default_dtype()
 assert model._common_args['pptrt_precision']=='fp32' and not model._common_args['enable_hpi']
 metadata={'native_paddle_inference_precision':'float32 default; enable_hpi=False','native_common_args':model._common_args,'model_name':a.model_id.split('/')[-1]}
else:
 import torch
 torch.set_num_threads(1);torch.manual_seed(0)
 if entry['backend']=='torchvision':
  import torchvision
  from torchvision import models
  weights=getattr(getattr(models,entry['weight_enum']),entry['weight_member']);preprocess=weights.transforms()
  kwargs={'image_size':weights.meta['min_size'][0]} if entry['builder']=='vit_l_16' else {}
  model=getattr(models,entry['builder'])(weights=None,**kwargs)
  model.load_state_dict(torch.load(directory/entry['filename'],map_location='cpu',weights_only=True),strict=True)
  labels=['tench','English springer','cassette player','chain saw','church','French horn','garbage truck','gas pump','golf ball','parachute'];indices=[weights.meta['categories'].index(label) for label in labels]
  index_tensor=torch.tensor(indices,dtype=torch.long,device=device);attention='native torchvision default'
  metadata={'official_weight_transform':str(preprocess),'fixed_imagenet_indices':indices,'classes':1000,'task_mode':'closed-set10 after ImageNet logit gather'}
 else:
  from transformers import TrOCRProcessor,VisionEncoderDecoderModel
  processor=TrOCRProcessor.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
  model,info=VisionEncoderDecoderModel.from_pretrained(directory,local_files_only=True,trust_remote_code=False,dtype=torch.float32,output_loading_info=True)
  assert not info.get('missing_keys') and not info.get('mismatched_keys'),info
  metadata={'native_generation_config':model.generation_config.to_dict(),'loading_info':json.loads(json.dumps(info,default=lambda x:sorted(x) if isinstance(x,set) else repr(x))),'recognition_only':True,'max_generation_policy':'native pinned checkpoint defaults, no beam/length tuning'}
  attention={'encoder':getattr(model.encoder.config,'_attn_implementation',None),'decoder':getattr(model.decoder.config,'_attn_implementation',None)}
 model.eval().float().to(device);param_count=sum(p.numel() for p in model.parameters());dtype=str(next(model.parameters()).dtype)
 metadata.update(TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,cuda_initialized=torch.cuda.is_initialized(),torch_threads=torch.get_num_threads())
 def sync():
  if not a.cpu_preflight:torch.cuda.synchronize()
versions={}
for package in ['torch','torchvision','transformers','tokenizers','paddlepaddle-gpu','paddleocr','paddlex','numpy','pillow','safetensors']:
 try:versions[package]=importlib.metadata.version(package)
 except importlib.metadata.PackageNotFoundError:pass
metadata.update(model_id=a.model_id,task=entry['task'],directory=str(directory),device=device,dtype=dtype,parameter_count=param_count,framework_versions=versions,attention=attention,load_success=True)
if a.cpu_preflight:
 if not is_paddle:assert not torch.cuda.is_initialized()
 print('CPU_PREFLIGHT|'+json.dumps(metadata),flush=True);sys.exit(0)

def norm(text):return re.sub(r'[^a-z0-9]','',text.lower())
def handler(request):
 assert request['task_type']==entry['task'];sample=request['sample']
 if entry['backend']=='torchvision':
  with torch.inference_mode():
   with Image.open(sample['input']) as image:inputs=preprocess(image.convert('RGB')).unsqueeze(0).to(device)
   logits=model(inputs).index_select(1,index_tensor);prediction_id=int(logits.argmax(-1).item())
   result={'prediction':labels[prediction_id],'prediction_id':prediction_id,'reference':sample['label'],'reference_id':int(sample['label_id']),'imagenet_index':indices[prediction_id],'correct':prediction_id==int(sample['label_id'])}
 elif is_paddle:
  image=formal.load_image_bgr(sample['input']);output=model.predict(input=image,batch_size=1);text,score=formal.extract_result(output);prediction=formal.normalize_ocr(text);reference=formal.normalize_ocr(sample.get('normalized_label',sample['label']))
  result={'prediction':prediction,'reference':reference,'score':score,'correct':prediction==reference,'recognition_only':True,'raw_response':text}
 else:
  with torch.inference_mode():
   with Image.open(sample['input']) as image:inputs=processor(images=image.convert('RGB'),return_tensors='pt').pixel_values.to(device)
   ids=model.generate(inputs,do_sample=False);text=processor.batch_decode(ids,skip_special_tokens=True)[0];prediction=norm(text);reference=norm(sample.get('normalized_label',sample['label']))
   result={'prediction':prediction,'reference':reference,'correct':prediction==reference,'raw_response':text,'recognition_only':True}
 sync();return dict(result,model=a.model_id)
sync()
if not is_paddle:torch.cuda.reset_peak_memory_stats()
run_worker(a.model_id,handler,metadata);sync()
if not is_paddle:print('MEMORY_END|'+json.dumps({'allocated_current_mib':torch.cuda.memory_allocated()/1048576,'reserved_current_mib':torch.cuda.memory_reserved()/1048576,'allocated_peak_mib':torch.cuda.max_memory_allocated()/1048576,'reserved_peak_mib':torch.cuda.max_memory_reserved()/1048576}),flush=True)
