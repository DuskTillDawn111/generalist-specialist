"""Resource-runner adapter for the already retained PP-OCRv6 native pipeline."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time


class Adapter:
    def __init__(self, spec, resources):
        source=Path(resources['inference_source'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==resources['inference_source_sha256']
        definition=importlib.util.spec_from_file_location('retained_ppocr_native',source)
        self.native=importlib.util.module_from_spec(definition);definition.loader.exec_module(self.native)
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        from transformers import AutoImageProcessor,AutoModelForObjectDetection,AutoModelForTextRecognition
        self.torch,self.cv2,self.np,self.Image=torch,cv2,np,Image
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.models={}
        for kind,cls in [('det',AutoModelForObjectDetection),('rec',AutoModelForTextRecognition)]:
            path=Path(resources['checkpoint'])/('PaddlePaddle--PP-OCRv6_medium_'+kind+'_safetensors')
            receipt=json.loads((path/'download-receipt.json').read_text())
            for f in receipt['files']:
                assert self.native.sha(path/f['file'])==f['sha256']
            model=cls.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32).eval().cuda()
            processor=AutoImageProcessor.from_pretrained(path,local_files_only=True,
                **({'limit_side_len':736} if kind=='det' else {}))
            self.models[kind]=model,processor
        params=[p for model,_ in self.models.values() for p in model.parameters()]
        self.parameter_count=sum(p.numel() for p in params)
        self.parameter_bytes=sum(p.numel()*p.element_size() for p in params)
        torch.cuda.synchronize()

    def memory(self):
        t=self.torch
        return dict(allocated_mib=t.cuda.memory_allocated()/2**20,
            reserved_mib=t.cuda.memory_reserved()/2**20,
            peak_allocated_mib=t.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=t.cuda.max_memory_reserved()/2**20)

    def run(self,sample):
        torch,cv2,np,Image=self.torch,self.cv2,self.np,self.Image
        torch.cuda.synchronize();start=time.perf_counter_ns()
        assert self.native.sha(sample['input'])==sample['input_sha256']
        with Image.open(sample['input']) as im: page=im.convert('RGB')
        array=np.asarray(page)
        det,dp=self.models['det'];rec,rp=self.models['rec']
        with torch.inference_mode():
            tensor=dp(images=page,return_tensors='pt').to('cuda')
            detection=det(**tensor)
            boxes=dp.post_process_object_detection(detection,
                target_sizes=torch.tensor([[page.height,page.width]]),threshold=.2,
                box_threshold=.45,unclip_ratio=1.4,max_candidates=3000)[0]
            regions=[]
            for i,points in enumerate(boxes['boxes'].cpu().tolist()):
                xs=[p[0] for p in points];ys=[p[1] for p in points]
                regions.append(dict(index=i,polygon=points,left=min(xs),
                    cy=(min(ys)+max(ys))/2,height=max(1,max(ys)-min(ys))))
            texts=[]
            for region in self.native.ordered(regions):
                image=Image.fromarray(self.native.crop(array,region['polygon'],cv2,np)[...,::-1].copy())
                batch=rp(images=[image],return_tensors='pt').to('cuda')
                logits=rec(**batch)
                assert torch.isfinite(logits.last_hidden_state).all()
                texts.extend(t['text'] for t in rp.post_process_text_recognition(logits))
            torch.cuda.synchronize()
        text='\n'.join(texts)
        return dict(task='ocr',sample_id=sample['id'],raw_output=text,prediction=text,
            output_sha256=hashlib.sha256(text.encode()).hexdigest(),success=bool(text.strip()),
            finish_reason='ctc_complete',service_ms=(time.perf_counter_ns()-start)/1e6)
