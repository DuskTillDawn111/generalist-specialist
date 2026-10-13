"""CPU/web metadata discovery; never imports a GPU framework."""
import concurrent.futures,datetime,hashlib,json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/module14_high_quality_heavy';SRC=OUT/'sources'
REPOS=['PaddlePaddle/PP-OCRv4_server_rec_doc','PaddlePaddle/PP-OCRv4_server_rec','PaddlePaddle/PP-OCRv5_server_rec','PaddlePaddle/ch_SVTRv2_rec','microsoft/trocr-large-str','microsoft/trocr-large-printed']
def get(repo):
    sess=requests.Session();sess.trust_env=False;errors=[]
    for host in ['hf-mirror.com','huggingface.co']:
        try:
            r=sess.get(f'https://{host}/api/models/{repo}',params={'blobs':'true'},timeout=60);r.raise_for_status();d=r.json();assert d.get('sha')
            (SRC/(repo.replace('/','--')+'.api.json')).write_text(json.dumps(d,indent=2)+'\n')
            print(repo,d['sha'],[(s['rfilename'],s.get('size')) for s in d['siblings'] if s['rfilename'].endswith(('.bin','.safetensors','.pdiparams','.pdmodel','.json','.yml'))],flush=True)
            for filename in ['README.md','config.json','generation_config.json','preprocessor_config.json','inference.yml']:
                if any(s['rfilename']==filename for s in d['siblings']):
                    rr=sess.get(f'https://{host}/{repo}/resolve/{d["sha"]}/{filename}',timeout=60);rr.raise_for_status();(SRC/(repo.replace('/','--')+'__'+filename)).write_bytes(rr.content)
            return {'model_id':repo,'revision':d['sha'],'success':True,'metadata_source':f'https://huggingface.co/{repo}/tree/{d["sha"]}'}
        except Exception as e:errors.append({'host':host,'reason':str(e).split('?')[0][:500]})
    print('METADATA_FAILED',repo,errors,flush=True);return {'model_id':repo,'success':False,'errors':errors}
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:records=list(ex.map(get,REPOS))
    (OUT/'discovery_metadata.json').write_text(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records},indent=2)+'\n')
