import sys,runpy
from pathlib import Path
ROOT=Path('/root/autodl-tmp/generalist-specialist')
sys.path.insert(0,str(ROOT))
sys.argv=[str(ROOT/'module13b/worker.py'),'--model-id','openai/whisper-large-v3','--task','asr','--directory',str(ROOT/'models/module13b/openai--whisper-large-v3')]
runpy.run_path(sys.argv[0],run_name='__main__')
