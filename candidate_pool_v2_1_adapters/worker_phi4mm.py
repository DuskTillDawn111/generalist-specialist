from pathlib import Path
import os, runpy
ROOT = Path(__file__).resolve().parents[1]
os.environ['AUDIT_ORIGINAL_WORKER'] = str(ROOT / 'joint_deploy/phi_worker.py')
runpy.run_path(str(ROOT / 'candidate_pool_v2_1_adapters/vqa_extension.py'), run_name='__main__')
