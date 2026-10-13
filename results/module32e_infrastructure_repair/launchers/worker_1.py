import sys,runpy
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/generalist-specialist/results/module32e_infrastructure_repair')
sys.argv=['/root/autodl-tmp/generalist-specialist/results/module32e_infrastructure_repair/restricted_worker.py','/root/autodl-tmp/generalist-specialist/candidate_pool_audit/worker_convnext_tiny.py']+sys.argv[1:]
runpy.run_path('/root/autodl-tmp/generalist-specialist/results/module32e_infrastructure_repair/restricted_worker.py',run_name='__main__')
