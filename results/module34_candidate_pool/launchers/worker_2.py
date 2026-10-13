import sys,runpy
sys.path.insert(0,'/root/autodl-tmp/generalist-specialist/results/module34_candidate_pool')
sys.argv=['/root/autodl-tmp/generalist-specialist/results/module34_candidate_pool/restricted_worker.py','/root/autodl-tmp/generalist-specialist/candidate_pool_v2_1_adapters/worker_minicpm.py']+sys.argv[1:]
runpy.run_path('/root/autodl-tmp/generalist-specialist/results/module34_candidate_pool/restricted_worker.py',run_name='__main__')
