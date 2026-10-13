import sys,runpy
sys.path.insert(0,'/root/autodl-tmp/generalist-specialist/results/module35_utility_routing')
sys.argv=['/root/autodl-tmp/generalist-specialist/results/module35_utility_routing/restricted_worker.py','/root/autodl-tmp/generalist-specialist/candidate_pool_audit/worker_roberta_sst2.py']+sys.argv[1:]
runpy.run_path('/root/autodl-tmp/generalist-specialist/results/module35_utility_routing/restricted_worker.py',run_name='__main__')
