import sys,runpy
sys.path.insert(0,'/root/autodl-tmp/generalist-specialist/results/module35_utility_routing')
sys.argv=['/root/autodl-tmp/generalist-specialist/results/module35_utility_routing/restricted_worker.py','/root/autodl-tmp/generalist-specialist/joint_deploy/whisper_worker.py']+sys.argv[1:]
runpy.run_path('/root/autodl-tmp/generalist-specialist/results/module35_utility_routing/restricted_worker.py',run_name='__main__')
