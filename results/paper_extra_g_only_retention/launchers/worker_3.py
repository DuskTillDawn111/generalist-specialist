import sys,runpy
sys.path.insert(0,'/root/autodl-tmp/generalist-specialist/results/paper_extra_g_only_retention')
sys.argv=['/root/autodl-tmp/generalist-specialist/results/paper_extra_g_only_retention/restricted_worker.py','/root/autodl-tmp/generalist-specialist/joint_deploy/whisper_worker.py']+sys.argv[1:]
runpy.run_path('/root/autodl-tmp/generalist-specialist/results/paper_extra_g_only_retention/restricted_worker.py',run_name='__main__')
