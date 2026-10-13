import sys,runpy
sys.path.insert(0,'/root/autodl-tmp/generalist-specialist/results/module32f_cuda_recovery')
sys.argv=['/root/autodl-tmp/generalist-specialist/results/module32f_cuda_recovery/restricted_worker.py','/root/autodl-tmp/generalist-specialist/results/module32_heavy_specialist/launchers/whisper_large_v3.py']+sys.argv[1:]
runpy.run_path('/root/autodl-tmp/generalist-specialist/results/module32f_cuda_recovery/restricted_worker.py',run_name='__main__')
