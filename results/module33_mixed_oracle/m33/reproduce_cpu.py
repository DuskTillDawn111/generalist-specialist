"""Reproduce CPU oracle in a NEW isolated subdirectory; never overwrite evidence."""
import argparse,os,shutil
from pathlib import Path
from .common import OUT,ref,read

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);args=parser.parse_args()
    if not args.name.replace('_','').replace('-','').isalnum():raise ValueError('Name must be alphanumeric, underscore or hyphen')
    dest=OUT/'reproductions'/args.name
    if dest.exists():raise FileExistsError('Refuse to overwrite reproduction')
    dest.mkdir(parents=True)
    shutil.copyfile(OUT/'frozen_problem_manifest.json',dest/'frozen_problem_manifest.json')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from . import cpu,verify_cpu,common
    common.OUT=dest;cpu.OUT=dest;verify_cpu.OUT=dest
    cpu.main();verify_cpu.main()
    (dest/'REPRODUCTION_SOURCE.txt').write_text(str(OUT)+'\nNo GPU launched; historical inputs unchanged.\n')
if __name__=='__main__':main()
