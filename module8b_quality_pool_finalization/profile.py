"""Reuse the actual unchanged Module 7A-2 wrapper and Stage-1 main.
Only authorization/output namespace and versioned launch registry differ.
"""
import argparse
from .common import POOL_OUT,registry
from candidate_pool_service_demand import profile as previous
if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--pool',required=True)
 parser.add_argument('--target-worker',required=True)
 parser.add_argument('--target-task',required=True)
 parser.add_argument('--target-count',type=int,default=30)
 parser.add_argument('--output',required=True)
 parser.add_argument('--timeout-s',type=float,default=3600.)
 previous.OUT=POOL_OUT
 previous.registry=registry
 previous.run(parser.parse_args())
