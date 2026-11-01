"""Exact v6 descriptor entry point for the shared CPU provenance engine."""
import argparse
import json

from .device_progress_actual_review_engine_v4 import review


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--raw-review',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(review(args.run,args.raw_review,args.output)))
