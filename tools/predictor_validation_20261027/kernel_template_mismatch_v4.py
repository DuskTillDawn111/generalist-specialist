"""Receipt-bound pattern attribution of a completed actual template comparison."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .kernel_baseline_intake_v3 import checked,_verify
from .native_collect import receipt,save


def analyze(comparison_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source=receipt(comparison_path);comparison=checked(source);model=checked(comparison['model'])
    for source_file in comparison['implementation_sources']:_verify(source_file)
    records=[]
    for result in comparison['comparisons']:
        actual=checked(result['observation'])['ordered_launch_signatures']
        candidate_source=next(s for s in model['candidate_instances'] if
            Path(s['path']).name==f'candidate-M{result["input_tokens"]}-{result["policy"]}.json')
        predicted=checked(candidate_source)['ordered_launch_signatures'];counts=Counter();mismatches={}
        for position,(left,right) in enumerate(zip(predicted,actual)):
            pattern=model['patterns'][model['slot_pattern_indices'][position]];same=left==right
            counts[pattern['kind']+(':MATCH' if same else ':MISMATCH')]+=1
            if not same:
                key=pattern['id']
                if key not in mismatches:mismatches[key]=dict(pattern_id=key,kind=pattern['kind'],
                    positions=[],predicted=left,actual=right,source_formula_verified=pattern['source_formula_verified'])
                if mismatches[key]['actual']!=right:raise ValueError('One pattern has multiple actual variants; split analysis required')
                mismatches[key]['positions'].append(position)
        records.append(dict(alias=result['alias'],request_id=result['request_id'],policy=result['policy'],
            kind_match_counts=dict(counts),mismatch_patterns=list(mismatches.values())))
    report=dict(schema='gsdeploy.kernel_template_pattern_mismatch_analysis.v4',comparison=source,
        model=comparison['model'],patterns=records,measured_durations_used=False,model_refitted=False,
        context_relationship=comparison['context_relationship'],GPU_workload_executed=False,
        implementation_sources=[receipt(__file__)])
    save(output/'report.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comparison',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.comparison,a.output)


if __name__=='__main__':main()
