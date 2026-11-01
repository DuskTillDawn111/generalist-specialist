"""Input-independent randomized exploration; all coins and guard fallbacks replay.

The original local policy estimates only select the specialist fallback. A coin
can expose G2 to easy and hard inputs; it is not an optimized routing policy.
"""
import copy
import hashlib
import random


class ProbePolicy:
    def __init__(self,base,spec,*,seed,phase):
        self.base=base;self.aliases=base.aliases;self.spec=copy.deepcopy(spec)
        material=f"{spec['namespace']}:{seed}:{phase}".encode()
        self.rng_seed=hashlib.sha256(material).hexdigest()
        self.rng=random.Random(int(self.rng_seed,16));self.index=0

    def choose(self,task,pool_id,now_ns,snapshot):
        result=copy.deepcopy(self.base.choose(task,pool_id,now_ns,snapshot))
        coin=self.rng.random();self.index+=1
        wants=coin<self.spec['probability']
        for c in result['audit']:
            if c.get('score') is None:continue
            c['pre_probe_score']=c['score']
            if c['family']=='generalist_2':
                c['score']=-1.0 if wants else None
                c['status']='RANDOMIZED_PROBE' if wants else 'PROBE_NOT_SELECTED'
        valid=[c for c in result['audit'] if c.get('score') is not None]
        chosen=min(valid,key=lambda c:(c['score'],c['predicted_completion_ms'],c['alias']))
        result.update(alias=chosen['alias'],probe=dict(spec=self.spec,stream_sha256=self.rng_seed,
            draw_index=self.index,coin=coin,requested_probe=wants,
            rule='One coin per text arrival; no reroll after guard rejection; independent of input pool/quality'),
            full_U=None,choice_scope='Randomized calibration exposure with specialist balancing fallback')
        return result
