import unittest

from tools.queue_aware_20261023.policy import QueueAwarePolicy
from tools.queue_aware_20261023.guarded_policy import RateLimitedPolicy


class GuardTests(unittest.TestCase):
    def policy(self):
        aliases={'g':dict(family='generalist_2',supported_tasks=['text','image']),
                 's':dict(family='text',supported_tasks=['text'])}
        base=QueueAwarePolicy(aliases,
            {'generalist_2':{'text':{'x':1},'image':{'i':1}},'text':{'text':{'x':10}}},
            {'generalist_2':{'text':0,'image':0},'text':{'text':0}},
            {'L':1,'Q':1},{'text':100,'image':100})
        return RateLimitedPolicy(base,dict(schema='queue_aware_token_bucket_v1',task='text',
            family='generalist_2',rate_rps=.25,burst=1.))

    def empty(self):
        return {a:dict(outstanding=[],last_completed_ns=None) for a in ['g','s']}

    def test_cap_fallback_refill_and_reset(self):
        p=self.policy();snapshot=self.empty()
        self.assertEqual(p.choose('text','x',0,snapshot)['alias'],'g')
        decision=p.choose('text','x',1_000_000_000,snapshot)
        self.assertEqual(decision['alias'],'s')
        self.assertFalse(decision['rate_guard']['allowed'])
        self.assertEqual(decision['rate_guard']['bucket_after_tokens'],.25)
        self.assertEqual(p.choose('text','x',4_000_000_000,snapshot)['alias'],'g')
        p.reset()
        self.assertEqual(p.choose('text','x',0,snapshot)['alias'],'g')

    def test_background_uses_no_text_token_and_no_backwards_clock(self):
        p=self.policy();snapshot=self.empty()
        p.choose('text','x',0,snapshot)
        self.assertEqual(p.choose('image','i',1,snapshot)['alias'],'g')
        with self.assertRaisesRegex(ValueError,'monotonic'):
            p.choose('text','x',0,snapshot)

    def test_unknown_path_does_not_consume_and_all_interval_exposure_bound(self):
        p=self.policy();snapshot=self.empty()
        decision=p.choose('text','missing',0,snapshot)
        self.assertIsNone(decision['alias'])
        self.assertEqual(p.tokens,1)
        chosen=[]
        for i in range(101):
            now=i*100_000_000
            if p.choose('text','x',now,snapshot)['alias']=='g':chosen.append(now)
        for i,start in enumerate(chosen):
            for end in chosen[i:]:
                count=sum(start<=v<=end for v in chosen)
                self.assertLessEqual(count,1+.25*(end-start)/1e9+1e-12)


if __name__=='__main__':unittest.main()
