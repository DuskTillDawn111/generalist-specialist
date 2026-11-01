"""Native seqNo may wrap; event identity includes timestamp and driver context."""
import unittest

from tools.predictor_validation_20261027 import whole57_shared_observable_risk_v39 as v


def pair(start,end,context,seq):
    return dict(begin=dict(timestamp=start,context_key=context),
        end=dict(timestamp=end,context_key=context,seqNo=seq))


class NativeSaveIdentityTests(unittest.TestCase):
    def test_repeated_seqNo_never_cross_links_different_timestamp_or_context(self):
        first=pair(10,20,[1,0,10,100],129)
        second=pair(30,40,[1,0,11,200],129)
        third=pair(50,60,[1,0,10,100],129)
        index=v.index_save_pairs([first,second,third])
        self.assertEqual(len(index),3)
        for original in [first,second,third]:
            self.assertIs(index[v.save_identity(original['end'])],original)
            self.assertLess(index[v.save_identity(original['end'])]['begin']['timestamp'],original['end']['timestamp'])

    def test_exact_duplicate_and_nonpositive_native_pair_rejected(self):
        original=pair(10,20,[1,0,10,100],129)
        for values in [[original,original],[pair(20,20,[1,0,10,100],129)],
                [pair(21,20,[1,0,10,100],129)]]:
            with self.assertRaises(ValueError):v.index_save_pairs(values)


if __name__=='__main__':unittest.main()
