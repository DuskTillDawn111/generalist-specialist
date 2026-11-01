"""Real proc-shaped files plus cwd transition faults; no process launch."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.predictor_validation_20261027 import process_inspection_v1 as m


class ProcessInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for pid,module in ((123,'wanted.module'),(124,'unrelated.module')):
            p=self.root/str(pid);p.mkdir()
            (p/'cmdline').write_bytes(b'python\0-m\0'+module.encode()+b'\0run\0')
            fields=['S']+['0']*18+['98765']
            (p/'stat').write_text(f'{pid} (python) '+' '.join(fields))
            (p/'cwd').symlink_to(self.root,target_is_directory=True)

    def test_unrelated_unreadable_cwd_is_never_resolved(self):
        original=Path.resolve
        def resolve(path,*args,**kwargs):
            if path==self.root/'124/cwd':raise PermissionError('exec transition')
            return original(path,*args,**kwargs)
        with patch.object(Path,'resolve',resolve):rows=m.modules('wanted.module',proc_root=self.root)
        self.assertEqual([r['pid'] for r in rows],[123])
        self.assertEqual(rows[0]['start_ticks'],98765)

    def test_relevant_unreadable_cwd_remains_an_error(self):
        original=Path.resolve
        def resolve(path,*args,**kwargs):
            if path==self.root/'123/cwd':raise PermissionError('relevant process unknown')
            return original(path,*args,**kwargs)
        with patch.object(Path,'resolve',resolve),self.assertRaises(PermissionError):
            m.modules('wanted.module',proc_root=self.root)

    def test_disappearing_relevant_process_is_not_a_false_match(self):
        (self.root/'123/stat').unlink()
        self.assertEqual(m.modules('wanted.module',proc_root=self.root),[])

    def test_duplicate_relevant_module_retains_both_identities(self):
        (self.root/'124/cmdline').write_bytes(b'python\0-m\0wanted.module\0run\0')
        self.assertEqual(sorted(r['pid'] for r in m.modules('wanted.module',proc_root=self.root)),[123,124])


if __name__=='__main__':unittest.main()
