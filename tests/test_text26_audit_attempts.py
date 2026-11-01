"""Audit discovery accepts numeric attempt directories, never sibling log files."""
import ast
from pathlib import Path
import tempfile
import unittest


class TextAttemptTests(unittest.TestCase):
    def test_only_numeric_attempt_directories_are_included(self):
        # Extract the pure path helper without importing the frozen research runtime.
        source=Path(__file__).resolve().parents[1]/'tools/spillover_discovery_20261022/audit_text26_development.py'
        function=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='attempt_directories')
        namespace={};exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
        with tempfile.TemporaryDirectory() as temp:
            runs=Path(temp);label='text26_n2_specialist_share_4of7'
            good=runs/(label+'-attempt1');good.mkdir();(good/'run.json').write_text('{}')
            (runs/(label+'-attempt1.log')).write_text('native worker log')
            (runs/(label+'-attempt2.log')).mkdir()
            (runs/(label+'-attempt3-extra')).mkdir()
            result=namespace['attempt_directories'](runs,label)
            self.assertEqual(result,[good])


if __name__=='__main__':unittest.main()
