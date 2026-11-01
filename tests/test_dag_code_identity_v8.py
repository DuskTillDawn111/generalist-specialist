"""Interpreter string/reference representation must not invalidate continuation."""
import marshal
import types
import unittest

from tools.predictor_validation_20261027.dag_checkpoint_v7 import code_identity, digest


def template():
    return ('constant with spaces', 'constant with spaces')


class CodeIdentityTests(unittest.TestCase):
    def test_equal_constant_values_with_different_reference_sharing_are_equal(self):
        first = ''.join(['constant ', 'with spaces'])
        second = ''.join(['constant with ', 'spaces'])
        self.assertIsNot(first, second)
        shared = template.__code__.replace(co_consts=(None, (first, first)))
        distinct = template.__code__.replace(co_consts=(None, (first, second)))
        self.assertEqual(types.FunctionType(shared, {})(), types.FunctionType(distinct, {})())
        self.assertNotEqual(marshal.dumps(shared), marshal.dumps(distinct))
        self.assertEqual(code_identity(shared), code_identity(distinct))

    def test_instruction_constant_nested_code_and_exception_metadata_still_bind(self):
        original = template.__code__
        changed = original.replace(co_consts=(None, ('different', 'values')))
        self.assertNotEqual(digest(code_identity(original)), digest(code_identity(changed)))
        nested_a = compile('def outer():\n def inner(): return 1\n return inner\n', __file__, 'exec')
        nested_b = compile('def outer():\n def inner(): return 2\n return inner\n', __file__, 'exec')
        self.assertNotEqual(code_identity(nested_a), code_identity(nested_b))
        if hasattr(original, 'co_exceptiontable'):
            self.assertNotEqual(code_identity(original),
                                code_identity(original.replace(co_exceptiontable=b'\x00')))

    def test_execution_does_not_change_semantic_identity(self):
        before = digest(code_identity(template.__code__))
        for _ in range(100):
            template()
        self.assertEqual(before, digest(code_identity(template.__code__)))
