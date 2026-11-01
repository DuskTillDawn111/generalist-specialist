import inspect
import unittest
from types import SimpleNamespace

from tools.predictor_validation_20261027.semantic_observer import observe_language_forwards, language_module


class TensorShape:
    def __init__(self, *shape):
        self.shape = shape


class FakeModule:
    """CPU fake implementing PyTorch hook contracts, including always_call."""
    def __init__(self):
        self.before, self.after = [], []
        self.outputs = []

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, fail=False):
        if fail:
            raise RuntimeError("native forward failure")
        output = object()
        self.outputs.append(output)
        return output

    def __call__(self, *args, **kwargs):
        output = None
        try:
            for hook in self.before:
                hook(self, args, kwargs)
            output = self.forward(*args, **kwargs)
            return output
        finally:
            for hook in self.after:
                hook(self, args, kwargs, output)

    def register_forward_pre_hook(self, hook, *, with_kwargs):
        assert with_kwargs
        self.before.append(hook)
        return SimpleNamespace(remove=lambda: self.before.remove(hook))

    def register_forward_hook(self, hook, *, with_kwargs, always_call):
        assert with_kwargs and always_call
        self.after.append(hook)
        return SimpleNamespace(remove=lambda: self.after.remove(hook))


class FakeNvtx:
    def __init__(self):
        self.active, self.labels = [], []

    def range_push(self, name):
        self.active.append(name); self.labels.append(name)

    def range_pop(self):
        return self.active.pop()


class SemanticObserverTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModule()
        self.executor = SimpleNamespace(family="chatqa", model=self.model)
        self.nvtx = FakeNvtx()

    def test_prefill_decode_positions_and_multitoken_decode(self):
        original = self.model.forward
        signature = inspect.signature(self.model.forward)
        emitted = []
        with observe_language_forwards(self.executor, "profile:7", nvtx=self.nvtx, sink=emitted.append) as observer:
            first = self.model(TensorShape(1, 200))
            second = self.model(input_ids=TensorShape(4, 3), past_key_values=object())
            third = self.model(inputs_embeds=TensorShape(1, 1, 4096), past_key_values=object())
            self.assertIs(self.model.forward.__func__, original.__func__)
            self.assertEqual(inspect.signature(self.model.forward), signature)
        rows = observer.records
        self.assertEqual([r["semantic_phase"] for r in rows], ["prefill", "decode", "decode"])
        self.assertEqual([r["input_sequence_length"] for r in rows], [200, 3, 1])
        self.assertEqual(rows[1]["input_token_positions"], 12)
        self.assertFalse(rows[0]["past_key_values_present"])
        self.assertTrue(rows[1]["past_key_values_present"])
        self.assertEqual(self.nvtx.labels[0], "GS_STAGE:profile:7:gpu:prefill_0000")
        self.assertEqual(len(emitted), 6)
        self.assertEqual(self.model.outputs, [first, second, third])
        self.assertEqual(self.nvtx.active, [])
        self.assertEqual(self.model.before+self.model.after, [])

    def test_single_token_first_call_still_prefill(self):
        with observe_language_forwards(self.executor, "r") as observer:
            self.model(input_ids=TensorShape(1, 1))
        self.assertEqual(observer.records[0]["semantic_phase"], "prefill")

    def test_exception_cleans_hooks_and_nvtx_then_next_request_resets(self):
        with self.assertRaisesRegex(RuntimeError, "native forward failure"):
            with observe_language_forwards(self.executor, "bad", nvtx=self.nvtx) as observer:
                self.model(input_ids=TensorShape(1, 10), fail=True)
        self.assertIn("end_ns", observer.records[0])
        self.assertEqual(self.nvtx.active, [])
        self.assertEqual(self.model.before+self.model.after, [])
        with observe_language_forwards(self.executor, "next") as next_observer:
            self.model(TensorShape(1, 8))
        self.assertEqual(next_observer.records[0]["semantic_phase"], "prefill")

    def test_gptq_proxy_resolves_real_llm(self):
        inner = SimpleNamespace(llm=self.model)
        executor = SimpleNamespace(family="minicpm45", model=SimpleNamespace(model=inner))
        self.assertIs(language_module(executor), self.model)
        with observe_language_forwards(executor, "g2") as observer:
            self.model(inputs_embeds=TensorShape(1, 40, 2048))
        self.assertEqual(observer.records[0]["input_kind"], "inputs_embeds")

    def test_peft_generate_bypasses_wrapper_but_base_hooks_observe(self):
        class CausalLM(FakeModule):
            def generate(self, input_ids):
                return self(input_ids=input_ids)
        class PeftWrapper(FakeModule):
            __module__ = "peft.fake"
            def __init__(self, base):
                super().__init__()
                self.base_model = SimpleNamespace(model=base)
            def get_base_model(self):
                return self.base_model.model
            def generate(self, input_ids):
                return self.base_model.model.generate(input_ids)
        base = CausalLM()
        wrapper = PeftWrapper(base)
        executor = SimpleNamespace(family="chatqa", model=wrapper)
        self.assertIs(language_module(executor), base)
        with observe_language_forwards(executor, "peft:bypass") as observer:
            output = wrapper.generate(TensorShape(1, 12))
        self.assertIs(output, base.outputs[0])
        self.assertEqual(len(observer.records), 1)
        self.assertEqual(observer.records[0]["input_sequence_length"], 12)
        self.assertEqual(wrapper.before+wrapper.after+base.before+base.after, [])

    def test_chatqa_does_not_unwrap_decoder_or_quantization_submodules(self):
        model = FakeModule()
        model.model = FakeModule()
        executor = SimpleNamespace(family="chatqa", model=model)
        self.assertIs(language_module(executor), model)

    def test_existing_hooks_retained_and_nested_observer_rejected(self):
        existing = lambda *args: None
        self.model.before.append(existing)
        with observe_language_forwards(self.executor, "outer"):
            with self.assertRaisesRegex(ValueError, "already"):
                with observe_language_forwards(self.executor, "inner"):
                    pass
        self.assertEqual(self.model.before, [existing])

    def test_bad_sink_does_not_change_native_output(self):
        def broken(_):
            raise RuntimeError("telemetry sink")
        with observe_language_forwards(self.executor, "r", sink=broken) as observer:
            result = self.model(TensorShape(1, 5))
        self.assertIs(result, self.model.outputs[0])
        self.assertEqual(len(observer.errors), 2)

    def test_no_labels_or_tensor_values_inspected(self):
        class ShapeOnly(TensorShape):
            def __getattr__(self, name):
                raise AssertionError("tensor value read: " + name)
        with observe_language_forwards(self.executor, "r") as observer:
            self.model(ShapeOnly(1, 9))
        self.assertEqual(observer.records[0]["input_sequence_length"], 9)


if __name__ == "__main__":
    unittest.main()
