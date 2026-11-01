"""Shared FIFO batch selection and actual one-call execution on CPU."""
import unittest
from gsdeploy.runtime.batching import BatchConfiguration, execute_selected_batch, select_batch


def row(rid, tokens, ready=0, **kw):
    return dict(request_id=rid, input_tokens=tokens, ready_ns=ready, owner="text",
                task="text", execution_key="greedy-v1", sample={"id": rid}, **kw)


class BatchContractTests(unittest.TestCase):
    def test_absolute_wait_does_not_reset_when_replanned(self):
        config = BatchConfiguration(2, 100, 10)
        self.assertEqual(select_batch([row("a", 8)], 3, config)["next_event_ns"], 10)
        self.assertEqual(select_batch([row("a", 8)], 9, config)["next_event_ns"], 10)
        selected = select_batch([row("a", 8)], 10, config)
        self.assertEqual(selected["formation_wait_ns"], {"a": 10})

    def test_full_batch_padding_and_per_request_wait(self):
        selected = select_batch([row("a", 5), row("b", 8, ready=4)], 4, BatchConfiguration(2, 16, 10))
        self.assertEqual(selected["members"], ["a", "b"])
        self.assertEqual(selected["padded_input_tokens"], 16)
        self.assertEqual(selected["padding_tokens"], 3)
        self.assertEqual(selected["formation_wait_ns"], {"a": 4, "b": 0})

    def test_fifo_bucket_barrier_flushes_head_without_bypass(self):
        selected = select_batch([row("a", 4), row("b", 9), row("c", 4)], 0,
                                BatchConfiguration(3, 100, 10, (4, 8, 16)))
        self.assertEqual(selected["members"], ["a"])
        self.assertEqual(selected["dispatch_reason"], "fifo_barrier")

    def test_padding_token_budget_is_not_sum_of_true_lengths(self):
        selected = select_batch([row("a", 4), row("b", 9)], 0, BatchConfiguration(3, 16, 10))
        self.assertEqual(selected["members"], ["a"])

    def test_oversize_requires_real_prefill_microbatch_executor(self):
        selected = select_batch([row("a", 17)], 0, BatchConfiguration(2, 16, 10))
        self.assertEqual(selected["status"], "UNSUPPORTED")

    def test_exact_selection_is_really_one_backend_call(self):
        class Backend:
            calls = []
            def run_batch(self, samples):
                self.calls.append(samples)
                return [s["id"].upper() for s in samples]
        rows = [row("a", 4), row("b", 9)]
        selection = select_batch(rows, 0, BatchConfiguration(2, 100, 10))
        backend = Backend()
        result = execute_selected_batch(selection, rows, backend)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual([r["result"] for r in result], ["A", "B"])
        self.assertEqual([r["request_id"] for r in result], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
